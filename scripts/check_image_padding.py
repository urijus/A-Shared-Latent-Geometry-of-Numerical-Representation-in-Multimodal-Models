"""CPU regression check for mixed-length image DAS batches.

Uses the production function bodies without importing model-loading dependencies.
"""

import ast
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = Path(__file__).resolve().parents[1] / "src/experiments/arithmetic_reference/das/image/das_image_core.py"
NAMES = {
    "inputs_to_device", "input_lengths", "padding_offsets", "patched_forward",
    "resolve_position_spec", "resolve_batch_positions", "resolve_prompt_positions", "answer_token_positions",
    "sequence_scores", "teacher_forced_batch_image",
}
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
tree.body = [node for node in tree.body if (isinstance(node, ast.FunctionDef) and node.name in NAMES)
             or (isinstance(node, ast.ClassDef) and node.name == "CachedGemma4Images")]
scope = {"torch": torch, "ExitStack": ExitStack, "OrderedDict": OrderedDict}
exec(compile(tree, str(SOURCE), "exec"), scope)


class Processor:
    def __init__(self, padding_side):
        self.padding_side = padding_side

    def __call__(self, texts, images):
        ids = [[ord(char) % 90 for char in text] for text in texts]
        width = max(map(len, ids))
        padded, masks = [], []
        for row in ids:
            extra = width - len(row)
            if self.padding_side == "left":
                padded.append([99] * extra + row)
                masks.append([0] * extra + [1] * len(row))
            else:
                padded.append(row + [99] * extra)
                masks.append([1] * len(row) + [0] * extra)
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(masks)}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Identity()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.activations = self.block(input_ids.float().unsqueeze(-1))
        ids = self.activations.squeeze(-1).long().clamp(0, 127)
        return SimpleNamespace(logits=torch.nn.functional.one_hot(ids, 128).float() * 4)


class CopySubspace:
    def patch(self, base, donor):
        return donor


scope.update({
    "make_inputs": lambda processor, texts, images: processor(texts, images),
    "sample_prompt": lambda processor, sample, prompt, enable_thinking: sample["prompt"],
    "load_rgb_image": lambda path: path,
    "image_path_for": lambda sample, root: sample["image_path"],
    "target_answers": lambda base, source, target: (
        base["answer"], source["answer"], (0, len(base["answer"])),
        (0, len(source["answer"])),
    ),
    "hook_module": lambda block, hook: (block, False),
    "hidden": lambda tensor: tensor,
    "replace_hidden": lambda original, updated: updated,
})


def check(padding_side):
    processor = Processor(padding_side)
    model = Model()
    pairs = [
        {"base": {"prompt": "A2", "answer": "2", "image_path": "a"},
         "source": {"prompt": "B1", "answer": "1", "image_path": "b"}},
        {"base": {"prompt": "AAA10", "answer": "10", "image_path": "c"},
         "source": {"prompt": "BB11", "answer": "11", "image_path": "d"}},
    ]
    args = (model, processor, None, [model.block], {"1": CopySubspace()}, [1],
            "resid_post", pairs, Path("."), "", -1, "result", False)
    batch = scope["teacher_forced_batch_image"](*args)
    batch_hidden = model.activations.detach().clone()
    prompts = ([item["base"]["prompt"] for item in pairs] * 2
               + [item["source"]["prompt"] for item in pairs])
    answers = ([item["source"]["answer"] for item in pairs]
               + [item["base"]["answer"] for item in pairs]
               + [item["source"]["answer"] for item in pairs])
    full_encoding = processor([prompt + answer for prompt, answer in zip(prompts, answers)], [None] * 6)
    offsets = scope["padding_offsets"](full_encoding)
    for position in (-1, -2, "last_input", 0):
        resolved = scope["resolve_prompt_positions"](
            processor, None, model, prompts, [None] * 6, position, full_encoding
        )
        expected = [
            offset + (len(prompt) - 1 if position == "last_input" else
                      len(prompt) + position if position < 0 else position)
            for prompt, offset in zip(prompts, offsets)
        ]
        assert resolved == expected
    individual = []
    for base_index, pair in enumerate(pairs):
        one = scope["teacher_forced_batch_image"](*args[:7], [pair], *args[8:])
        individual.append(one)
        one_hidden = model.activations.detach().clone()
        one_prompts = [pair["base"]["prompt"]] * 2 + [pair["source"]["prompt"]]
        one_answers = [pair["source"]["answer"], pair["base"]["answer"], pair["source"]["answer"]]
        one_encoding = processor([prompt + answer for prompt, answer in zip(one_prompts, one_answers)], [None] * 3)
        one_offsets = scope["padding_offsets"](one_encoding)
        for group, prompt in enumerate(one_prompts):
            batch_row = group * len(pairs) + base_index
            batch_position = offsets[batch_row] + len(prompt) - 1
            one_position = one_offsets[group] + len(prompt) - 1
            assert torch.equal(batch_hidden[batch_row, batch_position], one_hidden[group, one_position])
        batch_position = offsets[base_index] + len(pair["base"]["prompt"]) - 1
        answer_positions = list(range(batch_position + 1, batch_position + 1 + len(pair["source"]["answer"])))
        assert full_encoding["input_ids"][base_index, answer_positions].tolist() == [
            ord(char) % 90 for char in pair["source"]["answer"]
        ]
        assert torch.allclose(batch["source_variable_logprob"][base_index],
                              one["source_variable_logprob"][0], atol=1e-6)
        assert torch.equal(batch["variable_teacher_forced_iia"][base_index],
                           one["variable_teacher_forced_iia"][0])
    assert torch.allclose(batch["loss"], torch.stack([row["loss"] for row in individual]).mean(), atol=1e-6)


def check_processor_cache():
    class Batch(dict):
        def __init__(self, data):
            super().__init__(data)

    class ImageProcessor:
        calls = 0

        def __call__(self, images, **kwargs):
            self.calls += len(images)
            return Batch({"pixel_values": torch.tensor([[image.value] for image in images]),
                          "num_soft_tokens_per_image": [1] * len(images)})

    raw = ImageProcessor()
    cached = scope["CachedGemma4Images"](raw)
    a, b = SimpleNamespace(value=3), SimpleNamespace(value=7)
    expected = raw([a, b, a], return_tensors="pt")
    raw.calls = 0
    actual = cached([a, b, a], return_tensors="pt")
    cached([a, b, a], return_tensors="pt")
    assert torch.equal(actual["pixel_values"], expected["pixel_values"])
    assert actual["num_soft_tokens_per_image"] == expected["num_soft_tokens_per_image"]
    assert raw.calls == 2


if __name__ == "__main__":
    for side in ("left", "right"):
        check(side)
    check_processor_cache()
    print("Mixed one/two-digit image batch matches separate runs for both padding sides.")
