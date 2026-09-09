import argparse
from collections import Counter

from transformers import AutoProcessor, AutoTokenizer

from src.models import is_ministral3_model_name, resolve_model_for_loading
DEFAULT_EXAMPLES = ["2 + 4", "9 + 8", "10 + 3", "7 + 42", "98 + 3", "99 + 99"]
DEFAULT_COMPOSITE_EXAMPLES = [
    "(2 + 3) * (1 - 4) =",
    "(1 + (2 - 3)) * 4 =",
    "(12 - 3) * (4 + (5 * 6)) =",
]


def safe_repr(value):
    return ascii(value)


def tokenize(tokenizer, text, offsets=False):
    kwargs = {"add_special_tokens": False}
    if offsets:
        kwargs["return_offsets_mapping"] = True
    try:
        return tokenizer(text, **kwargs)
    except TypeError:
        kwargs.pop("return_offsets_mapping", None)
        encoding = tokenizer(text, **kwargs)
        if offsets and "offset_mapping" not in encoding:
            encoding["offset_mapping"] = [(None, None)] * len(encoding["input_ids"])
        return encoding


def token_label(tokenizer, token_id):
    try:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
    except (AttributeError, TypeError):
        token = None
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    return token if token is not None else decoded


def token_labels(tokenizer, token_ids):
    return [token_label(tokenizer, token_id) for token_id in token_ids]


def print_digits(tokenizer, max_number):
    print("\n1. DIGIT REPRESENTATION")
    print("number | without space | with leading space")
    for number in range(max_number + 1):
        plain = tokenize(tokenizer, str(number))["input_ids"]
        spaced = tokenize(tokenizer, f" {number}")["input_ids"]
        print(
            f"{number:>6} | {safe_repr(token_labels(tokenizer, plain)):<24} "
            f"| {safe_repr(token_labels(tokenizer, spaced))}"
        )


def classify_number_tokenization(tokenizer, number):
    text = str(number)
    ids = tokenize(tokenizer, text)["input_ids"]
    decoded_pieces = [tokenizer.decode([token_id]) for token_id in ids]
    if len(ids) == 1:
        return "one_token"
    if "".join(piece.strip() for piece in decoded_pieces) == text and all(
        piece.strip().isdigit() and len(piece.strip()) == 1
        for piece in decoded_pieces
    ):
        return "digit_tokens"
    return "other"


def print_number_pattern_summary(tokenizer):
    print("\n1b. ASSISTANT NUMERICAL ANSWER TOKENIZATION SUMMARY")
    for start, end, label in ((0, 9, "0-9"), (10, 99, "10-99")):
        rows = [
            (number, classify_number_tokenization(tokenizer, number))
            for number in range(start, end + 1)
        ]
        counts = Counter(pattern for _number, pattern in rows)
        print(f"{label}: {dict(counts)}")
        for pattern in ("one_token", "digit_tokens", "other"):
            examples = [number for number, row_pattern in rows if row_pattern == pattern][:10]
            if examples:
                print(f"  {pattern}: examples={examples}")


def print_prompt_tokens(tokenizer, text):
    encoding = tokenize(tokenizer, text, offsets=True)
    ids = encoding["input_ids"]
    tokens = token_labels(tokenizer, ids)

    for index, (token_id, token, offset) in enumerate(
        zip(ids, tokens, encoding["offset_mapping"])
    ):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        print(
            f"  {index:>2}  id={token_id:<8} token={safe_repr(token):<18} "
            f"decoded={safe_repr(decoded):<12} offset={tuple(offset)}"
        )


def print_alignment(tokenizer, text, arithmetic_tokens):
    encoding = tokenize(tokenizer, text, offsets=True)
    alignment_start = text.rfind("[INST]")
    cursor = alignment_start if alignment_start >= 0 else 0
    dataset_spans = []
    for token in arithmetic_tokens:
        start = text.index(token, cursor)
        end = start + len(token)
        dataset_spans.append((start, end))
        cursor = end

    token_to_hf_positions = []
    for dataset_start, dataset_end in dataset_spans:
        matching_positions = []
        for hf_position, (hf_start, hf_end) in enumerate(encoding["offset_mapping"]):
            if hf_start is None or hf_end is None:
                continue
            if hf_start < dataset_end and hf_end > dataset_start:
                matching_positions.append(hf_position)
        if not matching_positions:
            raise ValueError(
                f"No HF token found for dataset span {(dataset_start, dataset_end)} "
                f"in expression: {text}"
            )
        token_to_hf_positions.append(matching_positions)

    alignment = {
        "input_ids": encoding["input_ids"],
        "token_to_hf_positions": token_to_hf_positions,
    }
    hf_tokens = token_labels(tokenizer, alignment["input_ids"])

    for token, positions in zip(
        arithmetic_tokens,
        alignment["token_to_hf_positions"],
    ):
        mapped = [hf_tokens[position] for position in positions]
        print(
            f"  {safe_repr(token):<4} -> positions={positions!s:<10} "
            f"tokens={safe_repr(mapped):<24} chosen(last)={positions[-1]}"
        )


def parse_example(raw):
    a, b = raw.split("+", maxsplit=1)
    return int(a), int(b)


def make_prompt(tokenizer, expression, prompt, use_chat_template):
    text = f"{prompt} {expression}" if prompt else expression
    if not use_chat_template:
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def print_pre_generation_position(tokenizer, text):
    encoding = tokenize(tokenizer, text)
    ids = encoding["input_ids"]
    if not ids:
        print("Final pre-generation token: <empty prompt>")
        return

    final_position = len(ids) - 1
    token_id = ids[final_position]
    print(
        "Final pre-generation token: "
        f"position={final_position}, id={token_id}, "
        f"token={safe_repr(token_label(tokenizer, token_id))}, "
        f"decoded={safe_repr(tokenizer.decode([token_id], skip_special_tokens=False))}"
    )


def load_tokenizer(model_path, model_name):
    if is_ministral3_model_name(model_name) or is_ministral3_model_name(model_path):
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                fix_mistral_regex=True,
            )
        except ImportError as exc:
            raise ImportError(
                "Ministral 3 tokenization requires mistral-common. Install "
                "mistral-common>=1.8.6 in the active environment."
            ) from exc
        return getattr(processor, "tokenizer", processor)
    return AutoTokenizer.from_pretrained(model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gemma4_12b_it"])
    parser.add_argument("--max_number", type=int, default=20)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--examples", nargs="+", default=DEFAULT_EXAMPLES)
    parser.add_argument(
        "--composite_examples",
        nargs="+",
        default=DEFAULT_COMPOSITE_EXAMPLES,
    )
    parser.add_argument("--use_chat_template", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    for model_name in args.models:
        model_path, saved_name = resolve_model_for_loading(model_name)
        tokenizer = load_tokenizer(model_path, model_name)

        print("\n" + "=" * 90)
        print(f"MODEL: {model_name} ({saved_name})")
        print(f"TOKENIZER: {model_path}")

        print_digits(tokenizer, args.max_number)
        print_number_pattern_summary(tokenizer)

        prompts = []
        print("\n2. FULL PROMPT REPRESENTATION")
        for number, raw_example in enumerate(args.examples, start=1):
            a, b = parse_example(raw_example)
            expression = f"{a} + {b} ="
            text = make_prompt(
                tokenizer,
                expression,
                args.prompt,
                args.use_chat_template,
            )
            prompts.append((expression, text, [str(a), "+", str(b), "="]))
            print(f"\nExample {number}: {expression}")
            print(f"model text: {safe_repr(text)}")
            print_prompt_tokens(tokenizer, text)
            print_pre_generation_position(tokenizer, text)

        print("\n3. COMPOSITE EXPRESSION TOKENIZATION")
        for number, expression in enumerate(args.composite_examples, start=1):
            text = make_prompt(
                tokenizer,
                expression,
                args.prompt,
                args.use_chat_template,
            )
            print(f"\nComposite example {number}: {expression}")
            print(f"model text: {safe_repr(text)}")
            print_prompt_tokens(tokenizer, text)
            print_pre_generation_position(tokenizer, text)

        print("\n4. OFFSET ALIGNMENT")
        for number, (expression, text, arithmetic_tokens) in enumerate(
            prompts,
            start=1,
        ):
            print(f"\nExample {number}: {expression}")
            print_alignment(tokenizer, text, arithmetic_tokens)


if __name__ == "__main__":
    main()
