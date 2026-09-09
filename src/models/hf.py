import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ALIASES = {
    "pythia": {
        "hf_name": "EleutherAI/pythia-6.9b",
        "local_dirs": ["models/pythia-6.9b"],
        "slug": "pythia_6_9b",
    },
    "qwen3_8b": {
        "hf_name": "Qwen/Qwen3-8B",
        "local_dirs": ["models/Qwen3-8B", "models/qwen3-8b"],
        "slug": "qwen3_8b",
    },
    "qwen3_5_9b_base": {
        "hf_name": "Qwen/Qwen3.5-9B-Base",
        "local_dirs": ["models/Qwen3.5-9B-Base"],
        "slug": "qwen3_5_9b_base",
    },
    "gemma3_12b": {
        "hf_name": "google/gemma-3-12b-it",
        "local_dirs": ["models/gemma-3-12b-it", "models/Gemma-3-12B-IT"],
        "slug": "gemma3_12b",
    },
    "gemma4_12b": {
        "hf_name": "google/gemma-4-12B",
        "local_dirs": ["models/gemma-4-12B", "models/Gemma-4-12B"],
        "slug": "gemma4_12b",
    },
    "gemma4_12b_it": {
        "hf_name": "google/gemma-4-12B-it",
        "local_dirs": ["models/gemma-4-12B-it", "models/Gemma-4-12B-IT"],
        "slug": "gemma4_12b_it",
    },
    "gemma4_e4b": {
        "hf_name": "google/gemma-4-E4B",
        "local_dirs": ["models/gemma-4-E4B", "models/Gemma-4-E4B"],
        "slug": "gemma4_e4b",
    },
    "gemma4_e4b_it": {
        "hf_name": "google/gemma-4-E4B-it",
        "local_dirs": ["models/gemma-4-E4B-it", "models/Gemma-4-E4B-IT"],
        "slug": "gemma4_e4b_it",
    },
    "llama3_1_8b": {
        "hf_name": "meta-llama/Llama-3.1-8B",
        "local_dirs": [
            "models/Llama-3.1-8B",
            "models/llama-3.1-8b",
        ],
        "slug": "llama3_1_8b",
    },
    "glm4_1v_9b_base": {
        "hf_name": "zai-org/GLM-4.1V-9B-Base",
        "local_dirs": ["models/GLM-4.1V-9B-Base"],
        "slug": "glm4_1v_9b_base",
    },
    "smolvlm_base": {
        "hf_name": "HuggingFaceTB/SmolVLM-Base",
        "local_dirs": ["models/SmolVLM-Base"],
        "slug": "smolvlm_base",
    },
    "ministral3_14b_it_bf16": {
        "hf_name": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "local_dirs": [
            "models/Ministral-3-14B-Instruct-2512-BF16",
            "models/ministral-3-14b-instruct-2512-bf16",
        ],
        "slug": "ministral3_14b_it_bf16",
    },
}


def model_slug(model_name: str):
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]["slug"]
    for alias in MODEL_ALIASES.values():
        if model_name == alias["hf_name"]:
            return alias["slug"]

    return (
        model_name
        .replace("/", "__")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def resolve_model_for_loading(model_name: str):
    from pathlib import Path

    alias = MODEL_ALIASES.get(model_name)
    if alias is None:
        for candidate in MODEL_ALIASES.values():
            if model_name == candidate["hf_name"]:
                alias = candidate
                break

    if alias is not None:
        for local_dir_raw in alias["local_dirs"]:
            local_dir = Path(local_dir_raw)
            if local_dir.exists():
                print("Loading pre-loaded model.")
                return str(local_dir), alias["hf_name"]
        return alias["hf_name"], alias["hf_name"]

    local_dir = Path("models") / model_name.split("/")[-1]
    if local_dir.exists():
        print("Loading pre-loaded model.")
        return str(local_dir), model_name

    return model_name, model_name


def is_ministral3_model_name(model_name: str):
    alias = MODEL_ALIASES.get(model_name)
    names = [model_name]
    if alias is not None:
        names.append(alias["hf_name"])
        names.extend(alias["local_dirs"])
    return any("ministral-3" in name.lower() for name in names)


def get_num_hidden_layers(model):
    config = model.config
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        return config.text_config.num_hidden_layers
    raise AttributeError("Could not find num_hidden_layers in model config.")


def get_hidden_size(model):
    config = model.config
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise AttributeError("Could not find hidden_size in model config.")


def get_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers  # Gemma 4
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers                 # Llama/Qwen/Gemma 3
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers              # Pythia
    raise ValueError("Could not find transformer blocks.")


def validate_block_layers(model, layers=None):
    """Return valid 1-based transformer block numbers; embeddings are never layers."""
    n_blocks = len(get_blocks(model))
    layers = list(range(1, n_blocks + 1)) if layers is None else list(layers)
    invalid = [layer for layer in layers if layer < 1 or layer > n_blocks]
    if invalid:
        raise ValueError(
            f"Transformer block layers must be in 1..{n_blocks}; got {invalid}."
        )
    return layers


def validate_saved_block_layers(layers):
    """Reject saved layer axes that include embeddings or invalid block numbers."""
    layers = [int(layer) for layer in layers]
    if not layers or any(layer < 1 for layer in layers):
        raise ValueError(
            f"Saved layers must be 1-based transformer blocks, not embeddings: {layers}"
        )
    return layers


def load_hf_model_and_processor(model_name: str):
    if is_ministral3_model_name(model_name):
        try:
            from transformers import AutoProcessor, Mistral3ForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Ministral 3 requires a recent transformers install with "
                "Mistral3ForConditionalGeneration."
            ) from exc

        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                fix_mistral_regex=True,
            )
        except ImportError as exc:
            raise ImportError(
                "Ministral 3 tokenization requires mistral-common. Install "
                "mistral-common>=1.8.6 in the active environment."
            ) from exc

        tokenizer = getattr(processor, "tokenizer", processor)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map={"": 0},
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        model.eval()
        return model, processor, tokenizer

    if any(name in model_name.lower() for name in ("gemma-4", "glm-4.1v", "smolvlm", "qwen3.5")):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        print("Using AutoModelForMultimodal!")
        processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = getattr(processor, "tokenizer", processor)

        model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map={"": 0},
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        model.eval()
        return model, processor, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map={"": 0},
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    model.eval()
    return model, tokenizer, tokenizer


def load_hf_model(model_name: str):
    model, _processor, tokenizer = load_hf_model_and_processor(model_name)
    return model, tokenizer
