"""Content fingerprints for resumable causal experiments."""

import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_image_hash_cache = {}


def dataset_hash(path):
    """Hash the ordered JSONL and referenced image bytes, when present."""
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    images = []
    for row in rows:
        if "image_path" not in row:
            continue
        image = Path(row["image_path"])
        image = image if image.is_absolute() else path.parent / image
        image = image.resolve()
        stat = image.stat()
        key = (str(image), stat.st_size, stat.st_mtime_ns)
        if key not in _image_hash_cache:
            _image_hash_cache[key] = file_hash(image)
        images.append((str(image), _image_hash_cache[key]))
    return digest({"jsonl": file_hash(path), "images": images})


def model_identity(model_path):
    path = Path(model_path).resolve()
    if not path.is_dir():
        raise ValueError(f"A local model snapshot is required for a checked cache fingerprint: {model_path}")
    names = ("config.json", "processor_config.json", "preprocessor_config.json",
             "tokenizer_config.json", "chat_template.json", "model.safetensors.index.json")
    weights = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin"))
    return {"path": str(path), "files": {
        name: file_hash(path / name) for name in names if (path / name).is_file()
    }, "weights": [(item.name, item.stat().st_size, item.stat().st_mtime_ns)
                    for item in weights]}


def ordered_pairs_hash(path, count):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return digest(rows[:count] if count > 0 else rows)


def code_identity(*paths):
    return {str(Path(path).resolve()): file_hash(path) for path in paths}
