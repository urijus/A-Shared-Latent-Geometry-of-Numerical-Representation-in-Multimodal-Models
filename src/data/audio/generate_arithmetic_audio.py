"""Render arithmetic expressions as speech audio.

The generated metadata keeps the original arithmetic labels and adds audio
fields so text and audio versions can be matched by ``sample_id``.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from typing import Any

import numpy as np
from kokoro import KPipeline

from src.common.io import load_jsonl, save_jsonl


OPERATOR_SPEECH = {
    "+": "plus",
    "-": "minus",
    "*": "times",
    "×": "times",
    "=": "equals",
}


def token_for_audio(token: Any) -> str:
    return OPERATOR_SPEECH.get(str(token), str(token))


def expression_for_audio(sample: dict[str, Any]) -> str:
    """Return only the arithmetic expression, without instruction prompts."""
    tokens = sample.get("tokens")
    if isinstance(tokens, list) and tokens:
        return " ".join(token_for_audio(token) for token in tokens)

    expr = str(sample["expr"])
    expression = expr.split(". ", 1)[-1] if ". " in expr else expr
    for operator, spoken in OPERATOR_SPEECH.items():
        expression = expression.replace(operator, f" {spoken} ")
    return " ".join(expression.split())


def synthesize_audio(pipeline, text: str, voice: str, speed: float) -> np.ndarray:
    generator = pipeline(text, voice=voice, speed=speed)
    chunks = []
    for item in generator:
        audio = item.output.audio.detach().cpu().numpy()
        chunks.append(audio.astype(np.float32).reshape(-1))
    if not chunks:
        raise ValueError(f"Kokoro returned no audio for text: {text!r}")
    return np.concatenate(chunks)


def save_wav(audio: np.ndarray, output_path: Path, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def convert_samples_to_audio(
    samples: list[dict[str, Any]],
    output_dir: Path,
    metadata_name: str,
    model_id: str,
    lang_code: str,
    voice: str,
    speed: float,
    sample_rate: int,
    overwrite: bool,
) -> Path:
    pipeline = KPipeline(lang_code=lang_code, repo_id=model_id)
    audio_dir = output_dir / "wav"
    metadata_rows = []

    for index, sample in enumerate(samples):
        audio_text = expression_for_audio(sample)
        sample_id = sample.get("sample_id", index)
        audio_name = (
            f"{sample_id:06d}.wav"
            if isinstance(sample_id, int)
            else f"{index:06d}.wav"
        )
        audio_path = audio_dir / audio_name
        if overwrite or not audio_path.exists():
            audio = synthesize_audio(
                pipeline=pipeline,
                text=audio_text,
                voice=voice,
                speed=speed,
            )
            save_wav(audio=audio, output_path=audio_path, sample_rate=sample_rate)

        row = {
            **sample,
            "format": "audio_arithmetic",
            "source_format": sample.get("format", "text_arithmetic"),
            "source_expr": sample.get("expr"),
            "audio_text": audio_text,
            "audio_path": audio_path.relative_to(output_dir).as_posix(),
            "audio_sample_rate": sample_rate,
            "audio_format": "wav",
            "tts_model": model_id,
            "tts_voice": voice,
            "tts_lang_code": lang_code,
            "tts_speed": speed,
        }
        metadata_rows.append(row)

    metadata_path = output_dir / metadata_name
    save_jsonl(metadata_rows, metadata_path)
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_jsonl",
        type=Path,
        required=True,
        help="Existing arithmetic JSONL to render as speech.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("dataset/audio"))
    parser.add_argument("--model_id", default="hexgrad/Kokoro-82M")
    parser.add_argument("--lang_code", default="a")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        samples = samples[: args.limit]

    input_stem = args.input_jsonl.stem
    metadata_stem = (
        input_stem.removesuffix("_baseline")
        if input_stem.endswith("_baseline")
        else input_stem
    )
    metadata_name = f"{metadata_stem}_audio.jsonl"
    metadata_path = convert_samples_to_audio(
        samples=samples,
        output_dir=args.output_dir,
        metadata_name=metadata_name,
        model_id=args.model_id,
        lang_code=args.lang_code,
        voice=args.voice,
        speed=args.speed,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
    )
    print(f"Rendered {len(samples)} audio files")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
