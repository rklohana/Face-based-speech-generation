"""Copy reviewed VoxCeleb2 clips and transcripts into an MFA corpus.

Each WAV and matching .lab file is placed under its speaker directory.
The index records the exact TextGrid path expected after `mfa align`.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

from tools.prepare_manifests import read_jsonl, write_jsonl


SAFE_SPEAKER = re.compile(r"[A-Za-z0-9_-]+\Z")


def utterance_name(audio_path: Path) -> str:
    digest = hashlib.sha256(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"{audio_path.stem}_{digest}"


def build(rows: list[dict], corpus_dir: Path, alignment_dir: Path) -> list[dict]:
    output = []
    seen_audio = set()
    speaker_splits = {}
    for row in rows:
        speaker = row.get("speaker_id")
        if not isinstance(speaker, str) or not SAFE_SPEAKER.fullmatch(speaker):
            raise ValueError(f"Unsafe speaker ID: {speaker!r}")
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {speaker}")
        prior_split = speaker_splits.setdefault(speaker, row["split"])
        if prior_split != row["split"]:
            raise ValueError(f"Speaker occurs in multiple splits: {speaker}")
        if row.get("transcript_status") != "verified" or not str(row.get("text", "")).strip():
            raise ValueError(f"Unreviewed or empty transcript for {speaker}")
        source = Path(row["audio_path"]).resolve()
        if source.suffix.lower() != ".wav" or not source.is_file():
            raise ValueError(f"MFA input must be an existing WAV: {source}")
        if source in seen_audio:
            raise ValueError(f"Repeated audio in transcript manifest: {source}")
        seen_audio.add(source)
        name = utterance_name(source)
        target_dir = corpus_dir / speaker
        target_dir.mkdir(parents=True, exist_ok=True)
        wav = target_dir / f"{name}.wav"
        lab = target_dir / f"{name}.lab"
        shutil.copy2(source, wav)
        lab.write_text(row["text"].strip() + "\n", encoding="utf-8")
        output.append({**row, "alignment_path": str((alignment_dir / speaker / f"{name}.TextGrid").resolve()),
                       "mfa_audio_path": str(wav.resolve())})
    if not output:
        raise ValueError("No reviewed transcript records")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", required=True, type=Path)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    args = parser.parse_args()
    output = build(read_jsonl(args.reviewed), args.corpus_dir, args.alignment_dir)
    write_jsonl(args.output_index, output)
    print(f"Prepared {len(output)} WAV and .lab pairs for Montreal Forced Aligner")


if __name__ == "__main__":
    main()
