"""Build and validate speaker-disjoint manifests for Face2Speech training.

The script only indexes data already obtained by the user. It never downloads
datasets or guesses a mapping between different speaker ID namespaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


AUDIO_SUFFIXES = {".wav", ".flac"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = {"train", "val", "test"}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def files_by_speaker(root: Path, suffixes: set[str]) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise ValueError(f"Directory does not exist: {root}")
    result = {}
    for directory in sorted(root.iterdir()):
        if directory.is_dir():
            matches = sorted(
                path.resolve() for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in suffixes
            )
            if matches:
                result[directory.name] = matches
    return result


def read_mapping(path: Path) -> dict[str, str]:
    mapping = {}
    used_faces = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"audio_speaker_id", "face_speaker_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Mapping CSV needs audio_speaker_id,face_speaker_id columns")
        for row in reader:
            audio_id = row["audio_speaker_id"].strip()
            face_id = row["face_speaker_id"].strip()
            if not audio_id or not face_id or audio_id in mapping or face_id in used_faces:
                raise ValueError(f"Empty or repeated speaker mapping: {audio_id!r}")
            mapping[audio_id] = face_id
            used_faces.add(face_id)
    if not mapping:
        raise ValueError("Speaker mapping CSV is empty")
    return mapping


def read_vox2_meta(path: Path) -> dict[str, tuple[str, str]]:
    """Read the VoxCeleb2 ID -> VGGFace2 ID and official dev/test mapping."""
    result = {}
    used_faces = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        required = {"VoxCeleb2 ID", "VGGFace2 ID", "Set"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"VoxCeleb2 metadata needs columns: {sorted(required)}")
        for row in reader:
            audio_id = row["VoxCeleb2 ID"].strip()
            face_id = row["VGGFace2 ID"].strip()
            official_set = row["Set"].strip().lower()
            if not audio_id.startswith("id") or not face_id.startswith("n") or official_set not in {"dev", "test"}:
                raise ValueError(f"Invalid VoxCeleb2 metadata row: {row}")
            if audio_id in result or face_id in used_faces:
                raise ValueError(f"Repeated VoxCeleb2 or VGGFace2 ID: {row}")
            result[audio_id] = (face_id, official_set)
            used_faces.add(face_id)
    if not result:
        raise ValueError("VoxCeleb2 metadata CSV is empty")
    return result


def speaker_splits(ids: list[str], seed: int, val_fraction: float, test_fraction: float) -> dict[str, str]:
    if len(ids) < 3:
        raise ValueError("At least three matched speakers are needed for train/val/test splits")
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum below one")
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    test_count = max(1, round(len(shuffled) * test_fraction))
    if val_count + test_count >= len(shuffled):
        raise ValueError("Not enough speakers for the requested split fractions")
    return {
        speaker: "val" if index < val_count else "test" if index < val_count + test_count else "train"
        for index, speaker in enumerate(shuffled)
    }


def build_voxceleb(args: argparse.Namespace) -> None:
    audio = files_by_speaker(args.audio_root, AUDIO_SUFFIXES)
    faces = files_by_speaker(args.face_root, IMAGE_SUFFIXES)
    metadata = read_vox2_meta(args.vox2_meta) if args.vox2_meta else None
    mapping = ({speaker: face_id for speaker, (face_id, _) in metadata.items()}
               if metadata else read_mapping(args.mapping) if args.mapping
               else {speaker: speaker for speaker in audio})
    matched = {
        audio_id: face_id for audio_id, face_id in mapping.items()
        if audio_id in audio and face_id in faces
    }
    if not matched:
        raise ValueError("No speakers have both audio and face images; check the ID mapping")
    if metadata:
        if not 0 < args.val_fraction < 1:
            raise ValueError("Validation fraction must be between zero and one")
        dev_ids = [speaker for speaker in matched if metadata[speaker][1] == "dev"]
        if len(dev_ids) < 2:
            raise ValueError("At least two matched VoxCeleb2 dev speakers are needed")
        shuffled = sorted(dev_ids)
        random.Random(args.seed).shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * args.val_fraction))
        if val_count >= len(shuffled):
            raise ValueError("Validation split leaves no VoxCeleb2 dev speakers for training")
        splits = {speaker: "test" for speaker in matched if metadata[speaker][1] == "test"}
        splits.update({speaker: "val" if index < val_count else "train"
                       for index, speaker in enumerate(shuffled)})
    else:
        splits = speaker_splits(list(matched), args.seed, args.val_fraction, args.test_fraction)
    speech_rows = [
        {"speaker_id": audio_id, "split": splits[audio_id], "audio_path": str(path)}
        for audio_id in sorted(matched) for path in audio[audio_id]
    ]
    face_rows = [
        {"speaker_id": audio_id, "split": splits[audio_id], "image_path": str(path)}
        for audio_id in sorted(matched) for path in faces[matched[audio_id]]
    ]
    write_jsonl(args.output_dir / "speech.jsonl", speech_rows)
    write_jsonl(args.output_dir / "faces.jsonl", face_rows)
    print(f"Matched {len(matched)} speakers; wrote {len(speech_rows)} speech and {len(face_rows)} face records")


def build_libritts(args: argparse.Namespace) -> None:
    if not args.root.is_dir():
        raise ValueError(f"Directory does not exist: {args.root}")
    rows = []
    for text_path in sorted(args.root.rglob("*.normalized.txt")):
        relative = text_path.relative_to(args.root)
        if len(relative.parts) < 4:
            continue
        corpus_split, speaker = relative.parts[:2]
        split = "train" if corpus_split.startswith("train-") else "val" if corpus_split.startswith("dev-") else "test" if corpus_split.startswith("test-") else None
        if split is None:
            continue
        audio_path = text_path.with_name(text_path.name.removesuffix(".normalized.txt") + ".wav")
        if not audio_path.is_file():
            raise ValueError(f"Missing audio for {text_path}: {audio_path}")
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty transcript: {text_path}")
        rows.append({
            "speaker_id": f"libritts:{speaker}", "split": split,
            "audio_path": str(audio_path.resolve()), "text": text,
        })
    if not rows:
        raise ValueError("No LibriTTS normalized transcripts found")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} transcribed utterances")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(args: argparse.Namespace) -> None:
    seen_splits = defaultdict(set)
    seen_paths = {}
    manifest_speakers = {}
    for kind, path, file_key in (
        ("speech", args.speech, "audio_path"),
        ("faces", args.faces, "image_path"),
        ("tts", args.tts, "audio_path"),
    ):
        if path is None:
            continue
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty {kind} manifest: {path}")
        speakers = set()
        for number, row in enumerate(rows, start=1):
            speaker = row.get("speaker_id")
            split = row.get("split")
            item_path = row.get(file_key)
            if not isinstance(speaker, str) or not speaker or split not in SPLITS:
                raise ValueError(f"Invalid speaker or split at {path}:{number}")
            if not isinstance(item_path, str) or not Path(item_path).is_file():
                raise ValueError(f"Missing {file_key} at {path}:{number}: {item_path}")
            path_key = (kind, str(Path(item_path).resolve()))
            previous = seen_paths.setdefault(path_key, (speaker, split))
            if previous != (speaker, split):
                raise ValueError(f"A file belongs to multiple speakers or splits: {item_path}")
            if kind == "tts" and not str(row.get("text", "")).strip():
                raise ValueError(f"Missing real transcript at {path}:{number}")
            if kind == "tts" and row.get("transcript_source") == "faster-whisper" and row.get("transcript_status") != "verified":
                raise ValueError(f"ASR transcript still needs review at {path}:{number}")
            seen_splits[speaker].add(split)
            speakers.add(speaker)
        manifest_speakers[kind] = speakers
        print(f"{kind}: {len(rows)} records, {len(speakers)} speakers")
    leaked = sorted(speaker for speaker, splits in seen_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"Speakers occur in multiple splits: {leaked[:10]}")
    if "faces" in manifest_speakers and "speech" in manifest_speakers:
        missing = manifest_speakers["faces"] - manifest_speakers["speech"]
        if missing:
            raise ValueError(f"Face speakers without speech targets: {sorted(missing)[:10]}")
    print("Manifest validation passed; no speaker split leakage detected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    vox = subparsers.add_parser("voxceleb", help="Index matched speech and face folders")
    vox.add_argument("--audio-root", required=True, type=Path)
    vox.add_argument("--face-root", required=True, type=Path)
    vox_mapping = vox.add_mutually_exclusive_group()
    vox_mapping.add_argument("--mapping", type=Path, help="CSV for different audio and image ID namespaces")
    vox_mapping.add_argument("--vox2-meta", type=Path, help="vox2_meta.csv with VoxCeleb2/VGGFace2 IDs and dev/test sets")
    vox.add_argument("--output-dir", required=True, type=Path)
    vox.add_argument("--seed", type=int, default=42)
    vox.add_argument("--val-fraction", type=float, default=0.1)
    vox.add_argument("--test-fraction", type=float, default=0.1)
    vox.set_defaults(func=build_voxceleb)

    tts = subparsers.add_parser("libritts", help="Index official LibriTTS normalized text and WAV files")
    tts.add_argument("--root", required=True, type=Path)
    tts.add_argument("--output", required=True, type=Path)
    tts.set_defaults(func=build_libritts)

    check = subparsers.add_parser("validate", help="Validate manifests and speaker-disjoint splits")
    check.add_argument("--speech", type=Path)
    check.add_argument("--faces", type=Path)
    check.add_argument("--tts", type=Path)
    check.set_defaults(func=validate)

    args = parser.parse_args()
    if args.command == "validate" and not any((args.speech, args.faces, args.tts)):
        parser.error("validate needs at least one manifest")
    args.func(args)


if __name__ == "__main__":
    main()
