"""Stream selected cropped VGGFace2 images into speaker folders.

Requires: pip install datasets pillow fsspec
Only images with an ID in vox2_meta.csv are kept. Matching is by the
VGGFace2 ClassLabel name, never by its integer index or matching ID digits.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from numbers import Integral
from pathlib import Path

from tools.prepare_manifests import AUDIO_SUFFIXES, files_by_speaker, read_vox2_meta


DATASET = "chronopt-research/cropped-vggface2-224"


def save_image(payload: dict, destination: Path) -> None:
    from PIL import Image

    content = payload.get("bytes")
    if content is not None:
        source = io.BytesIO(content)
    elif payload.get("path"):
        path = payload["path"]
        if path.startswith("hf://"):
            import fsspec
            source = fsspec.open(path, "rb").open()
        else:
            source = open(path, "rb")
    else:
        raise ValueError("Hugging Face image row has neither bytes nor a path")
    with source:
        with Image.open(source) as image:
            image.convert("RGB").save(destination, format="JPEG", quality=95)


def export_rows(streams, selected_ids: set[str], output_dir: Path,
                max_per_speaker: int, max_rows: int | None = None) -> Counter:
    counts: Counter = Counter()
    visited = 0
    for stream in streams:
        features = stream.features
        if not features or "image" not in features or "label" not in features:
            raise ValueError("Expected image and label features in the Hugging Face dataset")
        label_feature = features["label"]
        if not hasattr(label_feature, "int2str"):
            raise ValueError("Expected a ClassLabel with VGGFace2 identity names")
        for row in stream:
            visited += 1
            label = row["label"]
            face_id = label_feature.int2str(int(label)) if isinstance(label, Integral) else str(label)
            if face_id in selected_ids and counts[face_id] < max_per_speaker:
                folder = output_dir / face_id
                folder.mkdir(parents=True, exist_ok=True)
                destination = folder / f"{counts[face_id]:05d}.jpg"
                save_image(row["image"], destination)
                counts[face_id] += 1
            if max_rows is not None and visited >= max_rows:
                print(f"Stopped after {visited} rows due to --max-rows")
                return counts
    print(f"Scanned {visited} image rows")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vox2-meta", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path,
                        help="Only export faces for speaker IDs with local WAV/FLAC clips")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="Pin a Hugging Face dataset commit for reproducibility")
    parser.add_argument("--max-per-speaker", type=int, default=10)
    parser.add_argument("--max-rows", type=int, help="Stop early for a smoke run")
    args = parser.parse_args()
    if args.max_per_speaker < 1 or (args.max_rows is not None and args.max_rows < 1):
        parser.error("--max-per-speaker and --max-rows must be positive")
    metadata = read_vox2_meta(args.vox2_meta)
    if args.audio_root:
        available = files_by_speaker(args.audio_root, AUDIO_SUFFIXES)
        selected_ids = {face_id for audio_id, (face_id, _) in metadata.items() if audio_id in available}
    else:
        selected_ids = {face_id for face_id, _ in metadata.values()}
    if not selected_ids:
        parser.error("No VGGFace2 IDs correspond to the available VoxCeleb2 audio")

    from datasets import load_dataset

    streams = [load_dataset(DATASET, split=split, revision=args.revision, streaming=True).decode(False)
               for split in ("train", "validation")]
    counts = export_rows(streams, selected_ids, args.output_dir,
                         args.max_per_speaker, args.max_rows)
    report = {
        "dataset": DATASET,
        "revision": args.revision,
        "selected_speakers": len(selected_ids),
        "exported_speakers": len(counts),
        "missing_face_ids": sorted(selected_ids - counts.keys()),
        "images_per_speaker": dict(sorted(counts.items())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {sum(counts.values())} crops for {len(counts)} speakers")


if __name__ == "__main__":
    main()
