"""Extract VoxCeleb2 WAV clips from locally unpacked MP4 video files.

Requires ffmpeg on PATH. Input paths must include a VoxCeleb2 idNNNNN
directory. The output mirrors the relative path below that identity, so
several clips from the same speaker cannot overwrite each other.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


SPEAKER_ID = re.compile(r"id\d{5}$")


def destination_for(video: Path, root: Path, output_root: Path) -> Path:
    relative = video.relative_to(root)
    parts = relative.parts
    matches = [index for index, part in enumerate(parts[:-1]) if SPEAKER_ID.fullmatch(part)]
    if len(matches) != 1:
        raise ValueError(f"Expected one VoxCeleb2 speaker directory in {video}")
    return output_root.joinpath(*parts[matches[0]:]).with_suffix(".wav")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, help="Limit conversion for a smoke run")
    args = parser.parse_args()
    if not args.video_root.is_dir():
        parser.error("--video-root is not a directory")
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be positive")
    videos = sorted(path for path in args.video_root.rglob("*")
                    if path.is_file() and path.suffix.lower() == ".mp4")
    if not videos:
        raise ValueError("No MP4 clips found")
    count = 0
    for video in videos:
        if args.max_clips is not None and count >= args.max_clips:
            break
        destination = destination_for(video, args.video_root, args.output_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            continue
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(destination)], check=True)
        count += 1
    print(f"Extracted {count} WAV clips")


if __name__ == "__main__":
    main()
