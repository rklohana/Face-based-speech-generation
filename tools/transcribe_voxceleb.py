"""Create draft VoxCeleb2 transcripts with faster-whisper for human review.

Requires: pip install faster-whisper
Drafts are not training-ready. Listen to and correct the clip, then set
transcript_status to "verified" before manifest validation / alignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.prepare_manifests import read_jsonl, write_jsonl


def transcribe_rows(rows: list[dict], model, language: str | None = None,
                    min_avg_logprob: float = -1.0) -> tuple[list[dict], list[dict]]:
    drafts = []
    rejected = []
    for row in rows:
        audio_path = Path(row["audio_path"])
        segments, info = model.transcribe(
            str(audio_path), beam_size=5, language=language,
            vad_filter=True, condition_on_previous_text=False,
        )
        segments = list(segments)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        logprob = (sum(segment.avg_logprob for segment in segments) / len(segments)
                   if segments else float("-inf"))
        if not text or logprob < min_avg_logprob:
            rejected.append({**row, "reason": "empty_or_low_confidence_asr",
                             "asr_avg_logprob": logprob if segments else None})
            continue
        drafts.append({
            **row,
            "text": text,
            "transcript_source": "faster-whisper",
            "transcript_status": "needs_review",
            "asr_language": info.language,
            "asr_avg_logprob": logprob,
        })
    return drafts, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech", required=True, type=Path, help="Output speech.jsonl from prepare_manifests.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="small.en", help="faster-whisper model name or local path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--language", help="Force a language; omit to detect it per clip")
    parser.add_argument("--min-avg-logprob", type=float, default=-1.0)
    parser.add_argument("--max-clips", type=int, help="Limit clips for a smoke run")
    args = parser.parse_args()
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be positive")
    rows = read_jsonl(args.speech)
    if args.max_clips:
        rows = rows[:args.max_clips]
    if not rows:
        parser.error("Speech manifest is empty")
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    drafts, rejected = transcribe_rows(rows, model, args.language, args.min_avg_logprob)
    write_jsonl(args.output, drafts)
    rejected_path = args.output.with_name(args.output.stem + "_rejected.jsonl")
    write_jsonl(rejected_path, rejected)
    print(f"Wrote {len(drafts)} draft transcripts and {len(rejected)} rejected clips")
    print("Review each accepted clip and mark transcript_status=verified before TTS preprocessing")


if __name__ == "__main__":
    main()
