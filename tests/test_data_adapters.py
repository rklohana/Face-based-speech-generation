import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from tools.export_hf_faces import export_rows
from tools.prepare_manifests import validate
from tools.transcribe_voxceleb import transcribe_rows


class FakeClassLabel:
    def int2str(self, value):
        return {0: "n000140", 1: "n000141"}[value]


class FakeStream:
    features = {"image": object(), "label": FakeClassLabel()}

    def __init__(self, image_bytes):
        self.image_bytes = image_bytes

    def __iter__(self):
        yield {"label": 1, "image": {"bytes": self.image_bytes}}
        yield {"label": 0, "image": {"bytes": self.image_bytes}}


class FakeASR:
    def transcribe(self, path, **kwargs):
        if "bad" in path:
            return iter([SimpleNamespace(text="Wrong", avg_logprob=-2.0)]), SimpleNamespace(language="en")
        return iter([SimpleNamespace(text="Hello world.", avg_logprob=-0.2)]), SimpleNamespace(language="en")


class AdapterTests(unittest.TestCase):
    def test_hf_export_uses_class_label_not_integer_id(self):
        image = Image.new("RGB", (8, 8), color="blue")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counts = export_rows([FakeStream(buffer.getvalue())], {"n000140"}, root, 1)
            self.assertEqual(counts["n000140"], 1)
            self.assertTrue((root / "n000140" / "00000.jpg").is_file())
            self.assertFalse((root / "n000141").exists())

    def test_asr_drafts_need_review_before_tts_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good.wav"
            bad = root / "bad.wav"
            good.write_bytes(b"wav")
            bad.write_bytes(b"wav")
            rows = [
                {"speaker_id": "id1", "split": "train", "audio_path": str(good)},
                {"speaker_id": "id2", "split": "test", "audio_path": str(bad)},
            ]
            drafts, rejected = transcribe_rows(rows, FakeASR())
            self.assertEqual(len(drafts), 1)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(drafts[0]["transcript_status"], "needs_review")
            manifest = root / "tts.jsonl"
            manifest.write_text(json.dumps(drafts[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "needs review"):
                validate(argparse.Namespace(speech=None, faces=None, tts=manifest))
            drafts[0]["transcript_status"] = "verified"
            manifest.write_text(json.dumps(drafts[0]) + "\n", encoding="utf-8")
            validate(argparse.Namespace(speech=None, faces=None, tts=manifest))


if __name__ == "__main__":
    unittest.main()
