import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_manifests import build_libritts, build_voxceleb, read_mapping, read_vox2_meta, validate


class ManifestTests(unittest.TestCase):
    def test_voxceleb_pairs_and_splits_are_speaker_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number in range(5):
                speaker = f"id{number:04d}"
                audio = root / "audio" / speaker / "clip.wav"
                image = root / "faces" / speaker / "face.jpg"
                audio.parent.mkdir(parents=True)
                image.parent.mkdir(parents=True)
                audio.write_bytes(b"wav")
                image.write_bytes(b"jpg")
            output = root / "manifests"
            build_voxceleb(argparse.Namespace(
                audio_root=root / "audio", face_root=root / "faces", mapping=None, vox2_meta=None,
                output_dir=output, seed=42, val_fraction=0.2, test_fraction=0.2,
            ))
            validate(argparse.Namespace(
                speech=output / "speech.jsonl", faces=output / "faces.jsonl", tts=None,
            ))
            speech = [json.loads(line) for line in (output / "speech.jsonl").read_text().splitlines()]
            faces = [json.loads(line) for line in (output / "faces.jsonl").read_text().splitlines()]
            self.assertEqual({row["speaker_id"] for row in speech}, {row["speaker_id"] for row in faces})
            self.assertEqual({row["split"] for row in speech}, {"train", "val", "test"})

    def test_mapping_cannot_reuse_a_face_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.csv"
            path.write_text("audio_speaker_id,face_speaker_id\na,b\nc,b\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_mapping(path)

    def test_vox2_metadata_maps_different_ids_and_keeps_test_held_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "vox2_meta.csv"
            metadata.write_text(
                "\ufeffVoxCeleb2 ID ,VGGFace2 ID ,Gender ,Set\n"
                "id00134 ,n000140 ,f ,dev\n"
                "id00135 ,n000135 ,m ,dev\n"
                "id00136 ,n000136 ,m ,test\n", encoding="utf-8"
            )
            self.assertEqual(read_vox2_meta(metadata)["id00134"], ("n000140", "dev"))
            for audio_id, face_id, _ in (("id00134", "n000140", "dev"),
                                         ("id00135", "n000135", "dev"),
                                         ("id00136", "n000136", "test")):
                audio = root / "audio" / audio_id / "clip.wav"
                face = root / "faces" / face_id / "face.jpg"
                audio.parent.mkdir(parents=True)
                face.parent.mkdir(parents=True)
                audio.write_bytes(b"wav")
                face.write_bytes(b"jpg")
            output = root / "out"
            build_voxceleb(argparse.Namespace(
                audio_root=root / "audio", face_root=root / "faces", mapping=None,
                vox2_meta=metadata, output_dir=output, seed=42,
                val_fraction=0.5, test_fraction=0.5,
            ))
            faces = [json.loads(line) for line in (output / "faces.jsonl").read_text().splitlines()]
            by_id = {row["speaker_id"]: row for row in faces}
            self.assertIn("n000140", by_id["id00134"]["image_path"])
            self.assertEqual(by_id["id00136"]["split"], "test")
            validate(argparse.Namespace(speech=output / "speech.jsonl", faces=output / "faces.jsonl", tts=None))

    def test_libritts_requires_real_text_and_wav(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "LibriTTS" / "train-clean-100" / "19" / "198"
            source.mkdir(parents=True)
            (source / "19_198_0001.normalized.txt").write_text("Hello world.\n", encoding="utf-8")
            (source / "19_198_0001.wav").write_bytes(b"wav")
            output = root / "tts.jsonl"
            build_libritts(argparse.Namespace(root=root / "LibriTTS", output=output))
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]["speaker_id"], "libritts:19")
            self.assertEqual(rows[0]["text"], "Hello world.")
            validate(argparse.Namespace(speech=None, faces=None, tts=output))

    def test_validator_rejects_split_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "one.wav"
            audio.write_bytes(b"wav")
            manifest = root / "speech.jsonl"
            rows = [
                {"speaker_id": "id1", "split": "train", "audio_path": str(audio)},
                {"speaker_id": "id1", "split": "test", "audio_path": str(audio)},
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate(argparse.Namespace(speech=manifest, faces=None, tts=None))


if __name__ == "__main__":
    unittest.main()
