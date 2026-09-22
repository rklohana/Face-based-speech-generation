import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.extract_ecapa_centroids import EMBEDDING_DIM, unit_vector
from tools.face_encoder import load_joined
from tools.extract_vox2_audio import destination_for


class EmbeddingJoinTests(unittest.TestCase):
    def test_face_join_uses_audio_identity_and_rejects_split_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            vector[0] = 1
            centroid = root / "centroid.npy"
            np.save(centroid, vector)
            centroids = root / "centroids.jsonl"
            centroids.write_text(json.dumps({"speaker_id": "id00134", "split": "train",
                                             "centroid_path": str(centroid)}) + "\n")
            faces = root / "faces.jsonl"
            row = {"speaker_id": "id00134", "split": "train",
                   "image_path": str(root / "n000140.jpg")}
            faces.write_text(json.dumps(row) + "\n")
            joined = load_joined(faces, centroids)
            self.assertEqual(joined[0][1:3], ("id00134", "train"))
            self.assertAlmostEqual(float(np.linalg.norm(joined[0][3])), 1.0)
            row["split"] = "test"
            faces.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "splits differ"):
                load_joined(faces, centroids)

    def test_ecapa_vector_checks_dimension_and_zero_norm(self):
        with self.assertRaises(ValueError):
            unit_vector(np.zeros(EMBEDDING_DIM))
        with self.assertRaises(ValueError):
            unit_vector(np.ones(EMBEDDING_DIM - 1))

    def test_video_conversion_preserves_speaker_and_clip_path(self):
        root = Path("video")
        result = destination_for(root / "dev" / "id00134" / "abc" / "clip.mp4",
                                 root, Path("audio"))
        self.assertEqual(result, Path("audio/id00134/abc/clip.wav"))


if __name__ == "__main__":
    unittest.main()
