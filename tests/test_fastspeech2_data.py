import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from tools.prepare_alignment_corpus import build
from tools.prepare_fastspeech2 import (
    durations_from_intervals, frame_pitch_energy, prepare, read_phone_intervals,
)


TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.4
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.4
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.2
            text = "HH"
        intervals [2]:
            xmin = 0.2
            xmax = 0.4
            text = "AH0"
'''


class FastSpeech2DataTests(unittest.TestCase):
    def test_textgrid_phone_durations_sum_to_mel_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip.TextGrid"
            path.write_text(TEXTGRID, encoding="utf-8")
            phones, durations = durations_from_intervals(read_phone_intervals(path), 26, 0.4)
            self.assertEqual(phones, ["HH", "AH0"])
            self.assertEqual(durations.tolist(), [13, 13])
            with self.assertRaisesRegex(ValueError, "cover"):
                durations_from_intervals(read_phone_intervals(path), 26, 0.7)

    def test_frame_features_are_finite_and_follow_mel_length(self):
        sample_rate = 16000
        time = np.arange(6400, dtype=np.float32) / sample_rate
        waveform = 0.1 * np.sin(2 * np.pi * 200 * time)
        pitch, energy = frame_pitch_energy(waveform, 26, sample_rate, 256, 1024)
        self.assertEqual(pitch.shape, (26,))
        self.assertTrue(np.isfinite(pitch).all())
        self.assertTrue(np.isfinite(energy).all())
        self.assertGreater(float(np.median(pitch)), 0)

    def test_prepared_row_joins_verified_audio_mel_alignment_and_centroid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wav = root / "clip.wav"
            wavfile.write(wav, 16000, np.zeros(6400, dtype=np.int16))
            row = {"speaker_id": "id00134", "split": "train", "audio_path": str(wav),
                   "text": "huh", "transcript_status": "verified"}
            index = build([row], root / "corpus", root / "aligned")
            alignment = Path(index[0]["alignment_path"])
            alignment.parent.mkdir(parents=True)
            alignment.write_text(TEXTGRID, encoding="utf-8")
            mel_path = root / "mel.npy"
            np.save(mel_path, np.zeros((80, 26), dtype=np.float32))
            centroid_path = root / "centroid.npy"
            vector = np.zeros(192, dtype=np.float32)
            vector[0] = 1
            np.save(centroid_path, vector)
            output, vocab = prepare(index, [{**row, "mel_path": str(mel_path)}],
                                    [{"speaker_id": "id00134", "split": "train",
                                      "centroid_path": str(centroid_path)}], root / "targets")
            self.assertEqual(vocab["<pad>"], 0)
            self.assertEqual(output[0]["speaker_id"], "id00134")
            with np.load(output[0]["features_path"], allow_pickle=False) as features:
                self.assertEqual(int(features["durations"].sum()), 26)
                self.assertEqual(len(features["pitch"]), 26)
            index[0]["transcript_status"] = "needs_review"
            with self.assertRaisesRegex(ValueError, "reviewed"):
                prepare(index, [{**row, "mel_path": str(mel_path)}],
                        [{"speaker_id": "id00134", "split": "train",
                          "centroid_path": str(centroid_path)}], root / "targets")


if __name__ == "__main__":
    unittest.main()
