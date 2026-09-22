import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class FastSpeech2ModelTests(unittest.TestCase):
    def test_aligned_training_step_and_speaker_conditioned_inference(self):
        import torch

        from fastspeech2_model import FaceConditionedFastSpeech2, fastspeech2_loss

        torch.manual_seed(7)
        model = FaceConditionedFastSpeech2(8, {
            "hidden": 32, "heads": 4, "encoder_layers": 1,
            "decoder_layers": 1, "dropout": 0.0, "max_frames": 100,
        })
        phones = torch.tensor([[1, 2, 3], [4, 5, 0]])
        durations = torch.tensor([[2, 2, 2], [3, 3, 0]])
        speaker = torch.randn(2, 192)
        pitch = torch.rand(2, 6)
        energy = torch.rand(2, 6)
        target_mel = torch.randn(2, 80, 6)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        output = model(phones, speaker, durations, pitch, energy)
        self.assertEqual(output["mel"].shape, (2, 80, 6))
        self.assertEqual(output["mel_lengths"].tolist(), [6, 6])
        losses = fastspeech2_loss(output, target_mel, durations, pitch, energy, phones)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(model.speaker_projection.weight.grad)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            inferred = model(phones[:1], speaker[:1])
        self.assertEqual(inferred["refined_mel"].shape[1], 80)
        self.assertGreater(int(inferred["mel_lengths"][0]), 0)
        self.assertTrue(torch.isfinite(inferred["refined_mel"]).all())


if __name__ == "__main__":
    unittest.main()
