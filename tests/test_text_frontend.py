import tempfile
import unittest
from pathlib import Path

from synthesize_face import phones_for_text, read_lexicon


class TextFrontendTests(unittest.TestCase):
    def test_mfa_lexicon_maps_text_to_training_phone_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "english.dict"
            path.write_text("HELLO 0.9 HH AH0 L OW1\nWORLD W ER1 L D\n", encoding="utf-8")
            vocabulary = {"<pad>": 0, "sp": 1, "HH": 2, "AH0": 3,
                          "L": 4, "OW1": 5, "W": 6, "ER1": 7, "D": 8}
            self.assertEqual(phones_for_text("Hello, world!", read_lexicon(path), vocabulary),
                             [2, 3, 4, 5, 1, 6, 7, 4, 8])
            with self.assertRaisesRegex(ValueError, "absent"):
                phones_for_text("Unknown word", read_lexicon(path), vocabulary)


if __name__ == "__main__":
    unittest.main()
