# Face based speech generation experiments

This repository contains two separate research scripts:

- `trainer.py` trains a conditional GAN that changes a spectrogram using a face embedding. It writes generated spectrograms as NumPy arrays; it does not reconstruct audio.
- `TTS_finetuning.py` trains a small text to mel spectrogram model on the `acul3/voxceleb2` Hugging Face dataset. VoxCeleb2 does not provide transcripts, so this script uses text made from each speaker ID. Its output is an experimental baseline, not a usable text to speech system.

## Setup

Use Python 3.10 or newer. Install a matching PyTorch and torchaudio build for your CPU or CUDA environment, then install the remaining packages:

```bash
python -m pip install numpy datasets tensorboard
```

Both scripts use PyTorch. The TTS script also needs `torchaudio`; the GAN script does not.

## Spectrogram GAN

Prepare the following files yourself; they are not included in this repository:

- A directory containing speaker folders with `.npy` spectrograms. Each spectrogram must be a nonempty, two dimensional numeric array. Arrays are cropped or zero padded to `image_size` by `image_size` (256 by default). Inputs should already be scaled to approximately `[-1, 1]`, matching the generator's `tanh` output.
- An `embeddings.npy` array with one fixed length embedding per row.
- An `ids.npy` string array with one speaker ID per embedding. The loader matches speakers using each ID and folder name after removing their first two characters, following the naming convention in the original data layout.

Train:

```bash
python trainer.py --data_dir path/to/spectrograms --embeddings_path path/to/embeddings.npy --ids_path path/to/ids.npy --num_iters 1000
```

Checkpoints are saved in `models/` by default. To generate spectrograms with a saved checkpoint:

```bash
python trainer.py --mode test --data_dir path/to/spectrograms --embeddings_path path/to/embeddings.npy --ids_path path/to/ids.npy --test_iters 1000
```

Generated two dimensional `.npy` arrays are saved in `results/`. Run `python trainer.py --help` for the remaining model and training options. Use the same model options when loading a checkpoint.

## TTS experiment

The script downloads `acul3/voxceleb2` through Hugging Face `datasets`. It expects each example to have an `audio` item with `array` and `sampling_rate`, plus a `speaker_id`. Audio is converted to mono, resampled to 16 kHz, and clipped or padded to eight seconds. Dataset access and enough storage for its cache are required.

```bash
python TTS_finetuning.py --num_epochs 10 --batch_size 16 --log_dir logs
```

The script writes TensorBoard logs, model checkpoints, and `speaker_ids.json` in `logs/`. It predicts mel spectrograms only; there is no waveform decoder or synthesis command. Training uses uniform token alignment because the dataset has no transcripts or token durations. Real text to speech training requires paired transcripts and a suitable alignment or duration model.
