# VoxCeleb2 and VGGFace2 data setup

This setup uses [Reverb's `vox2_meta.csv`](https://huggingface.co/datasets/Reverb/voxceleb2/blob/main/vox2_meta.csv) as the **identity join** between VoxCeleb2 audio and cropped VGGFace2 images. The `VoxCeleb2 ID` and `VGGFace2 ID` columns are different namespaces: for example, `id00134` maps to `n000140`. Never match them by their numeric suffix. `Set=dev/test` also determines the held-out test identities.

The [cropped VGGFace2 mirror](https://huggingface.co/datasets/chronopt-research/cropped-vggface2-224) provides 224-pixel images with VGGFace2 `n...` class labels. It is about 20 GB; the exporter streams rows and writes only requested identities. The [Reverb VoxCeleb2 mirror](https://huggingface.co/datasets/Reverb/voxceleb2) lists multipart MP4 video, speaker metadata, and the identity CSV. It does **not** supply verified word transcripts. Check the source datasets' use terms before downloading or redistributing data.

## Dependencies

Use Python 3.10 or newer. Install matching `torch`, `torchaudio`, and `torchvision` builds for your device, then:

```bash
python -m pip install numpy pillow datasets fsspec faster-whisper speechbrain
```

Install `ffmpeg` separately and put it on `PATH` to convert MP4 video to WAV. The metadata and manifest scripts use the Python standard library; `export_hf_faces.py` needs `datasets`, Pillow, and fsspec. The large datasets, speech checkpoint, and ResNet weights are downloaded only when the corresponding commands run.

## Obtain and arrange the data

Download the small identity CSV:

```bash
hf download Reverb/voxceleb2 vox2_meta.csv --repo-type dataset --local-dir data
```

Obtain and unpack the VoxCeleb2 MP4 archives following the [mirror's instructions](https://huggingface.co/datasets/Reverb/voxceleb2). Point `--video-root` at the unpacked directory containing speaker folders such as `id00134`. Convert clips, preserving their speaker ID:

```bash
python -m tools.extract_vox2_audio --video-root data/vox2_video --output-root data/vox2_audio --max-clips 100
```

Remove `--max-clips` for a full conversion. The mirror lists a dev archive; its separate test media may need to be obtained separately. With only dev speakers present, the manifest will have train and validation identities and no test records. Do not relabel dev identities as official test identities.

Export crops for identities that have local audio. `--max-rows` makes a small smoke run; remove it for full coverage. Review `export_report.json` for missing identities.

```bash
python -m tools.export_hf_faces --vox2-meta data/vox2_meta.csv --audio-root data/vox2_audio --output-dir data/vggface2_crops --max-per-speaker 10 --max-rows 10000
```

The exporter reads the image dataset's `ClassLabel` names (`n...`), not numeric label indices. For reproducibility, pass `--revision` with a dataset commit hash once selected. For large runs, leave enough disk space for the Hugging Face cache as well as the exported crops.

## Join and validate

```bash
python -m tools.prepare_manifests voxceleb --audio-root data/vox2_audio --face-root data/vggface2_crops --vox2-meta data/vox2_meta.csv --output-dir data/manifests
python -m tools.prepare_manifests validate --speech data/manifests/speech.jsonl --faces data/manifests/faces.jsonl
```

`speech.jsonl` has `speaker_id`, `split`, and `audio_path`; `faces.jsonl` has the **VoxCeleb2** `speaker_id`, the same split, and `image_path`. Only identities present in both datasets enter the manifests. A deterministic validation sample comes from `Set=dev`; `Set=test` stays held out. Validation rejects cross-split speaker leakage and missing files.

## Mel spectrograms and reviewed transcripts

Extract log-mels from the audio. `mel_config.json` records the exact definition; FastSpeech2 and its vocoder must use this same definition.

```bash
python -m tools.extract_mels --input data/manifests/speech.jsonl --output-dir data/mels --output-manifest data/manifests/speech_mels.jsonl
```

VoxCeleb2 has no verified text paired with these clips. Generate **draft** ASR text, listen to each retained clip, correct the words, and set `transcript_status` to `verified` in a reviewed copy of the JSONL. Reject clips with music, overlap, clipping, or unclear speech. A transcript must describe the audio clip, not its speaker ID.

```bash
python -m tools.transcribe_voxceleb --speech data/manifests/speech.jsonl --output data/manifests/tts_drafts.jsonl --model small.en --max-clips 100
python -m tools.prepare_manifests validate --speech data/manifests/speech.jsonl --faces data/manifests/faces.jsonl --tts data/manifests/tts_reviewed.jsonl
```

For multilingual speech, use a multilingual Whisper model and omit `small.en`. The ASR tool marks drafts `needs_review`; validation refuses those rows as TTS data. Review and alignment are necessary before FastSpeech2 training. VoxCeleb2 is noisy interview audio, so a clean transcribed corpus such as LibriTTS remains a useful TTS baseline. The optional loader is `python -m tools.prepare_manifests libritts --root data/LibriTTS --output data/manifests/libritts.jsonl`.

## Frozen speech targets and face embeddings

Extract one unit-normalized 192-dimensional ECAPA centroid per speaker from multiple clips, then train ResNet18 on the joined face images. The default [SpeechBrain ECAPA checkpoint](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) is frozen. Store its resolved revision in the command when possible.

```bash
python -m tools.extract_ecapa_centroids --speech data/manifests/speech.jsonl --output-dir data/ecapa --output-manifest data/manifests/centroids.jsonl --min-utterances 3
python -m tools.face_encoder train --faces data/manifests/faces.jsonl --centroids data/manifests/centroids.jsonl --output data/checkpoints/face_resnet18.pt --epochs 10
python -m tools.face_encoder embed --faces data/manifests/faces.jsonl --checkpoint data/checkpoints/face_resnet18.pt --output-dir data/face_embeddings --output-manifest data/manifests/face_embeddings.jsonl
```

The face model projects 224-pixel crops into the **same 192-dimensional, unit-normalized space** as ECAPA centroids. Its validation score uses speakers absent from training. The face embedding manifest retains the VoxCeleb2 identity and split, so the picture and mel/audio paths can be paired by `speaker_id` without assuming one specific image matches one specific utterance.

These commands prepare paired training records and a face embedding model. A working Face2Speech synthesizer still requires verified transcripts, measured phoneme durations, a speaker-conditioned FastSpeech2 implementation, and a mel-compatible vocoder. See [FACE2SPEECH_DESIGN.md](FACE2SPEECH_DESIGN.md).
