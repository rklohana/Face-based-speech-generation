# Face2Speech migration: ECAPA-TDNN + ResNet + FastSpeech2

This is a design for a **new text + face -> waveform pipeline**, not a claim that the current scripts implement it. It follows the three-stage training dependency in [Goto et al., Interspeech 2020](https://www.isca-archive.org/interspeech_2020/goto20_interspeech.pdf), while replacing their speaker encoder, face encoder, and TTS architecture.

## What differs from the paper and this repository

| Part | Paper | Current repository | Planned replacement |
| --- | --- | --- | --- |
| Speaker encoder | 3-layer LSTM, 256-dimensional normalized speaker vector, GE2E training | No speech encoder; `embeddings.npy` has unknown provenance | Frozen ECAPA-TDNN speaker encoder; use its native 192-dimensional output if using `speechbrain/spkrec-ecapa-voxceleb` |
| Face encoder | VGG19 trained to predict speech-speaker centroids | No image input or face encoder | Pretrained ResNet backbone plus 192-dimensional projection; train against frozen ECAPA centroids |
| TTS | Phoneme and speaker-vector conditioned duration/acoustic models, WORLD waveform synthesis | Speaker-ID lookup and synthetic speaker-ID text in `TTS_finetuning.py`; no waveform output | `train_fastspeech2.py` conditions phone, duration, pitch, energy, and mel prediction on the frozen ECAPA space |
| GAN | None | `trainer.py` changes an existing spectrogram with an embedding | Keep as a separate voice-conversion experiment; it cannot synthesize arbitrary text from a face |
| Vocoder | WORLD | None | `synthesize_face.py` uses inverse mel filtering and Griffin-Lim as a waveform baseline; add a mel-matched neural vocoder for quality |

The cited SpeechBrain ECAPA checkpoint uses 80-band filterbanks internally and emits **192-dimensional** embeddings. That is a different contract from the paper's 40-band input and 256-dimensional embedding. A dimension-only adapter cannot preserve speaker-space geometry automatically. Choose **one frozen speech embedding space**, train the face encoder to predict that space, and train TTS with embeddings from that same checkpoint. Do not train the three stages with embeddings from unrelated models.

## Data contracts

All manifests need stable `speaker_id` values. Split **by speaker**, before computing centroids or fitting any model. Do not allow a speaker, their face images, or their audio clips into more than one train/validation/test split.

1. **Speech encoder / centroid data:** 16 kHz mono speech with speaker IDs. VoxCeleb2 can provide the face-domain speakers. For each speaker, extract several utterance embeddings from the frozen ECAPA model, L2-normalize each vector, average them, then L2-normalize the centroid. Store the checkpoint identifier, embedding dimension, sample rate, and speaker IDs with the vectors.
2. **Face training data:** Detected, aligned face crops joined by speaker ID to the VoxCeleb2 centroids. The paper used VGGFace2 images for identities overlapping VoxCeleb2. Reject failed or ambiguous face detections. Use the preprocessing prescribed by the chosen ResNet weights for both training and inference.
3. **TTS training data:** Real `(transcript, audio, speaker_id)` records. The requested VoxCeleb2 path first needs ASR drafts, listening review, and correction because the linked metadata is an identity map, not a transcript source. LibriTTS remains a useful clean-speech baseline. Extract ECAPA embeddings with the **same** frozen checkpoint. Compute the FastSpeech2 mel targets, phoneme tokens, phoneme durations from a forced aligner, and pitch/energy targets. Verify that token count equals duration count and that duration sums match mel frame counts after any trimming or resampling. VoxCeleb2 speaker IDs are not transcripts.
4. **Waveform synthesis:** Use a vocoder trained for the same sample rate, mel band count, FFT/window/hop sizes, frequency limits, log transform, and normalization as the FastSpeech2 targets. Check waveform length and finite sample values on held-out utterances.

## Training order

```text
Vox2 audio -> frozen ECAPA -> 192-D speaker centroids -----+
VGGFace2 crop -> ResNet -> 192-D face embedding -----------+-> shared speaker space
reviewed Vox2 text + audio -> alignments -> FastSpeech2 ----+

inference: face crop -> ResNet -> speaker embedding -+
           text -> phonemes -> FastSpeech2 -> mel -> vocoder -> waveform
```

1. Freeze ECAPA after choosing the checkpoint. Extract speaker embeddings for both the TTS corpus and the face corpus.
2. Train and validate FastSpeech2 using **real text** and a speech-derived embedding. Condition both its duration predictor and mel decoder on the embedding. Use ground-truth durations, pitch, and energy during training; predict them at inference. The current uniform-duration target is not a forced alignment.
3. Train ResNet on face crops. Project its output to the ECAPA dimension and L2-normalize. Use a supervised cosine-softmax/contrastive loss over fixed speaker centroids (and monitor positive-vs-negative cosine separation); plain regression alone does not ensure discriminability.
4. At inference, replace the speech embedding with the face-derived vector. Hold FastSpeech2 and the vocoder fixed. Use one face crop and text; no source spectrogram or reference speech should be required.

## Required repository changes

- The new `train_fastspeech2.py` path uses reviewed transcripts, MFA phone alignments, real duration, pitch, energy, log-mel targets, and fixed ECAPA centroids. The older `TTS_finetuning.py` remains an unrelated experimental baseline.
- The data preparation commands now provide ECAPA centroid extraction and a ResNet18 face encoder trained on the `vox2_meta.csv` identity join. They have not yet been exercised against the full datasets; validate the exported report and run training before using their embeddings downstream.
- The new text/phoneme frontend reads the MFA pronunciation dictionary. `synthesize_face.py` accepts **text plus a face image** and writes a waveform using Griffin-Lim. It also accepts a speech centroid for comparison. The dictionary must include the requested words, and a better mel-compatible vocoder remains quality work.
- Keep `trainer.py`, `gan_model.py`, and `dataloader.py` out of the Face2Speech inference path. They solve spectrogram conversion, a different task.
- Add held-out-speaker tests for manifest joins, no split leakage, embedding dimensions/norms, duration-to-mel alignment, one TTS train step, and face-to-waveform inference. Compare speech-conditioned and face-conditioned outputs with the same text and TTS/vocoder weights.

## Acceptance criteria

The pipeline is working only when (1) all three training stages use the same frozen speaker space, (2) held-out speakers are absent from every training split, (3) real transcripts and measured alignments drive TTS, (4) one face image plus new text produces a valid waveform, and (5) both a speech-conditioned baseline and a face-conditioned run are evaluated for intelligibility, speaker similarity, and listening quality. Synthetic shape checks alone are not evidence of speech quality.

References: [Face2Speech paper](https://www.isca-archive.org/interspeech_2020/goto20_interspeech.pdf), [SpeechBrain ECAPA model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), [FastSpeech2 paper](https://arxiv.org/abs/2006.04558), [Torchvision ResNet18 weights](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18), [HiFi-GAN paper](https://arxiv.org/abs/2010.05646).
