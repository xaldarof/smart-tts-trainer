# Implementation Status

Last updated: 2026-11-16. Development machine: macOS 26 (Apple Silicon, 16 GB RAM), **no NVIDIA GPU**.
Target machine: Windows 11 + RTX 4060 Ti 16 GB. **Nothing has been run on Windows or CUDA yet.**

Legend: ✅ verified by running it (how is noted) · 🟡 implemented, not verifiable on this machine · ❌ not implemented

## 1. Backend decisions (checked against PyPI and Hugging Face on 2026-11-16)

| Component | Choice | Why |
|---|---|---|
| Python | 3.12 | Supported by every dependency. yt-dlp needs ≥3.11, librosa 1.0 needs ≥3.12, coqui-tts needs <3.15 |
| PyTorch | 2.11.0 + torchaudio 2.11.0, CUDA 12.8 wheels | Matched release pair (torchaudio 2.11 needs torch ≥2.11). cu128 Windows wheels exist for cp312. cu128 supports Ada (sm_89). Needs driver ≥570 |
| TTS + fine-tuning | **XTTS v2** via `coqui-tts` 0.28.0 (the maintained idiap fork) | Russian is supported natively. Ships an official GPT fine-tuning implementation (`GPTTrainer` + `coqui-tts-trainer`) that fits a 16 GB GPU. Weights pinned to `coqui/XTTS-v2@6c2b0d75` |
| F5-TTS | **not used** ❌ | Base weights cover English/Chinese only, so Russian would need a vocabulary extension and a long retraining rather than fine-tuning. Community Russian checkpoints have unclear licenses. Its pins also conflict with coqui-tts in one environment. The app has one backend selector, and adding F5-TTS would mean writing a new runner and engine |
| ASR | faster-whisper 1.2.1 (large-v3 by default), CTranslate2 4.9 | Word timestamps and per-word probabilities, which the segmenter and quality filter need |
| VAD | Silero VAD 6.2.3 | MIT, small, CPU |
| Enhancement | Demucs 4.1.0 (htdemucs vocals) + optional mild noisereduce | **DeepFilterNet was rejected**: its last release (0.5.6, 2023) pins numpy<2 and imports torchaudio APIs that no longer exist, so it can't be installed with this stack |
| Diarization | pyannote.audio 4.0.7 `speaker-diarization-community-1` (gated, needs an HF token) **or** a built-in ECAPA clustering fallback (SpeechBrain 1.1.2, ungated) | The app works without any account. pyannote adds overlap detection when a token is available |
| GUI | PySide6 6.12 | — |

Licenses: XTTS v2 weights are **CPML (non-commercial)**, and the app requires accepting it before download. coqui-tts code is MPL-2.0.
faster-whisper, Demucs and Silero are MIT. SpeechBrain ECAPA is Apache-2.0. pyannote community-1 is CC-BY-4.0 (gated).

The whole stack was installed with `uv` into a fresh Python 3.12 venv and imports together. The resolved versions are in
`requirements/lock-macos-dev.txt`. Problems found and handled along the way:
* coqui-tts 0.28 imports `six` without declaring it, so `six` is pinned explicitly.
* coqui-tts 0.28 changed `load_tts_samples()` to take a config instead of a list (the official recipe is outdated). The runner uses the new API.
* faster-whisper 1.2.1 can't decode *files* with PyAV 19 (`metadata_errors` kwarg). The app always passes decoded numpy arrays, so it isn't affected.
* HF `gradient_checkpointing_enable()` crashes on the XTTS GPT-2 because it needs the deleted `wte` embedding. The runner sets per-layer checkpointing directly. A unit test proves its gradients match a plain backward pass bit for bit.
* On CPU, SpeechBrain ECAPA at batch 32 hits a pathological slow path (32 clips: 20 s vs 0.6 s at batch 8). CPU now uses batch 4 with length sorting (~90× faster) and identical outputs.

## 2. Feature status

### Vertical slice (verified end-to-end with real models on CPU)
Run with `scripts/e2e_pipeline.py` on a synthetic two-speaker Russian recording (macOS `say`, 4:51, with a music bed), then
`scripts/gpu_smoke_test.py`-style steps:

| Step | Status | Evidence |
|---|---|---|
| Local media import | ✅ | `import_local_file` → video row, `downloaded` |
| Audio extraction (FFmpeg) | ✅ | 44.1 kHz mono work file, original kept |
| Demucs music removal | ✅ | htdemucs on CPU, 291 s audio |
| Silero VAD | ✅ | speech regions + silence stats |
| ECAPA diarization | ✅ | Found 2 speakers: 56 vs 10 turns (ground truth 56/12). Similarity to target 0.996 vs 0.33 |
| faster-whisper transcription | ✅ | `small` model, word timestamps |
| Speaker selection + consent gate | ✅ | `build_segments` refuses without consent (unit test) |
| Word/turn alignment, segmentation | ✅ | 60 segments. Found and **fixed** a bug where the first word of each phrase was dropped (turn edges) |
| Quality filtering + dedupe | ✅ | 14 repeated sentences rejected as duplicates. Numerals flagged for review |
| Dataset export + validation | ✅ | 35 train / 2 eval, 22.05 kHz. Parsed by Coqui's own `formatters.coqui` (unit test) |
| XTTS v2 download | ✅ | Pinned revision, gated on CPML acceptance |
| **Real fine-tuning** | ✅ (CPU) | 2 optimizer steps through `GPTTrainer`/`Trainer`. Loss 4.45 → 3.98. **All 426 GPT tensors changed, HiFi-GAN unchanged** (checked against base weights) |
| Graceful stop → checkpoint | ✅ | Stop-file → KeyboardInterrupt → `checkpoint_2.pth`, status `stopped` |
| Resume from checkpoint | ✅ | Resumed at step 3 with optimizer state, wrote `checkpoint_4.pth` |
| Model export | ✅ | Optimizer stripped (1.87 GB), base config/vocab, 4 reference clips, metadata |
| Trained-model inference | ✅ | 13.0 s of Russian from 2 chunks. **Whisper transcribed it back almost exactly** |
| Pretrained baseline inference | ✅ | Same refs, base weights |
| WAV/MP3 export | ✅ | FFmpeg libmp3lame |

### YouTube import (verified live against YouTube)
✅ Discovery of a single video and a channel (`@YouTube`, limit 5) · ✅ metadata (title, duration, size estimate) ·
✅ duplicate detection (within a project and across projects) · ✅ authorization gate · ✅ audio-only m4a download with
progress events · ✅ already-downloaded skip · ✅ unavailable-video error message ·
🟡 retry with backoff, resume of `.part` files and cancellation mid-download (the code paths exist, and only the error
classification is unit-tested) · 🟡 playlist discovery (shares the channel code path, not run live).

### GUI
✅ The app launches and all 8 pages build and refresh against real data (offscreen Qt, `--smoke`, pytest-qt).
Screenshots were reviewed and layout problems fixed (unstyled scrollbars and lists, column widths) ·
✅ background task manager: run, cancel and failure recording (pytest-qt) ·
🟡 interactive clicking through every workflow in a real windowed session (only offscreen construction and service-level runs were done) ·
🟡 QMediaPlayer playback on Windows.

### Persistence and recovery
✅ SQLite WAL, atomic race-free state transitions (8-thread test) · ✅ crash recovery (`recover_interrupted`: running jobs →
interrupted, downloads → queued, partial stages cleared, completed stages kept) · ✅ completed stages skipped on re-run
(unit test, and observed in e2e reruns: "processing done in 1s").

### Not verifiable here (needs the Windows + RTX machine) 🟡
* `scripts/install.ps1` and `run.bat` (written for Windows, never executed)
* CUDA paths: Demucs/Whisper/ECAPA/pyannote on GPU, `float16` Whisper, bf16 mixed-precision training,
  **actual VRAM use of the Low VRAM and Balanced presets** (the 8–10 / 12–14 GB figures are estimates),
  the OOM → gradient checkpointing → smaller batch auto-retry, Windows DLL registration for CTranslate2
* pyannote diarization (gated model; no token on this machine). The code follows the pyannote 4 API, and the
  automatic fallback to ECAPA is the path that was exercised.
* DataLoader workers > 0 on Windows (spawn). Tests used `num_workers=0`.
* Long runs: convergence and voice similarity after real multi-epoch fine-tuning on 20+ min of data. One full epoch with eval and best-model selection was verified on CPU (see §3)

**First thing to run on the target:** `.venv\Scripts\python.exe scripts\gpu_smoke_test.py --accept-cpml --steps 20`

## 3. Test results (this machine, final run)

| Suite | Command | Result |
|---|---|---|
| Unit + GUI (pytest-qt, offscreen) | `pytest -q` | **57 passed**, 3 skipped (the integration tests, which are opt-in) |
| Integration with real models, CPU | `LVT_INTEGRATION=1 LVT_ACCEPT_CPML=1 pytest tests/test_integration.py` | **3 passed** (142 s): Silero VAD + ECAPA diarization find 2 speakers in the fixture · Whisper tiny Russian · XTTS smoke path (load → infer → 2 fine-tuning steps with the **Low VRAM preset, gradient checkpointing on** → resume at step 3 → export → infer) |
| Full epoch with eval (CPU, 35 clips, batch 2 × accum 2, checkpointing on) | `scratchpad epoch run via TrainingProcess` | `completed`, 18 steps, train loss → 1.43, **eval loss 3.448** recorded, `best_model_18.pth` kept, duplicate `best_model.pth` removed, metrics stored. Bit-identical on re-run (seeded) |
| GUI launch | `python -m app.main --smoke` | exit 0, all pages cycled |
| Diagnostics | `python -m app.utils.system_check` | runs. On this Mac it reports no NVIDIA driver/CUDA, as expected |

The integration test caught a real bug that the unit tests missed: the Low VRAM preset crashed with
`GPT2Model has no attribute 'wte'`. It is fixed and covered by `tests/test_training_adapter.py`.

`data/` still contains the synthetic `e2e-test` project (the exported 2-step model and its generated WAV/MP3) as a
reference. Delete it from the Model Library, or delete `data/`, whenever you like. Training checkpoints from the test
runs were removed to free about 17 GB.

## 4. Known limitations
* Fine-tuning quality depends on data. 2 minutes is the hard minimum and 20+ minutes is recommended. The verification
  above proves the mechanics (gradients, checkpoints, resume, inference), **not** voice quality after hours of training.
* XTTS conditions on reference audio even after fine-tuning. The exported model bundles reference clips from its own
  training set. "Trained voice" uses fine-tuned weights with those clips, and "Cloning" uses base weights. The two are
  labelled separately in the UI and in the history.
* num2words produces nominative numerals ("от три до десять"), so clips containing digits always go to review.
* The ECAPA fallback can't detect real overlapped speech. It flags ambiguous windows instead, and pyannote is better
  for crosstalk.
* XTTS v2 output is non-commercial only (CPML).
