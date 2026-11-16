# Local Voice Trainer

A Windows desktop application (PySide6) that turns **authorized** recordings of a **consenting** speaker into a
fine-tuned Russian text-to-speech voice. Everything runs locally after the one-time model downloads: no paid APIs,
no cloud inference, no remote training.

```
YouTube URL / local file → audio extraction → music removal → speech detection → speaker diarization
→ target-speaker selection + consent → transcription → segment alignment & quality filtering
→ dataset review → XTTS v2 fine-tuning → trained-voice synthesis → WAV / MP3 export
```

> **Status:** the full pipeline has been run end-to-end on real models on macOS (CPU). The Windows installer,
> CUDA acceleration and VRAM sizing for the RTX 4060 Ti 16 GB **have not yet been run on that hardware**. See
> [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for exactly what was verified and how.

---

## 1. Responsible use

* Train only on a voice whose owner gave you permission. The app won't build or export a dataset until you
  record who the speaker is and how consent was obtained (**Audio Processing → Confirm speaker consent**).
* Download only content you are authorized to process. You have to confirm this before each download. The app uses
  yt-dlp's normal public extraction. It does **not** pass cookies, bypass logins, age gates or bot checks, or
  circumvent DRM. Restricted videos are reported as restricted.
* For media you own outside YouTube, use **YouTube Import → Import local file…**.
* **XTTS v2 weights are licensed under the Coqui Public Model License (CPML), which allows non-commercial use only.**
  This also covers models fine-tuned from it and the audio they generate. You must accept the CPML in Settings before
  the weights download.

## 2. Requirements

| | |
|---|---|
| OS | Windows 11 x64 |
| GPU | NVIDIA RTX with ≥ 8 GB VRAM (tuned for the RTX 4060 Ti 16 GB). Driver ≥ 570 (CUDA 12.8) |
| Python | 3.12 (installed by the script via winget if missing) |
| FFmpeg | on PATH (`winget install Gyan.FFmpeg`; the installer offers it) |
| Disk | ≥ 30 GB free. Models need ~6 GB, and a training run keeps the latest and best checkpoints (~5.5 GB each, plus one more while saving) |
| RAM | 16 GB minimum, 32 GB recommended |

## 3. Installation

```powershell
git clone <this repo> local-voice-trainer
cd local-voice-trainer
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The script:

1. checks for Python 3.12, FFmpeg and the NVIDIA driver version, and offers winget installs
2. creates `.venv` and installs **pinned** dependencies (`requirements/torch-cuda.txt` → PyTorch 2.11.0 + CUDA 12.8,
   then `requirements/app.txt` with `requirements/constraints.txt`)
3. verifies that PyTorch sees the GPU and runs a test kernel, then prints the system diagnostics
4. optionally pre-downloads the processing models and, **only after you accept the CPML**, XTTS v2

Options: `-Cpu` (no NVIDIA GPU), `-Dev` (pytest), `-SkipModels`.

Start the app with `run.bat` or `.venv\Scripts\python.exe -m app.main`.

### Optional: pyannote diarization (better quality, needs a free Hugging Face account)

By default the app uses an ungated ECAPA-embedding clustering diarizer. For better results, especially with
overlapping speech:

1. Accept the conditions at <https://huggingface.co/pyannote/speaker-diarization-community-1>.
2. Create a read token and either set `HF_TOKEN` in your user environment variables or run
   `.venv\Scripts\huggingface-cli login`. The app never stores tokens.
3. In Settings, keep the diarization backend on `auto` (pyannote if available, otherwise ECAPA) or choose
   `pyannote`.

### Manual model download

```powershell
.venv\Scripts\python.exe scripts\download_models.py --processing          # Whisper, Demucs, ECAPA, Silero
.venv\Scripts\python.exe scripts\download_models.py --xtts --accept-cpml  # XTTS v2 (CPML)
```

## 4. Usage

1. **Settings → Models**: accept the CPML and download XTTS v2. Run **System diagnostics**.
2. **Project selector** (sidebar): use one project per target voice.
3. **YouTube Import**: paste a video, playlist or channel URL and click **Discover**. The app shows the video list,
   total duration, an estimated download size and an estimated processing time before anything downloads. Select
   videos, tick the authorization box, then **Download selected**. Downloads run in parallel, resume from `.part`
   files, retry with exponential backoff, can be cancelled, and skip duplicates (same video ID, even across projects).
4. **Audio Processing**: select the downloaded items and click **Process selected**. Each stage is saved as it
   finishes, so a crash, cancel or reboot never repeats completed stages.
   * Listen to each detected speaker with **Preview speaker**. Mark the consenting voice as **TARGET** and exclude the
     others. Don't assume the speaker with the most talk time is the channel owner.
   * **Suggest matches across videos** compares voice embeddings with your confirmed target and asks you to confirm
     each match.
   * **Confirm speaker consent…**, then **Build dataset segments**.
5. **Dataset Manager**: review the clips marked **REVIEW**. Reasons include uncertain speaker identity, low-confidence
   words, auto-expanded numbers whose grammatical case may be wrong, Latin-script words and noise. Play a clip, drag
   the green/red handles on the waveform to trim it, edit the transcript, then accept or reject it. The header shows
   clip count, clean duration, average clip length, speaker distribution, rejection reasons, missing transcripts and
   training readiness. Click **Export training dataset…**. Aim for 20 or more minutes; the minimum is 2 minutes.
6. **Voice Training**: choose a dataset and preset, then **Start training**. Live step, epoch, loss, eval loss and
   GPU memory are shown, plus a loss chart and the trainer log. **Stop** saves a checkpoint, and **Resume selected
   run** continues from it. When you're done, **Export selected run as voice model…**.
7. **TTS Generator**: choose **Trained voice**, enter Russian text (up to 20,000 characters, split automatically into
   chunks of ≤170 characters at sentence and clause boundaries) and click **Generate**. Use **Play**, **Export WAV…**
   or **Export MP3…**. Generation history is kept. **Pretrained baseline** synthesizes the same text with the original
   XTTS weights for A/B comparison. **Reference-audio cloning** is zero-shot imitation, not a trained model.
8. **Model Library**: view model metadata, open the model folder, or delete a model.

### Training presets (XTTS v2 GPT decoder, effective batch 32)

| Preset | Micro-batch × accumulation | Grad. checkpointing | Precision | Expected peak VRAM* |
|---|---|---|---|---|
| Low VRAM | 2 × 16 | on | bf16 | ~8–10 GB |
| Balanced (RTX 4060 Ti 16 GB) | 4 × 8 | off | bf16 | ~12–14 GB |
| Custom | any | any | bf16 / fp16 / fp32 | — |

\* These are estimates and haven't been measured on the target GPU yet. The smoke test reports the real peak.
On CUDA out-of-memory, the run automatically switches on gradient checkpointing, then halves the micro-batch and
doubles accumulation (so the effective batch stays the same), and resumes from the last checkpoint.

## 5. GPU smoke test (run this first on the RTX machine)

```powershell
.venv\Scripts\python.exe scripts\gpu_smoke_test.py --accept-cpml --steps 4
```

The test detects the GPU, loads XTTS v2, runs a short Russian inference and checks the audio, runs N real
fine-tuning steps, resumes from the saved checkpoint, exports the model and synthesizes with it. Each step prints
PASS or FAIL, and the full results go to `data/smoke_test_result.json`.

## 6. Tests

```powershell
.venv\Scripts\python.exe -m pip install -r requirements\dev.txt
.venv\Scripts\python.exe -m pytest -q                                   # unit + GUI tests (no downloads)
$env:LVT_INTEGRATION=1; $env:LVT_ACCEPT_CPML=1
.venv\Scripts\python.exe -m pytest -q tests\test_integration.py -s      # real models
```

`scripts/e2e_pipeline.py` runs the whole pipeline headless on a media file you own.
`scripts/make_test_audio.py` builds a synthetic two-speaker Russian recording on macOS.

## 7. Architecture

```
app/
  main.py                    entry point (logging, crash recovery, Qt app)
  config/settings.py         settings + thresholds (data/settings.json)
  database/db.py             SQLite (WAL), atomic transitions, crash recovery
  services/                  business logic, no Qt imports
    youtube/                 URL parsing, yt-dlp discovery/download
    audio/                   enhancement (Demucs), VAD (Silero), pipeline orchestration
    diarization/             pyannote / ECAPA clustering, speaker embeddings, previews
    transcription/           faster-whisper, Russian normalization
    dataset/                 word→turn alignment, segmentation, quality scoring, export/validation
    training/                presets, XTTS runner (child process), run manager, base model downloads
    inference/               XTTS synthesis, chunking, generation history
  workers/tasks.py           Qt thread pools (io + single GPU queue), training process controller
  gui/                       main window, pages, widgets (dark theme)
  utils/                     audio I/O + FFmpeg, GPU helpers, logging, diagnostics, job context
data/                        downloads, audio, datasets, checkpoints, trained_models, generated_audio, logs, model_cache
scripts/                     install.ps1, download_models.py, gpu_smoke_test.py, e2e_pipeline.py, make_test_audio.py
tests/                       unit, GUI (pytest-qt), integration (real models)
```

GPU-heavy stages share one queue and run one at a time, each releasing its memory before the next. Training runs in a
separate process, so a crash can't take down the GUI and all VRAM is freed when it ends.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| "FFmpeg was not found" | `winget install Gyan.FFmpeg`, then open a new terminal |
| "CUDA not available" | Update the NVIDIA driver (≥ 570), then re-run `install.ps1` (it reinstalls the CUDA wheel) |
| YouTube asks for bot verification | The app won't bypass it. Wait and retry, or download the media yourself and use *Import local file* |
| pyannote "access denied" | Accept the model conditions on Hugging Face using the account that owns the token |
| Training OOM even on Low VRAM | Close browsers, games and other GPU apps. Lower the maximum segment length in Settings |
| Loss becomes NaN | Use precision `fp32` or a lower learning rate (Custom preset) |
| Disk full | Free some space. Interrupted runs and stages resume where they stopped |
