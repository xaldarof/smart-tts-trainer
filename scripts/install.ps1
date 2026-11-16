<#
.SYNOPSIS
  Local Voice Trainer - Windows setup (Windows 11 x64, NVIDIA RTX, Python 3.12).

.DESCRIPTION
  1. Verifies Python 3.12 (py launcher), FFmpeg, NVIDIA driver.
  2. Creates .venv and installs pinned dependencies (CUDA 12.8 PyTorch wheels).
  3. Verifies CUDA/PyTorch compatibility and runs first-run diagnostics.
  4. Optionally pre-downloads model weights.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Cpu        # no NVIDIA GPU
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Dev        # + test tools
#>
param(
    [switch]$Cpu,
    [switch]$Dev,
    [switch]$SkipModels
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "    [OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "    [WARN] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "    [FAIL] $m" -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------------ Python
Step "Checking Python 3.12"
$py = $null
try { $v = & py -3.12 -c "import sys; print(sys.version)" 2>$null; if ($LASTEXITCODE -eq 0) { $py = "py -3.12" } } catch {}
if (-not $py) {
    Warn "Python 3.12 not found via the 'py' launcher."
    $ans = Read-Host "Install Python 3.12 with winget now? [y/N]"
    if ($ans -match '^[yY]') {
        winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
        Fail "Python installed. Open a NEW PowerShell window and run this script again."
    }
    Fail "Install Python 3.12 (https://www.python.org/downloads/ or 'winget install Python.Python.3.12') and re-run."
}
Ok "Python $v"

# ------------------------------------------------------------------ FFmpeg
Step "Checking FFmpeg"
$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ff) {
    Warn "FFmpeg is not on PATH (required for audio extraction and MP3 export)."
    $ans = Read-Host "Install FFmpeg with winget (Gyan.FFmpeg) now? [y/N]"
    if ($ans -match '^[yY]') {
        winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
        Warn "FFmpeg installed. PATH changes apply to NEW terminals; restart PowerShell before launching the app."
    } else {
        Warn "Continuing without FFmpeg; the app will show an error until it is installed."
    }
} else { Ok (& ffmpeg -version | Select-Object -First 1) }

# ------------------------------------------------------------- NVIDIA GPU
Step "Checking NVIDIA driver"
$useCuda = -not $Cpu
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($useCuda) {
    if (-not $smi) {
        Warn "nvidia-smi not found: no NVIDIA driver. Installing CPU-only PyTorch (training will not be practical)."
        $useCuda = $false
    } else {
        $info = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        Ok $info
        $drv = [double](($info -split ",")[1].Trim().Split(".")[0])
        if ($drv -lt 570) { Warn "Driver $drv is older than 570; CUDA 12.8 wheels need >= 570. Update the NVIDIA driver." }
    }
}

# --------------------------------------------------------------- venv
Step "Creating virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & py -3.12 -m venv .venv; if ($LASTEXITCODE) { Fail "venv creation failed" } }
$vpy = Join-Path $Root ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip wheel | Out-Null
Ok "venv ready"

Step "Installing PyTorch ($(if ($useCuda) {'CUDA 12.8'} else {'CPU'}))"
$torchReq = if ($useCuda) { "requirements\torch-cuda.txt" } else { "requirements\torch-cpu.txt" }
& $vpy -m pip install -r $torchReq
if ($LASTEXITCODE) { Fail "PyTorch installation failed (network? disk space?)" }

Step "Installing application dependencies (pinned)"
& $vpy -m pip install -r requirements\app.txt -c requirements\constraints.txt
if ($LASTEXITCODE) {
    Warn "Install with full constraints failed (a transitive pin may lack a Windows wheel). Retrying with top-level pins only."
    & $vpy -m pip install -r requirements\app.txt
    if ($LASTEXITCODE) { Fail "Dependency installation failed. See the pip output above." }
}
if ($Dev) { & $vpy -m pip install -r requirements\dev.txt }
# Make sure pip did not replace the CUDA torch with a CPU build while resolving dependencies.
if ($useCuda) {
    $cudaBuild = & $vpy -c "import torch; print(torch.version.cuda or '')"
    if (-not $cudaBuild) {
        Warn "CPU torch build detected after dependency install; reinstalling CUDA wheels."
        & $vpy -m pip install --force-reinstall --no-deps -r requirements\torch-cuda.txt
    }
}

# --------------------------------------------------------- verification
Step "Verifying CUDA / PyTorch"
& $vpy -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
if ($useCuda) {
    & $vpy -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE) { Warn "PyTorch cannot use the GPU. Check the driver version, then re-run this script." }
}

Step "System diagnostics"
& $vpy -m app.utils.system_check

if (-not $SkipModels) {
    Step "Model weights"
    Write-Host "    Processing models (Whisper large-v3 ~3 GB, Demucs, ECAPA, Silero) download on first use."
    $ans = Read-Host "Download processing models now? [Y/n]"
    if ($ans -notmatch '^[nN]') { & $vpy scripts\download_models.py --processing }
    Write-Host ""
    Write-Host "    XTTS v2 (~1.9 GB) is licensed under the Coqui Public Model License: NON-COMMERCIAL use only."
    Write-Host "    https://coqui.ai/cpml"
    $ans = Read-Host "Do you accept the CPML and want to download XTTS v2 now? [y/N]"
    if ($ans -match '^[yY]') { & $vpy scripts\download_models.py --xtts --accept-cpml }
}

Step "Done"
Write-Host "    Start the app with:  run.bat   (or .venv\Scripts\python.exe -m app.main)"
Write-Host "    Optional GPU smoke test:  .venv\Scripts\python.exe scripts\gpu_smoke_test.py --accept-cpml"
