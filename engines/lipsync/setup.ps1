# Installe l'environnement isolé de lip-sync (Wav2Lip) pour la Phase 3 du doublage.
# Isolé du venv principal ; le dépôt Wav2Lip et les modèles ne sont pas versionnés.
# Usage :  .\engines\lipsync\setup.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$basePy = Join-Path $root ".venv\Scripts\python.exe"
$venv = Join-Path $PSScriptRoot ".venv"
$py = Join-Path $venv "Scripts\python.exe"
$w2l = Join-Path $PSScriptRoot "Wav2Lip"

# 1) venv + dépendances (torch 2.8 cu128 : compatible RTX 50xx + backends audio)
& $basePy -m venv $venv
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install numpy opencv-python librosa numba scipy tqdm soundfile

# 2) dépôt Wav2Lip + patches de compatibilité
if (-not (Test-Path $w2l)) {
    git clone --depth 1 https://github.com/Rudrabha/Wav2Lip $w2l
}
& $py (Join-Path $PSScriptRoot "patch.py")
New-Item -ItemType Directory -Force (Join-Path $w2l "temp"), (Join-Path $w2l "results"), (Join-Path $w2l "checkpoints") | Out-Null

# 3) modèles (miroir GitHub stable)
$ck = Join-Path $w2l "checkpoints\wav2lip_gan.pth"
$s3 = Join-Path $w2l "face_detection\detection\sfd\s3fd.pth"
if (-not (Test-Path $ck)) {
    Invoke-WebRequest "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth" -OutFile $ck
}
if (-not (Test-Path $s3)) {
    Invoke-WebRequest "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/s3fd.pth" -OutFile $s3
}
& $py -c "import torch; print('Lip-sync prêt — torch', torch.__version__, '| cuda', torch.cuda.is_available())"
