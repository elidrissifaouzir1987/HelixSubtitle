# Crée le venv isolé pour le clonage de voix XTTS (Phase 2b du doublage).
# Isolé du venv principal car coqui-tts tire transformers/hub incompatibles avec WhisperX.
# Usage :  .\engines\xtts\setup.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # racine du projet
$basePy = Join-Path $root ".venv\Scripts\python.exe"           # python de base pour créer le venv
$venv = Join-Path $PSScriptRoot ".venv"
$py = Join-Path $venv "Scripts\python.exe"

Write-Host "Création du venv isolé : $venv"
& $basePy -m venv $venv
& $py -m pip install --upgrade pip

# IMPORTANT (pièges rencontrés) :
#  - torch 2.8 cu128 : compatible RTX 50xx ET dispose des backends audio (torch >= 2.9
#    exige torchcodec, cassé sous Windows). On épingle donc 2.8.
#  - transformers < 5 : coqui-tts 0.27 utilise une API retirée dans transformers 5.
& $py -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install "coqui-tts" "transformers<5" soundfile

& $py -c "import torch, TTS; print('XTTS prêt — torch', torch.__version__, '| cuda', torch.cuda.is_available())"
Write-Host "OK. Le modèle XTTS (~1.8 Go) se télécharge au premier doublage cloné."
