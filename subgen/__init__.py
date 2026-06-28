"""subgen — pipeline de génération de sous-titres traduits et d'attache à une vidéo.

Étapes : extraction audio -> WhisperX (ASR + alignement + diarisation) ->
traduction (NLLB / LLM / API) -> écriture SRT/ASS -> attache ffmpeg (mux ou burn-in).
"""

import os as _os

# Windows sans droits admin / mode dev : les liens symboliques échouent (WinError 1314).
# On force Hugging Face à copier les fichiers de cache au lieu de les lier.
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

__version__ = "1.0.0"
