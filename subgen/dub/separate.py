"""Séparation voix / fond sonore via Demucs (préserve musique & bruitages)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..utils import log, run


def _extract_hifi(ffmpeg: str, video: Path, out_wav: Path) -> Path:
    """Extrait l'audio d'origine en qualité (stéréo 44.1 kHz) pour la séparation."""
    run([ffmpeg, "-y", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100",
         "-c:a", "pcm_s16le", str(out_wav)], "extraction audio hi-fi")
    return out_wav


def separate_background(ffmpeg: str, video: Path, tmp: Path, device: str = "cuda") -> Path | None:
    """Renvoie l'accompagnement (musique/bruitages sans voix), ou None si échec.

    Utilise Demucs en mode two-stems (vocals / no_vocals) via sous-processus,
    avec le Python du venv courant.
    """
    audio = _extract_hifi(ffmpeg, video, tmp / "orig.wav")
    outdir = tmp / "demucs"
    log.info("Séparation voix/fond (Demucs, %s)…", device)
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
           "-d", device, "-o", str(outdir), str(audio)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        log.warning("Demucs a échoué — fond non préservé.\n%s", "\n".join(tail))
        return None
    matches = list(outdir.glob("**/no_vocals.wav"))
    if not matches:
        log.warning("Demucs : accompagnement introuvable — fond non préservé.")
        return None
    log.info("Fond sonore extrait : %s", matches[0].name)
    return matches[0]
