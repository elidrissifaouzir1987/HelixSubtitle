"""Lip-sync (Phase 3) : aligne les bouches à l'écran sur l'audio doublé.

Utilise Wav2Lip dans un venv isolé (engines/lipsync) appelé en sous-processus.
S'applique APRÈS le doublage : synchronise la vidéo d'origine sur la voix doublée,
puis remux avec toutes les pistes audio + sous-titres du fichier doublé.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .attach import _iso
from .config import Config
from .utils import log, run

ROOT = Path(__file__).resolve().parent.parent
LS_PY = ROOT / "engines" / "lipsync" / ".venv" / "Scripts" / "python.exe"
W2L_DIR = ROOT / "engines" / "lipsync" / "Wav2Lip"
W2L_CKPT = W2L_DIR / "checkpoints" / "wav2lip_gan.pth"


def available() -> bool:
    return LS_PY.exists() and (W2L_DIR / "inference.py").exists() and W2L_CKPT.exists()


def _run_wav2lip(face_video: Path, speech_wav: Path, out_video: Path) -> None:
    """Lance Wav2Lip : visage (vidéo) + parole (wav) -> vidéo lip-syncée."""
    (W2L_DIR / "temp").mkdir(exist_ok=True)
    (W2L_DIR / "results").mkdir(exist_ok=True)
    cmd = [str(LS_PY), "inference.py", "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
           "--face", str(face_video), "--audio", str(speech_wav),
           "--outfile", str(out_video), "--nosmooth"]
    log.info("Synchronisation des lèvres (Wav2Lip)… (peut être long)")
    proc = subprocess.run(cmd, cwd=str(W2L_DIR), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not out_video.exists():
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise RuntimeError("Wav2Lip a échoué :\n" + "\n".join(tail))


def lipsync_combined(ffmpeg: str, original_video: Path, combined_video: Path,
                     target_lang: str, cfg: Config, out_dir: Path) -> str:
    """Lip-sync sur la voix doublée + remux des pistes du fichier doublé.

    Renvoie le chemin de la vidéo finale lip-syncée.
    """
    if not available():
        raise RuntimeError(
            "Lip-sync indisponible : venv isolé manquant (engines/lipsync). "
            "Lance engines/lipsync/setup.ps1."
        )
    container = cfg.get("attach", "container", default="mp4")
    out = out_dir / f"{original_video.stem}.lipsync.{container}"
    tmp = Path(tempfile.mkdtemp(prefix="subgen_ls_"))
    try:
        # 1) parole cible = piste audio doublée du fichier combiné (sélection par langue)
        speech = tmp / "speech.wav"
        run([ffmpeg, "-y", "-i", str(combined_video), "-map", f"0:m:language:{_iso(target_lang)}",
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(speech)],
            "extraction parole doublée")
        # 2) lip-sync (frames d'origine alignées sur la voix doublée)
        ls = tmp / "ls.mp4"
        _run_wav2lip(original_video, speech, ls)
        # 3) remux : image lip-syncée + toutes les pistes audio/sous-titres du doublage
        run([ffmpeg, "-y", "-i", str(ls), "-i", str(combined_video),
             "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
             "-c", "copy", str(out)], "remux lip-sync")
        log.info("Vidéo lip-syncée : %s", out)
        return str(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
