"""Utilitaires : logging, ffmpeg, formatage temps."""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("subgen")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    root = logging.getLogger("subgen")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg introuvable sur le PATH. Installe-le et relance "
            "(ex. winget install Gyan.FFmpeg)."
        )
    return exe


def run(cmd: list[str], desc: str = "") -> None:
    """Lance une commande et lève une erreur claire en cas d'échec."""
    log.debug("exec: %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-15:]
        raise RuntimeError(f"Échec {desc or cmd[0]} (code {proc.returncode}):\n" + "\n".join(tail))


def download_youtube(url: str, out_dir: Path, prefix: str, quality: str = "best") -> Path:
    """Télécharge une vidéo (YouTube ou autre site géré par yt-dlp) dans out_dir.

    quality : "best" (max) ou une hauteur max en pixels ("1080", "720", "480").
    """
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl = str(out_dir / f"{prefix}_%(title).80s.%(ext)s")
    if quality and quality.isdigit():
        fmt = f"bv*[height<={quality}]+ba/b[height<={quality}]/b"
    else:
        fmt = "bv*+ba/b"
    holder: dict = {}

    def hook(d):
        if d.get("status") == "downloading":
            pct = (d.get("_percent_str") or "").strip()
            if pct:
                log.info("Téléchargement YouTube %s", pct)
        elif d.get("status") == "finished":
            log.info("Téléchargement terminé, fusion…")

    ydl_opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": tmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    log.info("Récupération de la vidéo depuis le lien…")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        holder["name"] = ydl.prepare_filename(info)

    p = Path(holder["name"])
    if not p.exists():  # après fusion l'extension peut devenir .mp4
        cand = sorted(out_dir.glob(f"{prefix}_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not cand:
            raise RuntimeError("Le téléchargement a échoué (aucun fichier produit).")
        p = cand[0]
    return p


def expand_url(url: str) -> list[str]:
    """Renvoie la liste des URLs vidéo d'un lien (1 si vidéo simple, N si playlist)."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") if isinstance(info, dict) else None
    if not entries:
        return [url]
    urls: list[str] = []
    for e in entries:
        if not e:
            continue
        u = e.get("url") or e.get("webpage_url") or e.get("id")
        if not u:
            continue
        if not str(u).startswith("http"):
            u = f"https://www.youtube.com/watch?v={u}"
        urls.append(u)
    return urls or [url]


def media_duration(path: Path) -> float:
    """Durée en secondes d'un fichier média (via ffprobe ; 0.0 si inconnu)."""
    probe = shutil.which("ffprobe")
    if not probe:
        return 0.0
    proc = subprocess.run(
        [probe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        return float((proc.stdout or "").strip())
    except ValueError:
        return 0.0


def has_audio_stream(video: Path) -> bool:
    """Vrai si la vidéo contient au moins une piste audio (via ffprobe)."""
    probe = shutil.which("ffprobe")
    if not probe:  # pas de ffprobe : on laisse ffmpeg tenter
        return True
    proc = subprocess.run(
        [probe, "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return bool((proc.stdout or "").strip())


def extract_audio(ffmpeg: str, video: Path, out_wav: Path) -> Path:
    """Extrait un WAV 16 kHz mono (format attendu par Whisper)."""
    if not has_audio_stream(video):
        raise RuntimeError(
            "La vidéo ne contient aucune piste audio — impossible de générer "
            "des sous-titres. Vérifie le fichier (ce n'est peut-être qu'un extrait muet)."
        )
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run(
        [ffmpeg, "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(out_wav)],
        "extraction audio",
    )
    return out_wav


def fmt_timestamp(seconds: float, *, comma: bool = True) -> str:
    """Formate un temps en HH:MM:SS,mmm (SRT) ou HH:MM:SS.mmm (ASS/VTT)."""
    if seconds < 0:
        seconds = 0.0
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"
