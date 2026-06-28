"""Attache des sous-titres à la vidéo via ffmpeg : mux (soft) ou burn-in (hard)."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .utils import log, run

# ISO 639-1 -> 639-2/B (métadonnées de piste)
_ISO2 = {"fr": "fra", "en": "eng", "es": "spa", "de": "deu", "it": "ita", "pt": "por",
         "nl": "nld", "ru": "rus", "ar": "ara", "zh": "zho", "ja": "jpn", "ko": "kor"}


def _escape_filter(path: Path) -> str:
    """Échappe un chemin Windows pour le filtre ffmpeg 'subtitles='."""
    p = str(path.resolve()).replace("\\", "/")
    return p.replace(":", "\\:").replace("'", r"\'")


def attach(ffmpeg: str, video: Path, subtitle: Path, cfg: Config, out_dir: Path) -> Path:
    mode = cfg.get("attach", "mode", default="soft")
    container = cfg.get("attach", "container", default="mp4")
    lang = cfg.get("translate", "target_lang", default="fr").split("-")[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "soft":
        out = out_dir / f"{video.stem}.subbed.{container}"
        _soft(ffmpeg, video, subtitle, out, container, lang)
    elif mode == "hard":
        out = out_dir / f"{video.stem}.hardsub.{container}"
        _hard(ffmpeg, video, subtitle, out, cfg)
    else:
        raise ValueError(f"Mode d'attache inconnu : {mode}")
    log.info("Vidéo générée : %s", out)
    return out


def _soft(ffmpeg, video, subtitle, out, container, lang):
    # mkv -> srt ; mp4 -> mov_text
    scodec = "mov_text" if container == "mp4" else "srt"
    iso = _ISO2.get(lang, lang)
    log.info("Mux des sous-titres (soft, %s)…", container)
    run([ffmpeg, "-y", "-i", str(video), "-i", str(subtitle),
         "-map", "0", "-map", "1",
         "-c:v", "copy", "-c:a", "copy", "-c:s", scodec,
         "-metadata:s:s:0", f"language={iso}",
         "-disposition:s:0", "default", str(out)],
        "mux sous-titres")


def _hard(ffmpeg, video, subtitle, out, cfg: Config):
    encoder = "hevc_nvenc" if cfg.get("attach", "hevc", default=False) else "h264_nvenc"
    cq = str(cfg.get("attach", "crf_cq", default=23))
    sub = subtitle
    vf = f"subtitles='{_escape_filter(sub)}'"
    if sub.suffix.lower() == ".srt":  # applique un style aux .srt
        style = cfg.get("attach", "ass_style", default="")
        if style:
            vf += f":force_style='{style}'"
    log.info("Burn-in des sous-titres (hard, %s)…", encoder)
    run([ffmpeg, "-y", "-i", str(video), "-vf", vf,
         "-c:v", encoder, "-cq", cq, "-preset", "p5",
         "-c:a", "copy", str(out)],
        "burn-in sous-titres")
