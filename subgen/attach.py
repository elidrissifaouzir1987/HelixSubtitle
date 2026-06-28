"""Attache des sous-titres à la vidéo via ffmpeg : mux (soft, multi-pistes) ou burn-in (hard)."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .utils import log, run

# ISO 639-1 -> 639-2/B (métadonnées de piste)
_ISO2 = {"fr": "fra", "en": "eng", "es": "spa", "de": "deu", "it": "ita", "pt": "por",
         "nl": "nld", "ru": "rus", "ar": "ara", "zh": "zho", "ja": "jpn", "ko": "kor",
         "tr": "tur", "pl": "pol", "uk": "ukr", "hi": "hin", "vi": "vie", "id": "ind",
         "fa": "fas", "he": "heb", "sv": "swe", "cs": "ces", "ro": "ron", "el": "ell",
         "th": "tha", "hu": "hun", "fi": "fin", "da": "dan", "no": "nor", "ca": "cat"}


def _iso(lang: str) -> str:
    lang = lang.split("-")[0]
    return _ISO2.get(lang, lang)


def _escape_filter(path: Path) -> str:
    """Échappe un chemin Windows pour le filtre ffmpeg 'subtitles='."""
    p = str(path.resolve()).replace("\\", "/")
    return p.replace(":", "\\:").replace("'", r"\'")


def attach(ffmpeg: str, video: Path, subs: list[tuple[str, Path]], cfg: Config, out_dir: Path) -> Path:
    """subs : liste de (langue, fichier de sous-titres). Le premier est la piste par défaut."""
    if not subs:
        raise ValueError("Aucun sous-titre à attacher.")
    mode = cfg.get("attach", "mode", default="soft")
    container = cfg.get("attach", "container", default="mp4")
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "soft":
        out = out_dir / f"{video.stem}.subbed.{container}"
        _soft(ffmpeg, video, subs, out, container)
    elif mode == "hard":
        out = out_dir / f"{video.stem}.hardsub.{container}"
        _hard(ffmpeg, video, subs[0][1], out, cfg)
    else:
        raise ValueError(f"Mode d'attache inconnu : {mode}")
    log.info("Vidéo générée : %s", out)
    return out


def _soft(ffmpeg, video, subs, out, container):
    scodec = "mov_text" if container == "mp4" else "srt"
    log.info("Mux des sous-titres (soft, %s, %d piste(s))…", container, len(subs))
    cmd = [ffmpeg, "-y", "-i", str(video)]
    for _, path in subs:
        cmd += ["-i", str(path)]
    cmd += ["-map", "0:v:0", "-map", "0:a?"]
    for idx in range(len(subs)):
        cmd += ["-map", str(idx + 1)]
    cmd += ["-c:v", "copy", "-c:a", "copy", "-c:s", scodec]
    for idx, (lang, _) in enumerate(subs):
        cmd += [f"-metadata:s:s:{idx}", f"language={_iso(lang)}"]
    cmd += ["-disposition:s:0", "default", str(out)]
    run(cmd, "mux sous-titres")


def _hard(ffmpeg, video, subtitle, out, cfg: Config):
    encoder = "hevc_nvenc" if cfg.get("attach", "hevc", default=False) else "h264_nvenc"
    cq = str(cfg.get("attach", "crf_cq", default=23))
    sub = Path(subtitle)
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
