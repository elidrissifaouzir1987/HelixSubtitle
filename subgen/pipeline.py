"""Orchestration de bout en bout : vidéo -> sous-titres traduits -> vidéo sous-titrée."""
from __future__ import annotations

import tempfile
from pathlib import Path

from .attach import attach
from .config import Config
from .subtitles import SubtitleDoc, write
from .textnorm import clean_text
from .transcribe import transcribe
from .translate import build_translator
from .utils import extract_audio, log, require_ffmpeg


class Cancelled(RuntimeError):
    """Levée quand l'utilisateur annule le traitement."""


def process(video: Path, cfg: Config, cancel_event=None) -> dict:
    def ck():  # point de contrôle d'annulation (coopératif, entre étapes)
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled("Traitement annulé.")

    video = Path(video).resolve()
    if not video.exists():
        raise FileNotFoundError(f"Vidéo introuvable : {video}")
    ffmpeg = require_ffmpeg()
    out_dir = Path(cfg.get("io", "output_dir", default="output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="subgen_"))
    results: dict = {"subtitles": [], "video": None}

    try:
        # 1) audio
        ck()
        wav = extract_audio(ffmpeg, video, tmp / "audio.wav")

        # 2) ASR + alignement (+ diarisation)
        ck()
        doc: SubtitleDoc = transcribe(wav, cfg)
        if not doc.segments:
            raise RuntimeError("Aucun segment transcrit (audio vide ou silencieux ?).")

        # 3) traduction
        ck()
        if cfg.get("translate", "enabled", default=True):
            translator = build_translator(cfg)
            doc = translator.apply(doc, cfg.get("translate", "target_lang", default="fr"))

        # 4) nettoyage du texte (anti-carreaux : tatweel, harakat, invisibles)
        if cfg.get("subtitles", "clean_text", default=True):
            sd = cfg.get("subtitles", "strip_diacritics", default=True)
            for s in doc.segments:
                if s.translation is not None:
                    s.translation = clean_text(s.translation, strip_diacritics=sd)
                else:
                    s.text = clean_text(s.text, strip_diacritics=sd)

        # écriture des fichiers de sous-titres
        formats = cfg.get("subtitles", "formats", default=["srt"])
        mc = int(cfg.get("subtitles", "max_line_chars", default=42))
        ml = int(cfg.get("subtitles", "max_lines", default=2))
        style = cfg.get("attach", "ass_style", default="")
        lang = cfg.get("translate", "target_lang", default="fr")
        primary: Path | None = None
        for fmt in formats:
            path = out_dir / f"{video.stem}.{lang}.{fmt}"
            write(doc, path, fmt, max_chars=mc, max_lines=ml, ass_style=style)
            results["subtitles"].append(str(path))
            log.info("Sous-titres écrits : %s", path)
            primary = primary or path

        # 5) attache à la vidéo
        ck()
        if cfg.get("attach", "enabled", default=True) and cfg.get("attach", "mode") != "none":
            mode = cfg.get("attach", "mode", default="soft")
            sub = primary
            if mode == "hard":  # burn-in préfère ASS si dispo
                ass = next((Path(p) for p in results["subtitles"] if p.endswith(".ass")), None)
                sub = ass or primary
            results["video"] = str(attach(ffmpeg, video, sub, cfg, out_dir))
    finally:
        if not cfg.get("io", "keep_temp", default=False):
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            log.info("Fichiers temporaires conservés : %s", tmp)

    return results
