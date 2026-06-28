"""Orchestration : vidéo -> sous-titres traduits (multi-langues, bilingue) -> vidéo.

Exposé en briques réutilisables (prepare_source / translate_for / write_docs /
attach_docs) pour permettre un flux de révision dans l'interface web.
"""
from __future__ import annotations

import copy
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


def _ck(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled("Traitement annulé.")


def get_targets(cfg: Config) -> list[str]:
    """Liste des langues cibles (target_langs prioritaire sur target_lang)."""
    tl = cfg.get("translate", "target_langs")
    if tl:
        return [str(x) for x in tl]
    return [cfg.get("translate", "target_lang", default="fr")]


# ---- briques ----

def prepare_source(video: Path, cfg: Config, ffmpeg: str, tmp: Path, cancel_event=None) -> SubtitleDoc:
    """Extraction audio + transcription/alignement/diarisation -> document source."""
    _ck(cancel_event)
    wav = extract_audio(ffmpeg, video, tmp / "audio.wav")
    _ck(cancel_event)
    doc = transcribe(wav, cfg)
    if not doc.segments:
        raise RuntimeError("Aucun segment transcrit (audio vide ou silencieux ?).")
    return doc


def _clean(doc: SubtitleDoc, cfg: Config) -> None:
    if not cfg.get("subtitles", "clean_text", default=True):
        return
    sd = cfg.get("subtitles", "strip_diacritics", default=True)
    for s in doc.segments:
        if s.translation is not None:
            s.translation = clean_text(s.translation, strip_diacritics=sd)
        else:
            s.text = clean_text(s.text, strip_diacritics=sd)


def translate_for(src_doc: SubtitleDoc, translator, cfg: Config, lang: str,
                  cancel_event=None) -> SubtitleDoc:
    """Copie le document source et le traduit vers `lang`, puis nettoie."""
    _ck(cancel_event)
    doc = SubtitleDoc(segments=[copy.copy(s) for s in src_doc.segments],
                      language=src_doc.language)
    translator.apply(doc, lang)
    _clean(doc, cfg)
    return doc


def build_docs(src_doc: SubtitleDoc, cfg: Config, cancel_event=None) -> dict[str, SubtitleDoc]:
    """Construit un document par langue cible (ou le document source si pas de traduction)."""
    if not cfg.get("translate", "enabled", default=True):
        d = SubtitleDoc(segments=[copy.copy(s) for s in src_doc.segments],
                        language=src_doc.language)
        _clean(d, cfg)
        return {src_doc.language or "src": d}
    translator = build_translator(cfg)
    docs: dict[str, SubtitleDoc] = {}
    for lang in get_targets(cfg):
        docs[lang] = translate_for(src_doc, translator, cfg, lang, cancel_event)
    return docs


def write_docs(docs: dict[str, SubtitleDoc], video: Path, cfg: Config,
               out_dir: Path) -> dict[str, dict[str, Path]]:
    """Écrit les fichiers de sous-titres. Renvoie {langue: {format: chemin}}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = list(cfg.get("subtitles", "formats", default=["srt"]))
    attach_on = cfg.get("attach", "enabled", default=True) and cfg.get("attach", "mode") != "none"
    if attach_on and "srt" not in formats:  # srt requis pour le mux/burn
        formats.append("srt")
    bilingual = cfg.get("subtitles", "bilingual", default=False)
    mc = int(cfg.get("subtitles", "max_line_chars", default=42))
    ml = int(cfg.get("subtitles", "max_lines", default=2))
    style = cfg.get("attach", "ass_style", default="")
    written: dict[str, dict[str, Path]] = {}
    for lang, doc in docs.items():
        tag = f"{lang}.bi" if bilingual else lang
        written[lang] = {}
        for fmt in formats:
            path = out_dir / f"{video.stem}.{tag}.{fmt}"
            write(doc, path, fmt, max_chars=mc, max_lines=ml, ass_style=style, bilingual=bilingual)
            written[lang][fmt] = path
            log.info("Sous-titres écrits : %s", path)
    return written


def attach_docs(ffmpeg: str, video: Path, written: dict[str, dict[str, Path]],
                cfg: Config, out_dir: Path) -> str | None:
    """Attache les sous-titres écrits (mux multi-pistes ou burn-in de la 1re langue)."""
    if not (cfg.get("attach", "enabled", default=True) and cfg.get("attach", "mode") != "none"):
        return None
    mode = cfg.get("attach", "mode", default="soft")
    attach_list: list[tuple[str, Path]] = []
    for lang, fmts in written.items():
        if mode == "hard" and "ass" in fmts:
            chosen = fmts["ass"]
        else:
            chosen = fmts.get("srt") or next(iter(fmts.values()))
        attach_list.append((lang, chosen))
    return str(attach(ffmpeg, video, attach_list, cfg, out_dir))


def finalize_outputs(ffmpeg: str, video: Path, docs: dict, cfg: Config, out_dir: Path,
                     cancel_event=None) -> tuple[list[str], str | None, list[str]]:
    """Écrit les sous-titres puis produit la vidéo finale.

    Sans doublage : sous-titres + vidéo (mux/burn).
    Avec doublage : **un seul fichier** vidéo avec pistes audio sélectionnables
    (original + langue(s) doublée(s)) + sous-titres.
    Renvoie (sous-titres, vidéo, doublages).
    """
    written = write_docs(docs, video, cfg, out_dir)
    subs = [str(p) for fmts in written.values() for p in fmts.values()]
    if cfg.get("dub", "enabled", default=False):
        from .dub.build import combined_output
        combined = combined_output(ffmpeg, video, written, docs, cfg, out_dir, cancel_event)
        if cfg.get("lipsync", "enabled", default=False):
            _ck(cancel_event)
            from .lipsync import lipsync_combined
            final = lipsync_combined(ffmpeg, video, Path(combined),
                                     get_targets(cfg)[0], cfg, out_dir)
            return subs, final, []
        return subs, combined, []  # fichier unique : pistes audio dans la vidéo
    return subs, attach_docs(ffmpeg, video, written, cfg, out_dir), []


# ---- pipeline complet (CLI) ----

def process(video: Path, cfg: Config, cancel_event=None) -> dict:
    video = Path(video).resolve()
    if not video.exists():
        raise FileNotFoundError(f"Vidéo introuvable : {video}")
    ffmpeg = require_ffmpeg()
    out_dir = Path(cfg.get("io", "output_dir", default="output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="subgen_"))
    results: dict = {"subtitles": [], "video": None, "dubs": []}
    try:
        src = prepare_source(video, cfg, ffmpeg, tmp, cancel_event)
        docs = build_docs(src, cfg, cancel_event)
        _ck(cancel_event)
        subs, vid, dubs = finalize_outputs(ffmpeg, video, docs, cfg, out_dir, cancel_event)
        results["subtitles"], results["video"], results["dubs"] = subs, vid, dubs
    finally:
        if not cfg.get("io", "keep_temp", default=False):
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            log.info("Fichiers temporaires conservés : %s", tmp)
    return results
