"""Interface en ligne de commande."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .pipeline import process
from .utils import log, setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subgen",
        description="Génère des sous-titres traduits et les attache à une vidéo (WhisperX + ffmpeg).",
    )
    p.add_argument("video", help="Chemin de la vidéo d'entrée")
    p.add_argument("-c", "--config", help="Fichier config.yaml", default="config.yaml")
    p.add_argument("-s", "--source", help="Forcer la langue d'origine (ISO 639-1, ex. ja). Défaut : détection auto")
    p.add_argument("-t", "--target", help="Langue(s) cible(s) ISO 639-1, séparées par des virgules (ex. fr,ar)")
    p.add_argument("--bilingual", action="store_true", help="Sous-titres bilingues (source + traduction)")
    p.add_argument("-b", "--backend", choices=["nllb", "llm", "api"], help="Moteur de traduction")
    p.add_argument("-m", "--model", help="Modèle ASR (ex. large-v3, large-v3-turbo)")
    p.add_argument("--mode", choices=["soft", "hard", "none"], help="Attache : mux / burn-in / aucune")
    p.add_argument("--container", choices=["mp4", "mkv"], help="Conteneur de sortie")
    p.add_argument("--formats", help="Formats de sous-titres, séparés par des virgules (srt,ass,vtt)")
    p.add_argument("--diarize", action="store_true", help="Active la diarisation (token HF requis)")
    p.add_argument("--dub", action="store_true", help="Doublage : génère une voix synthétique dans la langue cible")
    p.add_argument("--voice", help="Voix Edge-TTS (ex. ar-SA-HamedNeural). Défaut : auto")
    p.add_argument("--clone", action="store_true", help="Doublage avec clonage de voix XTTS (venv isolé requis)")
    p.add_argument("--lipsync", action="store_true", help="Synchro labiale sur le doublage (venv isolé requis)")
    p.add_argument("--no-translate", action="store_true", help="Désactive la traduction")
    p.add_argument("--device", choices=["cuda", "cpu"], help="Périphérique de calcul")
    p.add_argument("-o", "--output", help="Dossier de sortie")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def apply_args(cfg: Config, a: argparse.Namespace) -> None:
    if a.source: cfg.override("asr.language", a.source)
    if a.target:
        langs = [x.strip() for x in a.target.split(",") if x.strip()]
        cfg.override("translate.target_lang", langs[0])
        cfg.override("translate.target_langs", langs if len(langs) > 1 else None)
    if a.bilingual: cfg.override("subtitles.bilingual", True)
    if a.dub: cfg.override("dub.enabled", True)
    if a.clone:
        cfg.override("dub.enabled", True)
        cfg.override("dub.backend", "xtts")
    if a.voice: cfg.override("dub.voice", a.voice)
    if a.lipsync:
        cfg.override("dub.enabled", True)   # le lip-sync nécessite l'audio doublé
        cfg.override("lipsync.enabled", True)
    if a.backend: cfg.override("translate.backend", a.backend)
    if a.model: cfg.override("asr.model", a.model)
    if a.mode: cfg.override("attach.mode", a.mode)
    if a.container: cfg.override("attach.container", a.container)
    if a.formats: cfg.override("subtitles.formats", [f.strip() for f in a.formats.split(",")])
    if a.diarize: cfg.override("asr.diarize", True)
    if a.no_translate: cfg.override("translate.enabled", False)
    if a.device:
        cfg.override("asr.device", a.device)
        cfg.override("translate.nllb.device", a.device)
    if a.output: cfg.override("io.output_dir", a.output)
    if a.verbose: cfg.override("verbose", True)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = Config.load(a.config)
    apply_args(cfg, a)
    setup_logging(cfg.get("verbose", default=False))
    try:
        res = process(Path(a.video), cfg)
    except KeyboardInterrupt:
        log.error("Interrompu.")
        return 130
    except Exception as e:
        log.error("Échec : %s", e)
        if cfg.get("verbose"):
            raise
        return 1
    log.info("Terminé. Sous-titres : %s | Vidéo : %s | Doublage : %s",
             ", ".join(res["subtitles"]) or "—", res["video"] or "—",
             ", ".join(res.get("dubs") or []) or "—")
    return 0


if __name__ == "__main__":
    sys.exit(main())
