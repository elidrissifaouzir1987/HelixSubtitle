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
    p.add_argument("-t", "--target", help="Langue cible (ISO 639-1, ex. fr)")
    p.add_argument("-b", "--backend", choices=["nllb", "llm", "api"], help="Moteur de traduction")
    p.add_argument("-m", "--model", help="Modèle ASR (ex. large-v3, large-v3-turbo)")
    p.add_argument("--mode", choices=["soft", "hard", "none"], help="Attache : mux / burn-in / aucune")
    p.add_argument("--container", choices=["mp4", "mkv"], help="Conteneur de sortie")
    p.add_argument("--formats", help="Formats de sous-titres, séparés par des virgules (srt,ass,vtt)")
    p.add_argument("--diarize", action="store_true", help="Active la diarisation (token HF requis)")
    p.add_argument("--no-translate", action="store_true", help="Désactive la traduction")
    p.add_argument("--device", choices=["cuda", "cpu"], help="Périphérique de calcul")
    p.add_argument("-o", "--output", help="Dossier de sortie")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def apply_args(cfg: Config, a: argparse.Namespace) -> None:
    if a.source: cfg.override("asr.language", a.source)
    if a.target: cfg.override("translate.target_lang", a.target)
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
    log.info("Terminé. Sous-titres : %s | Vidéo : %s",
             ", ".join(res["subtitles"]) or "—", res["video"] or "—")
    return 0


if __name__ == "__main__":
    sys.exit(main())
