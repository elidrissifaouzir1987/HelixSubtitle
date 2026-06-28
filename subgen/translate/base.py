"""Interface commune des moteurs de traduction + application au document."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..subtitles import SubtitleDoc
from ..utils import log


class Translator(ABC):
    name = "base"

    @abstractmethod
    def translate_batch(self, texts: list[str], src: str, tgt: str) -> list[str]:
        """Traduit une liste de textes ; renvoie une liste de même longueur."""

    def apply(self, doc: SubtitleDoc, target_lang: str) -> SubtitleDoc:
        src = (doc.language or "en").split("-")[0]
        tgt = target_lang.split("-")[0]
        if src == tgt:
            log.info("Langue source == cible (%s) — pas de traduction.", tgt)
            for s in doc.segments:
                s.translation = s.text
            return doc
        texts = [s.text for s in doc.segments]
        log.info("Traduction %s -> %s via %s (%d segments)…", src, tgt, self.name, len(texts))
        out = self.translate_batch(texts, src, tgt)
        if len(out) != len(texts):
            raise RuntimeError(
                f"Le moteur {self.name} a renvoyé {len(out)} lignes pour {len(texts)}."
            )
        for seg, t in zip(doc.segments, out):
            seg.translation = t.strip()
        return doc
