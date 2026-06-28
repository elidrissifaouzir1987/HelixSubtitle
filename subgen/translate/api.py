"""Moteur API en ligne — DeepL (clé via DEEPL_API_KEY)."""
from __future__ import annotations

import os

import requests

from .base import Translator
from ..utils import log


class DeepLTranslator(Translator):
    name = "deepl"

    def __init__(self):
        self.key = os.environ.get("DEEPL_API_KEY")
        if not self.key:
            raise RuntimeError("DEEPL_API_KEY manquant pour le backend api/deepl.")
        # clés ':fx' = offre gratuite -> endpoint api-free
        self.url = ("https://api-free.deepl.com/v2/translate"
                    if self.key.endswith(":fx")
                    else "https://api.deepl.com/v2/translate")

    def translate_batch(self, texts, src, tgt):
        out: list[str] = []
        # DeepL accepte plusieurs 'text' par requête (max ~50)
        for i in range(0, len(texts), 50):
            chunk = texts[i : i + 50]
            data = [("text", t) for t in chunk]
            data += [("source_lang", src.upper()), ("target_lang", tgt.upper())]
            r = requests.post(self.url, headers={"Authorization": f"DeepL-Auth-Key {self.key}"},
                              data=data, timeout=60)
            r.raise_for_status()
            out.extend(tr["text"] for tr in r.json()["translations"])
            log.debug("DeepL %d/%d", min(i + 50, len(texts)), len(texts))
        return out
