"""Moteur LLM (Anthropic API ou Ollama local) — traduction contextuelle de qualité."""
from __future__ import annotations

import json
import os

import requests

from .base import Translator
from ..utils import log

SYSTEM = (
    "Tu es un traducteur professionnel de sous-titres. On te donne un tableau JSON de "
    "répliques dans la langue {src}. Traduis chaque élément en {tgt} de façon naturelle, "
    "concise et adaptée à la lecture rapide à l'écran. Conserve le sens, le registre et la "
    "ponctuation. Ne fusionne ni ne sépare les éléments. Réponds UNIQUEMENT par un tableau "
    "JSON de chaînes, de même longueur et même ordre que l'entrée, sans aucun autre texte."
)


class LLMTranslator(Translator):
    name = "llm"

    def __init__(self, provider: str, model: str, ollama_host: str = "http://localhost:11434",
                 chunk: int = 20):
        self.provider = provider
        self.model = model
        self.ollama_host = ollama_host.rstrip("/")
        self.chunk = chunk
        if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Clé Anthropic manquante (Réglages) pour le backend llm/anthropic.")
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Clé OpenAI manquante (Réglages) pour le backend llm/openai.")

    def _call(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(system, user)
        if self.provider == "openai":
            return self._call_openai(system, user)
        return self._call_ollama(system, user)

    def translate_batch(self, texts, src, tgt):
        out: list[str] = []
        for i in range(0, len(texts), self.chunk):
            chunk = texts[i : i + self.chunk]
            out.extend(self._translate_chunk(chunk, src, tgt))
            log.debug("LLM %d/%d", min(i + self.chunk, len(texts)), len(texts))
        return out

    def _translate_chunk(self, chunk: list[str], src: str, tgt: str) -> list[str]:
        sys = SYSTEM.format(src=src, tgt=tgt)
        user = json.dumps(chunk, ensure_ascii=False)
        raw = self._call(sys, user)
        parsed = self._parse(raw, len(chunk))
        if parsed is None:
            log.warning("Réponse LLM non parsable, repli ligne-par-ligne sur ce bloc.")
            return [self._one(t, src, tgt) for t in chunk]
        return parsed

    def _one(self, text: str, src: str, tgt: str) -> str:
        sys = (f"Traduis ce sous-titre de {src} vers {tgt}. Réponds uniquement par la "
               f"traduction, sans guillemets ni commentaire.")
        raw = self._call(sys, text)
        return raw.strip()

    @staticmethod
    def _parse(raw: str, n: int):
        s = raw.strip()
        if "[" in s and "]" in s:
            s = s[s.index("[") : s.rindex("]") + 1]
        try:
            arr = json.loads(s)
            if isinstance(arr, list) and len(arr) == n:
                return [str(x) for x in arr]
        except Exception:
            pass
        return None

    def _call_anthropic(self, system: str, user: str) -> str:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": self.model, "max_tokens": 4096, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=120,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))

    def _call_openai(self, system: str, user: str) -> str:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                     "content-type": "application/json"},
            json={"model": self.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, system: str, user: str) -> str:
        r = requests.post(
            f"{self.ollama_host}/api/chat",
            json={"model": self.model, "stream": False,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=300,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
