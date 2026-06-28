"""Fabrique de moteurs de traduction selon la config."""
from __future__ import annotations

from ..config import Config
from .base import Translator


def build_translator(cfg: Config) -> Translator:
    backend = cfg.get("translate", "backend", default="nllb")

    if backend == "nllb":
        from .nllb import NLLBTranslator
        return NLLBTranslator(
            model_name=cfg.get("translate", "nllb", "model", default="facebook/nllb-200-distilled-600M"),
            device=cfg.get("translate", "nllb", "device", default="cuda"),
        )
    if backend == "llm":
        from .llm import LLMTranslator
        return LLMTranslator(
            provider=cfg.get("translate", "llm", "provider", default="anthropic"),
            model=cfg.get("translate", "llm", "model", default="claude-opus-4-8"),
            ollama_host=cfg.get("translate", "llm", "ollama_host", default="http://localhost:11434"),
        )
    if backend == "api":
        provider = cfg.get("translate", "api", "provider", default="deepl")
        if provider == "deepl":
            from .api import DeepLTranslator
            return DeepLTranslator()
        raise ValueError(f"Fournisseur API inconnu : {provider}")
    raise ValueError(f"Backend de traduction inconnu : {backend}")
