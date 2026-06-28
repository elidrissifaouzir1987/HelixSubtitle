"""Chargement de la configuration : defaults <- config.yaml <- CLI."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "asr": {
        "model": "large-v3",          # ou large-v3-turbo, medium...
        "device": "cuda",             # cuda | cpu
        "compute_type": "float16",    # float16 (GPU) | int8 (CPU/éco VRAM)
        "batch_size": 16,
        "language": None,             # None = détection auto
        "align": True,                # alignement mot-à-mot
        "diarize": False,             # nécessite un token Hugging Face
        "min_speakers": None,
        "max_speakers": None,
    },
    "translate": {
        "enabled": True,
        "target_lang": "fr",          # langue cible (ISO 639-1)
        "backend": "nllb",            # nllb | llm | api
        "nllb": {"model": "facebook/nllb-200-distilled-600M", "device": "cuda"},
        "llm": {
            "provider": "anthropic",  # anthropic | ollama
            "model": "claude-opus-4-8",
            "ollama_host": "http://localhost:11434",
            "context_window": 6,      # nb de lignes voisines données comme contexte
        },
        "api": {"provider": "deepl"}, # deepl (clé via DEEPL_API_KEY)
    },
    "subtitles": {
        "formats": ["srt"],           # srt | ass | vtt
        "max_line_chars": 42,
        "max_lines": 2,
        "max_cps": 17,                # caractères/seconde (lisibilité)
        "clean_text": True,           # retire tatweel/invisibles (anti-carreaux)
        "strip_diacritics": True,     # retire harakat arabes (sous-titres plus nets)
    },
    "attach": {
        "enabled": True,
        "mode": "soft",               # soft (mux) | hard (burn-in) | none
        "container": "mp4",           # mp4 | mkv
        "hevc": False,                # burn-in : True = hevc_nvenc, False = h264_nvenc
        "crf_cq": 23,
        "ass_style": "FontName=Arial,FontSize=22,Outline=2,Shadow=0",
    },
    "io": {"output_dir": "output", "keep_temp": False},
    "hf_token": None,                 # sinon variable d'env HF_TOKEN
    "verbose": False,
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    data: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS))

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        data = copy.deepcopy(DEFAULTS)
        if path and Path(path).exists():
            with open(path, encoding="utf-8") as f:
                data = _deep_merge(data, yaml.safe_load(f) or {})
        # tokens via environnement
        data["hf_token"] = data.get("hf_token") or os.environ.get("HF_TOKEN")
        return cls(data)

    def get(self, *keys, default=None):
        cur: Any = self.data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def override(self, dotted: str, value: Any) -> None:
        keys = dotted.split(".")
        cur = self.data
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
