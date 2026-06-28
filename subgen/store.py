"""Persistance locale : réglages (clés API) et historique des projets.

Fichiers sous data/ (gitignoré). Les clés ne quittent jamais la machine.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SETTINGS = DATA / "settings.json"
PROJECTS = DATA / "projects.json"
_LOCK = threading.Lock()

# clé de réglage -> variable d'environnement correspondante
ENV_MAP = {
    "anthropic_key": "ANTHROPIC_API_KEY",
    "openai_key": "OPENAI_API_KEY",
    "deepl_key": "DEEPL_API_KEY",
    "hf_token": "HF_TOKEN",
}
DEFAULT_SETTINGS = {
    "anthropic_key": "", "openai_key": "", "deepl_key": "", "hf_token": "",
    "llm_provider": "anthropic", "llm_model": "claude-opus-4-8",
}


def _read(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_settings() -> dict:
    return {**DEFAULT_SETTINGS, **_read(SETTINGS, {})}


def save_settings(data: dict) -> dict:
    DATA.mkdir(exist_ok=True)
    cur = load_settings()
    for k in DEFAULT_SETTINGS:
        if k in data and data[k] is not None:
            cur[k] = data[k]
    with _LOCK:
        SETTINGS.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    apply_to_env(cur)
    return cur


def apply_to_env(settings: dict | None = None) -> None:
    s = settings or load_settings()
    for key, env in ENV_MAP.items():
        if s.get(key):
            os.environ[env] = s[key]


def settings_status() -> dict:
    """Vue sûre pour l'UI : indique si chaque clé est configurée (sans l'exposer)."""
    s = load_settings()
    out = {"llm_provider": s.get("llm_provider"), "llm_model": s.get("llm_model")}
    for k in ENV_MAP:
        v = s.get(k) or ""
        out[k + "_set"] = bool(v)
        out[k + "_hint"] = ("•••• " + v[-4:]) if len(v) >= 4 else ("••••" if v else "")
    return out


def load_projects() -> list:
    return _read(PROJECTS, [])


def _write_projects(items: list) -> None:
    DATA.mkdir(exist_ok=True)
    PROJECTS.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def add_project(record: dict) -> None:
    with _LOCK:
        items = load_projects()
        items.insert(0, record)  # plus récent d'abord
        _write_projects(items[:200])  # borne l'historique


def delete_project(pid: str) -> list:
    """Supprime un projet de l'historique par son id. Renvoie la liste restante."""
    with _LOCK:
        items = [p for p in load_projects() if str(p.get("id", "")) != str(pid)]
        _write_projects(items)
        return items


def purge_projects() -> None:
    """Vide tout l'historique."""
    with _LOCK:
        _write_projects([])
