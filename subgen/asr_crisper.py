"""Moteur ASR alternatif : CrisperWhisper 2.0 (timestamps mot-à-mot très précis).

Renvoie un résultat au **format WhisperX** (`{"segments": [...], "language": ...}`)
pour que la suite du pipeline (diarisation, redécoupage, traduction) soit identique
quel que soit le moteur choisi.

Particularités du modèle :
- pas de détection de langue intégrée (paramètre `language` obligatoire) → on détecte
  avec faster-whisper si l'utilisateur a laissé « auto » ;
- pas de segments : il renvoie une liste de mots → on les regroupe en phrases ;
- modes `verbatim` (tout, y compris hésitations) et `intended` (texte nettoyé).
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import Config
from .utils import log

_SENT_END = re.compile(r"[.!?…。！？]$")

# tailles disponibles (les variantes *_pro sont commerciales)
SIZES = {"large", "turbo", "medium", "small"}
# correspondance depuis les noms de modèles WhisperX, pour un basculement sans friction
_FROM_WHISPERX = {
    "large-v3": "large", "large-v2": "large", "large": "large",
    "large-v3-turbo": "turbo", "turbo": "turbo",
    "medium": "medium", "small": "small", "base": "small", "tiny": "small",
}


def resolve_size(model: str | None) -> str:
    """Nom de taille CrisperWhisper à partir du réglage `asr.model`."""
    m = (model or "large").strip()
    if m in SIZES:
        return m
    return _FROM_WHISPERX.get(m, "large")


def detect_language(audio_path: Path, device: str = "cuda") -> str:
    """Détecte la langue parlée (CrisperWhisper ne le fait pas lui-même)."""
    from faster_whisper import WhisperModel

    log.info("Détection de la langue (faster-whisper base)…")
    compute = "float16" if device == "cuda" else "int8"
    try:
        m = WhisperModel("base", device=device, compute_type=compute)
    except Exception:  # repli CPU si la config GPU échoue
        m = WhisperModel("base", device="cpu", compute_type="int8")
    lang, prob, _ = m.detect_language(audio=_load_audio(audio_path))
    log.info("Langue détectée : %s (%.0f %%)", lang, (prob or 0) * 100)
    del m
    return lang


def _load_audio(path: Path):
    import whisperx
    return whisperx.load_audio(str(path))


def _group_words(words: list[dict], max_gap: float, max_dur: float) -> list[dict]:
    """Regroupe les mots en segments (phrases) : ponctuation, pause, durée max.

    CrisperWhisper renvoie une liste de mots à plat ; le reste du pipeline attend
    des segments. Le redécoupage fin (`resegment.py`) affinera ensuite si activé.
    """
    segments: list[dict] = []
    cur: list[dict] = []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            dur = w["end"] - cur[0]["start"]
            if gap > max_gap or dur > max_dur or _SENT_END.search(cur[-1]["word"]):
                segments.append(_mk_segment(cur))
                cur = []
        cur.append(w)
    if cur:
        segments.append(_mk_segment(cur))
    return segments


def _mk_segment(words: list[dict]) -> dict:
    text = " ".join(w["word"] for w in words)
    text = re.sub(r"\s+([,.!?…;:،؟])", r"\1", text).strip()
    return {"start": words[0]["start"], "end": words[-1]["end"],
            "text": text, "words": words}


def _norm_words(raw) -> list[dict]:
    """Normalise les mots du modèle (start/end peuvent être None)."""
    out: list[dict] = []
    last = 0.0
    for w in raw or []:
        txt = (getattr(w, "word", None) or "").strip()
        if not txt:
            continue
        st = getattr(w, "start", None)
        en = getattr(w, "end", None)
        st = float(st) if st is not None else last
        en = float(en) if en is not None else st
        last = en
        out.append({"word": txt, "start": st, "end": en})
    return out


def crisper_result(audio_path: Path, cfg: Config) -> dict:
    """Transcrit avec CrisperWhisper et renvoie un résultat au format WhisperX."""
    try:
        from crisperwhisper import CrisperWhisperModel
    except ImportError as e:
        raise RuntimeError(
            "CrisperWhisper n'est pas installé. Lance :\n"
            '  .\\.venv\\Scripts\\python.exe -m pip install "crisperwhisper[transformers]"'
        ) from e

    device = cfg.get("asr", "device", default="cuda")
    size = resolve_size(cfg.get("asr", "model"))
    mode = "verbatim" if cfg.get("asr", "verbatim", default=False) else "intended"
    lang = cfg.get("asr", "language") or detect_language(Path(audio_path), device)

    # Backend « transformers » par défaut : le backend ct2 exige le fork
    # ctranslate2-crisperwhisper (pas de build Windows) et son installation
    # remplacerait le ctranslate2 dont dépend WhisperX.
    backend = cfg.get("asr", "crisper_backend", default="transformers")
    log.info("Chargement de CrisperWhisper 2.0 (%s, %s, %s, mode %s)…", size, device, backend, mode)
    try:
        model = CrisperWhisperModel(
            size if size in SIZES else "large",
            backend=backend,
            device=device,
            compute_type=cfg.get("asr", "compute_type", default="float16"),
        )
    except Exception as e:
        raise RuntimeError(
            f"Chargement de CrisperWhisper impossible ({e}).\n"
            "Si le téléchargement est refusé : le modèle est sous licence non "
            "commerciale — accepte les conditions sur huggingface.co/nyralabs et "
            "renseigne ton token Hugging Face dans les Réglages."
        ) from e

    log.info("Transcription en cours (CrisperWhisper, langue %s)…", lang)
    res = model.transcribe(str(audio_path), language=lang, mode=mode, word_timestamps=True)

    words = _norm_words(getattr(res, "words", None))
    if not words:  # pas de mots -> on garde au moins le texte brut
        text = (getattr(res, "text", "") or "").strip()
        segments = [{"start": 0.0, "end": float(getattr(res, "duration", 0.0) or 0.0),
                     "text": text, "words": []}] if text else []
    else:
        segments = _group_words(
            words,
            max_gap=float(cfg.get("subtitles", "max_gap", default=0.7)),
            max_dur=float(cfg.get("subtitles", "max_duration", default=6.0)),
        )

    detected = getattr(res, "language", None) or lang
    log.info("Langue : %s — %d segments (%d mots)", detected, len(segments), len(words))
    _free(model)
    return {"segments": segments, "language": detected}


def _free(obj) -> None:
    try:
        import gc
        import torch
        del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
