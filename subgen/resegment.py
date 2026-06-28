"""Redécoupage des sous-titres à partir des timestamps mot-à-mot.

WhisperX renvoie des segments parfois très longs (plusieurs phrases dans un bloc).
On les recoupe en sous-titres lisibles en s'appuyant sur l'alignement mot-à-mot :
fins de phrase, pauses, durée/longueur max, et changement de locuteur (diarisation).
"""
from __future__ import annotations

import re

from .config import Config
from .subtitles import Segment, SubtitleDoc
from .utils import log

_SENT_END = re.compile(r"[.!?…。！？]$")  # fin de phrase (source latine/CJK)


def _norm_words(seg: Segment) -> list[dict]:
    """Normalise les mots : garantit start/end (report depuis le voisin si absent)."""
    out: list[dict] = []
    last = seg.start
    for w in seg.words:
        txt = (w.get("word") if isinstance(w, dict) else str(w)) or ""
        if not txt.strip():
            continue
        st = w.get("start") if isinstance(w, dict) else None
        en = w.get("end") if isinstance(w, dict) else None
        st = float(st) if st is not None else last
        en = float(en) if en is not None else st
        last = en
        out.append({"word": txt.strip(), "start": st, "end": en,
                    "speaker": (w.get("speaker") if isinstance(w, dict) else None) or seg.speaker})
    return out


def _flush(words: list[dict]) -> Segment:
    text = " ".join(w["word"] for w in words)
    text = re.sub(r"\s+([,.!?…;:،؟])", r"\1", text).strip()  # recolle la ponctuation
    spk = next((w["speaker"] for w in words if w["speaker"]), None)
    return Segment(start=words[0]["start"], end=words[-1]["end"], text=text, speaker=spk)


def resegment(doc: SubtitleDoc, cfg: Config) -> SubtitleDoc:
    if not cfg.get("subtitles", "resegment", default=True):
        return doc
    max_dur = float(cfg.get("subtitles", "max_duration", default=6.0))
    max_gap = float(cfg.get("subtitles", "max_gap", default=0.7))
    max_chars = int(cfg.get("subtitles", "max_line_chars", default=42)) * \
        int(cfg.get("subtitles", "max_lines", default=2))
    min_dur = 1.0  # ne pas créer de micro-cue sur une fin de phrase trop courte

    new: list[Segment] = []
    for seg in doc.segments:
        words = _norm_words(seg)
        if len(words) < 2:  # rien à recouper
            new.append(seg)
            continue
        cur: list[dict] = []
        for w in words:
            if cur:
                first = cur[0]
                cur_text = " ".join(x["word"] for x in cur)
                dur = w["end"] - first["start"]
                gap = w["start"] - cur[-1]["end"]
                spk_change = bool(w["speaker"] and first["speaker"] and w["speaker"] != first["speaker"])
                hard = (dur > max_dur or len(cur_text) + 1 + len(w["word"]) > max_chars
                        or gap > max_gap or spk_change)
                sentence = bool(_SENT_END.search(cur[-1]["word"])) and (cur[-1]["end"] - first["start"]) >= min_dur
                if hard or sentence:
                    new.append(_flush(cur))
                    cur = []
            cur.append(w)
        if cur:
            new.append(_flush(cur))

    if new:
        before, after = len(doc.segments), len(new)
        if after != before:
            log.info("Redécoupage : %d → %d sous-titres", before, after)
        doc.segments = new
    return doc
