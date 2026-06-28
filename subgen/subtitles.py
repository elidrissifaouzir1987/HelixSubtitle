"""Modèle de segment + écriture SRT / VTT / ASS, avec mise en forme lisible."""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from .utils import fmt_timestamp


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    translation: str | None = None  # rempli après traduction

    @property
    def out_text(self) -> str:
        return self.translation if self.translation is not None else self.text


@dataclass
class SubtitleDoc:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None

    @classmethod
    def from_whisperx(cls, result: dict) -> "SubtitleDoc":
        segs = []
        for s in result.get("segments", []):
            text = (s.get("text") or "").strip()
            if not text:
                continue
            segs.append(Segment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=text,
                speaker=s.get("speaker"),
            ))
        return cls(segments=segs, language=result.get("language"))


def _wrap(text: str, max_chars: int, max_lines: int) -> str:
    if len(text) <= max_chars:
        return text
    lines = textwrap.wrap(text, width=max_chars, break_long_words=False)
    if len(lines) > max_lines:  # regroupe l'excédent sur la dernière ligne autorisée
        lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1 :])]
    return "\n".join(lines)


def write(doc: SubtitleDoc, path: Path, fmt: str, *,
          max_chars: int = 42, max_lines: int = 2, ass_style: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "srt":
        _write_srt(doc, path, max_chars, max_lines)
    elif fmt == "vtt":
        _write_vtt(doc, path, max_chars, max_lines)
    elif fmt == "ass":
        _write_ass(doc, path, max_chars, max_lines, ass_style)
    else:
        raise ValueError(f"Format de sous-titre inconnu : {fmt}")
    return path


def _write_srt(doc, path, mc, ml):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(doc.segments, 1):
            f.write(f"{i}\n{fmt_timestamp(s.start)} --> {fmt_timestamp(s.end)}\n")
            f.write(_wrap(s.out_text, mc, ml) + "\n\n")


def _write_vtt(doc, path, mc, ml):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in doc.segments:
            a = fmt_timestamp(s.start, comma=False)
            b = fmt_timestamp(s.end, comma=False)
            f.write(f"{a} --> {b}\n{_wrap(s.out_text, mc, ml)}\n\n")


def _parse_style(style: str) -> dict:
    out = {}
    for part in (style or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _write_ass(doc, path, mc, ml, style):
    st = _parse_style(style)
    font = st.get("FontName", "Arial")
    size = st.get("FontSize", "22")
    outline = st.get("Outline", "2")
    shadow = st.get("Shadow", "0")
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{size},&H00FFFFFF,&H00000000,&H80000000,"
        f"0,0,1,{outline},{shadow},2,20,20,25,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for s in doc.segments:
            a = fmt_timestamp(s.start, comma=False)[:-1]  # ASS = centièmes
            b = fmt_timestamp(s.end, comma=False)[:-1]
            txt = _wrap(s.out_text, mc, ml).replace("\n", "\\N")
            f.write(f"Dialogue: 0,{a},{b},Default,,0,0,0,,{txt}\n")
