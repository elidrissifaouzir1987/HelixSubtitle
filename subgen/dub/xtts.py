"""Backend XTTS (clonage de voix) via le venv isolé engines/xtts.

Extrait une référence audio par locuteur (diarisation) puis synthétise chaque
segment dans la voix correspondante, en appelant engines/xtts/synth.py en
sous-processus (deps lourdes isolées).
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..utils import log, run

ROOT = Path(__file__).resolve().parent.parent.parent
XTTS_PY = ROOT / "engines" / "xtts" / ".venv" / "Scripts" / "python.exe"
XTTS_SCRIPT = ROOT / "engines" / "xtts" / "synth.py"
SR = 24000


def available() -> bool:
    return XTTS_PY.exists() and XTTS_SCRIPT.exists()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name))


def _read(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def _write(path: Path, data: np.ndarray) -> None:
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def build_speaker_refs(ffmpeg: str, video: Path, segments, tmp: Path,
                       target: float = 9.0) -> dict[str, Path]:
    """Construit un échantillon de référence (~target s) par locuteur."""
    src = tmp / "ref_src.wav"
    run([ffmpeg, "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", str(src)], "audio de référence")
    audio = _read(src)
    by_sp: dict[str, list] = defaultdict(list)
    for s in segments:
        by_sp[getattr(s, "speaker", None) or "_global"].append(s)
    refs: dict[str, Path] = {}
    for sp, segs in by_sp.items():
        chosen, tot = [], 0.0
        for s in sorted(segs, key=lambda x: x.end - x.start, reverse=True):
            chosen.append(s)
            tot += s.end - s.start
            if tot >= target:
                break
        parts = [audio[int(s.start * SR):int(s.end * SR)] for s in chosen]
        clip = np.concatenate(parts) if parts else audio[: int(target * SR)]
        ref = tmp / f"ref_{_safe(sp)}.wav"
        _write(ref, clip)
        refs[sp] = ref
    log.info("Références de voix extraites : %d locuteur(s)", len(refs))
    return refs


def xtts_synth_segments(ffmpeg: str, video: Path, segments, lang: str, tmp: Path,
                        cfg) -> list[tuple[int, str, Path]]:
    """Synthétise chaque segment avec la voix clonée du locuteur (sous-processus)."""
    if not available():
        raise RuntimeError(
            "Backend XTTS indisponible : venv isolé manquant "
            "(engines/xtts/.venv). Installe-le ou utilise dub.backend: edge."
        )
    refs = build_speaker_refs(ffmpeg, video, segments, tmp)
    fallback = next(iter(refs.values()))
    items = [(i, s.out_text, tmp / f"seg_{i:05d}.wav") for i, s in enumerate(segments)]
    spec = {"language": lang, "items": [
        {"text": s.out_text,
         "ref": str(refs.get(getattr(s, "speaker", None) or "_global", fallback)),
         "out": str(tmp / f"seg_{i:05d}.wav")}
        for i, s in enumerate(segments)
    ]}
    spec_path = tmp / "xtts_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    log.info("Synthèse vocale clonée (XTTS, %s, %d segments)…", lang, len(items))
    proc = subprocess.run([str(XTTS_PY), str(XTTS_SCRIPT), str(spec_path)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError("XTTS a échoué :\n" + "\n".join(tail))
    return items
