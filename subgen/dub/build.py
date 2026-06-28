"""Calage des clips TTS dans leur fenêtre + assemblage de la piste + mux ffmpeg."""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from ..attach import _iso
from ..config import Config
from ..utils import log, media_duration, run
from .tts import synth_segments

SR = 24000  # Edge-TTS sort en 24 kHz


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


def _write_wav(path: Path, data: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def _fit_to_wav(ffmpeg: str, mp3: Path, wav: Path, window: float, max_speedup: float) -> None:
    """Convertit le clip en WAV 24k mono, accéléré si trop long pour la fenêtre."""
    dur = media_duration(mp3)
    af = []
    if window > 0.1 and dur > window:
        speed = min(dur / window, max_speedup)
        if speed > 1.01:
            af = ["-filter:a", f"atempo={speed:.4f}"]
    run([ffmpeg, "-y", "-i", str(mp3), *af, "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", str(wav)], "calage clip TTS")


def build_track(ffmpeg: str, segments, lang: str, cfg: Config, tmp: Path,
                total_dur: float, cancel_event=None) -> Path:
    """Synthétise, cale et assemble la piste audio doublée -> WAV."""
    voice = cfg.get("dub", "voice", default="auto")
    max_su = float(cfg.get("dub", "max_speedup", default=1.3))
    items = synth_segments(segments, lang, tmp, voice)

    n = max(1, int(math.ceil(total_dur * SR)))
    buf = np.zeros(n, dtype=np.int16)
    log.info("Assemblage de la piste doublée (%s)…", lang)
    for s, (idx, text, mp3) in zip(segments, items):
        if cancel_event is not None and cancel_event.is_set():
            from ..pipeline import Cancelled
            raise Cancelled("Traitement annulé.")
        if not mp3.exists():
            continue
        wav = tmp / f"f_{idx:05d}.wav"
        _fit_to_wav(ffmpeg, mp3, wav, max(0.0, s.end - s.start), max_su)
        data = _read_wav(wav)
        off = int(s.start * SR)
        if off >= n:
            continue
        end = min(n, off + len(data))
        buf[off:end] = data[: end - off]
    out = tmp / f"dub_{lang}.wav"
    _write_wav(out, buf)
    return out


def mux_dub(ffmpeg: str, video: Path, dub_wav: Path, cfg: Config, out_dir: Path, lang: str) -> str:
    container = cfg.get("attach", "container", default="mp4")
    keep = cfg.get("dub", "keep_original", default=True)
    out = out_dir / f"{video.stem}.{lang}.dub.{container}"
    cmd = [ffmpeg, "-y", "-i", str(video), "-i", str(dub_wav),
           "-map", "0:v:0", "-map", "1:a:0"]
    if keep:
        cmd += ["-map", "0:a:0?"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-metadata:s:a:0", f"language={_iso(lang)}", "-disposition:a:0", "default"]
    if keep:
        cmd += ["-metadata:s:a:1", "language=orig", "-disposition:a:1", "0"]
    cmd += [str(out)]
    log.info("Mux de la vidéo doublée (%s)…", lang)
    run(cmd, "mux doublage")
    return str(out)


def dub_documents(ffmpeg: str, video: Path, docs: dict, cfg: Config, out_dir: Path,
                  cancel_event=None) -> list[str]:
    """Produit une vidéo doublée par langue cible. Renvoie les chemins."""
    import shutil
    import tempfile

    total_dur = media_duration(video)
    if total_dur <= 0:
        raise RuntimeError("Durée de la vidéo inconnue — doublage impossible.")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[str] = []
    for lang, doc in docs.items():
        tmp = Path(tempfile.mkdtemp(prefix="subgen_dub_"))
        try:
            track = build_track(ffmpeg, doc.segments, lang, cfg, tmp, total_dur, cancel_event)
            results.append(mux_dub(ffmpeg, video, track, cfg, out_dir, lang))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return results
