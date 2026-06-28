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


_DISP = {"ar": "Arabe", "en": "Anglais", "fr": "Français", "es": "Espagnol",
         "de": "Allemand", "it": "Italien", "pt": "Portugais", "ru": "Russe",
         "zh": "Chinois", "ja": "Japonais", "ko": "Coréen", "tr": "Turc",
         "nl": "Néerlandais", "pl": "Polonais", "hi": "Hindi", "fa": "Persan"}


def _disp(lang: str) -> str:
    return _DISP.get(lang.split("-")[0], lang.upper())


def combined_output(ffmpeg: str, video: Path, written: dict, docs: dict, cfg: Config,
                    out_dir: Path, cancel_event=None) -> str:
    """Un seul fichier : vidéo + audio original + pistes doublées + sous-titres.

    Les pistes audio sont sélectionnables dans le lecteur (Original / langue doublée),
    la première langue doublée étant la piste par défaut.
    """
    import shutil
    import tempfile
    from ..attach import _hard, _iso

    total = media_duration(video)
    if total <= 0:
        raise RuntimeError("Durée de la vidéo inconnue — doublage impossible.")
    container = cfg.get("attach", "container", default="mp4")
    mode = cfg.get("attach", "mode", default="soft")
    keep = cfg.get("dub", "keep_original", default=True)
    langs = list(docs.keys())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{video.stem}.dubbed.{container}"
    tmp = Path(tempfile.mkdtemp(prefix="subgen_dub_"))
    try:
        # 1) pistes audio doublées (un WAV par langue)
        tracks = {lg: build_track(ffmpeg, docs[lg].segments, lg, cfg, tmp, total, cancel_event)
                  for lg in langs}

        # 2) base vidéo : gravée si mode hard, sinon vidéo d'origine
        base = video
        if mode == "hard":
            primary = written[langs[0]].get("ass") or written[langs[0]].get("srt")
            base = tmp / f"burn.{container}"
            _hard(ffmpeg, video, primary, base, cfg)

        # 3) mux unique
        cmd = [ffmpeg, "-y", "-i", str(base)]
        for lg in langs:
            cmd += ["-i", str(tracks[lg])]
        soft = mode == "soft"
        sub_inputs = []
        if soft:
            for lg in langs:
                sp = written[lg].get("srt") or next(iter(written[lg].values()))
                sub_inputs.append((lg, sp))
                cmd += ["-i", str(sp)]

        cmd += ["-map", "0:v:0"]
        a_out = []  # (kind, lang)
        if keep:
            cmd += ["-map", "0:a:0?"]
            a_out.append(("orig", None))
        for di in range(len(langs)):
            cmd += ["-map", str(1 + di) + ":a:0"]
            a_out.append(("dub", langs[di]))
        if soft:
            for si in range(len(sub_inputs)):
                cmd += ["-map", str(1 + len(langs) + si)]

        cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
        if soft:
            cmd += ["-c:s", "mov_text" if container == "mp4" else "srt"]

        first_dub = next(i for i, (k, _) in enumerate(a_out) if k == "dub")
        for i, (k, lg) in enumerate(a_out):
            if k == "orig":
                cmd += [f"-metadata:s:a:{i}", "title=Original", f"-disposition:a:{i}", "0"]
            else:
                cmd += [f"-metadata:s:a:{i}", f"title={_disp(lg)} (doublage)",
                        f"-metadata:s:a:{i}", f"language={_iso(lg)}",
                        f"-disposition:a:{i}", "default" if i == first_dub else "0"]
        if soft:
            for si, (lg, _) in enumerate(sub_inputs):
                cmd += [f"-metadata:s:s:{si}", f"language={_iso(lg)}"]

        cmd += [str(out)]
        log.info("Montage de la vidéo doublée (%d langue(s), pistes sélectionnables)…", len(langs))
        run(cmd, "mux doublage combiné")
        return str(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
