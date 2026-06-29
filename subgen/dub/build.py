"""Calage des clips TTS dans leur fenêtre + assemblage de la piste + mux ffmpeg."""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

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


def _fit_to_wav(ffmpeg: str, mp3: Path, wav: Path, window: float, max_speedup: float,
                stretch: str = "rubberband") -> None:
    """Convertit le clip en WAV 24k mono, accéléré si trop long pour la fenêtre.

    stretch : 'rubberband' (préserve la hauteur de voix) ou 'atempo'.
    """
    dur = media_duration(mp3)
    af = []
    if window > 0.1 and dur > window:
        speed = min(dur / window, max_speedup)
        if speed > 1.01:
            if stretch == "rubberband":
                af = ["-filter:a", f"rubberband=tempo={speed:.4f}"]
            else:
                af = ["-filter:a", f"atempo={speed:.4f}"]
    run([ffmpeg, "-y", "-i", str(mp3), *af, "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", str(wav)], "calage clip TTS")


def _mix_voice_bg(ffmpeg: str, voice: Path, bg: Path, out: Path) -> None:
    """Mixe la voix doublée par-dessus le fond sonore (musique/bruitages)."""
    fc = ("[0:a]aresample=48000,aformat=channel_layouts=stereo,volume=1.6[v];"
          "[1:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.9[m];"
          "[v][m]amix=inputs=2:normalize=0:duration=first[a]")
    run([ffmpeg, "-y", "-i", str(voice), "-i", str(bg), "-filter_complex", fc,
         "-map", "[a]", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(out)],
        "mix voix + fond")


def build_track(ffmpeg: str, video: Path, segments, lang: str, cfg: Config, tmp: Path,
                total_dur: float, cancel_event=None) -> Path:
    """Synthétise, cale et assemble la piste audio doublée -> WAV."""
    max_su = float(cfg.get("dub", "max_speedup", default=1.3))
    stretch = cfg.get("dub", "timestretch", default="rubberband")
    backend = cfg.get("dub", "backend", default="edge")
    if backend == "xtts":
        from .xtts import xtts_synth_segments
        items = xtts_synth_segments(ffmpeg, video, segments, lang, tmp, cfg)
    else:
        items = synth_segments(segments, lang, tmp, cfg.get("dub", "voice", default="auto"))

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
        _fit_to_wav(ffmpeg, mp3, wav, max(0.0, s.end - s.start), max_su, stretch)
        data = _read_wav(wav)
        off = int(s.start * SR)
        if off >= n:
            continue
        end = min(n, off + len(data))
        buf[off:end] = data[: end - off]
    out = tmp / f"dub_{lang}.wav"
    _write_wav(out, buf)
    return out


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
        # 0) fond sonore (musique/bruitages) à préserver sous la voix doublée
        accomp = None
        if cfg.get("dub", "separate_background", default=True):
            from .separate import separate_background
            device = cfg.get("asr", "device", default="cuda")
            accomp = separate_background(ffmpeg, video, tmp, device)

        # 1) pistes audio doublées (un WAV par langue), voix mixée au fond si dispo
        tracks = {}
        for lg in langs:
            voice = build_track(ffmpeg, video, docs[lg].segments, lg, cfg, tmp, total, cancel_event)
            if accomp is not None:
                mixed = tmp / f"mix_{lg}.wav"
                _mix_voice_bg(ffmpeg, voice, accomp, mixed)
                tracks[lg] = mixed
            else:
                tracks[lg] = voice

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
