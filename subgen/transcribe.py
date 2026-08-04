"""Étape ASR : WhisperX ou CrisperWhisper (+ alignement et diarisation optionnels)."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .subtitles import SubtitleDoc
from .utils import log


def transcribe(audio_path: Path, cfg: Config) -> SubtitleDoc:
    import whisperx  # import tardif : lourd (torch/cuda)

    device = cfg.get("asr", "device", default="cuda")
    audio = whisperx.load_audio(str(audio_path))

    if cfg.get("asr", "engine", default="whisperx") == "crisperwhisper":
        # timestamps mot-à-mot natifs : ni modèle WhisperX ni alignement wav2vec2
        from .asr_crisper import crisper_result
        result = crisper_result(Path(audio_path), cfg)
        if cfg.get("asr", "diarize", default=False):
            _diarize(whisperx, result, audio, cfg, device)
        return SubtitleDoc.from_whisperx(result)

    compute = cfg.get("asr", "compute_type", default="float16")
    model_name = cfg.get("asr", "model", default="large-v3")
    batch_size = int(cfg.get("asr", "batch_size", default=16))
    language = cfg.get("asr", "language")

    log.info("Chargement du modèle ASR %s (%s/%s)…", model_name, device, compute)
    model = whisperx.load_model(model_name, device, compute_type=compute, language=language)

    log.info("Transcription en cours (batch=%d)…", batch_size)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    detected = result.get("language")
    log.info("Langue détectée : %s — %d segments", detected, len(result.get("segments", [])))

    # libère la VRAM du modèle ASR avant alignement/diarisation
    _free(model)

    if cfg.get("asr", "align", default=True):
        try:
            log.info("Alignement mot-à-mot…")
            amodel, meta = whisperx.load_align_model(language_code=detected, device=device)
            result = whisperx.align(result["segments"], amodel, meta, audio, device,
                                    return_char_alignments=False)
            result["language"] = detected
            _free(amodel)
        except Exception as e:  # langue non supportée par l'alignement -> on continue
            log.warning("Alignement ignoré (%s)", e)

    if cfg.get("asr", "diarize", default=False):
        _diarize(whisperx, result, audio, cfg, device)

    return SubtitleDoc.from_whisperx(result)


def _diarize(whisperx, result, audio, cfg: Config, device: str) -> None:
    token = cfg.get("hf_token")
    if not token:
        log.warning("Diarisation demandée mais aucun token HF (HF_TOKEN) — étape ignorée.")
        return
    try:
        log.info("Diarisation des locuteurs…")
        # API selon version de whisperx
        try:
            dia = whisperx.diarize.DiarizationPipeline(use_auth_token=token, device=device)
        except AttributeError:
            from whisperx.diarize import DiarizationPipeline
            dia = DiarizationPipeline(use_auth_token=token, device=device)
        kw = {}
        if cfg.get("asr", "min_speakers") is not None:
            kw["min_speakers"] = int(cfg.get("asr", "min_speakers"))
        if cfg.get("asr", "max_speakers") is not None:
            kw["max_speakers"] = int(cfg.get("asr", "max_speakers"))
        segs = dia(audio, **kw)
        result.update(whisperx.assign_word_speakers(segs, result))
    except Exception as e:
        log.warning("Diarisation échouée (%s) — on continue sans.", e)


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
