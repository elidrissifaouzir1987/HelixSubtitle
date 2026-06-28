"""Synthèse vocale via Edge-TTS (gratuit, en ligne, multilingue)."""
from __future__ import annotations

import asyncio
from pathlib import Path

from ..utils import log

# voix par défaut (Edge-TTS) par langue ISO 639-1
DEFAULT_VOICES = {
    "ar": "ar-SA-HamedNeural", "en": "en-US-AriaNeural", "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural", "de": "de-DE-KatjaNeural", "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural", "nl": "nl-NL-ColetteNeural", "ru": "ru-RU-SvetlanaNeural",
    "zh": "zh-CN-XiaoxiaoNeural", "ja": "ja-JP-NanamiNeural", "ko": "ko-KR-SunHiNeural",
    "tr": "tr-TR-EmelNeural", "pl": "pl-PL-ZofiaNeural", "uk": "uk-UA-PolinaNeural",
    "hi": "hi-IN-SwaraNeural", "vi": "vi-VN-HoaiMyNeural", "id": "id-ID-GadisNeural",
    "fa": "fa-IR-DilaraNeural", "he": "he-IL-HilaNeural", "sv": "sv-SE-SofieNeural",
    "cs": "cs-CZ-VlastaNeural", "ro": "ro-RO-AlinaNeural", "el": "el-GR-AthinaNeural",
    "th": "th-TH-PremwadeeNeural", "hu": "hu-HU-NoemiNeural", "fi": "fi-FI-NooraNeural",
    "da": "da-DK-ChristelNeural", "no": "nb-NO-PernilleNeural", "ca": "ca-ES-JoanaNeural",
}


def voice_for(lang: str, voice: str = "auto") -> str:
    if voice and voice != "auto":
        return voice
    v = DEFAULT_VOICES.get(lang.split("-")[0])
    if not v:
        raise RuntimeError(
            f"Aucune voix Edge-TTS par défaut pour '{lang}'. Précise une voix "
            f"(ex. dub.voice: xx-YY-NameNeural)."
        )
    return v


async def _synth_all(items: list[tuple[int, str, Path]], voice: str, conc: int = 4) -> None:
    import edge_tts

    sem = asyncio.Semaphore(conc)

    async def one(text: str, path: Path):
        async with sem:
            try:
                await edge_tts.Communicate(text, voice).save(str(path))
            except Exception as e:  # un segment qui échoue ne casse pas tout
                log.warning("TTS échec (segment ignoré) : %s", e)

    await asyncio.gather(*(one(t, p) for _, t, p in items if t.strip()))


def synth_segments(segments, lang: str, out_dir: Path, voice: str = "auto") -> list[tuple[int, str, Path]]:
    """Synthétise chaque segment -> fichier mp3. Renvoie [(idx, texte, chemin)]."""
    v = voice_for(lang, voice)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [(i, s.out_text, out_dir / f"seg_{i:05d}.mp3") for i, s in enumerate(segments)]
    log.info("Synthèse vocale (%s, voix %s, %d segments)…", lang, v, len(items))
    asyncio.run(_synth_all(items, v))
    return items
