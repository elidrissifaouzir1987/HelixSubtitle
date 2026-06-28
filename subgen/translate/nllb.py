"""Moteur NLLB-200 (Meta) via transformers, hors-ligne, GPU."""
from __future__ import annotations

from .base import Translator
from ..utils import log

# ISO 639-1 -> code NLLB (Flores-200). Étendre au besoin.
NLLB_CODES = {
    "en": "eng_Latn", "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "ru": "rus_Cyrl",
    "ar": "arb_Arab", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "tr": "tur_Latn", "pl": "pol_Latn", "uk": "ukr_Cyrl", "hi": "hin_Deva",
    "vi": "vie_Latn", "id": "ind_Latn", "fa": "pes_Arab", "he": "heb_Hebr",
    "sv": "swe_Latn", "cs": "ces_Latn", "ro": "ron_Latn", "el": "ell_Grek",
    "th": "tha_Thai", "hu": "hun_Latn", "fi": "fin_Latn", "da": "dan_Latn",
    "no": "nob_Latn", "ca": "cat_Latn",
}


def _code(lang: str) -> str:
    if lang in NLLB_CODES:
        return NLLB_CODES[lang]
    raise ValueError(
        f"Langue '{lang}' non mappée pour NLLB. Ajoute-la dans NLLB_CODES ou "
        f"utilise un autre backend (llm/api)."
    )


class NLLBTranslator(Translator):
    name = "nllb"

    def __init__(self, model_name: str, device: str = "cuda", batch: int = 16):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.batch = batch
        log.info("Chargement NLLB %s sur %s…", model_name, self.device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)

    def translate_batch(self, texts, src, tgt):
        import torch

        src_c, tgt_c = _code(src), _code(tgt)
        self.tok.src_lang = src_c
        bos = self.tok.convert_tokens_to_ids(tgt_c)
        out: list[str] = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i : i + self.batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=512).to(self.device)
            with torch.inference_mode():
                gen = self.model.generate(**enc, forced_bos_token_id=bos, max_length=512,
                                          num_beams=4)
            out.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
            log.debug("NLLB %d/%d", min(i + self.batch, len(texts)), len(texts))
        return out
