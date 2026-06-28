"""Nettoyage du texte des sous-titres (anti-carreaux / glyphes manquants).

Cible surtout les sorties de moteurs de traduction (NLLB) qui insèrent :
  - le TATWEEL (U+0640), élongation cosmétique souvent rendue en carreau ;
  - des harakat/diacritiques arabes superflus pour des sous-titres ;
  - des caractères invisibles / marques de direction (zero-width, BOM, soft hyphen).
"""
from __future__ import annotations

import re
import unicodedata

TATWEEL = 0x0640
# Tashkeel/harakat arabes + alef suscrit
HARAKAT = set(range(0x064B, 0x0653)) | {0x0670}
# Invisibles et marques bidi/contrôle à retirer
INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D,
     0x202E, 0x2066, 0x2067, 0x2068, 0x2069, 0xFEFF, 0x00AD],
    None,
)

_WS = re.compile(r"[ \t ]+")


def clean_text(text: str, *, strip_diacritics: bool = True) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.translate(INVISIBLE)
    drop = {TATWEEL}
    if strip_diacritics:
        drop |= HARAKAT
    if drop:
        text = "".join(ch for ch in text if ord(ch) not in drop)
    # remplace tout caractère de remplacement résiduel
    text = text.replace("�", "")
    # espaces multiples (sans toucher aux retours ligne)
    text = "\n".join(_WS.sub(" ", line).strip() for line in text.split("\n"))
    return text.strip()
