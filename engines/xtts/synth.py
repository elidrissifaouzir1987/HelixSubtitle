"""Pont XTTS (venv isolé) — clonage de voix multilingue.

Appelé en sous-processus par Helix. Lit un spec JSON et synthétise chaque item
avec la voix de référence indiquée.

Usage : python synth.py spec.json
spec = {
  "language": "ar",
  "items": [{"text": "...", "ref": "ref_speaker.wav", "out": "seg_00001.wav"}, ...]
}
"""
import json
import os
import sys

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # accepte la licence du modèle XTTS

# codes langue XTTS (2 lettres, sauf chinois)
XLANG = {"zh": "zh-cn"}
SUPPORTED = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs",
             "ar", "zh-cn", "ja", "hu", "ko", "hi"}


def main() -> int:
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    lang = spec["language"].split("-")[0]
    xlang = XLANG.get(lang, lang)
    if xlang not in SUPPORTED:
        print(f"ERROR: langue '{lang}' non supportée par XTTS", file=sys.stderr)
        return 2

    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"XTTS: chargement du modèle sur {device}…", file=sys.stderr)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    ok = 0
    for it in spec["items"]:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        try:
            tts.tts_to_file(text=text, speaker_wav=it["ref"], language=xlang,
                            file_path=it["out"])
            ok += 1
        except Exception as e:
            print(f"WARN: segment échoué ({e})", file=sys.stderr)
    print(f"DONE {ok}/{len(spec['items'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
