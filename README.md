# subgen — sous-titres traduits + attache vidéo

Pipeline robuste basé sur **WhisperX** (transcription, alignement, diarisation) et
**ffmpeg** (attache), avec traduction **configurable** : NLLB (local), LLM
(Anthropic/Ollama) ou API (DeepL).

```
vidéo ─▶ audio 16k ─▶ WhisperX (ASR+align+diar) ─▶ traduction ─▶ SRT/ASS ─▶ ffmpeg ─▶ vidéo sous-titrée
```

## Installation

L'environnement `.venv` contient déjà WhisperX + torch (cu128) + transformers. Sinon :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

ffmpeg doit être sur le PATH (`ffmpeg -version`).

## Utilisation

```powershell
# Le plus simple (config.yaml par défaut : NLLB local, cible fr, soft-subs mp4)
.\run.ps1 "C:\videos\conf.mp4"

# Choisir la langue, le moteur, le mode d'attache
.\run.ps1 "conf.mp4" -t es -b nllb --mode soft

# Burn-in (gravé) en HEVC NVENC, avec fichier .ass stylé
.\run.ps1 "conf.mp4" -t fr --mode hard --formats srt,ass

# Traduction LLM (qualité contextuelle) via Claude
$env:ANTHROPIC_API_KEY="sk-..."; .\run.ps1 "conf.mp4" -b llm -t fr

# Traduction LLM locale via Ollama
.\run.ps1 "conf.mp4" -b llm -t fr   # avec translate.llm.provider=ollama dans config.yaml

# DeepL
$env:DEEPL_API_KEY="..."; .\run.ps1 "conf.mp4" -b api -t fr

# Diarisation des locuteurs (token Hugging Face)
$env:HF_TOKEN="hf_..."; .\run.ps1 "conf.mp4" --diarize
```

Sorties dans `output/` : `nom.<lang>.srt` (+ formats demandés) et la vidéo
(`nom.subbed.mp4` en soft, `nom.hardsub.mp4` en hard).

## Interface web (Helix)

```powershell
.\web.ps1   # démarre le serveur et ouvre http://localhost:7860
```

Glisser-déposer une vidéo **ou** coller un **lien** (YouTube, Vimeo… via yt-dlp, avec
choix de qualité), choisir les langues et options, suivre la progression (étapes
parlantes + barre), annuler en cours, puis télécharger la vidéo sous-titrée et le `.srt`.

## Options CLI principales

| Option | Effet |
|---|---|
| `-t, --target` | langue cible (ISO 639-1) |
| `-b, --backend` | `nllb` \| `llm` \| `api` |
| `-m, --model` | modèle ASR (`large-v3`, `large-v3-turbo`…) |
| `--mode` | `soft` (mux) \| `hard` (burn-in) \| `none` |
| `--formats` | `srt,ass,vtt` |
| `--diarize` | identification des locuteurs |
| `--no-translate` | sous-titres dans la langue d'origine |
| `-v` | logs détaillés (+ traceback) |

Tout est aussi réglable dans `config.yaml`.

## Notes / robustesse

- **Python** : préférer python.org/conda au Python du Microsoft Store (chemins venv/CUDA plus fiables).
- **VRAM** : RTX 5090 → `batch_size: 24` et `large-v3` en `float16` sans souci ; passer en `int8` pour le CPU.
- **Traduction** : NLLB = hors-ligne et rapide ; LLM = meilleure qualité (contexte) ; DeepL = top mais en ligne.
- **Soft vs hard** : soft = activable/désactivable, instantané, sans ré-encodage (recommandé). Hard = gravé, lisible partout.
- Les fichiers temporaires (audio extrait) sont supprimés sauf `io.keep_temp: true`.
