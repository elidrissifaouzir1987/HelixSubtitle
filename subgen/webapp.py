"""Interface web (Flask) pour Helix.

Fonctions : upload (multi-fichiers) ou lien (vidéo/playlist), multi-langues,
sous-titres bilingues, révision/édition avant attache, file d'attente GPU.
Lancement :  python -m subgen.webapp   (puis http://localhost:7860)
"""
from __future__ import annotations

import copy
import logging
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

from .config import Config
from .pipeline import (Cancelled, build_docs, finalize_outputs, get_targets,
                       prepare_source)
from .subtitles import SubtitleDoc
from .translate.nllb import NLLB_CODES
from .utils import download_youtube, expand_url, require_ffmpeg, setup_logging

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUTPUT = ROOT / "output"
UPLOADS.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

LANG_NAMES = {
    "ar": "العربية (Arabe)", "en": "English", "fr": "Français", "es": "Español",
    "de": "Deutsch", "it": "Italiano", "pt": "Português", "nl": "Nederlands",
    "ru": "Русский", "zh": "中文", "ja": "日本語", "ko": "한국어", "tr": "Türkçe",
    "pl": "Polski", "uk": "Українська", "hi": "हिन्दी", "vi": "Tiếng Việt",
    "id": "Indonesia", "fa": "فارسی", "he": "עברית", "sv": "Svenska",
    "cs": "Čeština", "ro": "Română", "el": "Ελληνικά", "th": "ไทย",
    "hu": "Magyar", "fi": "Suomi", "da": "Dansk", "no": "Norsk", "ca": "Català",
}
LANGS = [(c, LANG_NAMES.get(c, c)) for c in sorted(NLLB_CODES)]
EXTRA = ["fr", "en", "es", "ar", "de", "it", "pt", "ru", "zh", "ja", "ko", "tr"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 * 1024  # 16 Go
jobs: dict[str, dict] = {}
GPU_LOCK = threading.Lock()  # sérialise le travail lourd (1 transcription à la fois)


class JobLogHandler(logging.Handler):
    def emit(self, record):
        buf = jobs.get(threading.current_thread().name, {}).get("log")
        if buf is not None:
            buf.append(self.format(record))


setup_logging(False)
_h = JobLogHandler()
_h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
logging.getLogger("subgen").addHandler(_h)


def _build_cfg(opts: dict) -> Config:
    cfg = Config.load(str(ROOT / "config.yaml"))
    targets = opts["targets"]
    cfg.override("translate.target_lang", targets[0])
    cfg.override("translate.target_langs", targets if len(targets) > 1 else None)
    cfg.override("subtitles.bilingual", opts["bilingual"])
    cfg.override("dub.enabled", opts.get("dub", False))
    if opts.get("voice"):
        cfg.override("dub.voice", opts["voice"])
    cfg.override("translate.backend", opts["backend"])
    cfg.override("attach.mode", opts["mode"])
    cfg.override("attach.container", opts["container"])
    cfg.override("asr.model", opts["model"])
    if opts.get("source"):
        cfg.override("asr.language", opts["source"])
    cfg.override("io.output_dir", str(OUTPUT))
    return cfg


def _finalize_and_attach(job: dict) -> None:
    """Écrit les sous-titres + vidéo finale (avec pistes doublées si demandé)."""
    subs, video_out, dubs = finalize_outputs(
        require_ffmpeg(), job["video"], job["docs"], job["cfg"], OUTPUT, job["cancel"])
    job["result"] = {"subtitles": subs, "video": video_out, "dubs": dubs}
    job["status"] = "done"
    job["log"].append("✅ Terminé.")


def _run_job(job_id: str, video: Path | None, opts: dict) -> None:
    import tempfile
    job = jobs[job_id]
    try:
        job["status"] = "running"
        if video is None:  # lien vidéo
            video = download_youtube(opts["url"], UPLOADS, job_id, opts.get("quality", "best"))
        stem = video.stem
        job["name"] = stem[len(job_id) + 1:] if stem.startswith(job_id + "_") else stem
        cfg = _build_cfg(opts)
        with GPU_LOCK:
            _ck(job)
            tmp = Path(tempfile.mkdtemp(prefix="subgen_"))
            try:
                src = prepare_source(video, cfg, require_ffmpeg(), tmp, job["cancel"])
            finally:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            docs = build_docs(src, cfg, job["cancel"])
            job.update({"docs": docs, "cfg": cfg, "video": video})
            if opts.get("review"):
                job["cues"] = _cues(docs, get_targets(cfg)[0])
                job["primary"] = get_targets(cfg)[0]
                job["status"] = "review"
                job["log"].append("✎ En attente de révision…")
                return
            _finalize_and_attach(job)
    except Cancelled:
        job["status"] = "cancelled"
        job["log"].append("⏹ Annulé.")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["log"].append(f"❌ Échec : {e}")


def _finalize_job(job_id: str, edits: list) -> None:
    job = jobs[job_id]
    try:
        job["status"] = "running"
        doc = job["docs"][job["primary"]]
        for e in edits:
            i = int(e.get("i", -1))
            if 0 <= i < len(doc.segments):
                doc.segments[i].translation = (e.get("text") or "").strip()
        with GPU_LOCK:
            _finalize_and_attach(job)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["log"].append(f"❌ Échec : {e}")


def _ck(job):
    if job["cancel"].is_set():
        raise Cancelled("Traitement annulé.")


def _cues(docs: dict[str, SubtitleDoc], primary: str) -> list[dict]:
    doc = docs[primary]
    return [{"i": i, "start": round(s.start, 2), "end": round(s.end, 2),
             "source": s.text, "target": s.out_text} for i, s in enumerate(doc.segments)]


def _new_job(name: str) -> str:
    jid = uuid.uuid4().hex[:12]
    jobs[jid] = {"status": "queued", "log": [], "result": None, "error": None,
                 "name": name, "cancel": threading.Event()}
    return jid


@app.post("/api/jobs")
def create_jobs():
    files = [f for f in request.files.getlist("video") if f and f.filename]
    url = request.form.get("youtube_url", "").strip()
    if not files and not url:
        return jsonify(error="Fournis une ou plusieurs vidéos, ou un lien."), 400

    targets = request.form.getlist("targets") or [request.form.get("target", "ar")]
    targets = list(dict.fromkeys([t for t in targets if t]))  # dédoublonne, garde l'ordre
    base = {
        "targets": targets,
        "bilingual": request.form.get("bilingual") == "1",
        "dub": request.form.get("dub") == "1",
        "voice": request.form.get("voice", "").strip(),
        "backend": request.form.get("backend", "nllb"),
        "model": request.form.get("model", "large-v3"),
        "mode": request.form.get("mode", "soft"),
        "container": request.form.get("container", "mp4"),
        "source": request.form.get("source", "").strip(),
        "quality": request.form.get("quality", "best"),
    }
    want_review = request.form.get("review") == "1"

    specs: list[tuple[str, Path | None, dict]] = []  # (job_id, video_path, opts)
    if files:
        for f in files:
            name = Path(f.filename).name
            jid = _new_job(name)
            dest = UPLOADS / f"{jid}_{name}"
            f.save(dest)
            specs.append((jid, dest, dict(base)))
    else:
        try:
            urls = expand_url(url)
        except Exception as e:
            return jsonify(error=f"Lien illisible : {e}"), 400
        for u in urls:
            jid = _new_job("vidéo en ligne")
            specs.append((jid, None, dict(base, url=u)))

    single = len(specs) == 1
    for _, _, opts in specs:
        opts["review"] = want_review and single
    for jid, vid, opts in specs:
        threading.Thread(target=_run_job, args=(jid, vid, opts), name=jid, daemon=True).start()

    return jsonify(job_ids=[s[0] for s in specs])


@app.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="introuvable"), 404
    if job["status"] in ("running", "queued", "review"):
        job["cancel"].set()
        if job["status"] == "review":  # rien en cours -> marque annulé
            job["status"] = "cancelled"
        job["log"].append("⏹ Annulation demandée…")
    return jsonify(ok=True)


@app.post("/api/jobs/<job_id>/finalize")
def finalize(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="introuvable"), 404
    if job["status"] != "review":
        return jsonify(error="job pas en révision"), 409
    edits = (request.get_json(silent=True) or {}).get("cues", [])
    threading.Thread(target=_finalize_job, args=(job_id, edits), name=job_id, daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="introuvable"), 404
    out = {"status": job["status"], "log": job["log"], "error": job["error"],
           "name": job.get("name"), "files": []}
    if job["status"] == "review":
        out["cues"] = job.get("cues", [])
        out["primary"] = job.get("primary")
    if job["status"] == "done" and job["result"]:
        if job["result"].get("video"):
            out["files"].append({"kind": "video", "name": Path(job["result"]["video"]).name})
        for d in job["result"].get("dubs", []):
            out["files"].append({"kind": "dub", "name": Path(d).name})
        for s in job["result"].get("subtitles", []):
            out["files"].append({"kind": "sub", "name": Path(s).name})
    return jsonify(out)


@app.get("/api/download/<path:name>")
def download(name: str):
    p = OUTPUT / Path(name).name
    if not p.exists():
        return jsonify(error="fichier introuvable"), 404
    return send_file(p, as_attachment=True)


@app.get("/")
def index():
    return Response(_PAGE, mimetype="text/html")


def _options(langs, with_auto=False):
    html = '<option value="">Détection auto</option>' if with_auto else ""
    for code, name in langs:
        sel = " selected" if (not with_auto and code == "ar") else ""
        html += f'<option value="{code}"{sel}>{name}</option>'
    return html


def _extra_chips():
    return "".join(
        f'<button type="button" class="lchip" data-l="{c}">{LANG_NAMES.get(c, c)}</button>'
        for c in EXTRA
    )


_PAGE = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Helix Subtitle Generator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0B0A1F;--bg2:#100E26;--panel:#17143A;--input:#0E0C24;--line:#2C2858;
  --fg:#ECEAF6;--mut:#9C97C9;--cyan:#3DE1D6;--mag:#FF5D8F;--gold:#FFC857;
  --err:#FF7A7A;--grad:linear-gradient(120deg,var(--cyan),var(--mag));
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:
  radial-gradient(1100px 520px at 80% -8%,rgba(255,93,143,.14),transparent 60%),
  radial-gradient(900px 480px at 0% 0%,rgba(61,225,214,.12),transparent 55%),
  var(--bg);
  color:var(--fg);font:15px/1.6 'IBM Plex Sans',system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:820px;margin:0 auto;padding:26px 20px 64px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:30px}
.brand{display:flex;align-items:center;gap:12px}
.mark{width:34px;height:34px;flex:none}
.word{font-family:'Space Grotesk';font-weight:700;font-size:19px;letter-spacing:.5px}
.word small{display:block;font-family:'IBM Plex Mono';font-weight:400;font-size:10px;
  letter-spacing:3px;color:var(--mut);text-transform:uppercase;margin-top:-2px}
.engine{font-family:'IBM Plex Mono';font-size:11px;color:var(--mut);display:flex;align-items:center;gap:7px;
  border:1px solid var(--line);border-radius:99px;padding:6px 12px}
.engine b{width:7px;height:7px;border-radius:50%;background:var(--cyan);box-shadow:0 0 10px var(--cyan)}
.hero{margin-bottom:26px}
.eyebrow{font-family:'IBM Plex Mono';font-size:11px;letter-spacing:4px;text-transform:uppercase;color:var(--mut);margin-bottom:14px}
.eyebrow span:first-child{color:var(--cyan)}.eyebrow span:last-child{color:var(--mag)}
h1{font-family:'Space Grotesk';font-weight:700;font-size:clamp(40px,8vw,68px);line-height:.98;margin:0 0 14px;letter-spacing:-1.5px}
h1 .g{background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.tag{color:var(--mut);font-size:16px;max-width:46ch;margin:0}
.card{background:linear-gradient(180deg,var(--panel),var(--bg2));border:1px solid var(--line);
  border-radius:18px;padding:24px;box-shadow:0 24px 60px -30px rgba(0,0,0,.8);margin-bottom:18px}
label{display:block;font-family:'IBM Plex Mono';font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:var(--mut);margin:16px 0 7px}
select,input[type=url]{width:100%;padding:12px 13px;background:var(--input);border:1px solid var(--line);border-radius:11px;color:var(--fg);font:14px 'IBM Plex Sans'}
select{appearance:none;cursor:pointer;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' fill='none'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%239C97C9' stroke-width='1.6'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 14px center}
select:focus,input[type=url]:focus{outline:none;border-color:var(--cyan);box-shadow:0 0 0 3px rgba(61,225,214,.16)}
input[type=url]::placeholder{color:#5b577e}
.row{display:flex;gap:14px;flex-wrap:wrap}.row>div{flex:1;min-width:150px}
.drop{border:1.5px dashed var(--line);border-radius:14px;padding:30px 20px;text-align:center;cursor:pointer;transition:.18s;background:var(--input)}
.drop:hover,.drop.hot{border-color:var(--cyan);background:rgba(61,225,214,.05)}
.drop.has{border-style:solid;border-color:var(--mag)}
.drop .ico{font-size:26px;display:block;margin-bottom:8px}
.drop b{font-family:'Space Grotesk';font-weight:600;font-size:16px}
.drop small{display:block;color:var(--mut);margin-top:5px;font-family:'IBM Plex Mono';font-size:12px}
.or{display:flex;align-items:center;gap:12px;margin:16px 0;color:var(--mut);font-family:'IBM Plex Mono';font-size:11px;letter-spacing:2px;text-transform:uppercase}
.or:before,.or:after{content:'';flex:1;height:1px;background:var(--line)}
.yt{display:flex;gap:10px;flex-wrap:wrap}.yt input{flex:1;min-width:180px}
.yt .ytico{flex:none;width:46px;display:flex;align-items:center;justify-content:center;background:var(--input);border:1px solid var(--line);border-radius:11px;font-size:18px}
.yt .qual{flex:none;width:120px;padding-right:30px}
.lchips{display:flex;flex-wrap:wrap;gap:8px}
.lchip{font:500 13px 'IBM Plex Sans';padding:7px 13px;border-radius:99px;cursor:pointer;
  background:var(--input);border:1px solid var(--line);color:var(--mut);transition:.15s}
.lchip:hover{border-color:var(--cyan);color:var(--fg)}
.lchip.on{background:rgba(61,225,214,.14);border-color:var(--cyan);color:var(--cyan)}
.toggles{display:flex;gap:22px;flex-wrap:wrap;margin-top:18px}
.tog{display:flex;align-items:center;gap:9px;cursor:pointer;font-family:'IBM Plex Sans';font-size:14px;
  color:var(--fg);text-transform:none;letter-spacing:0;margin:0}
.tog input{appearance:none;width:42px;height:24px;border-radius:99px;background:var(--input);border:1px solid var(--line);position:relative;cursor:pointer;transition:.2s;flex:none}
.tog input:checked{background:var(--grad);border-color:transparent}
.tog input:before{content:'';position:absolute;width:18px;height:18px;border-radius:50%;background:#fff;top:2px;left:2px;transition:.2s}
.tog input:checked:before{left:20px}
.go{margin-top:22px;width:100%;padding:15px;border:0;border-radius:13px;cursor:pointer;font:600 16px 'Space Grotesk';color:#0B0A1F;background:var(--grad);letter-spacing:.3px;transition:.18s}
.go:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 14px 34px -12px var(--mag)}
.go:disabled{opacity:.4;cursor:not-allowed}
.hidden{display:none!important}
/* job cards */
.job{background:linear-gradient(180deg,var(--panel),var(--bg2));border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:14px}
.stage{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
.stage h3{font-family:'Space Grotesk';font-weight:600;font-size:19px;margin:0}
.vidname{font-family:'IBM Plex Mono';font-size:12px;color:var(--mut);margin:2px 0 0;word-break:break-all}
.vidname b{color:var(--fg)}
.bar{height:8px;background:var(--input);border-radius:99px;overflow:hidden;margin:14px 0 10px}
.bar i{display:block;height:100%;width:0;background:var(--grad);border-radius:99px;transition:width .6s cubic-bezier(.2,.8,.2,1)}
.pct{font-family:'IBM Plex Mono';font-size:12px;color:var(--cyan);text-align:right;margin:0}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 0}
.chip{font-family:'IBM Plex Mono';font-size:11px;border:1px solid var(--line);border-radius:99px;padding:4px 10px;color:var(--mut)}
.chip.live{color:var(--cyan);border-color:rgba(61,225,214,.4)}
.badge{font-family:'IBM Plex Mono';font-size:12px;font-weight:500;padding:4px 11px;border-radius:99px;white-space:nowrap}
.b-run{background:rgba(255,200,87,.12);color:var(--gold)}
.b-done{background:rgba(61,225,214,.14);color:var(--cyan)}
.b-err{background:rgba(255,122,122,.14);color:var(--err)}
.b-review{background:rgba(255,93,143,.16);color:var(--mag)}
.files{margin-top:14px;display:grid;gap:10px}
.files a{display:flex;align-items:center;gap:11px;padding:13px 15px;background:var(--input);border:1px solid var(--line);border-radius:11px;color:var(--fg);text-decoration:none;transition:.15s}
.files a:hover{border-color:var(--cyan);transform:translateX(3px)}
.files a .k{font-size:18px}.files a .n{font-size:13px;color:var(--mut);word-break:break-all}
.files a .dl{margin-left:auto;font-family:'IBM Plex Mono';font-size:11px;color:var(--cyan)}
.act{margin-top:14px;display:flex;gap:10px}
.btn{flex:1;padding:11px;border-radius:11px;cursor:pointer;font:500 13px 'Space Grotesk';transition:.15s}
.btn-ghost{border:1px solid var(--line);background:transparent;color:var(--fg)}.btn-ghost:hover{border-color:var(--mag)}
.btn-cancel{border:1px solid rgba(255,122,122,.4);background:transparent;color:var(--err)}.btn-cancel:hover{background:rgba(255,122,122,.08)}
.btn-go{border:0;background:var(--grad);color:#0B0A1F}
details{margin-top:14px}
summary{font-family:'IBM Plex Mono';font-size:12px;color:var(--mut);cursor:pointer;list-style:none}
summary::-webkit-details-marker{display:none}summary:before{content:'▸ ';color:var(--cyan)}
details[open] summary:before{content:'▾ '}
.log{margin-top:10px;background:#070612;border:1px solid var(--line);border-radius:10px;padding:12px;font:12px/1.7 'IBM Plex Mono';color:#b9bce0;max-height:200px;overflow:auto;white-space:pre-wrap}
.err-box{color:var(--err);font-size:14px;margin-top:12px}
/* editor */
.editor{margin-top:14px;max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:11px}
.cue{display:grid;grid-template-columns:84px 1fr;gap:12px;padding:10px 12px;border-bottom:1px solid var(--line)}
.cue:last-child{border-bottom:0}
.cue .t{font-family:'IBM Plex Mono';font-size:11px;color:var(--mut);padding-top:8px}
.cue .src{font-size:11px;color:#6f6a98;margin-bottom:5px}
.cue textarea{width:100%;background:var(--input);border:1px solid var(--line);border-radius:8px;color:var(--fg);
  font:14px 'IBM Plex Sans';padding:7px 9px;resize:vertical;min-height:38px}
.cue textarea:focus{outline:none;border-color:var(--cyan)}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media (max-width:560px){.engine{display:none}}
</style></head><body><div class="wrap">

<header>
  <div class="brand">
    <svg class="mark" viewBox="0 0 40 40" fill="none">
      <defs><linearGradient id="hg" x1="0" y1="0" x2="40" y2="40">
        <stop offset="0" stop-color="#3DE1D6"/><stop offset="1" stop-color="#FF5D8F"/></linearGradient></defs>
      <path d="M11 4c0 9 18 9 18 16S11 27 11 36" stroke="url(#hg)" stroke-width="2.4" stroke-linecap="round"/>
      <path d="M29 4c0 9-18 9-18 16s18 9 18 16" stroke="url(#hg)" stroke-width="2.4" stroke-linecap="round" opacity=".55"/>
      <circle cx="11" cy="4" r="2.4" fill="#3DE1D6"/><circle cx="29" cy="36" r="2.4" fill="#3DE1D6"/>
      <circle cx="29" cy="4" r="2.4" fill="#FF5D8F"/><circle cx="11" cy="36" r="2.4" fill="#FF5D8F"/>
    </svg>
    <div class="word">Helix<small>subtitle&nbsp;generator</small></div>
  </div>
  <div class="engine"><b></b> moteur local · GPU</div>
</header>

<section class="hero" id="hero">
  <div class="eyebrow"><span>parole</span> &nbsp;↻&nbsp; <span>traduction</span></div>
  <h1>Deux brins,<br>une <span class="g">traduction</span>.</h1>
  <p class="tag">Helix écoute tes vidéos, transcrit chaque mot, tisse la traduction et la rattache à l'image — en local, sur ton GPU.</p>
</section>

<section class="card" id="form">
  <input type="file" id="video" accept="video/*" multiple class="hidden">
  <div class="drop" id="drop">
    <span class="ico">⇪</span>
    <b>Dépose une ou plusieurs vidéos</b>
    <small id="fname">ou clique pour parcourir · MP4 · MKV · MOV</small>
  </div>
  <div class="or">ou</div>
  <div class="yt">
    <span class="ytico">▶</span>
    <input type="url" id="yturl" placeholder="Lien vidéo ou playlist (YouTube, Vimeo…)">
    <select id="quality" class="qual" title="Qualité du téléchargement">
      <option value="best">Auto · max</option><option value="1080">1080p</option>
      <option value="720">720p</option><option value="480">480p</option>
    </select>
  </div>

  <div class="row">
    <div><label>Langue parlée</label><select id="source">__SRC__</select></div>
    <div><label>Traduire vers (principale)</label><select id="target">__TGT__</select></div>
  </div>
  <label>Langues supplémentaires</label>
  <div class="lchips" id="extra">__EXTRA__</div>

  <div class="row">
    <div><label>Moteur de traduction</label><select id="backend">
      <option value="nllb">NLLB · local hors-ligne</option>
      <option value="llm">LLM · Claude / Ollama</option>
      <option value="api">DeepL · clé API</option></select></div>
    <div><label>Transcription</label><select id="model">
      <option value="large-v3">large-v3 · qualité max</option>
      <option value="large-v3-turbo">large-v3-turbo · rapide</option>
      <option value="medium">medium · léger</option></select></div>
  </div>
  <div class="row">
    <div><label>Sous-titres</label><select id="mode">
      <option value="soft">Activables · piste mux</option>
      <option value="hard">Gravés · incrustés NVENC</option>
      <option value="none">Fichier .srt seul</option></select></div>
    <div><label>Conteneur</label><select id="container">
      <option value="mp4">MP4</option><option value="mkv">MKV</option></select></div>
  </div>
  <div class="toggles">
    <label class="tog"><input type="checkbox" id="bilingual"> Sous-titres bilingues</label>
    <label class="tog"><input type="checkbox" id="review"> Réviser avant d'attacher</label>
    <label class="tog"><input type="checkbox" id="dub"> Doublage (voix synthétique)</label>
  </div>
  <button class="go" id="go" disabled>Tisser les sous-titres</button>
  <p class="vidname" id="reviewNote" style="margin-top:10px"></p>
</section>

<div id="jobs"></div>
<button class="go hidden" id="again" style="background:transparent;border:1px solid var(--line);color:var(--fg)">↺ Nouvelle session</button>

<script>
const $=s=>document.querySelector(s);
let files=[],extras=new Set();

// extra langs chips
document.querySelectorAll('#extra .lchip').forEach(c=>{
  c.onclick=()=>{const l=c.dataset.l;if(extras.has(l)){extras.delete(l);c.classList.remove('on');}else{extras.add(l);c.classList.add('on');}};
});
// review note depends on count
function reviewNote(){
  const multi = files.length>1 || ($('#yturl').value.trim() && /list=|playlist/i.test($('#yturl').value));
  $('#reviewNote').textContent = ($('#review').checked && multi) ? "La révision ne s'applique qu'à une seule vidéo — ignorée en mode lot." : "";
}
$('#review').onchange=reviewNote;

// inputs
const drop=$('#drop'),inp=$('#video'),yt=$('#yturl');
drop.onclick=()=>inp.click();
inp.onchange=()=>setFiles([...inp.files]);
yt.oninput=()=>{refresh();reviewNote();};
function setFiles(fs){files=fs||[];
  if(files.length===1){$('#fname').textContent=files[0].name;}
  else if(files.length>1){$('#fname').textContent=files.length+' vidéos sélectionnées';}
  else{$('#fname').textContent='ou clique pour parcourir · MP4 · MKV · MOV';}
  drop.classList.toggle('has',files.length>0);$('#drop .ico').textContent=files.length?'🎞':'⇪';
  refresh();reviewNote();}
function refresh(){$('#go').disabled=!(files.length||yt.value.trim());}
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hot')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hot')}));
drop.addEventListener('drop',ev=>{const fs=[...ev.dataTransfer.files];if(fs.length){inp.files=ev.dataTransfer.files;setFiles(fs)}});

// stages
const STAGES=[
  {re:/vidéo doublée/i,label:'Montage de la vidéo doublée',pct:97},
  {re:/Assemblage de la piste/i,label:'Assemblage des voix',pct:93},
  {re:/Synthèse vocale/i,label:'Synthèse des voix',pct:90},
  {re:/Vidéo générée|gravé|Mux|Burn/i,label:'Incrustation dans la vidéo',pct:95},
  {re:/révision/i,label:'En attente de révision',pct:80},
  {re:/Sous-titres écrits/i,label:'Écriture des sous-titres',pct:86},
  {re:/NLLB|Traduction|LLM|DeepL/i,label:'Tissage de la traduction',pct:74},
  {re:/Alignement/i,label:'Calage de chaque mot',pct:58},
  {re:/Transcription/i,label:'Transcription de la parole',pct:42},
  {re:/Chargement.*ASR|modèle ASR/i,label:'Réveil du transcripteur',pct:24},
  {re:/bande-son|Extraction|audio/i,label:"À l'écoute de la bande-son",pct:13},
  {re:/YouTube|Téléchargement|lien|fusion/i,label:'Téléchargement de la vidéo',pct:6},
];
function stageFor(lg){for(const s of STAGES){for(let j=lg.length-1;j>=0;j--){if(s.re.test(lg[j]))return s;}}return{label:'En file…',pct:3};}
function detectLang(lg){const m=lg.join('\n').match(/Langue détectée\s*:\s*([a-zA-Z-]+)/);return m?m[1].toUpperCase():null;}
function fmtTime(s){s=Math.max(0,s);const m=Math.floor(s/60),sec=(s%60).toFixed(1);return m+':'+(sec<10?'0':'')+sec;}

// submit
$('#go').onclick=async()=>{
  const targets=[$('#target').value,...extras];
  const common=(fd)=>{fd.append('source',$('#source').value);targets.forEach(t=>fd.append('targets',t));
    ['backend','model','mode','container'].forEach(k=>fd.append(k,$('#'+k).value));
    fd.append('quality',$('#quality').value);
    fd.append('bilingual',$('#bilingual').checked?'1':'0');fd.append('review',$('#review').checked?'1':'0');
    fd.append('dub',$('#dub').checked?'1':'0');};
  $('#form').classList.add('hidden');$('#hero').classList.add('hidden');$('#again').classList.remove('hidden');
  const fd=new FormData();
  if(files.length){files.forEach(f=>fd.append('video',f));} else {fd.append('youtube_url',yt.value.trim());}
  common(fd);
  let r;try{r=await fetch('/api/jobs',{method:'POST',body:fd});}catch(e){return alert('Envoi impossible : '+e);}
  const j=await r.json();
  if(j.error){return alert(j.error);}
  (j.job_ids||[]).forEach((id,k)=>makeCard(id,files[k]?files[k].name:null));
};
$('#again').onclick=()=>location.reload();

// one card per job
function makeCard(id,localName){
  const el=document.createElement('div');el.className='job';el.dataset.id=id;
  el.innerHTML=`<div class="stage"><h3 class="st">En file…</h3><span class="badge b-run bd">en file</span></div>
    <p class="vidname nm">${localName?'<b>'+localName+'</b>':''}</p>
    <div class="bar"><i class="bf"></i></div><p class="pct pc">3 %</p>
    <div class="chips ch"></div>
    <div class="editor hidden ed"></div>
    <div class="files hidden fl"></div><div class="err-box hidden eb"></div>
    <div class="act"><button class="btn btn-cancel cx">⏹ Annuler</button></div>
    <details><summary>Journal technique</summary><div class="log lg"></div></details>`;
  $('#jobs').appendChild(el);
  el.querySelector('.cx').onclick=async()=>{el.querySelector('.cx').disabled=true;
    try{await fetch('/api/jobs/'+id+'/cancel',{method:'POST'});}catch(e){}};
  poll(id,el,!!localName);
}
function setCard(el,cls,txt){const b=el.querySelector('.bd');b.className='badge b-'+cls+' bd';b.textContent=txt;}
function setProg(el,s){el.querySelector('.st').textContent=s.label;const p=Math.round(s.pct);
  el.querySelector('.bf').style.width=p+'%';el.querySelector('.pc').textContent=p+' %';}

function poll(id,el,hasLocal){
  const t=setInterval(async()=>{
    let j;try{j=await(await fetch('/api/jobs/'+id)).json();}catch(e){return;}
    const lg=j.log||[];el.querySelector('.lg').textContent=lg.join('\n');
    if(j.name && !hasLocal)el.querySelector('.nm').innerHTML='<b>'+j.name+'</b>';
    if(j.status==='running'||j.status==='queued'){
      setProg(el,stageFor(lg));setCard(el,'run','en cours');
      const L=detectLang(lg);el.querySelector('.ch').innerHTML=L?('<span class="chip live">parole : '+L+'</span>'):'';
    }
    if(j.status==='review'){clearInterval(t);showEditor(el,id,j);}
    if(j.status==='done'){clearInterval(t);finishCard(el,j);}
    if(j.status==='error'){clearInterval(t);setCard(el,'err','échec');setProg(el,{label:"Échec",pct:0});
      el.querySelector('.cx').classList.add('hidden');const e=el.querySelector('.eb');e.classList.remove('hidden');e.textContent='⚠ '+(j.error||'Erreur');}
    if(j.status==='cancelled'){clearInterval(t);setCard(el,'err','annulé');setProg(el,{label:'Annulé',pct:0});
      el.querySelector('.cx').classList.add('hidden');}
  },1100);
}
function finishCard(el,j){
  setCard(el,'done','terminé');setProg(el,{label:"C'est tissé ✦",pct:100});
  el.querySelector('.cx').classList.add('hidden');el.querySelector('.ch').classList.add('hidden');
  const box=el.querySelector('.fl');box.innerHTML='';box.classList.remove('hidden');
  const ICO={video:'🎬',dub:'🔊',sub:'🅰'},LBL={video:'Vidéo sous-titrée',dub:'Vidéo doublée',sub:'Sous-titres'};
  (j.files||[]).forEach(f=>{const a=document.createElement('a');a.href='/api/download/'+encodeURIComponent(f.name);
    a.innerHTML='<span class="k">'+(ICO[f.kind]||'📄')+'</span><span><b>'+
      (LBL[f.kind]||'Fichier')+'</b><br><span class="n">'+f.name+'</span></span><span class="dl">télécharger</span>';
    box.appendChild(a);});
}
function showEditor(el,id,j){
  setCard(el,'review','à réviser');setProg(el,{label:'Révise puis génère',pct:80});
  el.querySelector('.cx').classList.add('hidden');
  const ed=el.querySelector('.ed');ed.classList.remove('hidden');ed.innerHTML='';
  (j.cues||[]).forEach(c=>{const row=document.createElement('div');row.className='cue';
    row.innerHTML='<div class="t">'+fmtTime(c.start)+'<br>'+fmtTime(c.end)+'</div>'+
      '<div><div class="src">'+(c.source||'').replace(/</g,'&lt;')+'</div>'+
      '<textarea data-i="'+c.i+'"></textarea></div>';
    ed.appendChild(row);row.querySelector('textarea').value=c.target||'';});
  const act=el.querySelector('.act');act.innerHTML='';
  const gen=document.createElement('button');gen.className='btn btn-go';gen.textContent='✦ Générer la vidéo';
  act.appendChild(gen);
  gen.onclick=async()=>{gen.disabled=true;gen.textContent='Génération…';
    const cues=[...ed.querySelectorAll('textarea')].map(t=>({i:+t.dataset.i,text:t.value}));
    ed.classList.add('hidden');
    try{await fetch('/api/jobs/'+id+'/finalize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cues})});}catch(e){}
    poll(id,el,true);};
}
</script>
</div></body></html>"""
_PAGE = (_PAGE.replace("__SRC__", _options(LANGS, with_auto=True))
              .replace("__TGT__", _options(LANGS))
              .replace("__EXTRA__", _extra_chips()))


def main():
    print("Helix Web UI -> http://localhost:7860  (Ctrl+C pour arrêter)")
    app.run(host="127.0.0.1", port=7860, threaded=True)


if __name__ == "__main__":
    main()
