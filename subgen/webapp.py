"""Interface web (Flask) pour Helix : upload vidéo ou lien -> choix langues -> sous-titrage.

Lancement :  python -m subgen.webapp   (puis http://localhost:7860)
Réutilise le pipeline de subgen.pipeline ; exécution en thread avec logs en direct.
"""
from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

from .config import Config
from .pipeline import Cancelled, process
from .translate.nllb import NLLB_CODES
from .utils import download_youtube, setup_logging

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUTPUT = ROOT / "output"
UPLOADS.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

# noms conviviaux pour les langues gérées par NLLB
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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024  # 8 Go max upload
jobs: dict[str, dict] = {}


class JobLogHandler(logging.Handler):
    """Route les logs du thread courant vers le tampon du job correspondant."""

    def emit(self, record):
        buf = jobs.get(threading.current_thread().name, {}).get("log")
        if buf is not None:
            buf.append(self.format(record))


setup_logging(False)
_h = JobLogHandler()
_h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
logging.getLogger("subgen").addHandler(_h)


def _run_job(job_id: str, video: Path | None, opts: dict) -> None:
    job = jobs[job_id]
    try:
        job["status"] = "running"
        if video is None:  # source = lien vidéo (YouTube, Vimeo…)
            video = download_youtube(opts["url"], UPLOADS, job_id, opts.get("quality", "best"))
        # nom convivial = nom du fichier sans le préfixe id ni l'extension
        stem = video.stem
        job["name"] = stem[len(job_id) + 1:] if stem.startswith(job_id + "_") else stem
        cfg = Config.load(str(ROOT / "config.yaml"))
        cfg.override("translate.target_lang", opts["target"])
        cfg.override("translate.backend", opts["backend"])
        cfg.override("attach.mode", opts["mode"])
        cfg.override("attach.container", opts["container"])
        cfg.override("asr.model", opts["model"])
        if opts.get("source"):
            cfg.override("asr.language", opts["source"])
        cfg.override("io.output_dir", str(OUTPUT))
        res = process(video, cfg, cancel_event=job["cancel"])
        job["result"] = res
        job["status"] = "done"
        job["log"].append("✅ Terminé.")
    except Cancelled:
        job["status"] = "cancelled"
        job["log"].append("⏹ Annulé par l'utilisateur.")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["log"].append(f"❌ Échec : {e}")


@app.post("/api/jobs")
def create_job():
    f = request.files.get("video")
    url = request.form.get("youtube_url", "").strip()
    if (not f or not f.filename) and not url:
        return jsonify(error="Fournis une vidéo ou un lien."), 400
    if url and not url.lower().startswith(("http://", "https://")):
        return jsonify(error="Lien invalide (doit commencer par http/https)."), 400

    job_id = uuid.uuid4().hex[:12]
    video_path: Path | None = None
    name = "vidéo en ligne"
    if f and f.filename:
        name = Path(f.filename).name
        video_path = UPLOADS / f"{job_id}_{name}"
        f.save(video_path)

    opts = {
        "url": url,
        "quality": request.form.get("quality", "best"),
        "source": request.form.get("source", "").strip(),
        "target": request.form.get("target", "ar"),
        "backend": request.form.get("backend", "nllb"),
        "model": request.form.get("model", "large-v3"),
        "mode": request.form.get("mode", "soft"),
        "container": request.form.get("container", "mp4"),
    }
    jobs[job_id] = {"status": "queued", "log": [], "result": None, "error": None,
                    "name": name, "cancel": threading.Event()}
    threading.Thread(target=_run_job, args=(job_id, video_path, opts),
                     name=job_id, daemon=True).start()
    return jsonify(job_id=job_id)


@app.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="introuvable"), 404
    if job["status"] in ("running", "queued"):
        job["cancel"].set()
        job["log"].append("⏹ Annulation demandée…")
    return jsonify(ok=True)


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="introuvable"), 404
    out = {"status": job["status"], "log": job["log"], "error": job["error"],
           "name": job.get("name"), "files": []}
    if job["status"] == "done" and job["result"]:
        if job["result"].get("video"):
            out["files"].append({"kind": "video", "name": Path(job["result"]["video"]).name})
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
  color:var(--fg);font:15px/1.6 'IBM Plex Sans',system-ui,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:780px;margin:0 auto;padding:26px 20px 64px}
/* ---- header ---- */
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:30px}
.brand{display:flex;align-items:center;gap:12px}
.mark{width:34px;height:34px;flex:none}
.word{font-family:'Space Grotesk';font-weight:700;font-size:19px;letter-spacing:.5px}
.word small{display:block;font-family:'IBM Plex Mono';font-weight:400;font-size:10px;
  letter-spacing:3px;color:var(--mut);text-transform:uppercase;margin-top:-2px}
.engine{font-family:'IBM Plex Mono';font-size:11px;color:var(--mut);display:flex;align-items:center;gap:7px;
  border:1px solid var(--line);border-radius:99px;padding:6px 12px}
.engine b{width:7px;height:7px;border-radius:50%;background:var(--cyan);box-shadow:0 0 10px var(--cyan)}
/* ---- hero ---- */
.hero{position:relative;margin-bottom:26px}
.eyebrow{font-family:'IBM Plex Mono';font-size:11px;letter-spacing:4px;text-transform:uppercase;
  color:var(--mut);margin-bottom:14px}
.eyebrow span{color:var(--cyan)}.eyebrow span+span{color:var(--mag)}
h1{font-family:'Space Grotesk';font-weight:700;font-size:clamp(40px,8vw,68px);line-height:.98;
  margin:0 0 14px;letter-spacing:-1.5px}
h1 .g{background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.tag{color:var(--mut);font-size:16px;max-width:46ch;margin:0}
/* ---- cards ---- */
.card{background:linear-gradient(180deg,var(--panel),var(--bg2));border:1px solid var(--line);
  border-radius:18px;padding:24px;box-shadow:0 24px 60px -30px rgba(0,0,0,.8)}
label{display:block;font-family:'IBM Plex Mono';font-size:11px;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--mut);margin:16px 0 7px}
select,input[type=url]{width:100%;padding:12px 13px;background:var(--input);border:1px solid var(--line);
  border-radius:11px;color:var(--fg);font:14px 'IBM Plex Sans'}
select{appearance:none;cursor:pointer;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' fill='none'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%239C97C9' stroke-width='1.6'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 14px center}
select:focus,input[type=url]:focus{outline:none;border-color:var(--cyan);box-shadow:0 0 0 3px rgba(61,225,214,.16)}
input[type=url]::placeholder{color:#5b577e}
.row{display:flex;gap:14px;flex-wrap:wrap}.row>div{flex:1;min-width:150px}
/* dropzone */
.drop{border:1.5px dashed var(--line);border-radius:14px;padding:30px 20px;text-align:center;
  cursor:pointer;transition:.18s;background:var(--input)}
.drop:hover,.drop.hot{border-color:var(--cyan);background:rgba(61,225,214,.05)}
.drop.has{border-style:solid;border-color:var(--mag)}
.drop .ico{font-size:26px;display:block;margin-bottom:8px}
.drop b{font-family:'Space Grotesk';font-weight:600;font-size:16px}
.drop small{display:block;color:var(--mut);margin-top:5px;font-family:'IBM Plex Mono';font-size:12px}
/* "ou" separator + youtube input */
.or{display:flex;align-items:center;gap:12px;margin:16px 0;color:var(--mut);
  font-family:'IBM Plex Mono';font-size:11px;letter-spacing:2px;text-transform:uppercase}
.or:before,.or:after{content:'';flex:1;height:1px;background:var(--line)}
.yt{display:flex;gap:10px;flex-wrap:wrap}.yt input{flex:1;min-width:180px}
.yt .ytico{flex:none;width:46px;display:flex;align-items:center;justify-content:center;
  background:var(--input);border:1px solid var(--line);border-radius:11px;font-size:18px}
.yt .qual{flex:none;width:120px;padding-right:30px}
/* CTA */
.go{margin-top:22px;width:100%;padding:15px;border:0;border-radius:13px;cursor:pointer;
  font:600 16px 'Space Grotesk';color:#0B0A1F;background:var(--grad);letter-spacing:.3px;
  transition:.18s}
.go:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 14px 34px -12px var(--mag)}
.go:disabled{opacity:.4;cursor:not-allowed}
.hidden{display:none!important}
/* ---- progress panel ---- */
.stage{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:4px}
.stage h3{font-family:'Space Grotesk';font-weight:600;font-size:21px;margin:0}
.bar{height:8px;background:var(--input);border-radius:99px;overflow:hidden;margin:16px 0 12px}
.bar i{display:block;height:100%;width:0;background:var(--grad);border-radius:99px;
  transition:width .6s cubic-bezier(.2,.8,.2,1)}
.pct{font-family:'IBM Plex Mono';font-size:13px;color:var(--cyan);text-align:right;margin:0 0 12px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 2px;min-height:26px}
.chip{font-family:'IBM Plex Mono';font-size:11px;border:1px solid var(--line);border-radius:99px;
  padding:4px 10px;color:var(--mut)}
.chip.live{color:var(--cyan);border-color:rgba(61,225,214,.4)}
.badge{font-family:'IBM Plex Mono';font-size:12px;font-weight:500;padding:4px 11px;border-radius:99px}
.b-run{background:rgba(255,200,87,.12);color:var(--gold)}
.b-done{background:rgba(61,225,214,.14);color:var(--cyan)}
.b-err{background:rgba(255,122,122,.14);color:var(--err)}
/* files + log */
.files{margin-top:18px;display:grid;gap:10px}
.files a{display:flex;align-items:center;gap:11px;padding:14px 16px;background:var(--input);
  border:1px solid var(--line);border-radius:12px;color:var(--fg);text-decoration:none;
  font-family:'IBM Plex Sans';transition:.15s}
.files a:hover{border-color:var(--cyan);transform:translateX(3px)}
.files a .k{font-size:18px}.files a .n{font-size:13px;color:var(--mut);word-break:break-all}
.files a .dl{margin-left:auto;font-family:'IBM Plex Mono';font-size:11px;color:var(--cyan)}
details{margin-top:18px}
summary{font-family:'IBM Plex Mono';font-size:12px;color:var(--mut);cursor:pointer;list-style:none}
summary::-webkit-details-marker{display:none}
summary:before{content:'▸ ';color:var(--cyan)}
details[open] summary:before{content:'▾ '}
#log{margin-top:12px;background:#070612;border:1px solid var(--line);border-radius:11px;padding:14px;
  font:12px/1.7 'IBM Plex Mono';color:#b9bce0;max-height:230px;overflow:auto;white-space:pre-wrap}
.again{margin-top:18px;width:100%;padding:13px;border:1px solid var(--line);border-radius:12px;cursor:pointer;
  background:transparent;color:var(--fg);font:500 14px 'Space Grotesk'}
.again:hover{border-color:var(--mag)}
.vidname{font-family:'IBM Plex Mono';font-size:12px;color:var(--mut);margin:2px 0 0;word-break:break-all}
.vidname b{color:var(--fg)}
.cancel{margin-top:16px;width:100%;padding:11px;border:1px solid rgba(255,122,122,.4);border-radius:11px;
  cursor:pointer;background:transparent;color:var(--err);font:500 13px 'Space Grotesk';transition:.15s}
.cancel:hover{background:rgba(255,122,122,.08)}
.cancel:disabled{opacity:.5;cursor:default}
.err-box{color:var(--err);font-size:14px;margin-top:14px;line-height:1.6}
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
  <p class="tag">Helix écoute ta vidéo, transcrit chaque mot, tisse la traduction et la rattache à l'image — en local, sur ton GPU.</p>
</section>

<section class="card" id="form">
  <input type="file" id="video" accept="video/*" class="hidden">
  <div class="drop" id="drop">
    <span class="ico">⇪</span>
    <b>Dépose ta vidéo ici</b>
    <small id="fname">ou clique pour parcourir · MP4 · MKV · MOV</small>
  </div>
  <div class="or">ou</div>
  <div class="yt">
    <span class="ytico">▶</span>
    <input type="url" id="yturl" placeholder="Colle un lien vidéo (YouTube, Vimeo…)">
    <select id="quality" class="qual" title="Qualité du téléchargement">
      <option value="best">Auto · max</option>
      <option value="1080">1080p</option>
      <option value="720">720p</option>
      <option value="480">480p</option>
    </select>
  </div>
  <div class="row">
    <div><label>Langue parlée</label><select id="source">__SRC__</select></div>
    <div><label>Traduire vers</label><select id="target">__TGT__</select></div>
  </div>
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
  <button class="go" id="go" disabled>Tisser les sous-titres</button>
</section>

<section class="card hidden" id="panel">
  <div class="stage">
    <h3 id="stageLabel">Préparation…</h3>
    <span class="badge b-run" id="badge">en cours</span>
  </div>
  <p class="vidname" id="vidName"></p>
  <div class="bar"><i id="barFill"></i></div>
  <p class="pct" id="pctText">0 %</p>
  <div class="chips" id="chips"></div>
  <div class="files hidden" id="files"></div>
  <div class="err-box hidden" id="errBox"></div>
  <details id="logWrap"><summary>Journal technique</summary><div id="log"></div></details>
  <button class="cancel hidden" id="cancel">⏹ Annuler</button>
  <button class="again hidden" id="again">↺ Nouvelle vidéo</button>
</section>

<script>
const $=s=>document.querySelector(s);
let file=null,timer=null,currentJob=null;

// ---- inputs ----
const drop=$('#drop'),inp=$('#video'),yt=$('#yturl');
drop.onclick=()=>inp.click();
inp.onchange=()=>setFile(inp.files[0]);
yt.oninput=refresh;
function setFile(f){if(!f)return;file=f;$('#fname').textContent=f.name;drop.classList.add('has');
  $('#drop .ico').textContent='🎞';refresh();}
function refresh(){$('#go').disabled = !(file || yt.value.trim());}
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hot')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hot')}));
drop.addEventListener('drop',ev=>{const f=ev.dataTransfer.files[0];if(f){inp.files=ev.dataTransfer.files;setFile(f)}});

// ---- stages (parlantes) déduites du journal réel du pipeline ----
const STAGES=[
  {re:/Vidéo générée|gravé|Mux|Burn/i,label:'Incrustation dans la vidéo',pct:95},
  {re:/Sous-titres écrits/i,label:'Écriture des sous-titres',pct:88},
  {re:/NLLB|Traduction|LLM|DeepL/i,label:'Tissage de la traduction',pct:76},
  {re:/Alignement/i,label:'Calage de chaque mot',pct:60},
  {re:/Transcription/i,label:'Transcription de la parole',pct:44},
  {re:/Chargement.*ASR|modèle ASR/i,label:'Réveil du transcripteur',pct:26},
  {re:/bande-son|Extraction|audio/i,label:"À l'écoute de la bande-son",pct:14},
  {re:/YouTube|Téléchargement|lien|fusion/i,label:'Téléchargement de la vidéo',pct:6},
];
function stageFor(logArr){
  for(const s of STAGES){for(let j=logArr.length-1;j>=0;j--){if(s.re.test(logArr[j]))return s;}}
  return {label:'Préparation…',pct:4};
}
function detectLang(logArr){const m=logArr.join('\n').match(/Langue détectée\s*:\s*([a-zA-Z-]+)/);return m?m[1].toUpperCase():null;}

// ---- run ----
$('#go').onclick=async()=>{
  const fd=new FormData();
  if(file)fd.append('video',file); else {fd.append('youtube_url',yt.value.trim());fd.append('quality',$('#quality').value);}
  ['source','target','backend','model','mode','container'].forEach(k=>fd.append(k,$('#'+k).value));
  $('#form').classList.add('hidden');$('#hero').classList.add('hidden');
  $('#panel').classList.remove('hidden');
  $('#cancel').classList.remove('hidden');$('#cancel').disabled=false;
  setBadge('run','en cours');setStage({label: file?'Envoi de la vidéo…':'Récupération du lien…',pct:3});
  if(file)$('#vidName').innerHTML='<b>'+file.name+'</b>';
  let r;try{r=await fetch('/api/jobs',{method:'POST',body:fd});}catch(e){return fail('Envoi impossible : '+e);}
  const j=await r.json();
  if(j.error){return fail(j.error);}
  currentJob=j.job_id;poll(j.job_id);
};
$('#cancel').onclick=async()=>{
  if(!currentJob)return;
  $('#cancel').disabled=true;setBadge('run','annulation…');
  try{await fetch('/api/jobs/'+currentJob+'/cancel',{method:'POST'});}catch(e){}
};
function setBadge(c,t){const b=$('#badge');b.className='badge b-'+c;b.textContent=t;}
function setStage(s){$('#stageLabel').textContent=s.label;const p=Math.round(s.pct);
  $('#barFill').style.width=p+'%';$('#pctText').textContent=p+' %';}
function poll(id){
  timer=setInterval(async()=>{
    const r=await fetch('/api/jobs/'+id);const j=await r.json();
    const lg=j.log||[];
    $('#log').textContent=lg.join('\n');$('#log').scrollTop=1e9;
    if(j.name && !file)$('#vidName').innerHTML='<b>'+j.name+'</b>';
    if(j.status==='running'||j.status==='queued'){
      setStage(stageFor(lg));
      const L=detectLang(lg);
      $('#chips').innerHTML = L?('<span class="chip live">parole : '+L+'</span><span class="chip">→ '+$('#target').value.toUpperCase()+'</span>'):'';
    }
    if(j.status==='done'){clearInterval(timer);finish(j);}
    if(j.status==='error'){clearInterval(timer);fail(j.error||'Erreur inconnue');}
    if(j.status==='cancelled'){clearInterval(timer);cancelled();}
  },1100);
}
function cancelled(){
  setBadge('err','annulé');setStage({label:'Traitement annulé',pct:0});
  $('#chips').classList.add('hidden');$('#cancel').classList.add('hidden');
  $('#again').classList.remove('hidden');
}
function finish(j){
  setBadge('done','terminé');setStage({label:"C'est tissé ✦",pct:100});
  $('#chips').classList.add('hidden');$('#cancel').classList.add('hidden');
  const box=$('#files');box.innerHTML='';box.classList.remove('hidden');
  (j.files||[]).forEach(f=>{const a=document.createElement('a');a.href='/api/download/'+encodeURIComponent(f.name);
    a.innerHTML='<span class="k">'+(f.kind==='video'?'🎬':'🅰')+'</span>'+
      '<span><b>'+(f.kind==='video'?'Vidéo sous-titrée':'Sous-titres')+'</b><br>'+
      '<span class="n">'+f.name+'</span></span><span class="dl">télécharger</span>';
    box.appendChild(a);});
  $('#again').classList.remove('hidden');
}
function fail(msg){
  setBadge('err','échec');setStage({label:"On s'arrête là",pct:0});
  $('#chips').classList.add('hidden');$('#cancel').classList.add('hidden');
  const e=$('#errBox');e.classList.remove('hidden');e.textContent='⚠ '+msg;
  $('#logWrap').open=true;$('#again').classList.remove('hidden');
}
$('#again').onclick=()=>location.reload();
</script>
</div></body></html>"""
_PAGE = _PAGE.replace("__SRC__", _options(LANGS, with_auto=True)).replace("__TGT__", _options(LANGS))


def main():
    print("Helix Web UI -> http://localhost:7860  (Ctrl+C pour arrêter)")
    app.run(host="127.0.0.1", port=7860, threaded=True)


if __name__ == "__main__":
    main()
