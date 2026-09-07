"""
app.py - a two-model PSA translation service.

Serves the two checkpoints this project actually settled on, each solving a
different translation direction:

    ekegusii      nllb_mixed_lora_only. English or Kiswahili PSAs into
                  Ekegusii. Mixed curriculum, single pass, layer-freeze - the
                  recorded winner in BEST_NLLB_CHECKPOINT.json.
    other_langs   nllb_combined_other_langs. English PSAs into Kiswahili,
                  Somali, or Dholuo - one multi-task model trained on all
                  three target languages at once (the design chosen over
                  three independent per-language models - see 06's Section
                  10/justification cell for why).

The two checkpoints are an implementation detail the client never sees or
chooses. The UI only offers a source language and a target language; ROUTES
below is the (src, tgt) -> system lookup that decides which checkpoint
actually handles a given request. Kiswahili as a source only ever routes to
Ekegusii (there is no Kiswahili->Somali/Dholuo model); English as a source
routes to all four targets.

Checkpoints load from a local directory by default (this project's own
deployment/{architecture}/{run_id}/ layout, already present on the training
node) - point HF_EKEGUSII / HF_OTHER_LANGS at a Hub repo instead (with the
matching _SUBFOLDER, since 07_publish_to_hub.ipynb pushes multiple
checkpoints into each repo as subfolders, not as separate repos) once you've
uploaded and want to serve from there instead.

Run it:
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000

Or, without any weights at all, to work on the front end:
    MOCK_MODE=1 uvicorn app:app --port 8000
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

log = logging.getLogger("psa-mt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
STARTED = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ---------------------------------------------------------------------------
# CONFIGURATION - everything tunable is an environment variable, because this
# runs in a container/notebook-adjacent process and its only interface is env.
# ---------------------------------------------------------------------------


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _flag(name: str, default: bool = False) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


MOCK_MODE = _flag("MOCK_MODE")

# Defaults point at this project's own deployment/ layout, already present on
# the training node right now - no Hub upload required to try this. Switch to
# a Hub repo (e.g. "username/nllb-ekegusii-ablation") plus the matching
# _SUBFOLDER once 07_publish_to_hub.ipynb has actually run - a repo there
# holds several checkpoints as subfolders, not one checkpoint per repo, so
# the subfolder is required for a Hub source and ignored for a local path.
DEPLOYMENT_DIR = _env("DEPLOYMENT_DIR", str(HERE.parent / "deployment"))
HF_EKEGUSII = _env("HF_EKEGUSII", f"{DEPLOYMENT_DIR}/nllb/nllb_mixed_lora_only")
HF_EKEGUSII_SUBFOLDER = _env("HF_EKEGUSII_SUBFOLDER", "")
HF_OTHER_LANGS = _env("HF_OTHER_LANGS", f"{DEPLOYMENT_DIR}/nllb/nllb_combined_other_langs")
HF_OTHER_LANGS_SUBFOLDER = _env("HF_OTHER_LANGS_SUBFOLDER", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# How many models may sit in memory at once. Only 2 systems exist, so this
# just controls whether both stay warm (2, the default) or the second one
# reloads on first use after the first (1) - correctness is identical either
# way, this only trades startup/reload time against idle memory.
MAX_RESIDENT = int(_env("MAX_RESIDENT", "2"))
PRELOAD = _flag("PRELOAD", True)

BEAMS = int(_env("BEAMS", "4"))
MAX_NEW_TOKENS = int(_env("MAX_NEW_TOKENS", "128"))   # matches MAX_LEN used throughout training
MAX_CHARS = int(_env("MAX_CHARS", "4000"))
MAX_SEGMENTS = int(_env("MAX_SEGMENTS", "40"))

# Which systems this instance serves. ENABLED_SYSTEMS=ekegusii to run only the
# Ekegusii direction (e.g. on a smaller box), or the default "both".
ENABLED_SYSTEMS = [s.strip() for s in _env("ENABLED_SYSTEMS", "ekegusii,other_langs").split(",") if s.strip()]

# Where corrections go. A Hugging Face DATASET repo id, e.g.
# "username/psa-mt-feedback". Unset, feedback is appended to a local file and
# lost when the process restarts - fine for trying this out, useless once the
# link is shared, so set it before sharing the link.
FEEDBACK_REPO = _env("FEEDBACK_REPO", "")
FEEDBACK_FILE = HERE / "feedback.jsonl"

# Requests per minute per client IP. Matters the moment this is reachable from
# beyond localhost: it's an unauthenticated endpoint in front of a GPU, and a
# single loop can occupy the card indefinitely. 0 disables it.
RATE_LIMIT_PER_MIN = int(_env("RATE_LIMIT_PER_MIN", "30"))

ENG, SWH, GUZ, SOM, LUO = "eng_Latn", "swh_Latn", "guz_Latn", "som_Latn", "luo_Latn"
LANG_LABELS = {ENG: "English", SWH: "Kiswahili", GUZ: "Ekegusii", SOM: "Somali", LUO: "Dholuo"}

SYSTEMS: "OrderedDict[str, dict]" = OrderedDict([
    ("ekegusii", {
        "repo": HF_EKEGUSII,
        "subfolder": HF_EKEGUSII_SUBFOLDER or None,
        "run_id": "nllb_mixed_lora_only",
        "target_codes": [GUZ],
    }),
    ("other_langs", {
        "repo": HF_OTHER_LANGS,
        "subfolder": HF_OTHER_LANGS_SUBFOLDER or None,
        "run_id": "nllb_combined_other_langs",
        "target_codes": [SWH, SOM, LUO],
    }),
])

unknown_systems = [s for s in ENABLED_SYSTEMS if s not in SYSTEMS]
if unknown_systems:
    raise SystemExit(f"ENABLED_SYSTEMS names systems that do not exist: {unknown_systems}")
SYSTEMS = OrderedDict((k, v) for k, v in SYSTEMS.items() if k in ENABLED_SYSTEMS)

# The (source, target) -> system lookup the UI is built around. Order here is
# display order: English's targets list Ekegusii first (the project's
# original focus), then Kiswahili/Somali/Dholuo.
_ALL_ROUTES = [
    {"src": ENG, "tgt": GUZ, "system": "ekegusii"},
    {"src": ENG, "tgt": SWH, "system": "other_langs"},
    {"src": ENG, "tgt": SOM, "system": "other_langs"},
    {"src": ENG, "tgt": LUO, "system": "other_langs"},
    {"src": SWH, "tgt": GUZ, "system": "ekegusii"},
]
ROUTES = [r for r in _ALL_ROUTES if r["system"] in SYSTEMS]


def route_for(src_lang: str, tgt_lang: str) -> "dict | None":
    return next((r for r in ROUTES if r["src"] == src_lang and r["tgt"] == tgt_lang), None)

# ---------------------------------------------------------------------------
# SEGMENTATION
# ---------------------------------------------------------------------------
# NLLB is a sentence-level model. Handing it a whole paragraph makes it drop
# clauses, so split, translate each piece, and rejoin. Blank lines are
# preserved as paragraph breaks because a PSA's shape carries meaning.

_SENT_END = re.compile(r"(?<=[.!?:;])\s+(?=[A-Z0-9\"'“])")


def segment(text: str) -> list:
    """Split into translatable units, remembering blank lines."""
    units = []
    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            units.append(None)          # a paragraph break, not a segment
            continue
        parts = [p.strip() for p in _SENT_END.split(stripped) if p.strip()]
        units.extend(parts or [stripped])
    while units and units[0] is None:
        units.pop(0)
    while units and units[-1] is None:
        units.pop()
    return units


def rejoin(units: list, translations: list) -> str:
    """Put translated segments back where their sources were."""
    out, it = [], iter(translations)
    for u in units:
        out.append("\n\n" if u is None else next(it, ""))
    text = ""
    for piece in out:
        if piece == "\n\n":
            text = text.rstrip() + "\n\n"
        else:
            text += piece + " "
    return text.strip()


# ---------------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------------


class Registry:
    """
    Least-recently-used cache of loaded models.

    Generation holds a lock. One 600M model saturates a GPU on its own, so
    letting two requests decode concurrently would not make either finish
    sooner - it would just double the peak memory and risk an OOM mid-demo.
    """

    def __init__(self):
        self._loaded: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = asyncio.Lock()
        self.device = "cpu"
        self.dtype = None
        self.errors: dict = {}

    # -- lifecycle ---------------------------------------------------------

    def describe(self) -> dict:
        return {"device": self.device, "dtype": str(self.dtype), "mock": MOCK_MODE,
                "resident": list(self._loaded), "max_resident": MAX_RESIDENT}

    def _load(self, name: str):
        """Blocking. Called in a worker thread."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        spec = SYSTEMS[name]
        repo = spec["repo"]
        is_local = Path(repo).is_dir()
        kwargs = {"token": HF_TOKEN} if (HF_TOKEN and not is_local) else {}
        if spec.get("subfolder") and not is_local:
            kwargs["subfolder"] = spec["subfolder"]

        log.info("loading %s from %s%s ...", name, repo,
                 f" (subfolder={spec['subfolder']})" if kwargs.get("subfolder") else "")
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(repo, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            repo, dtype=self.dtype, **kwargs).to(self.device).eval()

        for code in spec["target_codes"]:
            if tok.convert_tokens_to_ids(code) == tok.unk_token_id:
                raise RuntimeError(
                    f"{repo} has no {code!r} token. If this is a fine-tuned checkpoint, "
                    f"the tokenizer was not saved/uploaded alongside the weights.")
        log.info("loaded %s in %.1fs (vocab %d)", name, time.time() - t0, len(tok))
        return tok, model

    async def get(self, name: str):
        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]
        pair = await asyncio.to_thread(self._load, name)
        self._loaded[name] = pair
        self._loaded.move_to_end(name)
        while len(self._loaded) > MAX_RESIDENT:
            evicted, _ = self._loaded.popitem(last=False)
            log.info("evicted %s to stay under MAX_RESIDENT=%d", evicted, MAX_RESIDENT)
            self._free()
        return pair

    @staticmethod
    def _free():
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- inference ---------------------------------------------------------

    def _generate(self, tok, model, texts: list, src_lang: str, tgt_lang: str,
                  beams: int, max_new_tokens: int):
        """
        Blocking. Builds NLLB's input format by hand - [src_lang] tokens [eos].

        no_repeat_ngram_size/repetition_penalty guard against a real
        repetition-loop failure mode found in this project's checkpoints on
        weak-checkpoint x hard-input combinations (see inference.py) - beam
        search alone reduces but does not eliminate the risk, and the guard
        costs nothing on inputs that were never going to repeat anyway.

        Returns (texts, confidences), where a confidence is the geometric-mean
        per-token probability of the chosen output. See NOTE_CONFIDENCE below.
        """
        import torch

        eos, pad = tok.eos_token_id, tok.pad_token_id
        tgt_id = tok.convert_tokens_to_ids(tgt_lang)
        enc = [[tok.convert_tokens_to_ids(src_lang)] +
               tok(t, add_special_tokens=False, truncation=True,
                   max_length=max_new_tokens - 2)["input_ids"] + [eos] for t in texts]
        width = max(len(e) for e in enc)
        # Left-pad: the encoder is bidirectional so the side does not change
        # the result.
        ids = torch.tensor([[pad] * (width - len(e)) + e for e in enc]).to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=(ids != pad).long(),
                                 forced_bos_token_id=tgt_id, max_new_tokens=max_new_tokens,
                                 num_beams=beams, no_repeat_ngram_size=3, repetition_penalty=1.2,
                                 return_dict_in_generate=True, output_scores=True)

        seqs = out.sequences
        scores = getattr(out, "sequences_scores", None)
        if scores is None:
            # Greedy decoding does not populate sequences_scores; derive the
            # same quantity from per-step logits instead.
            trans = model.compute_transition_scores(
                seqs, out.scores, normalize_logits=True)
            finite = torch.isfinite(trans)
            scores = (trans.masked_fill(~finite, 0.0).sum(-1)
                      / finite.sum(-1).clamp(min=1))
        conf = scores.exp().clamp(0.0, 1.0).tolist()
        return tok.batch_decode(seqs, skip_special_tokens=True), conf

    async def translate(self, name: str, texts: list, src_lang: str, tgt_lang: str,
                        beams: int, max_new_tokens: int):
        if MOCK_MODE:
            await asyncio.sleep(0.25)
            tag = f"[{name}→{tgt_lang.split('_')[0]}]"
            return [f"{tag} {t}" for t in texts], [0.72] * len(texts)
        tok, model = await self.get(name)
        async with self._lock:
            return await asyncio.to_thread(
                self._generate, tok, model, texts, src_lang, tgt_lang,
                beams, max_new_tokens)


registry = Registry()


# NOTE_CONFIDENCE ------------------------------------------------------------
# What the number is: the geometric-mean per-token probability the model
# assigned to the output it chose. For beam search that is
# exp(sequences_scores), the length-normalised sum of log-probabilities.
#
# What it is NOT: a probability that the translation is correct. Neural MT is
# routinely confident and wrong. The band thresholds below are eyeballed, not
# calibrated against human judgment - they separate "the model was unsure"
# from "the model was not", nothing more, and the UI says so.
CONFIDENCE_BANDS = [(0.65, "high"), (0.45, "moderate"), (0.0, "low")]


def band(conf: float) -> str:
    return next(name for floor, name in CONFIDENCE_BANDS if conf >= floor)


# ---------------------------------------------------------------------------
# METRICS - one row per ROUTES entry (the exact 5 pairs the UI actually
# serves, in that order): source language, target language, zero-shot vs.
# fine-tuned chrF2++, "ALL" domains combined (not broken out by
# bible/psa/general or by PSA sub-domain). Driven directly by ROUTES rather
# than a separate hardcoded list, so this table can't drift out of sync with
# what the app actually routes.
# ---------------------------------------------------------------------------

NLLB_ALL_RESULTS = HERE.parent / "logs" / "nllb_all_results.csv"
OTHER_LANGS_RESULTS = HERE.parent / "logs" / "nllb_other_languages_all_results.csv"


def _first_present(columns: list, candidates: list) -> "str | None":
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _ekegusii_direction_metrics(src_lang: str, tgt_lang: str) -> "dict | None":
    """
    One row: chrF2++ weighted by each row's own n across every test_set, for
    ONE fixed direction (English->Ekegusii and Kiswahili->Ekegusii are two
    separate rows here, not pooled together the way
    BEST_NLLB_CHECKPOINT.json's single "weighted overall" number does).

    Column and direction-value names are a best-effort guess -
    nllb_all_results.csv was built by 02_nllb_training.ipynb before this app
    existed, so its exact schema hasn't been directly confirmed here. Fails
    safe: if a column or direction can't be matched, that row is just
    omitted (logged clearly), never shown wrong or crashed on.
    """
    if not NLLB_ALL_RESULTS.exists():
        return None
    try:
        with open(NLLB_ALL_RESULTS, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return None
        columns = list(rows[0].keys())
        run_col = _first_present(columns, ["run_id", "run", "config", "checkpoint", "name"])
        dir_col = _first_present(columns, ["direction"])
        n_col = _first_present(columns, ["n"])
        chrf_col = _first_present(columns, ["chrf2pp", "chrf2++", "chrf", "chrf_2_pp"])
        if not (run_col and dir_col and n_col and chrf_col):
            log.warning("%s columns don't match what /api/metrics expects (found %s) - "
                       "Ekegusii rows omitted until this is fixed", NLLB_ALL_RESULTS.name, columns)
            return None

        src_hints = {ENG: ["en"], SWH: ["sw", "swa"]}[src_lang]
        tgt_hints = {GUZ: ["guz", "gusii"]}[tgt_lang]
        directions_present = sorted({r.get(dir_col) for r in rows if r.get(dir_col)})
        direction = next((d for d in directions_present
                         if any(h in str(d).lower() for h in src_hints)
                         and any(h in str(d).lower() for h in tgt_hints)), None)
        if direction is None:
            log.warning("no direction in %s matches %s->%s (directions present: %s)",
                       NLLB_ALL_RESULTS.name, src_lang, tgt_lang, directions_present)
            return None

        def weighted(run_id: str) -> "float | None":
            matched = [r for r in rows if r.get(run_col) == run_id and r.get(dir_col) == direction]
            if not matched:
                return None
            total_n = sum(float(r[n_col]) for r in matched)
            if total_n <= 0:
                return None
            return sum(float(r[chrf_col]) * float(r[n_col]) for r in matched) / total_n

        zeroshot = weighted("zeroshot")
        finetuned = weighted("mixed_lora_only")
        if zeroshot is None or finetuned is None:
            log.warning("could not find both 'zeroshot' and 'mixed_lora_only' rows for direction "
                       "%r in %s (run ids present: %s)", direction, NLLB_ALL_RESULTS.name,
                       sorted({r.get(run_col) for r in rows}))
            return None
        return {"source": LANG_LABELS[src_lang], "target": LANG_LABELS[tgt_lang],
               "zeroshot": round(zeroshot, 2), "finetuned": round(finetuned, 2)}
    except Exception as exc:
        log.warning("could not read %s: %s", NLLB_ALL_RESULTS, exc)
        return None


def _other_langs_metrics(tgt_lang: str) -> "dict | None":
    """
    One row: domain == "ALL", baseline (zeroshot) vs. combined (the approach
    06_nllb_other_languages.ipynb's comparison actually chose). Schema known
    exactly, since this project built that notebook's Section 9 directly.
    """
    if not OTHER_LANGS_RESULTS.exists():
        return None
    lang_key = {SWH: "kiswahili", SOM: "somali", LUO: "dholuo"}.get(tgt_lang)
    if lang_key is None:
        return None
    try:
        with open(OTHER_LANGS_RESULTS, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        zeroshot = next((float(r["chrf2pp"]) for r in rows if r.get("language") == lang_key
                        and r.get("domain") == "ALL" and r.get("approach") == "baseline"), None)
        finetuned = next((float(r["chrf2pp"]) for r in rows if r.get("language") == lang_key
                         and r.get("domain") == "ALL" and r.get("approach") == "combined"), None)
        if zeroshot is None or finetuned is None:
            return None
        return {"source": LANG_LABELS[ENG], "target": LANG_LABELS[tgt_lang],
               "zeroshot": round(zeroshot, 2), "finetuned": round(finetuned, 2)}
    except Exception as exc:
        log.warning("could not read %s: %s", OTHER_LANGS_RESULTS, exc)
        return None


def load_all_metrics() -> list:
    rows = []
    for route in ROUTES:
        if route["system"] == "ekegusii":
            row = _ekegusii_direction_metrics(route["src"], route["tgt"])
        else:
            row = _other_langs_metrics(route["tgt"])
        if row:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MOCK_MODE:
        import torch
        registry.device = "cuda" if torch.cuda.is_available() else "cpu"
        registry.dtype = torch.float16 if registry.device == "cuda" else torch.float32
        log.info("device=%s dtype=%s", registry.device, registry.dtype)
        if registry.device == "cpu":
            log.warning("no GPU visible - expect several seconds per sentence")
        needs_token = [k for k, v in SYSTEMS.items()
                       if not Path(v["repo"]).is_dir()]
        if needs_token and not HF_TOKEN:
            log.warning("HF_TOKEN is unset and these load from the Hub, so a private "
                        "repo will 401: %s", ", ".join(needs_token))
        for k, v in SYSTEMS.items():
            if Path(v["repo"]).is_dir():
                log.info("%s loads from local disk (%s) - no token needed", k, v["repo"])
        if PRELOAD:
            for name in SYSTEMS:
                try:
                    await registry.get(name)
                except Exception as exc:      # a missing model must not kill the app
                    registry.errors[name] = str(exc)
                    log.error("could not preload %s: %s", name, exc)
    else:
        log.warning("MOCK_MODE - no weights loaded, outputs are placeholders")
    yield


app = FastAPI(title="PSA translation", lifespan=lifespan)

# ---------------------------------------------------------------------------
# RATE LIMIT
# ---------------------------------------------------------------------------
# A fixed window per IP, held in memory. Not distributed, not exact at window
# boundaries, and it resets on restart - all fine, since the job is to stop
# one script from monopolising the GPU, not to bill anyone.

_hits: dict = {}


def client_ip(request) -> str:
    """
    Behind a tunnel or proxy, request.client.host is 127.0.0.1 for everyone,
    which would rate-limit the whole internet as a single client. Cloudflare
    sets CF-Connecting-IP; most other proxies set X-Forwarded-For.
    """
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit(request, call_next):
    if RATE_LIMIT_PER_MIN <= 0 or not request.url.path.startswith("/api/"):
        return await call_next(request)
    if request.url.path in {"/api/health", "/api/languages", "/api/metrics"}:
        return await call_next(request)

    window = int(time.time() // 60)
    ip = client_ip(request)
    key = (ip, window)
    _hits[key] = _hits.get(key, 0) + 1
    if len(_hits) > 4096:                      # drop stale windows
        for k in [k for k in _hits if k[1] < window]:
            _hits.pop(k, None)
    if _hits[key] > RATE_LIMIT_PER_MIN:
        return JSONResponse(
            {"detail": f"Rate limit: {RATE_LIMIT_PER_MIN} requests per minute. "
                       f"This demo runs on a single shared GPU."}, status_code=429)
    return await call_next(request)


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    src_lang: str
    tgt_lang: str
    beams: int = BEAMS
    max_new_tokens: int = MAX_NEW_TOKENS


@app.get("/api/health")
async def health():
    return {
        "ok": True, **registry.describe(), "errors": registry.errors,
        "pid": os.getpid(),
        "started": STARTED,
        "enabled_systems": list(SYSTEMS),
        "feedback_repo": FEEDBACK_REPO or None,
        "feedback_durable": bool(FEEDBACK_REPO),
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
    }


@app.get("/api/languages")
async def languages():
    sources = list(dict.fromkeys(r["src"] for r in ROUTES))
    targets_by_source = {}
    for src in sources:
        targets = []
        for r in ROUTES:
            if r["src"] != src:
                continue
            targets.append({
                "code": r["tgt"], "label": LANG_LABELS[r["tgt"]],
                "available": r["system"] not in registry.errors,
                "error": registry.errors.get(r["system"]),
            })
        targets_by_source[src] = targets
    return {
        "sources": [{"code": s, "label": LANG_LABELS[s]} for s in sources],
        "targets_by_source": targets_by_source,
        "runtime": registry.describe(),
        "defaults": {"beams": BEAMS, "max_new_tokens": MAX_NEW_TOKENS, "max_chars": MAX_CHARS},
    }


@app.get("/api/metrics")
async def metrics():
    return {"rows": load_all_metrics()}


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    route = route_for(req.src_lang, req.tgt_lang)
    if route is None:
        raise HTTPException(400, f"no model translates {req.src_lang!r} -> {req.tgt_lang!r}")
    system = route["system"]

    if len(req.text) > MAX_CHARS:
        raise HTTPException(413, f"text is longer than MAX_CHARS ({MAX_CHARS})")

    units = segment(req.text)
    texts = [u for u in units if u is not None]
    if not texts:
        raise HTTPException(400, "nothing to translate")
    if len(texts) > MAX_SEGMENTS:
        raise HTTPException(413, f"{len(texts)} segments exceeds MAX_SEGMENTS "
                                 f"({MAX_SEGMENTS}); send it in smaller pieces")

    beams = max(1, min(req.beams, 8))
    max_new = max(16, min(req.max_new_tokens, 256))

    t0 = time.perf_counter()
    try:
        pieces, confs = await registry.translate(
            system, texts, req.src_lang, req.tgt_lang, beams, max_new)
        # Report the weakest segment, not the average - a paragraph whose
        # first sentence is solid and whose third is a guess should not read
        # as uniformly fine.
        worst = min(confs) if confs else 0.0
        result = {
            "output": rejoin(units, pieces),
            "segments": len(texts),
            "confidence": round(worst, 4),
            "confidence_band": band(worst),
            "per_segment": [round(c, 4) for c in confs],
            "ms": round((time.perf_counter() - t0) * 1000),
            "ok": True,
        }
    except Exception as exc:
        log.exception("%s failed", system)
        result = {"ok": False, "error": str(exc)}

    return {"system": system, "src_lang": req.src_lang, "tgt_lang": req.tgt_lang, "result": result}


class Feedback(BaseModel):
    system: str
    src_lang: str
    tgt_lang: str = ""
    source: str = Field(min_length=1, max_length=MAX_CHARS)
    machine: str = Field(default="", max_length=MAX_CHARS)
    correction: str = Field(default="", max_length=MAX_CHARS)
    rating: str = ""           # "good" | "usable" | "wrong"
    note: str = Field(default="", max_length=2000)


@app.post("/api/feedback")
async def feedback(item: Feedback):
    """
    Record a native speaker's correction.

    This is the most valuable thing the demo produces. Automatic metrics
    cannot tell you whether a translation reads as a public notice rather
    than as scripture; a fluent speaker typing the right sentence can, and
    every correction is a new training pair for the next round.
    """
    record = item.model_dump()
    record["received"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = json.dumps(record, ensure_ascii=False)

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    if not FEEDBACK_REPO:
        log.warning("FEEDBACK_REPO unset - correction kept only on local disk")
        return {"ok": True, "stored": "local", "durable": False}

    try:
        from huggingface_hub import upload_file
        await asyncio.to_thread(
            upload_file,
            path_or_fileobj=str(FEEDBACK_FILE), path_in_repo="feedback.jsonl",
            repo_id=FEEDBACK_REPO, repo_type="dataset", token=HF_TOKEN)
        return {"ok": True, "stored": FEEDBACK_REPO, "durable": True}
    except Exception as exc:
        log.exception("could not upload feedback")
        return {"ok": True, "stored": "local", "durable": False, "error": str(exc)}


@app.get("/")
async def index():
    path = STATIC / "index.html"
    if not path.exists():
        return JSONResponse({"error": "static/index.html is missing"}, 500)
    return FileResponse(path)


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
