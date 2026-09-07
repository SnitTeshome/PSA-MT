# PSA translator - serving app

A small FastAPI service (plus a single-page vanilla-JS front end) around this
project's two chosen checkpoints:

- **`ekegusii`** (`nllb_mixed_lora_only`) - English or Kiswahili into Ekegusii.
- **`other_langs`** (`nllb_combined_other_langs`) - English into Kiswahili,
  Somali, or Dholuo (one multi-task model, target chosen per request).

**Which checkpoint handles a request is never a user choice.** The UI only
offers a source language and a target language; `ROUTES` in `app.py` is the
(source, target) -> system lookup that decides which model actually runs.
Kiswahili as a source only ever routes to Ekegusii (there's no
Kiswahili->Somali/Dholuo model); English as a source routes to all four
targets. Picking Ekegusii as the target - from either source - always lands
on the `ekegusii` checkpoint; picking Kiswahili/Somali/Dholuo (only offered
under an English source) always lands on `other_langs`.

Adapted from [SamAbr/public-service-anouncement-MT/serve](https://github.com/SamAbr/public-service-anouncement-MT/tree/main/serve),
which serves four checkpoints that all share one target language and a
model-picker UI. This project's `other_langs` checkpoint doesn't fit that
shape (one model, three selectable targets, and a second checkpoint with an
overlapping-but-different source language), so the request model and the
front end were rebuilt around routing by language pair instead of by model;
the `Registry` load/cache/generation pattern, segmentation, rate limiting,
and feedback collection carry over close to as-is.

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

By default both systems load straight from this project's own
`deployment/nllb/{run_id}/` folders on local disk - no Hugging Face token or
upload needed to try this. Open `http://localhost:8000`.

To work on the front end without loading any weights:

```bash
MOCK_MODE=1 uvicorn app:app --port 8000
```

To serve from the Hub instead (once `07_publish_to_hub.ipynb` has actually
run), copy `.env.example` to `.env` and set `HF_EKEGUSII` /
`HF_OTHER_LANGS` to your repo ids - see the comments in that file for the
subfolder requirement.

## Get a shareable link

```bash
bash serve/run_public_demo.sh
```

Starts the API and opens a Cloudflare quick tunnel to it - no account, no
install beyond a single static binary it downloads to `serve/.cloudflared`
on first run. Prints a `https://<random>.trycloudflare.com` URL (also
written to `serve/PUBLIC_URL.txt`) that works from anywhere, using this
node's own GPU.

**This is a demo link, not a permanent one.** It's random and changes every
time the script runs, and it stops working the moment this process, this
node, or your terminal session ends - there's nothing to configure to make
it outlive the session; a tunnel like this fundamentally can't. For a link
that has to keep working (e.g. one cited in a paper), the service needs to
run somewhere that isn't tied to this node - a Hugging Face Space is the
natural fit, since the checkpoints are already going to the Hub via
`07_publish_to_hub.ipynb`; ask if you want that built out.

Ctrl-C stops both the tunnel and the API together. Logs land in
`serve/api.log` and `serve/tunnel.log` if anything needs debugging - the
script tails them itself on failure (a stuck/blocked tunnel most often means
the node's network blocks the QUIC/UDP path, which is why the script forces
`--protocol http2` instead).

## What's here

- `app.py` - the whole backend: `ROUTES` (the language-pair -> checkpoint
  lookup), a model registry with an LRU cache (both checkpoints fit in memory
  at once, so in practice neither ever evicts), sentence segmentation,
  `/api/translate`, `/api/feedback`, `/api/languages` (sources, and which
  targets are valid for each source), `/api/metrics` - one row per `ROUTES`
  entry, in that exact order (English->Ekegusii, English->Kiswahili,
  English->Somali, English->Dholuo, Kiswahili->Ekegusii): source language,
  target language, zero-shot chrF2++, fine-tuned chrF2++, "ALL" domains
  combined. Driven directly by `ROUTES` so this table can't drift out of
  sync with what the app actually serves. Reads `logs/nllb_all_results.csv`
  (the two Ekegusii rows) and `logs/nllb_other_languages_all_results.csv`
  (the three Kiswahili/Somali/Dholuo rows).

  **One honest gap**: `_ekegusii_direction_metrics()`'s column-name guesses
  (`run_id`/`test_set`/`direction`/`n`/`chrf2pp`) and its `en-guz`/`sw-guz`
  direction-value matching for `nllb_all_results.csv` haven't been checked
  against the real file - that CSV was built by `02_nllb_training.ipynb`
  before this app existed. It fails safe (a row is just omitted from
  `/api/metrics`, logged clearly, nothing crashes) if a guess is wrong -
  check the server log on first run and tell me the real column/direction
  names if a row doesn't show up. `nllb_other_languages_all_results.csv`'s
  schema is exact, since this project built that notebook directly.
- `static/index.html` - the UI: source/target language selectors (target
  options change based on the selected source), one output card, a
  confidence pill (the model's own score, shown as a percentage), a feedback
  form, and a collapsible metrics table.
- `.env.example` - every tunable, documented.
- `run_public_demo.sh` - starts the API and a Cloudflare quick tunnel
  together, prints the public URL. See "Get a shareable link" above.

## Not included yet

The reference repo's `Dockerfile`(s), `docker-compose.yml`, `deploy_space.py`,
and Space-specific files weren't ported over - this was scoped to "get the
two models translating behind a UI," not deployment packaging. Ask if you
want those adapted too.
