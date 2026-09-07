# PSA Machine Translation for Four Kenyan Languages

Machine translation of Kenyan public service announcements (PSAs) — health,
agriculture, education, security, and governance notices — into four
under-resourced languages: **Ekegusii, Kiswahili, Somali, and Dholuo**. Built
by fine-tuning [NLLB-200-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
on real and PSA-register parallel text, with a fixed seed (`SEED = 42`,
`arch_config.py`) throughout for reproducibility.

This repository is the companion to a separate ablation study of NLLB
fine-tuning strategies on English/Kiswahili→Ekegusii (documented and
published separately) — the Ekegusii checkpoint that study produced is
reused here as one of this system's two models, rather than retrained.

## Two models, five language pairs

Which checkpoint handles a request is never a user choice — it's decided
automatically by the (source language, target language) pair, exactly as
implemented in `serve/app.py`'s `ROUTES` table:

| Source | Target | Model |
|---|---|---|
| English | Ekegusii | `nllb_mixed_lora_only` |
| Kiswahili | Ekegusii | `nllb_mixed_lora_only` |
| English | Kiswahili | `nllb_combined_other_langs` |
| English | Somali | `nllb_combined_other_langs` |
| English | Dholuo | `nllb_combined_other_langs` |

There's no Kiswahili→Somali/Dholuo model — `nllb_combined_other_langs` only
ever sees English as a source in training.

- **`nllb_mixed_lora_only`** — English or Kiswahili PSAs into Ekegusii. Mixed
  curriculum (Bible + storybooks + PSAs in one pass, no staged curriculum),
  layer-freeze rather than LoRA despite the checkpoint's own naming. The
  recorded winner of the companion ablation study — see
  `BEST_NLLB_CHECKPOINT.json` for the full selection methodology (chrF2++
  weighted by each test-set/direction row's own sample size, not an
  unweighted average).
- **`nllb_combined_other_langs`** — English PSAs into Kiswahili, Somali, or
  Dholuo. One multi-task model trained on all three target languages at
  once, chosen over three independent per-language models after directly
  comparing both (`06_nllb_other_languages.ipynb`) — the combined model won
  2 of 3 languages, and the independent model's only advantage was a
  statistically insignificant 0.07 chrF2++ overall edge driven by a single
  language (Dholuo). See that notebook's justification cell for the full
  reasoning.

## Results

chrF2++ (word-order 2), zero-shot vs. fine-tuned, "ALL" domains combined —
the same table `serve/app.py`'s `/api/metrics` serves:

| Source | Target | Zero-shot chrF2++ | Fine-tuned chrF2++ |
|---|---|---|---|
| English | Ekegusii | — | — |
| Kiswahili | Ekegusii | — | — |
| English | Kiswahili | — | — |
| English | Somali | — | — |
| English | Dholuo | — | — |

*(Fill in from `logs/nllb_all_results.csv` and
`logs/nllb_other_languages_all_results.csv`, or just run `serve/app.py` and
read them off `/api/metrics`.)*

## Try it

```bash
cd serve
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Both checkpoints load from `deployment/nllb/{run_id}/` on local disk by
default — see `serve/README.md` for Hub-hosted loading, mock mode, and
`run_public_demo.sh` (a one-command shareable link via a Cloudflare quick
tunnel).

## Models on the Hub

Pushed via `07_publish_to_hub.ipynb`, one subfolder per checkpoint (a repo
holds several related checkpoints, not one model per repo):

- `<hf-username>/nllb-ekegusii-ablation` — subfolder `nllb_mixed_lora_only`
  (this system's Ekegusii model) plus the other 7 ablation configs and the
  mT5 comparison run, kept for the companion study's reproducibility.
- `<hf-username>/nllb-kiswahili-somali-luo` — subfolder `nllb_combined_other_langs`
  (this system's Kiswahili/Somali/Dholuo model, what `serve/` actually uses) plus
  the three independent per-language models kept for comparison, plus
  `nllb_combined_other_langs_v2` (the gentler-recipe retrain from
  `06_nllb_other_languages_v2.ipynb` - an in-progress improvement attempt, not
  yet wired into `serve/` in place of the original - see that notebook's
  section above for its current, not fully resolved status).

*(Replace `<hf-username>` with the actual namespace once pushed.)*

## Setup

```bash
pip install -r requirements.txt
```

See `requirements.txt` for which versions are pinned because they were
confirmed against a real bug this project hit, versus which are unpinned
because no version-specific issue was ever found for them.

`raw_data/` and the large derived files under `data/` aren't in this
repository (see `.gitignore` — `data/mixed.jsonl` alone is 108MB, over
GitHub's 100MB per-file limit). To reproduce from scratch: download
`raw_data/`'s source corpora from `<hf-username>/<raw-data-dataset-repo>`
*(fill in once hosted)*, then run `00_data_prep.ipynb` — it rebuilds every
large file locally in a few minutes, no GPU needed. The small final
evaluation splits (`test_psa.csv`, `validation_psa.csv`, `test_bible.csv`,
etc.) are committed as-is.

## Layout

```
Final_Training/
  raw_data/        NOT in git - the 6 original source corpora; see Setup
  data/            00_data_prep.ipynb's output - curriculum splits + 3 test sets.
                   Large derived files (mixed/stage1/stage2.jsonl, whole_dataset.*,
                   train.csv) are NOT in git - regenerate via 00_data_prep.ipynb.
                   The small final splits (test_*.csv, validation_*.csv) are.
  data/other_langs/  other_langs_{train,test,validation}.csv (05's output) +
                   Filtered_PSA_final.csv (Kiswahili/Somali/Dholuo PSA source data)
  checkpoints/     NOT in git - every trained model, one subfolder per run_id
  logs/            per-notebook results CSVs (chrF2++/BLEU tables) - small, committed
  deployment/      NOT in git - merged, deployment-ready checkpoints; see
                   "Models on the Hub" above instead
  mlflow.db        NOT in git - isolated sqlite MLflow tracking store
  serve/           the FastAPI + vanilla-JS serving app - see serve/README.md
  Notebooks/       the 8 notebooks below, run in order
  arch_config.py, data_io.py, eda.py, peft_setup.py, metrics.py, train.py, inference.py
                   shared modules every notebook imports from
  experiment_log.csv  one row per run: zero-shot + every fine-tuned configuration
  BEST_NLLB_CHECKPOINT.json  explicit record of the winning NLLB run (mixed_lora_only)
                   and why - read by 02's demo and 06 (cross-check only)
```

## Pipeline order

1. `00_data_prep.ipynb` -> builds `data/`
2. `01_eda.ipynb` -> reports on `data/`
3. `seed_language_check.ipynb` -> decides `arch_config.py`'s `related_code`
4. `02_nllb_training.ipynb` -> the 8-run NLLB ablation (needs `related_code` from step 3);
   produces `nllb_mixed_lora_only`, this system's Ekegusii model

   *(Numbering jumps from `02` to `05` on purpose - `03_nllb_other_languages.ipynb` is an
   early, superseded draft of the next step, kept for history but not run in sequence.
   See its own section below.)*
5. `05_other_languages_preprocess.ipynb` -> cleans `Filtered_PSA_final.csv` against
   this project's own `test_psa.csv`/`validation_psa.csv` (real, confirmed leakage -
   see that section below)
6. `06_nllb_other_languages.ipynb` -> Kiswahili/Somali/Dholuo, **independent** of `02` -
   raw base checkpoint. Trains both a combined multi-task model and three independent
   per-language models, compares them, and proceeds with the combined model (the
   winner) - both sets of checkpoints are saved either way. Produces
   `nllb_combined_other_langs`, this system's other-languages model.

   `06_nllb_other_languages_v2.ipynb` is a follow-up retrain of the combined model only -
   see its own section below for why, and its current (not yet fully resolved) status.
7. `07_publish_to_hub.ipynb` -> pushes deployment-ready checkpoints to the Hub
8. `serve/` -> the translation UI, backed by the two checkpoints above

---

## `00_data_prep.ipynb`

Builds every file under `data/` from the 6 raw corpora in `raw_data/`. Nothing downstream
re-splits or re-merges this output — it's the one place splitting happens.

1. **Environment setup** — project-root resolution, seeded RNG.
2. **Robust CSV loading (encoding repair)** — UTF-8 with a per-byte CP1252 fallback
   (registered `codecs` error handler) for the mixed-encoding source files.
3. **Loading and normalizing the 6 raw corpora** — one loader per source file, unified
   into a single schema.
4. **Shuffling and de-duplication** — fixed-seed shuffle, exact-duplicate removal.
5. **Three-part test construction: PSA -> Bible -> general Ekegusii** — carved out in
   that order: PSA test (2,000 rows, 400/domain across 5 domains, from native-speaker
   `psa_ke_test`/`psa_ke_train` only) first, then Bible test (2,000 random rows), then the
   general Ekegusii test (bible 5%, storybooks 25%, PSA 10%, dictionary/lughayangu 25% —
   each a percentage of that source's own *remaining* pool, after the first two tests are
   already removed; the PSA slice here explicitly includes the mostly-synthetic
   `kenyan_psa_multilingual_dataset`).
6. **Validation decomposition (general vs. PSA)** — the two validation sets Section 7's
   curriculum stages evaluate against.
7. **Curriculum construction: Stage 1, Stage 2, Mixed** — Stage 1 (general/non-PSA) ->
   Stage 2 (PSA upsampled x4 + 25% Stage-1 replay) as a sequential curriculum; Mixed as a
   single-pass control over the same pool.
8. **Held-out set construction: length gating and leakage verification** — confirms no
   test-set sentence leaked into any training split.
9. **Serialization and export** — every `.jsonl`/`.csv` under `data/`, plus
   `whole_dataset.csv`/`.parquet` and `mixture.json`.
10. **Cross-architecture portability notes** — how NLLB and mT5 each consume this same
    output differently (language-code tags vs. instruction prefixes).
11. **Reproducibility summary** — final row counts, seed, and file manifest.

## `01_eda.ipynb`

Read-only reporting on `00_data_prep.ipynb`'s output — never merges or re-splits.

1. **Environment setup**
2. **Loading everything `00_data_prep.ipynb` wrote**
3. **Corpus-level composition (`whole_dataset`)** — size and source breakdown before any
   splitting.
4. **Per-split EDA via the shared `eda.py`** — length distributions, token statistics.
5. **PSA test: domain stratification** — confirms the 400/domain x 5 domains balance held.
6. **Bible test and general test: composition breakdown**
7. **Vocabulary and out-of-vocabulary analysis** — found the PSA test's OOV rate (29.1%)
   far exceeds Bible's (9.2%), the main empirical motivation for treating PSA as its own
   evaluation domain rather than folding it into "general."
8. **Curriculum size report + independent leakage re-verification** — re-runs Section 8's
   check independently, on the finished files rather than in-memory state.
9. **Dataset datasheet summary** — a single reference table of every split's size/purpose.

## `seed_language_check.ipynb`

Empirically decides which existing NLLB-200 language `guz_Latn`'s embedding should be
seeded from, rather than assuming one. Its result (`related_code = "lug_Latn"`) is read
directly by `arch_config.py` — nothing downstream re-derives this.

1. **Environment setup**
2. **A small, fixed-seed subsample of the three test sets** — 100 rows/cell, cheap enough
   to run three candidates zero-shot without real training cost.
3. **Candidate seed languages** — `kik_Latn` (Kikuyu), `lug_Latn` (Luganda), `swh_Latn`
   (Swahili); all three are Bantu languages already in NLLB-200.
4. **Zero-shot generation and scoring, one fresh model load per candidate** — chrF2++ on
   all 6 test-set/direction combinations per candidate, embedding copied fresh each time.
5. **Results and recommendation** — Luganda won all 6 combinations (mean chrF2++ 15.4 vs.
   14.8 Swahili vs. 14.5 Kikuyu) — the actual result now encoded in `arch_config.py`.

## `02_nllb_training.ipynb`

The core ablation: 8 fine-tuned NLLB configurations (Stage1/Stage2/Mixed/PSA-only, each
paired `_lora_only`/`_lora_emb`) plus a zero-shot baseline, evaluated on all three test
sets, both directions. `mixed_lora_only` — this system's Ekegusii model — is the winner.

1. **Setup** — prints every resolved output path with a loud warning if `Final_Training`
   isn't in `PROJECT_ROOT`, since a silent path mistake here would misdirect 8 training
   runs' worth of checkpoints/logs.
2. **Data overview** — a quick row-count sanity check against `01_eda.ipynb`'s numbers,
   not a repeat of the EDA itself.
3. **Load the already-split curriculum data** — plus the three test sets individually
   (`test_psa`/`test_bible`/`test_general`), not the single pooled `test` this project
   moved away from.
4. **Tokenizer: add the real `guz_Latn` token** — a genuine new token (not a placeholder
   reusing Swahili's), embedding seeded from `arch_config.py`'s `related_code`.
5. **Tokenization**
6. **Run all NLLB configurations, one cell per run** — one cell per run_id, not a loop
   over configs, so a failure or an odd result is traceable to one specific cell.
7. **Evaluate every configuration on all three test sets** — writes
   `logs/nllb_all_results.csv` and the pivoted `logs/nllb_chrf_summary.csv`.
8. **Save deployment-ready checkpoints** — `deployment/nllb/{run_id}/`, each verified to
   actually contain a `.safetensors` file.
9. **Demonstration** — translates a handful of real PSA-style sentences with whichever
   `DEMO_RUN` Section 7's table names as the winner.

## `03_nllb_other_languages.ipynb` (superseded - kept for history, not part of the active pipeline)

An earlier draft of the Kiswahili/Somali/Dholuo specialization, predating `05`/`06`. It
evaluated on a plain 80/10/10 split of the same file used for training, rather than
matching against an independently-sourced test set the way `05` does. A review of that
style of split found roughly 78% of its "test" rows had a near-duplicate template in
"train" - same sentence structure (e.g. an institution/county name swapped, "KUCCPS
advises applicants in Murang'a County..." vs. "HELB advises applicants in Murang'a
County..."), which inflates apparent fine-tuning gains without reflecting genuine
generalization to new content. That finding is exactly why `05_other_languages_preprocess.
ipynb` and `06_nllb_other_languages.ipynb` were built with a cross-corpus shared test set
instead (verified independently to have only ~5% near-duplicate overlap with training).
Kept in this repo for transparency about how the current approach was arrived at -
**use `05`/`06`, not this file, to reproduce results.**

## `05_other_languages_preprocess.ipynb`

Data-only - no training, no model loading. Builds a **shared parallel test/validation
set** used by `06`: English+Kiswahili come from this project's own
`test_psa.csv`/`validation_psa.csv`; Somali+Dholuo are looked up from
`Filtered_PSA_final.csv` by matching English text, keeping only rows where both are
available. Same underlying content across all three languages, so a score difference
between languages reflects the language, not different test-set difficulty.

Confirmed directly, not assumed: despite zero shared `concept_id`, 985 of `test_psa.csv`'s
1,948 rows (50.6%) have a usable Kiswahili+Somali+Dholuo match this way - that's the
resulting shared test set size (1,589 of 1,751 for validation). Kiswahili's own text
coverage in `test_psa.csv` is ~40% blank, so it's backfilled from the same
`Filtered_PSA_final.csv` lookup where blank, the identical mechanism Somali/Dholuo already
use - bringing Kiswahili to the same 985/985 coverage rather than a smaller, non-random
subset. Rows not used by either shared set fold into training instead of sitting unused.

1. **Setup**
2. **Load the three source files** — `Filtered_PSA_final.csv` plus this project's
   own `test_psa.csv`/`validation_psa.csv`.
3. **Data quality: same-language mislabeling check** — English vs. each of
   Kiswahili/Somali/Dholuo (found 7 English=Somali and 119 English=Dholuo rows to drop).
4. **Build the shared parallel test set and validation set** — matched by normalized
   English text; Kiswahili backfilled from the lookup where blank; all three languages
   must be non-blank to survive into the shared set.
5. **Build the training pool** — everything from `Filtered_PSA_final.csv` not used by
   either shared set.
6. **Save all three outputs** — `other_langs_train.csv`, `other_langs_test.csv`,
   `other_langs_validation.csv` under `data/other_langs/`, read directly by `06`.

## `06_nllb_other_languages.ipynb`

One notebook, both training designs, run in a fixed order so the comparison is decided
before anything downstream depends on it: **combined multi-task model first, then the
three independent per-language models**, then a single evaluation pass scores every
approach, then an n-weighted comparison picks a winner, then domain ablation runs only on
the winner - while **both** sets of checkpoints get saved regardless of which one won.
Genuinely independent of `02`/`BEST_NLLB_CHECKPOINT` either way - raw
`facebook/nllb-200-distilled-600M`, not any Final_Training-produced checkpoint. Reads
`05`'s three output files directly.

- **Combined**: one `train_stage()` call on all three languages' training rows
  concatenated (each row keeps its own correct target-language code; tokenization reads
  `tgt_lang` per row rather than one fixed value per call). One checkpoint
  (`nllb_combined_other_langs`) - **this system's chosen model.**
- **Independent**: three separate `train_stage()` calls, one per target language, each in
  its own cell, each seeing only its own language's data. Three checkpoints
  (`nllb_kiswahili_target`/`nllb_somali_target`/`nllb_dholuo_target`) - kept for
  comparison, not used by `serve/`.

This tests a real, undecided question: does training all three together help (shared
encoder learns "how to read this PSA-register English sentence" from three signals) or
hurt (NLLB's tied embedding/output layer serves all three target languages' gradients at
once, which can interfere)? Not knowable in advance - which is why the notebook trains
both instead of assuming an answer. The winner is decided by chrF2++ weighted by each
`language x approach` row's own test-set `n` (the same weight-by-actual-sample-size
principle `BEST_NLLB_CHECKPOINT.json` was chosen by), not an unweighted mean across
languages - combined won on that basis, but by a margin (0.07 chrF2++ overall) inside
normal run-to-run noise and driven by a single language (Dholuo); combined actually won
2 of 3 languages outright (Kiswahili, Somali). See the notebook's own justification cell
for the full reasoning, including the operational case for one checkpoint over three.

Note: within the shared 985-row test set, Kiswahili's own text coverage would have been
much sparser than Somali/Dholuo's without `05`'s backfill (see `05`'s section above) - all
three now reach 985/985.

1. **Setup**
2. **Verify the language codes against the real NLLB tokenizer**
3. **Load `05`'s three outputs**
4. **Build training records** — per-language lists plus a combined concatenation built
   from the same records (test records stay per-language, since evaluation always scores
   each language separately).
5. **Tokenization** — combined (per-row `tgt_lang`) and independent (fixed `tgt_lang` per
   call), both from the same records.
6. **Zero-shot baseline** — raw checkpoint, saved once.
7. **Train the combined model** — one call.
8. **Train the independent models** — three calls, one per cell.
9. **Evaluate everything in one pass** — zero-shot, combined, and independent, overall and
   by domain, in one results table (`nllb_other_languages_all_results.csv`).
10. **Decide the winner** — n-weighted chrF2++, combined vs. independent; overridden to
    "combined" with the reasoning above, not taken as the raw metric's literal pick.
11. **Domain ablation** — winning approach only, from Section 9's already-computed scores.
12. **Save deployment-ready checkpoints** — combined and all three independent models,
    unconditionally, regardless of which approach won.
13. **Demonstration** — using the combined model.

## `06_nllb_other_languages_v2.ipynb`

A combined-only retrain of the other-languages model, addressing two problems found in
`06`'s original combined checkpoint: it regressed Somali and Dholuo below their own
zero-shot baseline (chrF2++ -2.29 and -4.21 respectively), and `train.py`'s internal
evaluation never set `forced_bos_token_id`, beam search, or anti-repetition guards -
meaning the metric used to pick a "best" checkpoint couldn't be trusted for a
three-target-language combined validation set in the first place. This notebook (with
`train_other_langs_v2.py` - a separate training entry point; `train.py` itself is
untouched) fixes both: a lower learning rate (1e-5, down from 5e-5), a warmup schedule,
and real per-language generation during evaluation (the same forced_bos_token_id/beam/
repetition settings Section 9 uses), with training stopped once Somali *or* Dholuo -
individually, not averaged - fails to beat its own best chrF2++ for several consecutive
checks.

Result so far: the regression shrank substantially (Somali -2.29 -> -0.42, Dholuo -4.21
-> -0.64) without being fully eliminated. Follow-up investigation found the *validation*
set this notebook's stopping decision relies on (`shared_val`, from `05`) has an ~86.5%
near-duplicate rate against the training pool - almost the same contamination `03`'s
naive split had - while the *test* set used for final reporting (`shared_test`) has only
~5%. So the "best step" this run picked was itself chosen using a partly-misleading
signal, and the true per-language trajectory earlier than the first checkpoint isn't yet
known. **This is not yet a fully resolved result** - the next planned fix is splitting a
low-leakage monitoring subset out of `shared_test` itself for the stopping decision,
rather than relying on `shared_val`.

## `07_publish_to_hub.ipynb`

Pushes deployment-ready checkpoints to two Hugging Face Hub repos, one subfolder per
checkpoint (a repo root can only hold one model's files, so multiple checkpoints in one
repo need the subfolder pattern) - see "Models on the Hub" above for which checkpoints go
where. Verifies every source folder actually has a `.safetensors` file before uploading
anything, creates both repos (private by default), uploads each checkpoint to its
subfolder, writes a `README.md` into each repo explaining the subfolder layout, and
verifies the upload against the Hub afterward.

## `serve/`

The FastAPI + vanilla-JS app that actually serves the two chosen checkpoints - see
`serve/README.md` for the full breakdown (routing, mock mode, the Cloudflare tunnel
script, the metrics endpoint). Briefly: the UI only ever asks for a source and target
language: `ROUTES` in `serve/app.py` is the single source of truth for which of the two
checkpoints handles a given pair, matching the table at the top of this file exactly.

---

## Shared modules

- **`arch_config.py`** — every architecture-specific setting (checkpoint, LoRA targets,
  learning rate, `related_code`, etc.) in one place, so no notebook carries a copy that
  could drift. Also `EXPERIMENT_PREFIX` (`"final-training"`, namespacing every MLflow
  experiment this project creates) and `MLFLOW_DB`.
- **`data_io.py`** — `load_jsonl`, `load_curriculum`, `derive_psa_only`, `load_flat_splits`.
- **`eda.py`** — the reporting functions `01_eda.ipynb` calls.
- **`peft_setup.py`** — LoRA wrapping, the embedding-unfreeze mechanism, and
  `extend_tokenizer_for_new_language` (version-tolerant across transformers releases that
  removed `additional_special_tokens`/`lang_code_to_id`).
- **`metrics.py`** — `score()`: chrF2++ (word_order=2) as the primary metric, not BLEU —
  Ekegusii's agglutinative morphology penalizes BLEU's exact whole-word matching even for
  correct translations; BLEU is still reported alongside it.
- **`train.py`** — `train_stage()`, the one training entry point every notebook calls.
  Forces MLflow's tracking store to an isolated sqlite file under this project root
  regardless of any ambient `MLFLOW_TRACKING_URI`, and guarantees `mlflow.end_run()` fires
  even on exception (a `with mlflow.start_run(...)` block, not separate calls).
- **`inference.py`** — `generate_one`, `save_for_deployment` (verifies a `.safetensors`
  file is actually present before calling a checkpoint deployment-ready), `mt5_prefix`.
- **`experiment_log.csv`** — one row per run: zero-shot + every fine-tuned configuration
  across both architectures, with the actual PEFT method used per row.

## Reproducibility

- `SEED = 42` everywhere (`arch_config.py`), used for every shuffle/split/subsample in
  every notebook.
- MLflow tracking is forced to a sqlite file under this project root
  (`_ensure_isolated_mlflow_tracking()` in `train.py`), so this project's runs can never
  land in the same store as any other project's on this node, regardless of environment
  variables.
- Every experiment name is prefixed `final-training-*` (`EXPERIMENT_PREFIX`), never the
  bare `psa-translation-*` names an earlier, separate project used.
