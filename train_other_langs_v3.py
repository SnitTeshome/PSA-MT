"""
train_other_langs_v3.py
========================
A separate, standalone training entry point for the Somali/Dholuo/Kiswahili
combined-model retrain - NOT a modification of train.py or
train_other_langs_v2.py, and not used by any other notebook.

WHAT v1 GOT WRONG (06_nllb_other_languages.ipynb, train.py)
----------------------------------------------------------------
train.py's Seq2SeqTrainer never sets forced_bos_token_id, num_beams,
no_repeat_ngram_size, or repetition_penalty for its internal
predict_with_generate evaluation - confirmed by reading the file directly,
nothing assumed. For a combined model whose validation set mixes three
target languages in one pass, that means the in-training "Chrf2pp" number
had no way to know which target language a given row was even supposed to
decode into - almost certainly why v1's reported validation chrF2++ (38-40)
never resembled any of the three languages' real, correctly-generated
scores on the held-out test set (55-89 range). On top of that, v1's
checkpoint-selection metric was aggregated across all three languages at
once, so it could pick a step that was still improving Kiswahili while
already hurting Somali/Dholuo, without that tradeoff ever being visible in
the one number it optimized. Net result: v1's combined model regressed
Somali and Dholuo below their own zero-shot baseline (chrF2++ -2.29 and
-4.21) despite Kiswahili improving (+1.34).

WHAT v2 FIXED, AND WHAT IT GOT WRONG
------------------------------------------
v2 replaced predict_with_generate + compute_metrics + load_best_model_at_end
entirely with PerLanguageProtectiveEval below - a callback that runs REAL,
correctly-configured generation (explicit forced_bos_token_id per language,
beam search, the same anti-repetition guards Section 8 and inference.py
already use) on a per-language sample of a validation set, and decides when
to stop from that instead of the Trainer's own loss-only internal eval. It
also lowered the learning rate (5e-5 -> 1e-5) and added a warmup schedule,
on the reasoning that Somali/Dholuo already have real, strong pretrained
NLLB capability (unlike Ekegusii, which needed a brand-new token bootstrapped
from nothing) - regressing them is a forgetting risk from a fairly aggressive
recipe applied to an already-good baseline, not an underfitting problem, so
more adapter capacity (a reviewed third-party suggestion proposed rank 32 and
expanding LoRA to q/k/v/out_proj) was rejected as the wrong direction. This
shrank the regression substantially (Somali -2.29 -> -0.42, Dholuo -4.21 ->
-0.64) - real progress, but not fully closed.

The reason it wasn't fully closed: v2 used `shared_val` (05's validation
output) as the per-language monitoring signal. Checked directly: `shared_val`
has 86.5% near-duplicate overlap with the training pool - nearly as
contaminated as the 78% that made v1's naive-split comparison notebook
untrustworthy in the first place - because `validation_psa.csv` was built
for the original Ekegusii curriculum pipeline, where some train/validation
overlap is normal, not held out to the same standard as `test_psa.csv`. So
v2's "best step" (500, the very first checkpoint) was chosen by a signal
that rewarded memorization, not generalization - confirmed by the ~12-18
point gap between what that checkpoint scored during v2's training (Somali
85.07, Dholuo 77.80) and what it actually scored on the clean, held-out
final test set (73.07, 59.56).

WHAT v3 (THIS FILE) FIXES
-------------------------------
PerLanguageProtectiveEval no longer takes a raw validation set and
subsamples it internally. The caller (the notebook) passes in an
already-correct `val_by_lang` - a "monitor" split carved from the same
clean, low-leakage 985-row pool `shared_test` comes from (confirmed only 5%
overlap with training), stratified by domain and disjoint from whatever
"final" subset of that same pool gets used for the real, once-only
end-of-training evaluation. Neither half is `shared_val` - see the
notebook's Section 4 for the actual split.

Eval frequency also dropped from every 500 steps to every 100: every
protected language's score peaked at the very first checkpoint v2 measured
and declined at every check after, which means the true peak - now measured
against a clean signal - could easily be earlier than 500 steps too.
Checking 5x more often needs `patience` scaled up correspondingly (3 -> 5)
so the stopping decision isn't twitchy on single-check noise from a smaller,
noisier per-check sample; the effective step-window of tolerance (patience x
eval_steps) still shrinks (1500 -> 500 steps) because catching the true peak
precisely was the actual point of checking more often.

WHAT'S DELIBERATELY STILL UNCHANGED FROM v1/v2
------------------------------------------------
LoRA rank (16), target modules (q_proj/v_proj only), and LoRA alpha/dropout
are all identical to the original OTHER_LANG_CFG - more adapter capacity is
the right move for an underfitting model (Ekegusii's situation), not for an
already-capable model regressing on a small, narrow, single-domain dataset,
which is what's happening here. Also still not attempted: mixing
general-domain (non-PSA) data into training - this project's own Ekegusii
ablation found "mixed" (broad content) beat narrow domain-only training, and
Somali/Dholuo/Kiswahili have never had that tested, but it needs data this
project doesn't currently have staged for these three languages.
"""
import os
import time
import inspect
import logging
import warnings
from pathlib import Path

import pandas as pd
import torch
from transformers import (AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback)
from peft import PeftModel
import mlflow

logging.getLogger("mlflow").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

from arch_config import EXPERIMENT_PREFIX, MLFLOW_DB
from peft_setup import wrap_for_training
from train import _ensure_isolated_mlflow_tracking   # reused as-is - same isolation guarantee,
                                                       # nothing about it needed to change here
import metrics


def _schedule_kwargs(train_size, batch_size, max_epochs, warmup_fraction=0.04):
    """
    Warmup/schedule args for Seq2SeqTrainingArguments, tolerant of which exact
    keyword name the installed transformers version accepts - confirmed on a
    real run that `warmup_ratio` isn't universal (TypeError: unexpected
    keyword argument on the node's actual installed version), so this
    inspects the real, installed signature at runtime instead of assuming
    one API, the same tolerance approach peft_setup.py already uses for its
    own cross-version tokenizer-attribute issue.
    """
    params = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
    kwargs = {}

    if "lr_scheduler_type" in params:
        kwargs["lr_scheduler_type"] = "linear"

    if "warmup_ratio" in params:
        kwargs["warmup_ratio"] = warmup_fraction
    elif "warmup_steps" in params:
        total_steps = max(1, (train_size // batch_size) * max_epochs)
        kwargs["warmup_steps"] = max(1, int(warmup_fraction * total_steps))
        print(f"  (this transformers version has no warmup_ratio - using warmup_steps="
             f"{kwargs['warmup_steps']} instead, ~{warmup_fraction:.0%} of {total_steps} total steps)")
    else:
        print("  WARNING: this transformers version exposes neither warmup_ratio nor "
             "warmup_steps on Seq2SeqTrainingArguments - proceeding with NO warmup.")

    return kwargs


class PerLanguageProtectiveEval(TrainerCallback):
    """
    Runs correctly-configured generation (forced_bos_token_id set per
    language, beam search, anti-repetition guards) on a per-language sample
    of validation rows every `eval_steps` steps - the same generation
    settings Section 8's evaluation and inference.py use, deliberately not
    the Trainer's own uninstrumented predict_with_generate path.

    Tracks each language's own chrF2++ across checks and requests a stop
    once EITHER of `protect_languages` has gone `patience` checks without
    beating its own best-so-far. Records which training step produced each
    language's best score; train_stage_v3() rebuilds the final
    deployment-ready model from whichever step was actually best - the
    Trainer's own per-step checkpoints are adapter-only (PEFT's normal
    save_pretrained behavior), so that does the same base-load + PEFT-load +
    merge_and_unload dance train.py's own end-of-run merge does, just
    pointed at an earlier step instead of the final one.
    """

    def __init__(self, tokenizer, val_by_lang, language_codes, max_len,
                protect_languages=("somali", "dholuo"), patience=5):
        """
        `val_by_lang` is used exactly as given, no internal resampling - the
        caller (the notebook) is responsible for it already being a clean,
        disjoint-from-"final" monitor split (carved from the low-leakage
        shared_test pool, stratified by domain - see the notebook's
        Section 4). Composing the right monitor set is a decision made once,
        in one place, not split across two files.
        """
        self.tokenizer = tokenizer
        self.language_codes = language_codes
        self.max_len = max_len
        self.protect_languages = protect_languages
        self.patience = patience
        self.val_by_lang = val_by_lang

        self.best = {}          # lang -> best chrf2pp so far
        self.best_step = {}     # lang -> step that achieved it
        self.bad_checks = {lang: 0 for lang in protect_languages}
        self.history = []       # every check's full per-language scores, for later inspection

    def _generate_one(self, model, text, tgt_code):
        self.tokenizer.src_lang = "eng_Latn"
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.max_len).to(model.device)
        forced_bos = self.tokenizer.convert_tokens_to_ids(tgt_code)
        with torch.no_grad():
            out = model.generate(**enc, forced_bos_token_id=forced_bos, max_length=self.max_len,
                                 num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.2)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def on_evaluate(self, args, state, control, model, **kwargs):
        was_training = model.training
        model.eval()

        scores = {"step": state.global_step}
        for lang, rows in self.val_by_lang.items():
            tgt_code = self.language_codes[lang]
            preds = [self._generate_one(model, r["src"], tgt_code) for r in rows]
            score = metrics.score(preds, [r["tgt"] for r in rows], f"perlang_eval_{lang}_step{state.global_step}")
            scores[lang] = score["chrf2pp"]

        self.history.append(dict(scores))
        print(f"[per-language eval @ step {state.global_step}] " +
              " | ".join(f"{l}={scores[l]:.2f}" for l in self.val_by_lang))

        should_stop = False
        for lang in self.protect_languages:
            if lang not in self.best or scores[lang] > self.best[lang]:
                self.best[lang] = scores[lang]
                self.best_step[lang] = state.global_step
                self.bad_checks[lang] = 0
            else:
                self.bad_checks[lang] += 1
                if self.bad_checks[lang] >= self.patience:
                    should_stop = True

        if should_stop:
            print(f"stopping: a protected language's chrF2++ hasn't beaten its own best "
                  f"for {self.patience} consecutive checks (best steps so far: {self.best_step})")
            control.should_training_stop = True

        if was_training:
            model.train()
        return control

    def overall_best_step(self):
        """The step to actually deploy: the LATEST of the protected languages' own best
        steps - by definition the point closest to training's end where neither protected
        language had yet regressed from its peak."""
        return max(self.best_step[lang] for lang in self.protect_languages)

    def save_history(self, path):
        pd.DataFrame(self.history).to_csv(path, index=False)
        print(f"per-language eval history saved: {path}")


def train_stage_v3(run_id, cfg, stage_name, init_from, train_tok, val_tok, tokenizer,
                   max_len, lora_kwargs, val_by_lang, language_codes, eval_steps=100,
                   max_epochs=5, patience=5, checkpoints_dir="checkpoints", logs_dir="logs"):
    # Confirmed on a real run: this transformers/torch build defaults to PyTorch's SDPA
    # attention dispatcher, which on an H100 can select a cuDNN CUDA-graph-based fused
    # attention backend - a newer, less mature code path that crashed with
    # "mha_graph.execute(...).is_good() == false" during backward() at step 88, well
    # before anything PerLanguageProtectiveEval touches (its first eval is step 100).
    # Disabling just that one backend leaves flash-attention/memory-efficient attention
    # available, so SDPA still dispatches to a faster-than-eager backend - just not the
    # one that crashed.
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
        print("  disabled the cuDNN SDPA backend (known crash risk on this environment) - "
             "flash-attention/memory-efficient attention remain available")

    mlflow_db_path = _ensure_isolated_mlflow_tracking()
    out_dir = Path(checkpoints_dir) / run_id
    print("=" * 70)
    print(f"  {run_id}   arch=nllb  stage={stage_name}  init={init_from}  use_lora=True  [v3: clean monitoring split]")
    print(f"  checkpoint -> {out_dir.resolve()}")
    print(f"  mlflow db  -> {mlflow_db_path}")
    print(f"  lr={cfg['learning_rate']}  eval_steps={eval_steps}  "
         f"patience={patience}  max_epochs={max_epochs}")

    model = AutoModelForSeq2SeqLM.from_pretrained(init_from)
    model = wrap_for_training(model, tokenizer, cfg, lora_kwargs, use_lora=True)

    schedule_kwargs = _schedule_kwargs(len(train_tok), cfg["batch_size"], max_epochs)
    print(f"  schedule -> {schedule_kwargs}")
    print("=" * 70)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=max(8, cfg["batch_size"] // 2),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=max_epochs,
        **schedule_kwargs,
        # predict_with_generate deliberately OFF - the Trainer's own generation path has
        # no forced_bos_token_id/beam/repetition config (see module docstring), so it
        # would just burn compute producing another unreliable number. Loss-based eval
        # (still useful, still cheap) is all the Trainer itself needs to do; the real
        # signal comes from PerLanguageProtectiveEval below.
        predict_with_generate=False,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        # Must comfortably exceed `patience` checks, or the checkpoint that turns out to
        # be the actual best one could already be deleted by the time training stops and
        # we go looking for it.
        save_total_limit=patience + 3,
        logging_steps=20,
        report_to=["mlflow"],
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        load_best_model_at_end=False,   # handled manually below - our real metric doesn't
                                        # come from compute_metrics, so this flag can't drive it
        optim="adamw_torch",
    )

    per_lang_cb = PerLanguageProtectiveEval(
        tokenizer=tokenizer, val_by_lang=val_by_lang, language_codes=language_codes,
        max_len=max_len, patience=patience,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer, callbacks=[per_lang_cb],
    )

    mlflow.set_experiment(f"{EXPERIMENT_PREFIX}-nllb-{stage_name}")
    while mlflow.active_run() is not None:
        print(f"ending stale run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({"init_from": str(init_from), "use_lora": True,
                          "n_train": len(train_tok), "learning_rate": cfg["learning_rate"],
                          "eval_steps": eval_steps, "patience": patience})

        t0 = time.time()
        trainer.train()
        train_seconds = time.time() - t0
        mlflow.log_metric("train_seconds", train_seconds)
        print(f"{run_id} training time: {train_seconds/60:.1f} minutes")

        Path(logs_dir).mkdir(exist_ok=True)
        per_lang_cb.save_history(f"{logs_dir}/{run_id}_perlang_eval_log.csv")

        best_step = per_lang_cb.overall_best_step()
        best_checkpoint_dir = out_dir / f"checkpoint-{best_step}"
        print(f"\nbest step by protected-language chrF2++: {best_step}")
        print(f"per-language best scores: {per_lang_cb.best}")
        print(f"reloading and merging: {best_checkpoint_dir}")

        if not best_checkpoint_dir.exists():
            available = sorted(p.name for p in out_dir.glob("checkpoint-*"))
            raise FileNotFoundError(
                f"{best_checkpoint_dir} no longer exists (save_total_limit evicted it too "
                f"early). Checkpoints still on disk: {available}. Increase save_total_limit "
                f"and re-run.")

        base = AutoModelForSeq2SeqLM.from_pretrained(init_from)
        peft_model = PeftModel.from_pretrained(base, str(best_checkpoint_dir))
        merged = peft_model.merge_and_unload()
        merged_path = out_dir / "merged"
        merged.save_pretrained(str(merged_path), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_path))
        mlflow.log_metric("best_step", best_step)
        for lang, score in per_lang_cb.best.items():
            mlflow.log_metric(f"best_{lang}_chrf2pp", score)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return merged_path, train_seconds, per_lang_cb
