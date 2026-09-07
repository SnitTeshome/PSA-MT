"""
train.py
========
The generalized train_stage() function, called identically by both the NLLB
and mT5 notebooks. Every checkpoint is saved with safe_serialization=True
(safetensors) explicitly set - this is the format the deployment demo loads,
and setting it explicitly means it's guaranteed regardless of what a given
environment's transformers version defaults to.

MLflow isolation from the earlier, independent project happens two ways, both
needed - neither alone is enough:
  1. Experiment naming is namespaced by arch_config.EXPERIMENT_PREFIX, so this
     project's runs never land in the same MLflow *experiment* as the earlier
     project's - "psa-translation-nllb-stage1" was already created by that
     project; this project's runs must not append to it.
  2. _ensure_isolated_mlflow_tracking() below forces the MLflow tracking *store*
     itself to an absolute path under this project's own root, overriding any
     ambient MLFLOW_TRACKING_URI. Without this, (1) alone still isn't sufficient
     if the two projects happen to share a tracking server or a tracking URI set
     in the shell environment rather than by this code - different experiment
     name, same physical store, still browsable side by side in one MLflow UI
     in a way that invites mixing them up at review/deployment time.

     This uses a sqlite database (arch_config.MLFLOW_DB), not a plain
     "file:./mlruns" URI - confirmed against the installed MLflow version that
     the filesystem backend now raises MlflowException ("in maintenance mode...
     migrate to a database backend") unless MLFLOW_ALLOW_FILE_STORE=true is set.
     sqlite needs no server and no extra dependency beyond mlflow itself
     (verified), and sidesteps relying on an opt-out flag for a backend MLflow's
     own message says won't receive further updates.
"""
import os
import time
import logging
import warnings
from pathlib import Path
import pandas as pd
import torch
from transformers import (AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback)
import mlflow

# The sqlite backend logs "database is locked" as a caught, non-fatal ERROR (mlflow's
# async logging queue retries internally - training itself never sees an exception),
# but at high logging_steps frequency this floods stdout enough to trip Jupyter's
# IOPub rate limit. Silencing mlflow's own logger (not the root logger, so unrelated
# warnings from other libraries still show) is the direct fix rather than reducing
# logging_steps, which would also throw away legitimate metrics.
logging.getLogger("mlflow").setLevel(logging.CRITICAL)

# Library UserWarnings (e.g. peft's save_embedding_layers notice) are expected, already
# confirmed harmless, and just add to the same console-flood risk - blanket-silenced per
# explicit request rather than filtered one module at a time.
warnings.filterwarnings("ignore")

from arch_config import EXPERIMENT_PREFIX, MLFLOW_DB
from peft_setup import wrap_for_training
from metrics import build_compute_metrics


def _ensure_isolated_mlflow_tracking():
    """Force MLflow's tracking store under THIS project's own root (the current
    working directory - the calling notebook already os.chdir()'d to PROJECT_ROOT
    before any training starts), regardless of any ambient MLFLOW_TRACKING_URI.
    Cheap and idempotent - called at the top of every train_stage() rather than
    once at import time, so there's no shared init-order state to get out of sync,
    and printed every time so the destination is visible in each cell's own output,
    not just once at the top of the notebook."""
    db_path = Path(os.path.abspath(MLFLOW_DB))
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    return db_path


def train_stage(run_id, architecture, cfg, stage_name, init_from, train_tok, val_tok,
                tokenizer, max_len, lora_kwargs, use_lora=True, epochs=8,
                checkpoints_dir="checkpoints", logs_dir="logs"):
    mlflow_db_path = _ensure_isolated_mlflow_tracking()
    out_dir_preview = Path(checkpoints_dir) / run_id
    print("=" * 70)
    print(f"  {run_id}   arch={architecture}  stage={stage_name}  init={init_from}  use_lora={use_lora}")
    print(f"  checkpoint -> {out_dir_preview.resolve()}")
    print(f"  mlflow db  -> {mlflow_db_path}")
    print("=" * 70)

    model = AutoModelForSeq2SeqLM.from_pretrained(init_from)
    model = wrap_for_training(model, tokenizer, cfg, lora_kwargs, use_lora=use_lora)

    out_dir = Path(checkpoints_dir) / run_id
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=max(8, cfg["batch_size"] // 2),   # was hardcoded to 8 -
                                                                     # left disconnected from
                                                                     # the raised train batch
                                                                     # size, wasting the same
                                                                     # H100 headroom during
                                                                     # every epoch's generation-
                                                                     # based evaluation
        learning_rate=cfg["learning_rate"],
        num_train_epochs=epochs,
        predict_with_generate=True,
        generation_max_length=max_len,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=20,
        report_to=["mlflow"],
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        load_best_model_at_end=True,
        metric_for_best_model="chrf2pp",   # primary metric for this project - see metrics.py
        optim="adamw_torch",
        # No save_safetensors kwarg: transformers dropped it (TypeError: unexpected keyword
        # argument, confirmed against the currently installed version - it's not renamed,
        # just gone). Verified empirically that this isn't a silent behavior change to
        # compensate for: model.save_pretrained() with no format argument at all still
        # writes model.safetensors, not a .bin - safetensors is the only format now, not
        # a default that happens to be toggleable. inference.save_for_deployment()'s
        # .safetensors check downstream still holds without this line.
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer, compute_metrics=build_compute_metrics(tokenizer),
        # Stops a run once validation chrF2++ hasn't improved for 2 consecutive epochs,
        # rather than always burning the full epoch count regardless of convergence -
        # this is what actually reduces total time across all runs, without cutting
        # any of the comparisons themselves.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    mlflow.set_experiment(f"{EXPERIMENT_PREFIX}-{architecture}-{stage_name}")

    # Defensive cleanup: mlflow.start_run() below raises "Run ... is already active" if a
    # PREVIOUS call to this function (in an earlier cell execution, this same kernel
    # session) died between start_run() and end_run() without going through the
    # `with` block's automatic cleanup - a cell error, a manual interrupt, an OOM mid-
    # training, anything. mlflow's active-run tracking is in-memory and kernel-lifetime,
    # so a stale run from an unrelated earlier attempt otherwise blocks every subsequent
    # train_stage() call, not just the one that actually failed.
    while mlflow.active_run() is not None:
        print(f"ending stale run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

    # `with` (not separate start_run()/end_run() calls) is what actually fixes this
    # going forward: mlflow.end_run() is guaranteed to run on the way out - including on
    # exception - so a run this function starts can never again be the stale one a later
    # call has to clean up.
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({"init_from": str(init_from), "use_lora": use_lora, "n_train": len(train_tok)})

        t0 = time.time()
        trainer.train()
        train_seconds = time.time() - t0
        mlflow.log_metric("train_seconds", train_seconds)
        print(f"{run_id} training time: {train_seconds/60:.1f} minutes")

        if use_lora:
            trainer.model.save_pretrained(str(out_dir / "adapter"), safe_serialization=True)
            if cfg["unfreeze_embeddings"]:
                print("Note: the embedding update lives outside peft's tracking (see peft_setup.py), "
                      "so this adapter-only save does NOT include it. The merged checkpoint below is "
                      "the complete, deployment-ready artifact for this run.")
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(str(out_dir / "merged"), safe_serialization=True)
            tokenizer.save_pretrained(str(out_dir / "merged"))
            merged_path = out_dir / "merged"
        else:
            trainer.save_model(str(out_dir / "merged"))
            tokenizer.save_pretrained(str(out_dir / "merged"))
            merged_path = out_dir / "merged"

        Path(logs_dir).mkdir(exist_ok=True)
        log_df = pd.DataFrame(trainer.state.log_history)
        log_path = f"{logs_dir}/{run_id}_training_log.csv"
        log_df.to_csv(log_path, index=False)
        mlflow.log_artifact(log_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trainer, merged_path, train_seconds
