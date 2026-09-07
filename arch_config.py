"""
arch_config.py
===============
All architecture-specific settings live here, in one place, so nllb_training.ipynb
and mt5_training.ipynb both import from the same source of truth instead of each
carrying their own copy that could silently drift out of sync.
"""

ARCH_CONFIG = {
    "nllb": {
        "checkpoint": "facebook/nllb-200-distilled-600M",
        "lora_target_modules": ["q_proj", "v_proj"],   # M2M100-style attention naming
        "freeze_layers": 6,                             # used only for the LoRA-off fallback
        "tgt_code": "guz_Latn",                          # a genuinely new token, not a placeholder
        "related_code": "lug_Latn",                      # set from seed_language_check.ipynb's real run:
                                                          # Luganda won on all 6 test-set/direction slices
                                                          # (mean zero-shot chrF2++ 15.4 vs. 14.8 for Swahili,
                                                          # 14.5 for Kikuyu) - margins were modest on some
                                                          # slices (razor-thin on general/sw-guz), but the
                                                          # win was consistent, not noise. Re-run that check
                                                          # if the data changes enough to make this stale.
        "unfreeze_embeddings": True,                     # LoRA alone can't bootstrap a token from near-nothing
        "learning_rate": 5e-5,
        "batch_size": 32,   # more conservative than mT5's - the unfrozen embedding adds
                           # full-rank gradient memory on top of LoRA's usual small footprint.
                           # Not measured, estimated from the earlier ~12-18GB peak at
                           # batch_size=16; watch actual VRAM on the first run before trusting
                           # this across all NLLB configurations.
        "dataloader_num_workers": 8,
    },
    "mt5": {
        "checkpoint": "google/mt5-small",
        "lora_target_modules": ["q", "v"],               # T5-style attention naming
        "freeze_layers": 4,
        "tgt_code": None,                                 # mT5 uses an instruction prefix - no new token
        "related_code": None,
        "unfreeze_embeddings": False,                     # no new token exists here to unfreeze
        "learning_rate": 3e-4,
        "batch_size": 64,   # raised from 32 for the same reason as NLLB above
        "dataloader_num_workers": 8,
    },
}

MAX_LEN = 128
LORA_KWARGS = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
REPLAY_FRACTION = 0.25   # informational here - the actual replay split already happened
                         # when stage2.jsonl was built; see data_io.derive_psa_only()
SEED = 42

DATA_DIR = "data"           # where stage1.jsonl etc. already live - see data_io.py
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"
DEPLOYMENT_DIR = "deployment"
MLFLOW_DB = "mlflow.db"      # sqlite tracking store, see train.py's
                             # _ensure_isolated_mlflow_tracking(). Not a plain "file:./mlruns"
                             # URI - the installed MLflow puts that backend behind a
                             # maintenance-mode guard (MlflowException unless
                             # MLFLOW_ALLOW_FILE_STORE=true is set) and recommends a database
                             # backend instead; sqlite needs no server and no extra
                             # dependency beyond mlflow itself (verified). Forced to an
                             # absolute path under THIS project root every run, regardless
                             # of any ambient MLFLOW_TRACKING_URI, so this project's runs can
                             # never land in the same store as the earlier, independent
                             # project's runs. Run artifacts (e.g. the training log CSV) still
                             # default to a local mlruns/ folder alongside this db file -
                             # that part of MLflow's layout didn't change.

# Identifies every MLflow experiment this project creates as belonging to THIS
# independent training run - e.g. "final-training-nllb-stage1", never
# "psa-translation-nllb-stage1" (a separate, earlier project that used different
# data and must not be mixed with this one's runs, checkpoints, or reported
# metrics). Keeping this as one config value rather than a string baked into
# train.py means renaming the whole project later is a one-line change, not a
# find-and-replace across every module.
EXPERIMENT_PREFIX = "final-training"
