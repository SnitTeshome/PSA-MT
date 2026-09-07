"""
data_io.py
==========
Loads the ALREADY-SPLIT, already-built curriculum files produced once by the
data-prep notebook (00_data_prep.ipynb). This module deliberately contains NO
merging or splitting logic. The train/validation/test partition (test built as
three independent sets - PSA / Bible / general Ekegusii, per this project's own
design) was built once with a fixed seed and leakage checks already verified;
re-running that merge+split inside every training notebook would risk a subtly
different split each run and defeat the point of having one canonical,
reproducible split. Every function here only reads what already exists.
"""
import json
from pathlib import Path
import pandas as pd


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_curriculum(data_dir):
    """
    Loads the direction-expanded training/eval files built by the data-prep
    notebook. Returns a dict of lists-of-dicts, keyed by name.

    Expects these files to already exist in data_dir:
        stage1.jsonl, stage2.jsonl, mixed.jsonl,
        validation_general.jsonl, validation_psa.jsonl, test.jsonl

    Note: "test.jsonl" here is the union of test_psa/test_bible/test_general
    (each row's "corpus" field says which). Load those three individually via
    load_jsonl() directly if you need them scored separately - see
    02_nllb_training.ipynb's Section 3 for exactly that.
    """
    data_dir = Path(data_dir)
    names = ["stage1", "stage2", "mixed", "validation_general", "validation_psa", "test"]
    curriculum = {}
    for name in names:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. This module only LOADS the already-built curriculum "
                f"files - run the data-prep notebook first if they don't exist yet, rather "
                f"than adding merge/split logic here."
            )
        curriculum[name] = load_jsonl(path)
    return curriculum


def derive_psa_only(stage2_rows):
    """
    Derives the PSA-only training set directly from stage2's own rows, by
    dropping every row tagged corpus == "replay". stage2 already contains
    exactly (upsampled PSA rows, tagged "psa") + (a replay slice of stage1,
    tagged "replay") - filtering out the replay tag reconstructs the pure
    PSA-only set with the same PSA volume/composition stage2 and mixed use,
    with zero new data generation, merging, or re-splitting required.
    """
    psa_only = [r for r in stage2_rows if r.get("corpus") != "replay"]
    n_replay_dropped = len(stage2_rows) - len(psa_only)
    print(f"derive_psa_only: kept {len(psa_only):,} psa-tagged rows, "
          f"dropped {n_replay_dropped:,} replay-tagged rows "
          f"({100 * n_replay_dropped / len(stage2_rows):.0f}% of stage2)")
    print("  cross-check: this count should exactly match the 'psa_up' figure "
          "printed when stage2.jsonl was originally built - if it doesn't, the "
          "file has likely changed since then.")
    return psa_only


def load_flat_splits(data_dir):
    """
    Loads the row-level (pre-direction-expansion) CSVs - these are what the
    EDA section reads, since domain/PSA composition and sentence-length stats
    are clearer at the row level than after English/Kiswahili have each been
    expanded into separate directional records.

    "test" here is the combined test_psa+test_bible+test_general union: load
    those three individually with pandas.read_csv() if you need them separately.
    """
    data_dir = Path(data_dir)
    return {
        "train": pd.read_csv(data_dir / "train.csv"),
        "validation_general": pd.read_csv(data_dir / "validation_general.csv"),
        "validation_psa": pd.read_csv(data_dir / "validation_psa.csv"),
        "test": pd.read_csv(data_dir / "test.csv"),
    }
