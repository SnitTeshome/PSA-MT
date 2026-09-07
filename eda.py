"""
eda.py
======
Exploratory data analysis helpers, called identically from both training
notebooks. The numbers this prints (domain composition, sentence-length
distributions, vocabulary size) are exactly what a dataset datasheet needs to
report - keeping this in one shared module means the datasheet always reflects
the same computation both notebooks actually saw, not two independently
re-derived versions that could disagree.
"""
import pandas as pd
import matplotlib.pyplot as plt


def eda_report(df, name="dataset", show_plots=True):
    print(f"=== EDA: {name} ===")
    print(f"rows: {len(df):,}")

    summary = {"n_rows": len(df)}

    if "PSA_non_PSA" in df.columns:
        counts = df["PSA_non_PSA"].value_counts()
        print("\nPSA / Non-PSA composition:")
        print(counts.to_string())
        summary["psa_counts"] = counts.to_dict()

    if "dataset_origin" in df.columns:
        n_origin = df["dataset_origin"].nunique()
        print(f"\ndistinct source files (dataset_origin): {n_origin}")
        summary["n_source_files"] = n_origin

    en_lens = df["English"].dropna().astype(str).str.split().str.len() if "English" in df.columns else None
    guz_lens = df["Ekegusii"].dropna().astype(str).str.split().str.len() if "Ekegusii" in df.columns else None

    if en_lens is not None and guz_lens is not None and len(en_lens) and len(guz_lens):
        print(f"\nEnglish length  - mean {en_lens.mean():.1f}, median {en_lens.median():.0f}, "
              f"max {en_lens.max():.0f}")
        print(f"Ekegusii length - mean {guz_lens.mean():.1f}, median {guz_lens.median():.0f}, "
              f"max {guz_lens.max():.0f}")
        summary["en_len_mean"] = float(en_lens.mean())
        summary["guz_len_mean"] = float(guz_lens.mean())

        if show_plots:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            axes[0].hist(en_lens, bins=40, color="#378ADD")
            axes[0].set_title(f"{name}: English length (words)")
            axes[1].hist(guz_lens, bins=40, color="#1D9E75")
            axes[1].set_title(f"{name}: Ekegusii length (words)")
            plt.tight_layout()
            plt.show()

    if "Ekegusii" in df.columns:
        vocab = set()
        for s in df["Ekegusii"].dropna().astype(str):
            vocab.update(s.lower().split())
        print(f"\nEkegusii vocabulary size (whitespace tokens): {len(vocab):,}")
        summary["ekegusii_vocab_size"] = len(vocab)

    return summary


def curriculum_size_report(curriculum, psa_only_rows=None):
    print("=== Curriculum sizes (direction-expanded) ===")
    for name, rows in curriculum.items():
        print(f"  {name:20s} {len(rows):>8,} examples")
    if psa_only_rows is not None:
        print(f"  {'psa_only':20s} {len(psa_only_rows):>8,} examples  "
              f"(derived from stage2, replay rows excluded - no new file)")
