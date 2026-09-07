"""
metrics.py
==========
chrF2++ (word_order=2) is the PRIMARY metric for this project, not BLEU.
Ekegusii is agglutinative - a single concept can surface as several valid
surface forms through affixation - and BLEU's exact whole-word matching
penalizes this kind of morphological variation even when a translation is
correct. chrF's character-level matching tolerates exactly this variation.
BLEU is still computed and reported alongside it, for a reader used to seeing
it, but no training or model-selection decision is made based on it.
"""
import numpy as np
import evaluate

sacrebleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")


def score(preds, refs, label):
    bleu = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])
    c = chrf.compute(predictions=preds, references=[[r] for r in refs], word_order=2)
    print(f"{label:35s} chrF2++={c['score']:.2f}  BLEU={bleu['score']:.2f}")
    return {"name": label, "chrf2pp": c["score"], "bleu": bleu["score"]}


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu = sacrebleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        c = chrf.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels], word_order=2)
        return {"bleu": bleu["score"], "chrf2pp": c["score"]}
    return compute_metrics
