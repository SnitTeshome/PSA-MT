"""
inference.py
============
Generation for the demo section, plus save_for_deployment() - the step that
bundles a merged checkpoint into a clearly labeled, self-contained folder
(model + tokenizer, verified safetensors) so the application has exactly one
path to load from.
"""
import json
import shutil
from pathlib import Path


def nllb_src_code(source_lang):
    return "eng_Latn" if source_lang == "en" else "swh_Latn"


def mt5_prefix(source_lang):
    return "translate English to Ekegusii: " if source_lang == "en" else "translate Kiswahili to Ekegusii: "


def generate_one(model, tokenizer, text, source_lang, cfg, architecture, max_len):
    # no_repeat_ngram_size/repetition_penalty: plain greedy decoding (num_beams=1, the
    # default) has no way to escape a repetition loop once "repeat the last n-gram"
    # becomes the highest-probability next token - it just runs to max_length and gets
    # cut off mid-word. Confirmed this actually happens on a weak checkpoint x hard-input
    # combination (mixed_lora_emb on a Kiswahili-source PSA sentence): 30+ repeats of the
    # same word, truncated mid-token, no EOS. Both params are standard, low-risk guards -
    # they only suppress degenerate repetition, a genuinely correct translation essentially
    # never needs to repeat the same 3-gram.
    if architecture == "nllb":
        tokenizer.src_lang = nllb_src_code(source_lang)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        forced_bos = tokenizer.convert_tokens_to_ids(cfg["tgt_code"])
        out = model.generate(**enc, forced_bos_token_id=forced_bos, max_length=max_len,
                             no_repeat_ngram_size=3, repetition_penalty=1.2)
    else:
        enc = tokenizer(mt5_prefix(source_lang) + text, return_tensors="pt", truncation=True,
                        max_length=max_len).to(model.device)
        out = model.generate(**enc, max_length=max_len,
                             no_repeat_ngram_size=3, repetition_penalty=1.2)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def save_for_deployment(model_path, architecture, run_id, deployment_dir="deployment"):
    """
    Copies a merged checkpoint into deployment/{architecture}/{run_id}/ and
    verifies it actually contains a .safetensors file before calling it done -
    a missing safetensors file here means save_safetensors=True didn't take
    effect somewhere upstream, and the app would silently get a checkpoint it
    can't use.
    """
    src = Path(model_path)
    dst = Path(deployment_dir) / architecture / run_id
    dst.mkdir(parents=True, exist_ok=True)

    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, dst / f.name)

    safetensor_files = list(dst.glob("*.safetensors"))
    if not safetensor_files:
        print(f"WARNING: no .safetensors file found in {dst.resolve()} - check "
              f"save_safetensors=True was set during training before trusting this "
              f"checkpoint for deployment.")
    else:
        print(f"deployment-ready: {dst.resolve()}  ({len(safetensor_files)} safetensors file(s))")

    manifest = {"architecture": architecture, "run_id": run_id, "path": str(dst.resolve())}
    with open(dst / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return dst
