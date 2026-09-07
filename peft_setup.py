"""
peft_setup.py
=============
LoRA wrapping, including the verified-correct approach to unfreezing NLLB's new
guz_Latn embedding. This is not a naive implementation - it replaced a subtly
broken first attempt, and the reasoning below matters if this module is ever
"simplified" without re-testing.

CRITICAL, VERIFIED FINDINGS:

1. LoraConfig(modules_to_save=["shared"]) looks like the textbook way to also
   train the embedding alongside LoRA adapters. On NLLB's tied-embedding
   architecture, it silently breaks: encoder.embed_tokens / decoder.embed_tokens
   / lm_head stop being the same tensor as "shared" the moment get_peft_model()
   wraps it - BEFORE any training happens. Confirmed with a real gradient step:
   lm_head.weight.requires_grad stays False and lm_head never changes, meaning
   the model would learn to encode the new language but never to generate it.
   This does not raise an error - it silently trains the wrong thing.

2. The correct fix: skip modules_to_save for the embedding entirely, and call
   model.get_input_embeddings().weight.requires_grad_(True) directly, AFTER
   get_peft_model(). Because encoder.embed_tokens / decoder.embed_tokens /
   lm_head are the literal same Parameter object before any peft wrapping, and
   this bypasses peft's tracking for that one piece, all three stay genuinely
   tied throughout training. Confirmed: lm_head DOES change after a real
   training step this way, and the tie survives both merge_and_unload() and a
   full save/reload from disk.

3. One consequence: the adapter-only save (trainer.model.save_pretrained(...))
   no longer captures the embedding update, since it was never registered with
   peft's tracking. Only the MERGED checkpoint is a complete artifact when
   embeddings are unfrozen - train.py's train_stage() prints this explicitly
   rather than leaving it as a silent gap.

4. resize_model_for_new_language() must be called on EVERY fresh load of the
   original base checkpoint (Stage 1, Mixed), but must NOT re-run its copy-init
   step on a checkpoint that already has the extended vocabulary (Stage 2,
   loading Stage 1's merged output) - doing so would silently overwrite
   whatever Stage 1 learned for guz_Latn with a fresh copy of kik_Latn. The
   size check below is what prevents that.
"""
import torch


def freeze_encoder_layers(model, num_layers_to_freeze):
    """Fallback path when use_lora=False - not the primary technique for this project."""
    encoder = model.get_encoder()
    layers = encoder.block if hasattr(encoder, "block") else encoder.layers
    for i, layer in enumerate(layers):
        if i < num_layers_to_freeze:
            for p in layer.parameters():
                p.requires_grad = False
    return model


def extend_tokenizer_for_new_language(tokenizer, cfg):
    """NLLB only: add the new guz_Latn token to the tokenizer, if not already present.
    No-op for mT5 (cfg["tgt_code"] is None there - no new token needed).

    Tolerates two different transformers APIs, confirmed by actually loading both:
    transformers <5.0's NllbTokenizer tracked language codes in `additional_special_tokens`
    (a real list attribute) plus separate `lang_code_to_id`/`id_to_lang_code` dicts that
    src_lang/tgt_lang lookups depended on; transformers >=5.0 dropped all three attributes
    (confirmed via `dir(tokenizer)` - replaced by `all_special_tokens`/`_extra_special_tokens`
    internals) and resolves src_lang/tgt_lang purely through the ordinary added-special-token
    vocabulary instead. Empirically verified: under >=5.0, `add_special_tokens(...)` alone
    is sufficient - src_lang/tgt_lang-based encoding produces the correct prefix token with
    no dict population at all. Written to use whichever attributes actually exist rather
    than assuming one API, so this keeps working if the pinned transformers version changes.
    """
    new_code = cfg["tgt_code"]
    if new_code is None:
        return tokenizer
    if new_code in tokenizer.all_special_tokens:
        print(f"{new_code!r} already present in tokenizer - skipping token addition")
        return tokenizer

    try:
        existing = tokenizer.additional_special_tokens  # transformers <5.0
    except AttributeError:
        existing = []  # transformers >=5.0: no such attribute - add_special_tokens merges
                       # into the existing special-token set rather than replacing it, so
                       # passing just the new code (not the old list) is safe here.
    tokenizer.add_special_tokens({"additional_special_tokens": existing + [new_code]})
    new_id = tokenizer.convert_tokens_to_ids(new_code)

    if hasattr(tokenizer, "lang_code_to_id"):   # transformers <5.0 only - see docstring
        tokenizer.lang_code_to_id[new_code] = new_id
        tokenizer.id_to_lang_code[new_id] = new_code

    print(f"Added new token {new_code!r} at id {new_id}")
    return tokenizer


def resize_model_for_new_language(model, tokenizer, cfg):
    """See point 4 in the module docstring - this guard is load-bearing, not defensive fluff."""
    if cfg["tgt_code"] is None:
        return model  # mT5: nothing to resize

    current_size = model.get_input_embeddings().weight.shape[0]
    target_size = len(tokenizer)
    if current_size == target_size:
        print(f"embedding matrix already sized for the extended vocab ({current_size}) "
              f"- assuming {cfg['tgt_code']!r} is already initialized from a prior stage, "
              f"not re-copying")
        return model

    model.resize_token_embeddings(target_size)
    new_id = tokenizer.convert_tokens_to_ids(cfg["tgt_code"])
    related_id = tokenizer.convert_tokens_to_ids(cfg["related_code"])
    with torch.no_grad():
        model.get_input_embeddings().weight[new_id] = \
            model.get_input_embeddings().weight[related_id].clone()
    print(f"resized embeddings {current_size} -> {target_size} and initialized "
          f"{cfg['tgt_code']!r} from {cfg['related_code']!r}")
    return model


def wrap_for_training(model, tokenizer, cfg, lora_kwargs, use_lora=True):
    """
    Applies resize (if NLLB), then either LoRA (+ direct embedding unfreeze, if
    configured) or the layer-freezing fallback. See points 1-2 above for why
    the embedding unfreeze is done the way it is, not via modules_to_save.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    model = resize_model_for_new_language(model, tokenizer, cfg)

    if not use_lora:
        return freeze_encoder_layers(model, cfg["freeze_layers"])

    lora_config = LoraConfig(
        target_modules=cfg["lora_target_modules"], bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM, **lora_kwargs,
        # Deliberately NOT using modules_to_save for the embedding - see point 1 above.
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()   # required if gradient checkpointing is ever turned on

    if cfg["unfreeze_embeddings"]:
        # Direct unfreeze of the ORIGINAL tied parameter, bypassing peft's tracking
        # for this piece entirely - see point 2 above for why this is correct and
        # modules_to_save is not.
        model.get_input_embeddings().weight.requires_grad_(True)

    model.print_trainable_parameters()   # validation check - expect a large jump for NLLB
                                          # once the embedding is included; that's expected,
                                          # not a sign LoRA "stopped being efficient"
    return model
