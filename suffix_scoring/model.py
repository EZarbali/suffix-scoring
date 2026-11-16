"""Model / tokenizer loading and the character -> token-id mapping."""
from __future__ import annotations

import string

import torch

MODEL_ID = "Qwen/Qwen3-0.6B"
START_TOKEN = "<|endoftext|>"   # context token for length-0 prefixes (documented assumption)
LOWER, UPPER, DIGITS = string.ascii_lowercase, string.ascii_uppercase, string.digits
N_SUFFIXES = len(LOWER) * len(UPPER) * len(DIGITS)  # 6760

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_tokenizer(model_id: str = MODEL_ID):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def char_token_ids(tokenizer) -> dict[str, list[int]]:
    """Token id of every bare character (no leading space). Fails if any is not a single token."""
    out = {}
    for name, chars in (("lower", LOWER), ("upper", UPPER), ("digit", DIGITS)):
        ids = []
        for c in chars:
            enc = tokenizer.encode(c, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"character {c!r} is not a single token: {enc}")
            ids.append(enc[0])
        out[name] = ids
    return out


def tiny_char_token_ids() -> dict[str, list[int]]:
    """Fake mapping for the tiny random test model (no download needed)."""
    return {"lower": list(range(10, 36)), "upper": list(range(36, 62)), "digit": list(range(62, 72))}


def load_model(model_id: str = MODEL_ID, dtype: str | torch.dtype = "bf16", device: str = "cuda",
               attn_implementation: str = "sdpa"):
    from transformers import AutoModelForCausalLM
    dtype = DTYPES.get(dtype, dtype)
    try:  # transformers >= 4.56
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype,
                                                     attn_implementation=attn_implementation)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                     attn_implementation=attn_implementation)
    return model.to(device).eval()


def load_tiny_model(dtype: str | torch.dtype = "fp32", device: str = "cpu", seed: int = 0):
    """Randomly initialised Qwen3 with the same architecture but tiny sizes (CPU tests)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    dtype = DTYPES.get(dtype, dtype)
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=4096, tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg).to(dtype=dtype, device=device).eval()
