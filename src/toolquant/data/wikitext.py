"""Fixed WikiText-2 slice for the perplexity probe: 64 sequences of 512 tokens."""

from __future__ import annotations

import math
from typing import Any

N_SEQ = 64
SEQ_LEN = 512


def load_slice(tok: Any, n_seq: int = N_SEQ, seq_len: int = SEQ_LEN) -> Any:
    """First n_seq*seq_len tokens of the WikiText-2 (raw) test split, shaped (n_seq, seq_len)."""
    import torch
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = n_seq * seq_len
    if ids.numel() < need:
        raise ValueError("WikiText-2 slice too short")
    return ids[:need].view(n_seq, seq_len).to(torch.long)


def perplexity(model: Any, ids: Any, batch_size: int = 8) -> float:
    import torch

    device = next(model.parameters()).device
    nll, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], batch_size):
            x = ids[i : i + batch_size].to(device)
            logits = model(x).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum"
            )
            nll += float(loss)
            count += x[:, 1:].numel()
    return math.exp(nll / count)
