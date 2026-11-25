"""Batched greedy generation with the chat template and tools."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np

from toolquant.data.bfcl import Task

MAX_NEW_TOKENS = 256
SEED = 0


def seed_everything(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_prompt(tok: Any, task: Task) -> str:
    return tok.apply_chat_template(
        task.messages,
        tools=task.tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def load_hf(path: str, dtype: str = "float32", device: str = "cpu") -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype)).eval()
    return model.to(device), tok


def generate_onnx(
    model_dir: str,
    tok: Any,
    tasks: list[Task],
    max_new_tokens: int = MAX_NEW_TOKENS,
    progress: Callable[[int, int], None] | None = None,
    return_ids: bool = False,
) -> dict[str, Any]:
    """Greedy decoding with onnxruntime-genai (one prompt at a time). Stops at end-of-turn."""
    import onnxruntime_genai as og

    model = og.Model(str(model_dir))
    out: dict[str, Any] = {}
    for i, t in enumerate(tasks):
        ids = tok(build_prompt(tok, t), return_tensors="pt").input_ids[0].tolist()
        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False, max_length=len(ids) + max_new_tokens, batch_size=1
        )
        gen = og.Generator(model, params)
        gen.append_tokens(ids)
        new: list[int] = []
        while not gen.is_done() and len(new) < max_new_tokens:
            gen.generate_next_token()
            new.append(int(gen.get_next_tokens()[0]))
        out[t.id] = new if return_ids else tok.decode(new, skip_special_tokens=True)
        if progress:
            progress(i + 1, len(tasks))
    return out


def generate_hf(
    model: Any,
    tok: Any,
    tasks: list[Task],
    batch_size: int = 8,
    max_new_tokens: int = MAX_NEW_TOKENS,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, str]:
    """Greedy decoding. Prompts are length-sorted into batches (left padded); output per id."""
    import torch

    seed_everything()
    device = next(model.parameters()).device
    prompts = {t.id: build_prompt(tok, t) for t in tasks}
    order = sorted(tasks, key=lambda t: len(prompts[t.id]))
    out: dict[str, str] = {}
    for i in range(0, len(order), batch_size):
        batch = order[i : i + batch_size]
        enc = tok([prompts[t.id] for t in batch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        for t, row in zip(batch, gen, strict=True):
            out[t.id] = tok.decode(row[enc.input_ids.shape[1] :], skip_special_tokens=True)
        if progress:
            progress(min(i + batch_size, len(order)), len(order))
    return out
