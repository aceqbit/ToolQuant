"""Spike: first 20 greedy tokens, PyTorch vs onnxruntime-genai, on 5 prompts."""

import shutil
import sys
from pathlib import Path

import onnxruntime_genai as og
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF = "data/cache/qwen3-0.6b-hf"
PROMPTS = [
    "What is the weather in Paris today?",
    "Add 17 and 25.",
    "Book a table for two at 7pm.",
    "Translate 'hello' to French.",
    "Get the stock price of AAPL.",
]


def main(onnx_dir: str) -> None:
    d = Path(onnx_dir)
    for f in Path("data/cache/qwen3-0.6b-fp32").glob("*"):
        if f.suffix in {".json", ".jinja"} and not (d / f.name).exists():
            shutil.copy(f, d / f.name)
    tok = AutoTokenizer.from_pretrained(HF)
    pt = AutoModelForCausalLM.from_pretrained(HF, torch_dtype=torch.float32).eval()
    model = og.Model(str(d))
    agree = []
    for p in PROMPTS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            out = pt.generate(ids, max_new_tokens=20, do_sample=False)[0, ids.shape[1] :].tolist()
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=ids.shape[1] + 20)
        gen = og.Generator(model, params)
        gen.append_tokens(ids[0].tolist())
        got = []
        while not gen.is_done() and len(got) < 20:
            gen.generate_next_token()
            got.append(int(gen.get_next_tokens()[0]))
        n = min(len(out), len(got))
        first_div = next((i for i in range(n) if out[i] != got[i]), n)
        agree.append(first_div)
        print(f"{p!r}: match_prefix={first_div}/{n}")
    print("mean matching prefix", sum(agree) / len(agree))


if __name__ == "__main__":
    main(sys.argv[1])
