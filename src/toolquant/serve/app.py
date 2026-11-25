"""FastAPI server: ONNX Runtime GenAI model, one tool-call endpoint, health check.

Environment: MODEL_DIR (ONNX dir), TOKENIZER_DIR (HF dir), MAP_ID, ORT_INTRA_OP_THREADS,
ORT_GRAPH_OPT_LEVEL, OTEL_EXPORTER_OTLP_ENDPOINT.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from toolquant.scoring.match import schema_valid
from toolquant.scoring.parse import parse_tool_calls
from toolquant.serve import otel

HEALTH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]
HEALTH_MESSAGES = [{"role": "user", "content": "What is the weather in Paris right now?"}]


class GenerateRequest(BaseModel):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_new_tokens: int = 256


class Engine:
    def __init__(self, model_dir: str, tokenizer_dir: str, map_id: str) -> None:
        import onnxruntime_genai as og
        from transformers import AutoTokenizer

        self.og = og
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        cfg = og.Config(model_dir)
        opts: dict[str, Any] = {}
        if os.environ.get("ORT_INTRA_OP_THREADS"):
            opts["intra_op_num_threads"] = int(os.environ["ORT_INTRA_OP_THREADS"])
        if os.environ.get("ORT_GRAPH_OPT_LEVEL"):
            opts["graph_optimization_level"] = os.environ["ORT_GRAPH_OPT_LEVEL"]
        if opts:
            cfg.overlay(json.dumps({"model": {"decoder": {"session_options": opts}}}))
        self.model = og.Model(cfg)
        self.map_id = map_id

    def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_new: int
    ) -> dict[str, Any]:
        text = self.tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = self.tok(text).input_ids
        params = self.og.GeneratorParams(self.model)
        params.set_search_options(do_sample=False, max_length=len(ids) + max_new)
        t0 = time.perf_counter()
        gen = self.og.Generator(self.model, params)
        gen.append_tokens(ids)
        out: list[int] = []
        ttft = 0.0
        while not gen.is_done() and len(out) < max_new:
            gen.generate_next_token()
            out.append(int(gen.get_next_tokens()[0]))
            if len(out) == 1:
                ttft = (time.perf_counter() - t0) * 1000
        return {
            "text": self.tok.decode(out, skip_special_tokens=True),
            "prompt_tokens": len(ids),
            "output_tokens": len(out),
            "ttft_ms": ttft,
        }


def create_app(engine: Engine | None = None) -> FastAPI:
    tracer = otel.setup()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or Engine(
            os.environ["MODEL_DIR"],
            os.environ.get("TOKENIZER_DIR", "data/cache/qwen3-0.6b-hf"),
            os.environ.get("MAP_ID", "unknown"),
        )
        yield

    app = FastAPI(title="toolquant", lifespan=lifespan)

    def run(messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_new: int) -> dict:
        eng: Engine = app.state.engine
        with tracer.start_as_current_span("generate") as span:
            r = eng.generate(messages, tools, max_new)
            span.set_attribute("prompt_tokens", r["prompt_tokens"])
            span.set_attribute("output_tokens", r["output_tokens"])
            span.set_attribute("ttft_ms", r["ttft_ms"])
            span.set_attribute("map_id", eng.map_id)
        return r

    @app.post("/v1/generate")
    def generate(req: GenerateRequest) -> dict[str, Any]:
        r = run(req.messages, req.tools, req.max_new_tokens)
        parsed = parse_tool_calls(r["text"])
        return {"tool_calls": parsed.calls, "malformed": parsed.malformed, "raw_text": r["text"]}

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        r = run(HEALTH_MESSAGES, HEALTH_TOOLS, 128)
        p = parse_tool_calls(r["text"])
        ok = bool(p.calls) and all(schema_valid(c, HEALTH_TOOLS) for c in p.calls)
        if not ok:
            raise HTTPException(status_code=503, detail={"raw_text": r["text"]})
        return {"status": "ok", "call": p.calls[0]}

    return app


def app_factory() -> FastAPI:  # uvicorn --factory toolquant.serve.app:app_factory
    return create_app()
