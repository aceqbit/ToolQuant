import json
import socket
import threading
import time
import urllib.error
import urllib.request

import uvicorn

from toolquant.serve import otel
from toolquant.serve.app import create_app

CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'


class StubEngine:
    map_id = "stub-map"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, messages, tools, max_new):
        return {"text": self.text, "prompt_tokens": 11, "output_tokens": 7, "ttft_ms": 3.5}


def serve(engine):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(
        uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="error")
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}"


def test_generate_healthz_and_span():
    server, base = serve(StubEngine(CALL))
    try:
        body = json.dumps(
            {"messages": [{"role": "user", "content": "weather?"}], "tools": []}
        ).encode()
        req = urllib.request.Request(
            base + "/v1/generate", body, {"Content-Type": "application/json"}
        )
        r = json.load(urllib.request.urlopen(req))
        assert r["tool_calls"] == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
        assert r["raw_text"] == CALL
        h = json.load(urllib.request.urlopen(base + "/healthz"))
        assert h["status"] == "ok"
        spans = [s for s in otel.finished_spans() if s.name == "generate"]
        assert spans
        a = spans[-1].attributes
        assert a["prompt_tokens"] == 11 and a["output_tokens"] == 7
        assert a["ttft_ms"] == 3.5 and a["map_id"] == "stub-map"
    finally:
        server.should_exit = True


def test_healthz_fails_without_valid_call():
    server, base = serve(StubEngine("I cannot help with that."))
    try:
        try:
            urllib.request.urlopen(base + "/healthz")
            raise AssertionError("expected 503")
        except urllib.error.HTTPError as e:
            assert e.code == 503
    finally:
        server.should_exit = True
