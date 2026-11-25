"""Spike: inspect mixed-precision ONNX and run it with onnxruntime-genai."""
import collections, sys
from pathlib import Path
import onnx
import onnxruntime_genai as og

d = Path(sys.argv[1])
m = onnx.load(str(d / "model.onnx"), load_external_data=False)
c = collections.Counter()
for n in m.graph.node:
    if n.op_type in ("MatMul", "MatMulNBits"):
        bits = next((a.i for a in n.attribute if a.name == "bits"), 0)
        grp = "layers.0" if "/layers.0/" in n.name else "other"
        c[(grp, n.op_type, bits)] += 1
for k, v in sorted(c.items()):
    print(k, v)
for p in Path("data/cache/qwen3-0.6b-fp32").glob("*"):
    if p.suffix in {".json", ".jinja"} and not (d / p.name).exists():
        import shutil; shutil.copy(p, d / p.name)
model = og.Model(str(d))
tok = og.Tokenizer(model)
params = og.GeneratorParams(model)
params.set_search_options(do_sample=False, max_length=40)
g = og.Generator(model, params)
g.append_tokens(tok.encode("The capital of France is"))
out = []
while not g.is_done():
    g.generate_next_token(); out.append(int(g.get_next_tokens()[0]))
print("generated:", tok.decode(out))
