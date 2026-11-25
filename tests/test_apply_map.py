import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from toolquant.export.apply_map import apply_map, group_weight_params, inspect, node_groups

NAMES = [
    "/model/layers.0/attn/qkv_proj/MatMul",
    "/model/layers.0/attn/o_proj/MatMul",
    "/model/layers.0/mlp/down_proj/MatMul",
    "/model/layers.1/attn/qkv_proj/MatMul",
    "/model/layers.1/mlp/down_proj/MatMul",
    "/lm_head/MatMul",
]


def toy_model(tmp_path):
    rng = np.random.default_rng(0)
    nodes, inits = [], []
    prev = "x"
    for i, name in enumerate(NAMES):
        w = f"w{i}"
        inits.append(numpy_helper.from_array(rng.standard_normal((128, 128)).astype(np.float32), w))
        out = f"y{i}"
        nodes.append(helper.make_node("MatMul", [prev, w], [out], name=name))
        prev = out
    g = helper.make_graph(
        nodes,
        "toy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 128])],
        [helper.make_tensor_value_info(prev, TensorProto.FLOAT, [1, 128])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    d = tmp_path / "fp"
    d.mkdir()
    onnx.save(m, str(d / "model.onnx"))
    (d / "genai_config.json").write_text("{}")
    return d


def test_node_groups_block_and_attn_mlp(tmp_path):
    m = onnx.load(str(toy_model(tmp_path) / "model.onnx"))
    assert list(node_groups(m, "block")) == ["layer00", "layer01"]
    assert len(node_groups(m, "attn_mlp")) == 4
    assert not any("lm_head" in n for v in node_groups(m).values() for n in v)
    assert group_weight_params(m, node_groups(m))["layer00"] == 3 * 128 * 128


@pytest.mark.parametrize(
    "bits",
    [{"layer00": 4, "layer01": 8}, {"layer00": 4, "layer01": 4}, {"layer00": 8, "layer01": 8}],
)
def test_apply_map_sets_requested_precision(tmp_path, bits):
    fp = toy_model(tmp_path)
    out = apply_map(bits, tmp_path / "out", fp_dir=fp)
    info = inspect(out)
    for g, ops in info["per_group"].items():
        assert ops == [{"op": "MatMulNBits", "bits": bits[g]}]
    m = onnx.load(str(out / "model.onnx"), load_external_data=False)
    assert [n.op_type for n in m.graph.node if "lm_head" in n.name] == ["MatMul"]


def test_two_groups_differ_in_one_model(tmp_path):
    out = apply_map({"layer00": 4, "layer01": 8}, tmp_path / "o", fp_dir=toy_model(tmp_path))
    per = inspect(out)["per_group"]
    assert per["layer00"] != per["layer01"]


def test_map_group_mismatch_raises(tmp_path):
    with pytest.raises(ValueError):
        apply_map({"layer00": 4}, tmp_path / "o", fp_dir=toy_model(tmp_path))
