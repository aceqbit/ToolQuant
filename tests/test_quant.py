import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from toolquant.quant.fakequant import apply_map, fake_quantize_tensor, quantized
from toolquant.quant.groups import derive_groups, group_param_counts, other_param_count
from toolquant.quant.sizing import budget, group_bits, to_mb, total_bits, uniform_bits


def tiny() -> Qwen3ForCausalLM:
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


# ---- groups
@pytest.mark.parametrize("gran,expected", [("block", 3), ("attn_mlp", 6)])
def test_group_count(gran, expected):
    assert len(derive_groups(tiny(), gran)) == expected


@pytest.mark.parametrize("gran", ["block", "attn_mlp"])
def test_every_decoder_linear_in_exactly_one_group(gran):
    m = tiny()
    groups = derive_groups(m, gran)
    members = [n for names in groups.values() for n in names]
    assert len(members) == len(set(members))
    decoder = [
        n
        for n, mod in m.named_modules()
        if isinstance(mod, torch.nn.Linear) and ".layers." in f".{n}"
    ]
    assert sorted(members) == sorted(decoder)
    assert not any("lm_head" in n or "embed" in n for n in members)
    assert len(decoder) == 3 * 7


def test_unknown_granularity():
    with pytest.raises(ValueError):
        derive_groups(tiny(), "nope")


def test_param_accounting_sums_to_model():
    m = tiny()
    g = derive_groups(m)
    total = sum(group_param_counts(m, g).values()) + other_param_count(m, g)
    assert total == sum(p.numel() for p in m.parameters())


# ---- fake quantization
def test_8bit_error_smaller_than_4bit():
    w = torch.randn(64, 256)
    e4 = (fake_quantize_tensor(w, 4) - w).abs().mean()
    e8 = (fake_quantize_tensor(w, 8) - w).abs().mean()
    assert e8 < e4


@pytest.mark.parametrize("bits", [4, 8])
def test_values_on_grid(bits):
    w = torch.randn(8, 256)
    wq = fake_quantize_tensor(w, bits, 128)
    qmax = 2 ** (bits - 1) - 1
    for start in (0, 128):
        g, gq = w[:, start : start + 128], wq[:, start : start + 128]
        scale = g.abs().amax(dim=1, keepdim=True) / qmax
        q = gq / scale
        assert torch.allclose(q, q.round(), atol=1e-4)
        assert q.max() <= qmax + 1e-4 and q.min() >= -(qmax + 1) - 1e-4


def test_shape_and_dtype_preserved():
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        w = torch.randn(16, 300).to(dt)  # 300 is not a multiple of 128
        wq = fake_quantize_tensor(w, 4)
        assert wq.shape == w.shape and wq.dtype == dt


def test_identical_values_stay_exact():
    w = torch.full((4, 128), 0.37)
    assert torch.allclose(fake_quantize_tensor(w, 4), w, atol=1e-6)
    assert torch.equal(fake_quantize_tensor(torch.zeros(4, 128), 4), torch.zeros(4, 128))


def test_bad_bits():
    with pytest.raises(ValueError):
        fake_quantize_tensor(torch.randn(2, 128), 3)


# ---- restore
def test_weights_restored_and_output_back_to_baseline():
    m = tiny()
    ids = torch.randint(0, 128, (2, 10))
    before = m(ids).logits.clone()
    groups = derive_groups(m)
    first = next(iter(groups.values()))
    with quantized(m, first, 4):
        during = m(ids).logits
        assert not torch.allclose(before, during)
    assert torch.equal(before, m(ids).logits)


def test_restore_after_exception():
    m = tiny()
    ids = torch.randint(0, 128, (1, 6))
    before = m(ids).logits.clone()
    g = next(iter(derive_groups(m).values()))
    with pytest.raises(RuntimeError), quantized(m, g, 4):
        raise RuntimeError("boom")
    assert torch.equal(before, m(ids).logits)


def test_embedding_and_lm_head_untouched_by_map():
    m = tiny()
    emb = m.model.embed_tokens.weight.clone()
    groups = derive_groups(m)
    with apply_map(m, groups, {g: 4 for g in groups}):
        assert torch.equal(emb, m.model.embed_tokens.weight)
        assert torch.equal(m.lm_head.weight, m.model.embed_tokens.weight)  # tied, unquantized


# ---- sizing
def test_size_model():
    m = tiny()
    g = derive_groups(m)
    gp = group_param_counts(m, g)
    other = other_param_count(m, g)
    s4, s8 = uniform_bits(gp, 4, other), uniform_bits(gp, 8, other)
    assert s4 < s8
    for f in (0.25, 0.5):
        assert s4 < budget(f, gp, other) < s8
    assert budget(0.0, gp, other) == s4 and budget(1.0, gp, other) == s8
    assert group_bits(128, 4) == 128 * (4 + 16 / 128)
    mixed = {name: (4 if i == 0 else 8) for i, name in enumerate(gp)}
    assert s4 < total_bits(gp, mixed, other) < s8
    assert to_mb(8e6) == 1.0
