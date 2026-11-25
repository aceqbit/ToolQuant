from toolquant import tracking


def test_standard_params_contains_required_keys():
    p = tracking.standard_params(model="m", revision="r", seed=1)
    for key in ("model", "revision", "seed", "git_sha"):
        assert key in p
    assert any(k.startswith("pkg.") for k in p)


def test_optional_params_omitted_when_none():
    p = tracking.standard_params(model="m", revision="r", seed=1, budget=0.5)
    assert p["budget"] == 0.5
    assert "granularity" not in p
