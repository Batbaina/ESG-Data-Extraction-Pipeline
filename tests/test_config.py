from esg_pipeline.config import Settings


def test_generation_options_are_explicit():
    s = Settings()
    opts = s.generation_options()
    assert set(opts) == {"temperature", "top_p", "top_k", "seed", "num_ctx", "num_predict"}
    assert 0 <= opts["temperature"]
    assert 0 < opts["top_p"] <= 1
