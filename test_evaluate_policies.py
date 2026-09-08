from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.evaluate_policies import PolicySpec, evaluate_spec


def _args(mode):
    return Namespace(
        bosco_mode=mode,
        config=Path("config/mappo_baseline.yaml"),
        episodes=8,
        seed=3,
        max_steps=None,
        progress_every=0,
        batch_size=8,
        chunk_steps=64,
        stochastic=False,
    )


def test_jax_bosco_uses_batched_evaluator():
    spec = PolicySpec("BOSCO JAX", "bosco", 8)
    with patch("src.evaluate_policies.evaluate_marl", return_value=["jax"]) as batched:
        with patch("src.evaluate_policies.evaluate_bosco") as host:
            assert evaluate_spec(spec, _args("jax"), object()) == ["jax"]
    batched.assert_called_once()
    host.assert_not_called()


def test_host_bosco_preserves_reference_evaluator():
    spec = PolicySpec("Plain BOSCO", "bosco", 8)
    with patch("src.evaluate_policies.evaluate_bosco", return_value=["host"]) as host:
        with patch("src.evaluate_policies.evaluate_marl") as batched:
            assert evaluate_spec(spec, _args("host"), object()) == ["host"]
    host.assert_called_once()
    batched.assert_not_called()
