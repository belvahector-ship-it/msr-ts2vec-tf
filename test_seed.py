"""
tests/test_utils/test_seed.py
==============================
Unit tests for src.utils.seed.

Covers:
- Reproducibility: same seed → same random outputs across all sources
- Distinctness: different seeds → different random outputs
- Input validation: negative / non-integer seeds raise ValueError
- Non-standard seed warning is emitted
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.utils.seed import VALID_SEEDS, get_torch_generator, set_all_seeds


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same seed must produce identical random outputs from all sources."""

    def _sample_python(self) -> float:
        return random.random()

    def _sample_numpy(self) -> float:
        return float(np.random.rand())

    def _sample_torch(self) -> float:
        return float(torch.rand(1).item())

    def test_python_random_reproducible(self) -> None:
        set_all_seeds(42)
        a = self._sample_python()
        set_all_seeds(42)
        b = self._sample_python()
        assert a == b, "Python random not reproducible with same seed."

    def test_numpy_reproducible(self) -> None:
        set_all_seeds(42)
        a = self._sample_numpy()
        set_all_seeds(42)
        b = self._sample_numpy()
        assert a == b, "NumPy random not reproducible with same seed."

    def test_torch_reproducible(self) -> None:
        set_all_seeds(42)
        a = self._sample_torch()
        set_all_seeds(42)
        b = self._sample_torch()
        assert a == b, "PyTorch random not reproducible with same seed."

    def test_all_sources_reproducible_in_sequence(self) -> None:
        """Verify all three sources together in a single call sequence."""
        set_all_seeds(123)
        py1 = self._sample_python()
        np1 = self._sample_numpy()
        t1  = self._sample_torch()

        set_all_seeds(123)
        py2 = self._sample_python()
        np2 = self._sample_numpy()
        t2  = self._sample_torch()

        assert py1 == py2
        assert np1 == np2
        assert t1 == t2


# ---------------------------------------------------------------------------
# Distinctness
# ---------------------------------------------------------------------------


class TestDistinctness:
    """Different seeds must produce different outputs."""

    def test_different_seeds_produce_different_torch_outputs(self) -> None:
        set_all_seeds(42)
        a = torch.rand(100)
        set_all_seeds(123)
        b = torch.rand(100)
        # Probability of collision is astronomically low
        assert not torch.equal(a, b), "Different seeds produced identical tensors."

    def test_different_seeds_produce_different_numpy_outputs(self) -> None:
        set_all_seeds(42)
        a = np.random.rand(100)
        set_all_seeds(123)
        b = np.random.rand(100)
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_raises_on_negative_seed(self) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            set_all_seeds(-1)

    def test_raises_on_float_seed(self) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            set_all_seeds(3.14)  # type: ignore[arg-type]

    def test_raises_on_string_seed(self) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            set_all_seeds("42")  # type: ignore[arg-type]

    def test_zero_seed_is_valid(self) -> None:
        # Zero is a valid non-negative integer
        set_all_seeds(0, warn_if_nonstandard=False)


# ---------------------------------------------------------------------------
# Non-standard seed warning
# ---------------------------------------------------------------------------


class TestNonStandardSeedWarning:
    def test_warns_on_nonstandard_seed(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="src.utils.seed"):
            set_all_seeds(9999, warn_if_nonstandard=True)
        assert any("9999" in r.message for r in caplog.records)

    def test_no_warning_for_valid_seeds(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="src.utils.seed"):
            set_all_seeds(42, warn_if_nonstandard=True)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 0

    def test_valid_seeds_constant(self) -> None:
        """Confirm VALID_SEEDS matches the protocol definition (ADR-019)."""
        assert VALID_SEEDS == (42, 123, 456, 789, 1024)


# ---------------------------------------------------------------------------
# get_torch_generator
# ---------------------------------------------------------------------------


class TestGetTorchGenerator:
    def test_generator_is_reproducible(self) -> None:
        gen1 = get_torch_generator(42)
        gen2 = get_torch_generator(42)
        a = torch.rand(10, generator=gen1)
        b = torch.rand(10, generator=gen2)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self) -> None:
        gen1 = get_torch_generator(42)
        gen2 = get_torch_generator(99)
        a = torch.rand(10, generator=gen1)
        b = torch.rand(10, generator=gen2)
        assert not torch.equal(a, b)
