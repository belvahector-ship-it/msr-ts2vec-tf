"""
src/utils/seed.py — Deterministic Seed Control
===============================================

Purpose
-------
Set random seeds across all randomness sources (Python, NumPy, PyTorch, CUDA)
to guarantee reproducible execution for a given seed value.

This module is the ONLY place in the codebase where seeds are set.
All training runs, embedding computations, and clustering runs must
call set_all_seeds() before any randomized operation.

Design (ADR-019, INV-007)
--------------------------
- Five seeds are defined in base.yaml: [42, 123, 456, 789, 1024].
- set_all_seeds() covers: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
- Known non-determinism on GPU (certain CUDA operations) is documented
  in the function's docstring and in the checkpoint metadata.
- VALID_SEEDS is exposed to allow callers to assert they are using
  only the protocol-defined seeds.

Usage
-----
    from src.utils.seed import set_all_seeds, VALID_SEEDS

    for seed in VALID_SEEDS:
        set_all_seeds(seed)
        # ... run experiment with this seed ...
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# The five seeds defined in ADR-019 and base.yaml experiment.seeds.
# Kept here as a constant for runtime assertion in the experiment runner.
VALID_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1024)


def set_all_seeds(seed: int, warn_if_nonstandard: bool = True) -> None:
    """
    Set random seeds across all randomness sources for reproducible execution.

    Sources controlled:
      - Python built-in random module
      - NumPy random generator
      - PyTorch CPU random generator
      - PyTorch CUDA random generator (all devices, if available)
      - PYTHONHASHSEED environment variable (for Python hash randomization)
      - torch.backends.cudnn determinism flags

    Parameters
    ----------
    seed : int
        Random seed value. Must be a non-negative integer.
    warn_if_nonstandard : bool
        If True, log a WARNING when seed is not in VALID_SEEDS.
        The protocol defines exactly five valid seeds (ADR-019).
        Seeds outside this set are permitted for testing but should
        not appear in final experiment runs.

    Notes
    -----
    GPU non-determinism:
        Even with all seeds set, some CUDA operations (e.g., certain
        atomics in PyTorch scatter operations) may produce slightly
        different results across runs. This is documented as a known
        limitation in V-INV-007 and in every checkpoint artifact.

        To maximize determinism, torch.use_deterministic_algorithms(True)
        is applied where possible. Operations that cannot be made
        deterministic will raise a RuntimeError, which is caught and
        logged as a WARNING rather than aborting — the researcher must
        decide whether to accept the non-determinism for that operation.

    Raises
    ------
    ValueError
        If seed is negative or not an integer.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(
            f"Seed must be a non-negative integer, got: {seed!r}. "
            "Check the seeds defined in configs/base.yaml."
        )

    if warn_if_nonstandard and seed not in VALID_SEEDS:
        logger.warning(
            "Seed %d is not one of the protocol-defined seeds %s. "
            "This seed should only be used for testing or development, "
            "not for final experiment runs (ADR-019).",
            seed,
            VALID_SEEDS,
        )

    # Python built-in
    random.seed(seed)

    # PYTHONHASHSEED (affects dict/set ordering in some contexts)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Maximize CUDA determinism
        # cudnn.deterministic = True disables non-deterministic algorithms
        # cudnn.benchmark = False prevents runtime auto-tuning (which is non-deterministic)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms globally where possible
    # Errors for operations without a deterministic implementation are
    # caught and logged, not suppressed.
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:
        logger.warning(
            "torch.use_deterministic_algorithms(True) raised RuntimeError: %s. "
            "Some operations may remain non-deterministic. "
            "Document this in the paper's Reproducibility section.",
            exc,
        )

    logger.info(
        "All seeds set to %d "
        "(Python, NumPy, PyTorch CPU%s)",
        seed,
        ", PyTorch CUDA" if torch.cuda.is_available() else "",
    )


def get_torch_generator(seed: int) -> torch.Generator:
    """
    Create and seed a PyTorch Generator for use in DataLoaders or
    random matrix generation.

    Parameters
    ----------
    seed : int
        Seed value.

    Returns
    -------
    torch.Generator
        A seeded PyTorch Generator instance.

    Notes
    -----
    Used by the fusion module (ADR-003) to generate the fixed random
    projection matrix deterministically.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
