"""
src/utils/config.py — Configuration Loader with Schema Validation
=================================================================

Purpose
-------
Load and validate YAML configuration files. Merges condition-specific
experiment configs on top of the base config (base.yaml).

Every experiment run receives a fully resolved, validated config dict.
A snapshot of this dict is saved in every experiment artifact.

Design (ADR-020)
----------------
- No hardcoded hyperparameters exist anywhere in source code.
- base.yaml is the single source of truth for all shared parameters.
- Experiment configs override only condition-specific fields.
- Schema validation runs at load time. Missing required fields raise
  ConfigValidationError before any computation begins.
- The resolved config is a plain Python dict (not a DotMap or Namespace)
  to avoid YAML parsing ambiguity in saved snapshots.

Usage
-----
    from src.utils.config import load_config, load_experiment_config

    # Load base config
    cfg = load_config()

    # Load base + experiment override merged
    cfg = load_experiment_config("1tf")

    # Access values
    seeds = cfg["experiment"]["seeds"]
    W     = cfg["windows"]["W"]
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import Paths


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigValidationError(ValueError):
    """Raised when a required configuration field is missing or invalid."""


# ---------------------------------------------------------------------------
# Required field schema
# ---------------------------------------------------------------------------

# Flat list of dot-notation keys that must be present after merging.
# Any missing key raises ConfigValidationError before any computation begins.
_REQUIRED_FIELDS: list[str] = [
    # Project
    "project.name",
    "project.version",
    # TS2Vec
    "ts2vec.commit_hash",
    "ts2vec.primary_repo",
    # Data
    "data.symbol",
    "data.exchange",
    "data.timeframes",
    "data.start_date",
    "data.end_date",
    "data.raw_dir",
    "data.interim_dir",
    "data.processed_dir",
    "data.max_gap_ratio",
    # Split
    "split.train_end",
    "split.test_start",
    # Features
    "features.n_features",
    "features.volume_zscore_window",
    # Windows
    "windows.W",
    "windows.stride",
    "windows.epsilon",
    # Encoder
    "encoder.input_dim",
    "encoder.output_dim",
    "encoder.hidden_dim",
    "encoder.depth",
    "encoder.mask_ratio",
    # Fusion
    "fusion.output_dim",
    "fusion.projection_seed",
    # Training
    "training.optimizer",
    "training.learning_rate",
    "training.weight_decay",
    "training.batch_size",
    "training.max_epochs",
    "training.early_stopping_patience",
    "training.checkpoint_dir",
    # Experiment
    "experiment.seeds",
    "experiment.experiments_dir",
    # Clustering
    "clustering.metric",
    "clustering.min_clusters",
    "clustering.max_clusters",
    "clustering.grid_search.min_cluster_size",
    "clustering.grid_search.min_samples",
    # Evaluation
    "evaluation.primary_metric",
    "evaluation.significance_level",
    # Outputs
    "outputs.base_dir",
    "outputs.figures.dpi",
    "outputs.figures.formats",
    # Logging
    "logging.level",
    "logging.log_dir",
]

# Fields that must be present in experiment configs (on top of base)
_REQUIRED_EXPERIMENT_FIELDS: list[str] = [
    "condition",
    "active_branches",
]


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _get_nested(d: dict[str, Any], key: str) -> Any:
    """
    Retrieve a value from a nested dict using dot-notation.

    Parameters
    ----------
    d : dict
        The config dictionary.
    key : str
        Dot-separated key, e.g. 'training.learning_rate'.

    Returns
    -------
    Any
        The value at the specified key path.

    Raises
    ------
    KeyError
        If any level of the key path is absent.
    """
    parts = key.split(".")
    current: Any = d
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key)
        current = current[part]
    return current


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge override into base. Values in override take precedence.
    Returns a new dict; neither input is mutated.

    Parameters
    ----------
    base : dict
        Base configuration dictionary.
    override : dict
        Override values (from experiment config).

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_required_fields(
    config: dict[str, Any],
    required: list[str],
    context: str = "config",
) -> None:
    """
    Verify that all required dot-notation keys exist in the config dict.

    Parameters
    ----------
    config : dict
        The fully merged config dictionary.
    required : list of str
        List of required dot-notation key paths.
    context : str
        Human-readable label for the config source, used in error messages.

    Raises
    ------
    ConfigValidationError
        If any required key is missing. Lists ALL missing keys in one message
        so the researcher does not need to fix one field at a time.
    """
    missing: list[str] = []
    for key in required:
        try:
            _get_nested(config, key)
        except KeyError:
            missing.append(key)

    if missing:
        joined = "\n  - ".join(missing)
        raise ConfigValidationError(
            f"Configuration validation failed for '{context}'.\n"
            f"The following required fields are missing or unreachable:\n"
            f"  - {joined}\n"
            f"Check that these keys exist in 'configs/base.yaml' or "
            f"the relevant experiment config."
        )


def _validate_types(config: dict[str, Any]) -> None:
    """
    Validate semantic constraints beyond field presence.

    Raises
    ------
    ConfigValidationError
        If any value has an unexpected type or out-of-range value.
    """
    errors: list[str] = []

    # Seeds must be a non-empty list of integers
    seeds = _get_nested(config, "experiment.seeds")
    if not isinstance(seeds, list) or not seeds:
        errors.append("'experiment.seeds' must be a non-empty list.")
    elif not all(isinstance(s, int) for s in seeds):
        errors.append("'experiment.seeds' must contain only integers.")

    # Window size must be positive integer
    W = _get_nested(config, "windows.W")
    if not isinstance(W, int) or W <= 0:
        errors.append(f"'windows.W' must be a positive integer, got: {W!r}.")

    # Stride must be positive integer
    stride = _get_nested(config, "windows.stride")
    if not isinstance(stride, int) or stride <= 0:
        errors.append(f"'windows.stride' must be a positive integer, got: {stride!r}.")

    # Output dimensions
    branch_dim = _get_nested(config, "encoder.output_dim")
    fusion_dim = _get_nested(config, "fusion.output_dim")
    if not isinstance(branch_dim, int) or branch_dim <= 0:
        errors.append(f"'encoder.output_dim' must be a positive integer, got: {branch_dim!r}.")
    if not isinstance(fusion_dim, int) or fusion_dim <= 0:
        errors.append(f"'fusion.output_dim' must be a positive integer, got: {fusion_dim!r}.")

    # max_clusters must be >= min_clusters
    min_k = _get_nested(config, "clustering.min_clusters")
    max_k = _get_nested(config, "clustering.max_clusters")
    if max_k < min_k:
        errors.append(
            f"'clustering.max_clusters' ({max_k}) must be >= "
            f"'clustering.min_clusters' ({min_k})."
        )

    # Figure formats must be a non-empty list
    formats = _get_nested(config, "outputs.figures.formats")
    if not isinstance(formats, list) or not formats:
        errors.append("'outputs.figures.formats' must be a non-empty list.")

    if errors:
        joined = "\n  - ".join(errors)
        raise ConfigValidationError(
            f"Configuration type/range validation failed:\n  - {joined}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a single YAML file and return its contents as a dict.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If the file contains invalid YAML syntax.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: '{path}'. "
            "Ensure the project is correctly set up and all config files are present."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load and validate the base configuration.

    Parameters
    ----------
    config_path : Path or None
        Path to the base config YAML. Defaults to configs/base.yaml.

    Returns
    -------
    dict
        Validated base configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ConfigValidationError
        If any required field is missing or invalid.
    """
    path = config_path or Paths.base_config()
    config = load_yaml(path)
    _validate_required_fields(config, _REQUIRED_FIELDS, context=str(path))
    _validate_types(config)
    return config


def load_experiment_config(
    condition: str,
    base_config_path: Path | None = None,
    experiment_config_path: Path | None = None,
) -> dict[str, Any]:
    """
    Load the base config and merge an experiment-specific override on top.

    Parameters
    ----------
    condition : str
        Condition identifier, e.g. '1tf', '2tf', 'bl_15m'.
        Used to locate configs/experiment_{condition}.yaml.
    base_config_path : Path or None
        Override path to base.yaml.
    experiment_config_path : Path or None
        Override path to the experiment YAML.

    Returns
    -------
    dict
        Fully merged and validated configuration dictionary.
        The 'condition' and 'active_branches' keys are guaranteed to be present.

    Raises
    ------
    FileNotFoundError
        If either config file does not exist.
    ConfigValidationError
        If any required field is missing or has an invalid value.
    """
    base = load_config(base_config_path)

    exp_path = experiment_config_path or Paths.experiment_config(condition)
    exp_overrides = load_yaml(exp_path)

    merged = _deep_merge(base, exp_overrides)

    # Validate all base fields still present after merge
    _validate_required_fields(merged, _REQUIRED_FIELDS, context=str(exp_path))
    # Validate experiment-specific fields
    _validate_required_fields(
        merged,
        _REQUIRED_EXPERIMENT_FIELDS,
        context=str(exp_path),
    )
    _validate_types(merged)

    return merged


def get_config_value(config: dict[str, Any], key: str) -> Any:
    """
    Retrieve a value from a config dict using dot-notation.

    Parameters
    ----------
    config : dict
        Config dictionary.
    key : str
        Dot-separated key, e.g. 'training.learning_rate'.

    Returns
    -------
    Any
        Value at the specified key path.

    Raises
    ------
    ConfigValidationError
        If the key is not found.
    """
    try:
        return _get_nested(config, key)
    except KeyError:
        raise ConfigValidationError(
            f"Key '{key}' not found in configuration. "
            "Check that base.yaml defines this field."
        )
