"""
tests/test_utils/test_config.py
================================
Unit tests for src.utils.config.

Covers:
- Successful load of base.yaml
- Deep merge behavior
- Missing required field detection (all-at-once error report)
- Type/range validation
- Experiment config merging
"""

from __future__ import annotations

import copy
import textwrap
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    ConfigValidationError,
    _deep_merge,
    load_config,
    load_experiment_config,
    load_yaml,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_valid_config() -> dict:
    """
    A minimal config dict that satisfies all required fields.
    Used to build targeted failure cases without needing full YAML files.
    """
    return {
        "project": {"name": "test", "version": "1.0"},
        "ts2vec": {
            "commit_hash": "abc123",
            "primary_repo": "https://github.com/yuezhihan/ts2vec",
        },
        "data": {
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "timeframes": ["15m", "1h", "4h", "1d"],
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "raw_dir": "data/raw",
            "interim_dir": "data/interim",
            "processed_dir": "data/processed",
            "max_gap_ratio": 0.05,
        },
        "split": {
            "train_end": "2022-12-31 23:00:00",
            "test_start": "2023-01-01 00:00:00",
        },
        "features": {"n_features": 7, "volume_zscore_window": 20},
        "windows": {"W": 48, "stride": 1, "epsilon": 1e-8},
        "encoder": {
            "input_dim": 7,
            "output_dim": 64,
            "hidden_dim": 64,
            "depth": 10,
            "mask_ratio": 0.5,
        },
        "fusion": {"output_dim": 256, "projection_seed": 42},
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 8,
            "max_epochs": 50,
            "early_stopping_patience": 10,
            "checkpoint_dir": "checkpoints",
        },
        "experiment": {
            "seeds": [42, 123, 456, 789, 1024],
            "experiments_dir": "experiments",
        },
        "clustering": {
            "metric": "euclidean",
            "min_clusters": 2,
            "max_clusters": 8,
            "grid_search": {
                "min_cluster_size": [50, 100, 200],
                "min_samples": [5, 10, 20],
            },
        },
        "evaluation": {"primary_metric": "silhouette", "significance_level": 0.05},
        "outputs": {
            "base_dir": "outputs",
            "figures": {"dpi": 300, "formats": ["png", "pdf", "svg"]},
        },
        "logging": {
            "level": "INFO",
            "log_dir": "logs",
            "format": "%(asctime)s | %(message)s",
            "date_format": "%Y-%m-%d %H:%M:%S",
        },
    }


@pytest.fixture()
def tmp_base_config(tmp_path: Path, minimal_valid_config: dict) -> Path:
    """Write a valid minimal base config to a temp file and return its path."""
    config_path = tmp_path / "base.yaml"
    with config_path.open("w") as f:
        yaml.dump(minimal_valid_config, f)
    return config_path


@pytest.fixture()
def tmp_experiment_config(tmp_path: Path) -> Path:
    """Write a valid experiment override config to a temp file."""
    override = {
        "condition": "1TF",
        "active_branches": ["1h"],
        "fusion": {"input_dim_override": 64},
    }
    config_path = tmp_path / "experiment_1tf.yaml"
    with config_path.open("w") as f:
        yaml.dump(override, f)
    return config_path


# ---------------------------------------------------------------------------
# load_yaml
# ---------------------------------------------------------------------------


class TestLoadYaml:
    def test_loads_valid_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.yaml"
        p.write_text("key: value\nnested:\n  a: 1\n")
        result = load_yaml(p)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        result = load_yaml(p)
        assert result == {}


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_override_scalar(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_merge(self) -> None:
        base = {"training": {"lr": 1e-3, "batch_size": 8}}
        override = {"training": {"lr": 5e-4}}
        result = _deep_merge(base, override)
        assert result["training"]["lr"] == 5e-4
        assert result["training"]["batch_size"] == 8  # preserved

    def test_new_key_added(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_inputs(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"x": 99}}
        base_copy = copy.deepcopy(base)
        _deep_merge(base, override)
        assert base == base_copy  # base unchanged


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_valid_base_config(self, tmp_base_config: Path) -> None:
        cfg = load_config(tmp_base_config)
        assert cfg["windows"]["W"] == 48
        assert cfg["experiment"]["seeds"] == [42, 123, 456, 789, 1024]

    def test_raises_on_missing_required_field(
        self, tmp_path: Path, minimal_valid_config: dict
    ) -> None:
        # Remove a required field
        del minimal_valid_config["fusion"]["projection_seed"]
        p = tmp_path / "bad.yaml"
        with p.open("w") as f:
            yaml.dump(minimal_valid_config, f)

        with pytest.raises(ConfigValidationError, match="fusion.projection_seed"):
            load_config(p)

    def test_error_lists_all_missing_fields(
        self, tmp_path: Path, minimal_valid_config: dict
    ) -> None:
        # Remove multiple required fields
        del minimal_valid_config["fusion"]
        del minimal_valid_config["encoder"]
        p = tmp_path / "bad.yaml"
        with p.open("w") as f:
            yaml.dump(minimal_valid_config, f)

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(p)
        # Both missing sections should appear in the single error message
        assert "fusion" in str(exc_info.value)
        assert "encoder" in str(exc_info.value)

    def test_raises_on_empty_seeds(
        self, tmp_path: Path, minimal_valid_config: dict
    ) -> None:
        minimal_valid_config["experiment"]["seeds"] = []
        p = tmp_path / "bad.yaml"
        with p.open("w") as f:
            yaml.dump(minimal_valid_config, f)

        with pytest.raises(ConfigValidationError, match="seeds"):
            load_config(p)

    def test_raises_on_negative_window_size(
        self, tmp_path: Path, minimal_valid_config: dict
    ) -> None:
        minimal_valid_config["windows"]["W"] = -1
        p = tmp_path / "bad.yaml"
        with p.open("w") as f:
            yaml.dump(minimal_valid_config, f)

        with pytest.raises(ConfigValidationError, match="W"):
            load_config(p)

    def test_raises_when_max_clusters_less_than_min(
        self, tmp_path: Path, minimal_valid_config: dict
    ) -> None:
        minimal_valid_config["clustering"]["min_clusters"] = 10
        minimal_valid_config["clustering"]["max_clusters"] = 2
        p = tmp_path / "bad.yaml"
        with p.open("w") as f:
            yaml.dump(minimal_valid_config, f)

        with pytest.raises(ConfigValidationError, match="max_clusters"):
            load_config(p)


# ---------------------------------------------------------------------------
# load_experiment_config
# ---------------------------------------------------------------------------


class TestLoadExperimentConfig:
    def test_merges_correctly(
        self,
        tmp_base_config: Path,
        tmp_experiment_config: Path,
    ) -> None:
        cfg = load_experiment_config(
            condition="1tf",
            base_config_path=tmp_base_config,
            experiment_config_path=tmp_experiment_config,
        )
        # Base fields preserved
        assert cfg["windows"]["W"] == 48
        # Experiment override applied
        assert cfg["condition"] == "1TF"
        assert cfg["active_branches"] == ["1h"]

    def test_raises_if_condition_missing(
        self,
        tmp_path: Path,
        tmp_base_config: Path,
        minimal_valid_config: dict,
    ) -> None:
        # Experiment config missing 'condition'
        override = {"active_branches": ["1h"]}
        p = tmp_path / "bad_exp.yaml"
        with p.open("w") as f:
            yaml.dump(override, f)

        with pytest.raises(ConfigValidationError, match="condition"):
            load_experiment_config(
                condition="1tf",
                base_config_path=tmp_base_config,
                experiment_config_path=p,
            )

    def test_raises_if_active_branches_missing(
        self,
        tmp_path: Path,
        tmp_base_config: Path,
    ) -> None:
        override = {"condition": "1TF"}
        p = tmp_path / "bad_exp.yaml"
        with p.open("w") as f:
            yaml.dump(override, f)

        with pytest.raises(ConfigValidationError, match="active_branches"):
            load_experiment_config(
                condition="1tf",
                base_config_path=tmp_base_config,
                experiment_config_path=p,
            )

    def test_base_values_not_mutated_across_calls(
        self,
        tmp_base_config: Path,
        tmp_experiment_config: Path,
    ) -> None:
        cfg1 = load_experiment_config("1tf", tmp_base_config, tmp_experiment_config)
        cfg2 = load_config(tmp_base_config)
        # Base should not have 'condition' from experiment override
        assert "condition" not in cfg2
        assert "condition" in cfg1
