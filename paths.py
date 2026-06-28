"""
src/utils/paths.py — Centralized Path Management
=================================================

Purpose
-------
Provides the canonical path constants for the entire project.
All modules must import paths from here. No module may construct
paths using hardcoded strings or os.path manipulation outside
of this file.

Design
------
- All paths are derived from PROJECT_ROOT, which is detected
  automatically at import time by walking up from this file.
- Functions return pathlib.Path objects, never strings.
- Directories are created on first access if they do not exist.

Usage
-----
    from src.utils.paths import Paths

    raw_dir = Paths.raw_data()          # data/raw/
    train_windows = Paths.processed_file("train_windows_1h.npy")
    checkpoint = Paths.checkpoint("branch_1h", "seed_42", "best_model.pt")
"""

from __future__ import annotations

from pathlib import Path


def _find_project_root() -> Path:
    """
    Walk upward from this file until a directory containing
    'configs/base.yaml' is found. That directory is the project root.

    Raises
    ------
    RuntimeError
        If the project root cannot be located, which typically means
        the package is being used outside of the expected repository structure.
    """
    candidate = Path(__file__).resolve()
    for parent in [candidate, *candidate.parents]:
        if (parent / "configs" / "base.yaml").exists():
            return parent
    raise RuntimeError(
        "Project root could not be located. "
        "Expected to find 'configs/base.yaml' in an ancestor directory of "
        f"'{__file__}'. "
        "Ensure you are running from within the project repository."
    )


PROJECT_ROOT: Path = _find_project_root()


class Paths:
    """
    Namespace class providing static accessors for all canonical project paths.

    All path methods return pathlib.Path objects.
    Directory-returning methods create the directory if it does not exist.
    """

    # ------------------------------------------------------------------
    # Root
    # ------------------------------------------------------------------

    @staticmethod
    def root() -> Path:
        """Return the absolute path to the project root directory."""
        return PROJECT_ROOT

    # ------------------------------------------------------------------
    # Configs
    # ------------------------------------------------------------------

    @staticmethod
    def configs() -> Path:
        """Return the path to the configs/ directory."""
        return PROJECT_ROOT / "configs"

    @staticmethod
    def base_config() -> Path:
        """Return the path to configs/base.yaml."""
        return PROJECT_ROOT / "configs" / "base.yaml"

    @staticmethod
    def experiment_config(condition: str) -> Path:
        """
        Return the path to the experiment config for a given condition.

        Parameters
        ----------
        condition : str
            Condition label, e.g. '1tf', '2tf', 'bl_15m'.
            Case-insensitive; converted to lowercase.

        Returns
        -------
        Path
            Path to configs/experiment_{condition}.yaml
        """
        return PROJECT_ROOT / "configs" / f"experiment_{condition.lower()}.yaml"

    # ------------------------------------------------------------------
    # Data — raw
    # ------------------------------------------------------------------

    @staticmethod
    def raw_data() -> Path:
        """Return data/raw/, creating it if necessary."""
        p = PROJECT_ROOT / "data" / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def raw_file(filename: str) -> Path:
        """Return the path to a file in data/raw/."""
        return Paths.raw_data() / filename

    # ------------------------------------------------------------------
    # Data — interim
    # ------------------------------------------------------------------

    @staticmethod
    def interim_data() -> Path:
        """Return data/interim/, creating it if necessary."""
        p = PROJECT_ROOT / "data" / "interim"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def interim_file(filename: str) -> Path:
        """Return the path to a file in data/interim/."""
        return Paths.interim_data() / filename

    # ------------------------------------------------------------------
    # Data — processed
    # ------------------------------------------------------------------

    @staticmethod
    def processed_data() -> Path:
        """Return data/processed/, creating it if necessary."""
        p = PROJECT_ROOT / "data" / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def processed_file(filename: str) -> Path:
        """Return the path to a file in data/processed/."""
        return Paths.processed_data() / filename

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    @staticmethod
    def checkpoints() -> Path:
        """Return checkpoints/, creating it if necessary."""
        p = PROJECT_ROOT / "checkpoints"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def branch_checkpoint_dir(timeframe: str, seed: int) -> Path:
        """
        Return the checkpoint directory for a specific branch and seed.

        Parameters
        ----------
        timeframe : str
            Branch timeframe, e.g. '1h', '4h'.
        seed : int
            Random seed for this training run.

        Returns
        -------
        Path
            checkpoints/branch_{timeframe}/seed_{seed}/
        """
        p = PROJECT_ROOT / "checkpoints" / f"branch_{timeframe}" / f"seed_{seed}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def checkpoint(timeframe: str, seed: int, filename: str) -> Path:
        """
        Return the path to a specific checkpoint file.

        Parameters
        ----------
        timeframe : str
            Branch timeframe.
        seed : int
            Random seed.
        filename : str
            Checkpoint filename, e.g. 'best_model.pt' or 'latest_model.pt'.
        """
        return Paths.branch_checkpoint_dir(timeframe, seed) / filename

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    @staticmethod
    def experiments() -> Path:
        """Return experiments/, creating it if necessary."""
        p = PROJECT_ROOT / "experiments"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def experiment_dir(exp_id: str) -> Path:
        """
        Return the directory for a specific experiment run.

        Parameters
        ----------
        exp_id : str
            Unique experiment identifier, e.g. '20260601_1TF_seed42'.
        """
        p = PROJECT_ROOT / "experiments" / exp_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def experiment_embeddings(exp_id: str, kind: str = "fused") -> Path:
        """
        Return the embeddings subdirectory for an experiment.

        Parameters
        ----------
        exp_id : str
            Unique experiment identifier.
        kind : str
            'branch' for per-branch 64-dim embeddings,
            'fused' for 256-dim fused embeddings.
        """
        p = Paths.experiment_dir(exp_id) / "embeddings" / kind
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def experiment_clustering(exp_id: str) -> Path:
        """Return the clustering subdirectory for an experiment."""
        p = Paths.experiment_dir(exp_id) / "clustering"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def experiment_evaluation(exp_id: str) -> Path:
        """Return the evaluation subdirectory for an experiment."""
        p = Paths.experiment_dir(exp_id) / "evaluation"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    @staticmethod
    def outputs() -> Path:
        """Return outputs/, creating it if necessary."""
        p = PROJECT_ROOT / "outputs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def figures(figure_name: str | None = None) -> Path:
        """
        Return outputs/final/figures/ or a named figure subdirectory.

        Parameters
        ----------
        figure_name : str or None
            If provided, returns outputs/final/figures/{figure_name}/.
        """
        base = PROJECT_ROOT / "outputs" / "final" / "figures"
        p = base / figure_name if figure_name else base
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def tables() -> Path:
        """Return outputs/final/tables/, creating it if necessary."""
        p = PROJECT_ROOT / "outputs" / "final" / "tables"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def paper_artifacts() -> Path:
        """Return outputs/final/paper_artifacts/, creating it if necessary."""
        p = PROJECT_ROOT / "outputs" / "final" / "paper_artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    @staticmethod
    def logs() -> Path:
        """Return logs/, creating it if necessary."""
        p = PROJECT_ROOT / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def log_file(name: str) -> Path:
        """
        Return the path to a named log file.

        Parameters
        ----------
        name : str
            Log file name, e.g. 'training_branch_1h_seed42.log'.
        """
        return Paths.logs() / name

    # ------------------------------------------------------------------
    # Docs
    # ------------------------------------------------------------------

    @staticmethod
    def docs() -> Path:
        """Return docs/."""
        return PROJECT_ROOT / "docs"

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    @staticmethod
    def experiment_registry() -> Path:
        """Return the path to the experiment registry JSON file."""
        return Paths.experiments() / "registry.json"
