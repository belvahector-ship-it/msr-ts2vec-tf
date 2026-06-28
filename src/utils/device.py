"""
src/utils/device.py — Device-Agnostic Execution (ADR-007)
==========================================================

Purpose
-------
Provide a single, authoritative device detection function used by
all training, inference, and evaluation modules.

Design (ADR-007)
----------------
- One code path only: no separate CPU or GPU branches anywhere in source.
- Device is detected once at startup and propagated via config / argument.
- GPU is used when available; CPU is the fallback with no code changes.
- No torch.cuda.* calls that bypass the device abstraction are permitted.
- Checkpoints saved on GPU must load on CPU via map_location='cpu'.

Usage
-----
    from src.utils.device import get_device, DeviceInfo

    device = get_device()               # torch.device("cuda") or "cpu"
    info   = DeviceInfo.collect()       # Full device summary for logging/checkpoints

    model = MyModel().to(device)
    tensor = tensor.to(device)

    # Safe checkpoint loading (CPU always works regardless of save device)
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any

import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Returns CUDA if a CUDA-capable GPU is available, otherwise CPU.
    This is the only function that may inspect torch.cuda.is_available().

    Returns
    -------
    torch.device
        torch.device("cuda") if CUDA is available,
        torch.device("cpu") otherwise.

    Notes
    -----
    The returned device should be stored once and passed through the
    pipeline as a configuration value. Device detection should not be
    called repeatedly in inner loops.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Device selected: CUDA (%s)", gpu_name)
    else:
        device = torch.device("cpu")
        logger.info(
            "Device selected: CPU. "
            "Training will be significantly slower than on GPU. "
            "Expected training time per branch: see README.md for benchmarks."
        )
    return device


# ---------------------------------------------------------------------------
# Device info — for checkpoint metadata and logging
# ---------------------------------------------------------------------------


@dataclass
class DeviceInfo:
    """
    Snapshot of the execution environment.

    Stored in every checkpoint bundle (ADR-010) to document the
    hardware and software context of each training run.

    Attributes
    ----------
    device_type : str
        'cuda' or 'cpu'.
    gpu_name : str or None
        GPU name if CUDA is available, None otherwise.
    gpu_count : int
        Number of available CUDA devices (0 on CPU).
    cuda_version : str or None
        CUDA version string if available, None otherwise.
    torch_version : str
        PyTorch version string.
    python_version : str
        Python version string.
    platform_info : str
        OS and hardware platform description.
    mixed_precision_available : bool
        True if the GPU supports automatic mixed precision (AMP).
    """

    device_type: str
    gpu_name: str | None
    gpu_count: int
    cuda_version: str | None
    torch_version: str
    python_version: str
    platform_info: str
    mixed_precision_available: bool
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def collect(cls, device: torch.device | None = None) -> "DeviceInfo":
        """
        Collect device and environment information.

        Parameters
        ----------
        device : torch.device or None
            The device in use. If None, get_device() is called.

        Returns
        -------
        DeviceInfo
            Populated DeviceInfo instance.
        """
        if device is None:
            device = get_device()

        gpu_name: str | None = None
        cuda_version: str | None = None
        gpu_count: int = 0
        amp_available: bool = False

        if device.type == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda  # type: ignore[attr-defined]
            # AMP requires compute capability >= 7.0 (Volta+)
            cap = torch.cuda.get_device_capability(0)
            amp_available = cap[0] >= 7

        return cls(
            device_type=device.type,
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            cuda_version=cuda_version,
            torch_version=torch.__version__,
            python_version=platform.python_version(),
            platform_info=platform.platform(),
            mixed_precision_available=amp_available,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dict suitable for JSON/checkpoint storage.

        Returns
        -------
        dict
            Serializable representation of device information.
        """
        return {
            "device_type": self.device_type,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "mixed_precision_available": self.mixed_precision_available,
            **self.extra,
        }

    def log_summary(self) -> None:
        """
        Write a human-readable device summary to the logger.
        Called once at the start of training runs.
        """
        logger.info("=" * 60)
        logger.info("Execution Environment")
        logger.info("  Device:          %s", self.device_type.upper())
        if self.gpu_name:
            logger.info("  GPU:             %s (count: %d)", self.gpu_name, self.gpu_count)
            logger.info("  CUDA version:    %s", self.cuda_version)
            logger.info("  AMP available:   %s", self.mixed_precision_available)
        logger.info("  PyTorch:         %s", self.torch_version)
        logger.info("  Python:          %s", self.python_version)
        logger.info("  Platform:        %s", self.platform_info)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Checkpoint loading utility
# ---------------------------------------------------------------------------


def safe_load_checkpoint(path: Any, device: torch.device | None = None) -> dict[str, Any]:
    """
    Load a PyTorch checkpoint safely, mapping to CPU by default.

    Checkpoints may have been saved on GPU but must be loadable on any
    device. map_location="cpu" ensures this without requiring CUDA.

    Parameters
    ----------
    path : Path-like
        Path to the checkpoint file (.pt).
    device : torch.device or None
        Target device. Defaults to CPU for maximum compatibility.
        The caller is responsible for moving tensors to the correct
        device after loading.

    Returns
    -------
    dict
        Loaded checkpoint dictionary.

    Notes
    -----
    After loading, call model.to(device) to move to the target device.
    """
    map_location = device if device is not None else torch.device("cpu")
    checkpoint: dict[str, Any] = torch.load(path, map_location=map_location)
    return checkpoint
