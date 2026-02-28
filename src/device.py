"""
Device management with GPU/CPU fallback.
All modules import `get_device()` and use the returned device.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device(verbose: bool = True) -> torch.device:
    """
    Return the best available device: CUDA GPU → CPU fallback.
    Logs a warning if falling back to CPU.
    """
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            if verbose:
                props = torch.cuda.get_device_properties(device)
                vram_gb = props.total_memory / 1e9
                logger.info(
                    f"Using GPU: {props.name}  |  VRAM: {vram_gb:.1f} GB  |  "
                    f"CUDA {torch.version.cuda}"
                )
            return device
    except Exception as e:
        logger.warning(f"CUDA initialisation failed ({e}). Falling back to CPU.")

    logger.warning("CUDA not available — running on CPU (expect slower inference).")
    return torch.device("cpu")


def move(obj, device: torch.device):
    """Move a tensor, module, or dict/list of tensors to `device`."""
    if isinstance(obj, (torch.Tensor, torch.nn.Module)):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move(v, device) for v in obj]
        return type(obj)(moved)
    return obj  # non-tensor passthrough


def safe_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Move tensor to device; if it fails (e.g. OOM) fall back to CPU.
    """
    try:
        return tensor.to(device)
    except RuntimeError as e:
        logger.warning(f"Failed to move tensor to {device} ({e}). Using CPU.")
        return tensor.cpu()
