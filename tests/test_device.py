"""
Tests for src/device.py

Covers:
  - get_device() returns a valid torch.device
  - move() handles tensors, modules, dicts, lists
  - safe_to_device() falls back to CPU on failure
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Make sure src/ is on the path regardless of where pytest is run from
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from device import get_device, move, safe_to_device


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_torch_device(self):
        """get_device() must always return a torch.device instance."""
        device = get_device(verbose=False)
        assert isinstance(device, torch.device)

    def test_device_is_cuda_or_cpu(self):
        """Device type must be one of the two valid options."""
        device = get_device(verbose=False)
        assert device.type in ("cuda", "cpu")

    def test_cuda_device_when_available(self):
        """If CUDA is available the returned device should be CUDA."""
        if torch.cuda.is_available():
            device = get_device(verbose=False)
            assert device.type == "cuda"

    def test_repeated_calls_consistent(self):
        """Calling get_device() multiple times should return the same type."""
        d1 = get_device(verbose=False)
        d2 = get_device(verbose=False)
        assert d1.type == d2.type


# ---------------------------------------------------------------------------
# move
# ---------------------------------------------------------------------------

class TestMove:
    def setup_method(self):
        self.device = get_device(verbose=False)

    def test_move_tensor(self):
        """move() should place a tensor on the target device."""
        t = torch.randn(4, 4)
        moved = move(t, self.device)
        assert moved.device.type == self.device.type

    def test_move_module(self):
        """move() should move all parameters of a nn.Module."""
        model = nn.Linear(8, 8)
        moved = move(model, self.device)
        param_device = next(moved.parameters()).device
        assert param_device.type == self.device.type

    def test_move_dict_of_tensors(self):
        """move() should recursively move every value in a dict."""
        d = {"a": torch.randn(2, 2), "b": torch.randn(3, 3)}
        moved = move(d, self.device)
        for v in moved.values():
            assert v.device.type == self.device.type

    def test_move_list_of_tensors(self):
        """move() should recursively move every element in a list."""
        lst = [torch.randn(2), torch.randn(3)]
        moved = move(lst, self.device)
        assert isinstance(moved, list)
        for t in moved:
            assert t.device.type == self.device.type

    def test_move_tuple_of_tensors(self):
        """move() should preserve tuple type after moving."""
        tup = (torch.randn(2), torch.randn(3))
        moved = move(tup, self.device)
        assert isinstance(moved, tuple)

    def test_move_passthrough_non_tensor(self):
        """move() should return non-tensor objects unchanged."""
        assert move(42, self.device) == 42
        assert move("hello", self.device) == "hello"
        assert move(None, self.device) is None


# ---------------------------------------------------------------------------
# safe_to_device
# ---------------------------------------------------------------------------

class TestSafeToDevice:
    def setup_method(self):
        self.device = get_device(verbose=False)

    def test_normal_move(self):
        """safe_to_device() should successfully move a small tensor to device."""
        t = torch.randn(4, 4)
        moved = safe_to_device(t, self.device)
        assert moved.device.type == self.device.type

    def test_cpu_fallback_on_bad_device(self):
        """safe_to_device() should return a CPU tensor when the target is invalid."""
        t = torch.randn(4, 4)
        bad_device = torch.device("cuda:99")  # almost certainly doesn't exist
        result = safe_to_device(t, bad_device)
        assert result.device.type == "cpu"

    def test_data_preserved_after_move(self):
        """Tensor values must be unchanged after a device move."""
        t = torch.tensor([1.0, 2.0, 3.0])
        moved = safe_to_device(t, self.device)
        assert torch.allclose(moved.cpu(), t.cpu())
