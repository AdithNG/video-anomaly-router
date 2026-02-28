"""
Tests for src/preprocessing.py

Covers:
  - preprocess_frame(): shape, dtype, value range, scene-cut flag
  - frames_to_tensor(): shape transposition (T,H,W,C) → (C,T,H,W)
  - build_clips(): sliding window count, clip shapes, scene-cut propagation
  - VideoClipDataset: length, item shapes when loading pre-saved frames
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing import (
    VideoClipDataset,
    _MEAN,
    _STD,
    build_clips,
    frames_to_tensor,
    preprocess_frame,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_bgr(h: int = 128, w: int = 128) -> np.ndarray:
    """Return a random uint8 BGR frame (H, W, 3)."""
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def _save_frames(n: int, directory: str, h: int = 64, w: int = 64):
    """Save `n` random PNG frames into `directory`."""
    for i in range(n):
        frame = _make_random_bgr(h, w)
        path = os.path.join(directory, f"frame_{i:06d}.png")
        cv2.imwrite(path, frame)
    return sorted(Path(directory).glob("frame_*.png"))


# ---------------------------------------------------------------------------
# preprocess_frame
# ---------------------------------------------------------------------------

class TestPreprocessFrame:
    def test_output_shape(self):
        """Output array should be (H, W, 3) matching the requested size."""
        frame = _make_random_bgr(128, 128)
        out, _ = preprocess_frame(frame, size=(64, 64))
        assert out.shape == (64, 64, 3)

    def test_output_dtype(self):
        """Output should be float32."""
        frame = _make_random_bgr()
        out, _ = preprocess_frame(frame, size=(64, 64))
        assert out.dtype == np.float32

    def test_normalised_range(self):
        """After ImageNet normalisation values should span roughly [-3, 3]."""
        frame = _make_random_bgr()
        out, _ = preprocess_frame(frame, size=(64, 64))
        assert out.min() > -5.0 and out.max() < 5.0

    def test_scene_cut_not_detected_identical_frames(self):
        """Same frame presented as current and previous → no scene cut.
        Uses a solid-color frame so the 3-D histogram is fully concentrated
        in one bin, giving a correlation of 1.0 with itself.
        """
        # Solid mid-grey: every pixel identical → deterministic histogram
        frame = np.full((128, 128, 3), 128, dtype=np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, cut = preprocess_frame(frame, size=(64, 64), detect_scene_cut=True, prev_frame=rgb)
        assert cut is False

    def test_scene_cut_detected_on_totally_different_frames(self):
        """A pure-black vs pure-white frame should register as a scene cut."""
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        white = np.ones((64, 64, 3), dtype=np.uint8) * 255
        white_rgb = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
        _, cut = preprocess_frame(black, size=(64, 64), detect_scene_cut=True, prev_frame=white_rgb)
        assert cut is True

    def test_no_scene_cut_flag_when_disabled(self):
        """detect_scene_cut=False should always return False regardless of content."""
        frame = _make_random_bgr()
        _, cut = preprocess_frame(frame, size=(64, 64), detect_scene_cut=False)
        assert cut is False


# ---------------------------------------------------------------------------
# frames_to_tensor
# ---------------------------------------------------------------------------

class TestFramesToTensor:
    def test_shape_transposition(self):
        """(T, H, W, C) numpy stack → (C, T, H, W) tensor."""
        T, H, W, C = 8, 32, 32, 3
        frames = [np.random.rand(H, W, C).astype(np.float32) for _ in range(T)]
        tensor = frames_to_tensor(frames)
        assert tensor.shape == (C, T, H, W)

    def test_output_is_tensor(self):
        frames = [np.random.rand(32, 32, 3).astype(np.float32) for _ in range(4)]
        assert isinstance(frames_to_tensor(frames), torch.Tensor)

    def test_values_preserved(self):
        """Pixel values must survive the transposition unchanged."""
        frame = np.random.rand(16, 16, 3).astype(np.float32)
        tensor = frames_to_tensor([frame])       # T=1
        # channel 0, time 0 should equal frame[:, :, 0]
        assert np.allclose(tensor[0, 0].numpy(), frame[:, :, 0])


# ---------------------------------------------------------------------------
# build_clips
# ---------------------------------------------------------------------------

class TestBuildClips:
    def setup_method(self):
        # Write 32 random frames to a temp dir for reuse across tests
        self.tmp = tempfile.mkdtemp()
        self.paths = [str(p) for p in _save_frames(32, self.tmp)]

    def test_clip_count_sliding_window(self):
        """Number of clips = floor((N - clip_len) / stride) + 1."""
        clip_len, stride = 16, 8
        clips, _ = build_clips(self.paths, clip_len=clip_len, stride=stride, frame_size=(32, 32))
        expected = (len(self.paths) - clip_len) // stride + 1
        assert len(clips) == expected

    def test_clip_shape(self):
        """Each clip should be (C=3, T=clip_len, H, W)."""
        clips, _ = build_clips(self.paths, clip_len=16, stride=8, frame_size=(32, 32))
        for clip in clips:
            assert clip.shape == (3, 16, 32, 32)

    def test_scene_cut_flags_length_matches_clips(self):
        """There should be one scene-cut flag per clip."""
        clips, flags = build_clips(self.paths, clip_len=16, stride=8, frame_size=(32, 32))
        assert len(clips) == len(flags)

    def test_scene_cut_flags_are_bool(self):
        _, flags = build_clips(self.paths, clip_len=16, stride=8, frame_size=(32, 32))
        for flag in flags:
            assert isinstance(flag, bool)

    def test_no_clips_when_too_few_frames(self):
        """Fewer frames than clip_len should yield zero clips."""
        tmp = tempfile.mkdtemp()
        paths = [str(p) for p in _save_frames(4, tmp)]
        clips, _ = build_clips(paths, clip_len=16, stride=8, frame_size=(32, 32))
        assert len(clips) == 0


# ---------------------------------------------------------------------------
# VideoClipDataset
# ---------------------------------------------------------------------------

class TestVideoClipDataset:
    def setup_method(self):
        # Create a dataset directory with one sub-folder of 32 frames
        self.root = tempfile.mkdtemp()
        scene_dir = os.path.join(self.root, "scene_01")
        os.makedirs(scene_dir)
        _save_frames(32, scene_dir, h=64, w=64)

    def test_dataset_not_empty(self):
        ds = VideoClipDataset(self.root, clip_len=16, stride=8, frame_size=(64, 64))
        assert len(ds) > 0

    def test_item_shape(self):
        """Each item should be a (C=3, T=16, H=64, W=64) tensor."""
        ds = VideoClipDataset(self.root, clip_len=16, stride=8, frame_size=(64, 64))
        clip, meta = ds[0]
        assert clip.shape == (3, 16, 64, 64)

    def test_item_meta_has_required_keys(self):
        """Metadata dict must contain 'source' and 'scene_cut' keys."""
        ds = VideoClipDataset(self.root, clip_len=16, stride=8, frame_size=(64, 64))
        _, meta = ds[0]
        assert "source" in meta
        assert "scene_cut" in meta

    def test_clip_values_are_finite(self):
        """No NaN or Inf values in any clip."""
        ds = VideoClipDataset(self.root, clip_len=16, stride=8, frame_size=(64, 64))
        for i in range(len(ds)):
            clip, _ = ds[i]
            assert torch.isfinite(clip).all()
