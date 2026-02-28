"""
Preprocessing module.
Extracts fixed-rate frame sequences from video clips using FFmpeg,
then resizes, normalises, and stacks them into spatiotemporal tensors.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _get_ffmpeg_exe() -> str:
    """
    Resolve the FFmpeg executable path.
    Priority:
      1. System PATH (ffmpeg / ffmpeg.exe)
      2. imageio-ffmpeg bundled binary (installed with the project venv)
    Raises RuntimeError if neither is found.
    """
    import shutil
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(
        "FFmpeg not found. Install imageio-ffmpeg: pip install imageio-ffmpeg"
    )

# ImageNet-style normalisation constants (standard for pre-trained backbones)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames_ffmpeg(
    video_path: str,
    fps: int = 4,
    out_dir: Optional[str] = None,
) -> List[str]:
    """
    Decode `video_path` at `fps` frames per second using FFmpeg.
    Returns a sorted list of saved frame paths (PNG).
    Falls back gracefully if FFmpeg is unavailable.
    """
    video_path = str(video_path)
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="frames_")
    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(out_dir, "frame_%06d.png")
    ffmpeg_exe = _get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"FFmpeg failed ({e}). Falling back to OpenCV extraction.")
        return _extract_frames_opencv(video_path, fps, out_dir)

    frames = sorted(Path(out_dir).glob("frame_*.png"), key=lambda p: p.name)
    return [str(f) for f in frames]


def _extract_frames_opencv(video_path: str, fps: int, out_dir: str) -> List[str]:
    """OpenCV fallback for frame extraction."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    saved, idx, frame_idx = [], 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            path = os.path.join(out_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(path, frame)
            saved.append(path)
            frame_idx += 1
        idx += 1

    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(
    frame: np.ndarray,
    size: Tuple[int, int] = (64, 64),
    detect_scene_cut: bool = False,
    prev_frame: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Resize and normalise a single BGR frame (H×W×3, uint8).
    Returns (normalised float32 RGB array, scene_cut_flag).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    normed = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD  # (H, W, 3)

    scene_cut = False
    if detect_scene_cut and prev_frame is not None:
        # Simple histogram-based scene cut detector
        hist_curr = cv2.calcHist([rgb], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist_prev = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        corr = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CORREL)
        scene_cut = corr < 0.5

    return normed, scene_cut


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """
    Stack list of (H, W, C) float32 arrays → tensor (C, T, H, W).
    This is the spatiotemporal format expected by 3-D convolutions.
    """
    stacked = np.stack(frames, axis=0)       # (T, H, W, C)
    stacked = stacked.transpose(3, 0, 1, 2)  # (C, T, H, W)
    return torch.from_numpy(stacked)


# ---------------------------------------------------------------------------
# Clip builder
# ---------------------------------------------------------------------------

def build_clips(
    frame_paths: List[str],
    clip_len: int = 16,
    stride: int = 8,
    frame_size: Tuple[int, int] = (64, 64),
    detect_scene_cuts: bool = True,
) -> Tuple[List[torch.Tensor], List[bool]]:
    """
    Slide a window over `frame_paths` to produce overlapping clips.

    Returns:
        clips      – list of (C, T, H, W) tensors
        scene_cuts – bool flag per clip (True if a scene cut was detected)
    """
    clips, scene_cut_flags = [], []
    prev_bgr = None

    # Load all frames once
    raw_frames, cut_flags = [], []
    for path in frame_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            logger.warning(f"Cannot read frame: {path}. Skipping.")
            continue
        normed, cut = preprocess_frame(
            bgr, size=frame_size,
            detect_scene_cut=detect_scene_cuts,
            prev_frame=prev_bgr,
        )
        raw_frames.append(normed)
        cut_flags.append(cut)
        prev_bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Sliding window
    for start in range(0, max(1, len(raw_frames) - clip_len + 1), stride):
        end = start + clip_len
        if end > len(raw_frames):
            break
        clip_frames = raw_frames[start:end]
        has_cut = any(cut_flags[start:end])
        clips.append(frames_to_tensor(clip_frames))
        scene_cut_flags.append(has_cut)

    return clips, scene_cut_flags


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class VideoClipDataset(Dataset):
    """
    Dataset that loads pre-extracted frame clips from a directory.
    Expected structure:
        root/
          video_001/frame_000001.png
          video_001/frame_000002.png
          ...
    """

    def __init__(
        self,
        root: str,
        clip_len: int = 16,
        stride: int = 8,
        frame_size: Tuple[int, int] = (64, 64),
        fps: int = 4,
        extract: bool = False,
    ):
        self.clip_len = clip_len
        self.stride = stride
        self.frame_size = frame_size
        self.clips: List[torch.Tensor] = []
        self.meta: List[dict] = []

        root = Path(root)
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}

        for item in sorted(root.iterdir()):
            if item.is_file() and item.suffix.lower() in video_exts and extract:
                frame_dir = root / f"_frames_{item.stem}"
                paths = extract_frames_ffmpeg(str(item), fps=fps, out_dir=str(frame_dir))
                self._add_clips(paths, source=item.name)
            elif item.is_dir():
                paths = sorted(item.glob("*.png")) + sorted(item.glob("*.jpg"))
                paths = [str(p) for p in paths]
                if paths:
                    self._add_clips(paths, source=item.name)

    def _add_clips(self, frame_paths: List[str], source: str):
        clips, cuts = build_clips(
            frame_paths,
            clip_len=self.clip_len,
            stride=self.stride,
            frame_size=self.frame_size,
        )
        for clip, cut in zip(clips, cuts):
            self.clips.append(clip)
            self.meta.append({"source": source, "scene_cut": cut})

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return self.clips[idx], self.meta[idx]
