from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, List, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np


logger = logging.getLogger(__name__)

Cv2Error = cv2.error if cv2 is not None else RuntimeError

EVIDENCE_KEYFRAME = "keyframe_single"
EVIDENCE_CLIP = "video_clip"
EVIDENCE_PREFIX = "video_prefix"
EVIDENCE_CLIP_PAIR = "video_clip_pair"

CLIP_FRAME_COUNT = 50
PREFIX_FRAME_COUNT = 100
CLIP_PAIR_FRAMES_PER_CLIP = 50
MAX_DIM_SINGLE = 2048
MAX_DIM_STITCHED = 2048


class BlackFrameError(RuntimeError):
    pass


_BLACK_FRAME_BRIGHTNESS_THRESHOLD: int = 10
_BLACK_FRAME_RATIO_THRESHOLD: float = 0.80


def _resize_image_if_needed(image_path: str, max_dim: int = 2048) -> bytes:
    if cv2 is None:
        raise ValueError("cv2 (OpenCV) is required for image evidence extraction but is not installed.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot decode image: {image_path}")

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"Failed to JPEG-encode image: {image_path}")
    return buf.tobytes()


def _count_black_frames(frames: List[np.ndarray]) -> Tuple[int, int]:
    n_black = 0
    for frame in frames:
        if frame.mean() < _BLACK_FRAME_BRIGHTNESS_THRESHOLD:
            n_black += 1
    return n_black, len(frames)


def _check_black_frames(frames: List[np.ndarray], video_path: str) -> None:
    if not frames:
        return
    n_black, n_total = _count_black_frames(frames)
    if n_total > 0 and n_black / n_total >= _BLACK_FRAME_RATIO_THRESHOLD:
        raise BlackFrameError(
            f"Black-frame detector: {n_black}/{n_total} frames "
            f"({n_black / n_total:.0%}) are near-black "
            f"(mean brightness < {_BLACK_FRAME_BRIGHTNESS_THRESHOLD}/255) "
            f"in {video_path}"
        )


def _check_black_image(image_path: str) -> None:
    if cv2 is None:
        return
    img = cv2.imread(image_path)
    if img is not None and img.mean() < _BLACK_FRAME_BRIGHTNESS_THRESHOLD:
        raise BlackFrameError(
            f"Black-frame detector: keyframe is near-black "
            f"(mean brightness {img.mean():.1f}/255 < "
            f"{_BLACK_FRAME_BRIGHTNESS_THRESHOLD}) in {image_path}"
        )


def _extract_video_frames(video_path: str, n_frames: int) -> List[np.ndarray]:
    if cv2 is None:
        raise RuntimeError("cv2 (OpenCV) is required for video frame extraction but is not installed.")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video reports 0 frames: {video_path}")

        actual_n = min(n_frames, total_frames)
        if actual_n == 1:
            indices = [total_frames // 2]
        else:
            indices = [
                int(round(i * (total_frames - 1) / (actual_n - 1)))
                for i in range(actual_n)
            ]

        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        if not frames:
            raise RuntimeError(f"Could not read any frames from: {video_path}")
        if len(frames) < actual_n:
            logger.warning(
                "[extract_video_frames] only %d/%d frames read from %s",
                len(frames), actual_n, video_path,
            )
        return frames
    finally:
        cap.release()


def _stitch_frame_pair(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    max_h = max(h1, h2)

    if h1 < max_h:
        pad = np.zeros((max_h - h1, w1, 3), dtype=frame1.dtype)
        frame1 = np.vstack([frame1, pad])
    if h2 < max_h:
        pad = np.zeros((max_h - h2, w2, 3), dtype=frame2.dtype)
        frame2 = np.vstack([frame2, pad])

    return np.hstack([frame1, frame2])


def _frames_to_base64_images(
    frames: List[np.ndarray],
    max_dim: int,
    stitch_pairs: bool = False,
) -> List[str]:
    images_to_encode: List[np.ndarray] = []

    if stitch_pairs:
        for idx in range(0, len(frames) - 1, 2):
            images_to_encode.append(_stitch_frame_pair(frames[idx], frames[idx + 1]))
        if len(frames) % 2 == 1:
            images_to_encode.append(frames[-1])
    else:
        images_to_encode = list(frames)

    b64_images: List[str] = []
    for img in images_to_encode:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            b64_images.append(base64.b64encode(buf.tobytes()).decode("ascii"))

    return b64_images


def resolve_evidence_frames(sample: Any) -> List[str]:
    evidence_type = sample.evidence_type

    if evidence_type == EVIDENCE_KEYFRAME:
        if not sample.image:
            raise ValueError(f"keyframe_single sample has no images: task={sample.task_name}")
        _check_black_image(sample.image[0])
        img_bytes = _resize_image_if_needed(sample.image[0], MAX_DIM_SINGLE)
        return [base64.b64encode(img_bytes).decode("ascii")]

    if evidence_type == EVIDENCE_CLIP:
        if not sample.video:
            raise ValueError(f"video_clip sample has no video: task={sample.task_name}")
        frames = _extract_video_frames(sample.video, CLIP_FRAME_COUNT)
        _check_black_frames(frames, sample.video)
        return _frames_to_base64_images(frames, max_dim=MAX_DIM_SINGLE, stitch_pairs=False)

    if evidence_type == EVIDENCE_PREFIX:
        if not sample.video:
            raise ValueError(f"video_prefix sample has no video: task={sample.task_name}")
        frames = _extract_video_frames(sample.video, PREFIX_FRAME_COUNT)
        _check_black_frames(frames, sample.video)
        return _frames_to_base64_images(frames, max_dim=MAX_DIM_STITCHED, stitch_pairs=True)

    if evidence_type == EVIDENCE_CLIP_PAIR:
        if not sample.video:
            raise ValueError(f"video_clip_pair sample has no video: task={sample.task_name}")
        try:
            paths = json.loads(sample.video)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"video_clip_pair: cannot parse video field: {exc}")
        if not isinstance(paths, list) or len(paths) != 2:
            raise ValueError(f"video_clip_pair expects [path1, path2], got: {sample.video!r}")

        panels: List[str] = []
        for clip_path in paths:
            clip_frames = _extract_video_frames(clip_path, CLIP_PAIR_FRAMES_PER_CLIP)
            _check_black_frames(clip_frames, clip_path)
            panels.extend(
                _frames_to_base64_images(
                    clip_frames,
                    max_dim=MAX_DIM_STITCHED,
                    stitch_pairs=True,
                )
            )
        return panels

    raise ValueError(f"Unknown evidence type: {evidence_type!r}")
