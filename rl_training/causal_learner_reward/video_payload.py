

from __future__ import annotations

import base64
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

try:
    from .cache import VIDEO_PREPROCESS_CACHE, file_fingerprint, stable_hash
    from .config import JudgeAPIConfig
except ImportError:
    from cache import VIDEO_PREPROCESS_CACHE, file_fingerprint, stable_hash                
    from config import JudgeAPIConfig                


class MissingVideoError(RuntimeError):
    pass


class UnsupportedVideoInputError(RuntimeError):
    pass


@dataclass
class VideoPayload:
    video_key: str
    source: str
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    frames_base64: Optional[List[str]] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    preprocess_count: int = 0


def resolve_video_source(extra_info: Mapping[str, Any]) -> Dict[str, Any]:


    video_path = _first_text(extra_info.get("video_path"), extra_info.get("raw_video_path"))
    videos = extra_info.get("videos")
    if not video_path and isinstance(videos, list) and videos:
        if len(videos) > 1:
            raise UnsupportedVideoInputError("multiple_videos_not_supported")
        first = videos[0]
        if isinstance(first, Mapping):
            video_path = _first_text(first.get("video"), first.get("path"), first.get("url"))
        else:
            video_path = _first_text(first)
    return {
        "video_path": video_path,
        "video_data": extra_info.get("video_frames"),
    }


def prepare_video_payload(extra_info: Mapping[str, Any], config: JudgeAPIConfig) -> VideoPayload:
    config.validate()
    source = resolve_video_source(extra_info)
    video_path = source["video_path"]
    video_data = source["video_data"]

    if config.mode == "openai" and config.video_transport == "native_video" and not config.openai_native_video:
        raise UnsupportedVideoInputError("unsupported_native_video_input")

    if config.video_transport == "frame_fallback":
        frames, preprocess_count = _prepare_frames(video_path, video_data, config)
        if not frames:
            raise MissingVideoError("missing_video_frames")
        return VideoPayload(
            video_key=_video_cache_key(video_path, video_data, config, "frames"),
            source="frames",
            video_path=video_path,
            frames_base64=frames,
            preprocess_count=preprocess_count,
        )

    if video_data:
        video_url, mm_kwargs, preprocess_count = _prepare_encoded_video_data(video_path, video_data, config)
        if not video_url:
            raise MissingVideoError("video_data_encoding_failed")
        return VideoPayload(
            video_key=_video_cache_key(video_path, video_data, config, "video_data"),
            source="video_data",
            video_path=video_path,
            video_url=video_url,
            mm_processor_kwargs=mm_kwargs,
            preprocess_count=preprocess_count,
        )

    if video_path:
        video_url, preprocess_count = _prepare_raw_video_url(video_path, config)
        return VideoPayload(
            video_key=_video_cache_key(video_path, None, config, "raw_video"),
            source="raw_video",
            video_path=video_path,
            video_url=video_url,
            mm_processor_kwargs={"fps": config.fps} if config.mode == "vllm" else None,
            preprocess_count=preprocess_count,
        )

    raise MissingVideoError("missing_video")


def _prepare_encoded_video_data(
    video_path: Optional[str],
    video_data: Any,
    config: JudgeAPIConfig,
) -> tuple[Optional[str], Dict[str, Any], int]:
    cache_key = _video_cache_key(video_path, video_data, config, "video_data")
    if config.enable_video_cache:
        cached = VIDEO_PREPROCESS_CACHE.get(cache_key)
        if cached is not None:
            data_uri, mm_kwargs = cached
            return data_uri, mm_kwargs, 0

    encoder = _load_video_data_encoder()
    actual_fps = _actual_sample_fps(video_data) or config.fps
    data_uri = encoder(video_data, target_fps=actual_fps)
    mm_kwargs = {"fps": actual_fps, "do_sample_frames": False}
    result = (data_uri, mm_kwargs)
    if config.enable_video_cache:
        VIDEO_PREPROCESS_CACHE.set(cache_key, result)
    return data_uri, mm_kwargs, 1


def _prepare_raw_video_url(video_path: str, config: JudgeAPIConfig) -> tuple[str, int]:
    normalized = video_path.removeprefix("file://")
    if config.allow_file_url:
        if video_path.startswith("file://"):
            return video_path, 0
        return f"file://{normalized}", 0

    cache_key = _video_cache_key(video_path, None, config, "raw_video")
    if config.enable_video_cache:
        cached = VIDEO_PREPROCESS_CACHE.get(cache_key)
        if cached is not None:
            return cached, 0

    with open(normalized, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    data_uri = f"data:video/mp4;base64,{encoded}"
    if config.enable_video_cache:
        VIDEO_PREPROCESS_CACHE.set(cache_key, data_uri)
    return data_uri, 1


def _prepare_frames(video_path: Optional[str], video_data: Any, config: JudgeAPIConfig) -> tuple[List[str], int]:
    cache_key = _video_cache_key(video_path, video_data, config, "frames")
    if config.enable_video_cache:
        cached = VIDEO_PREPROCESS_CACHE.get(cache_key)
        if cached is not None:
            return cached, 0

    frames: List[str] = []
    if video_data:
        arrays = [item[0] for item in video_data if isinstance(item, tuple) and len(item) == 2]
        frames = _load_numpy_frames_encoder()(arrays, max_frames=config.max_frames_for_judge)
    elif video_path:
        frames = _load_frame_extractor()(video_path, max_frames=config.max_frames_for_judge, fps=config.fps)

    if config.enable_video_cache:
        VIDEO_PREPROCESS_CACHE.set(cache_key, frames)
    return frames, 1 if frames else 0


def _video_cache_key(video_path: Optional[str], video_data: Any, config: JudgeAPIConfig, mode: str) -> str:
    return stable_hash(
        {
            "video_path": file_fingerprint(video_path),
            "video_data": _video_data_fingerprint(video_data) if video_data is not None else None,
            "fps": config.fps,
            "max_frames": config.max_frames_for_judge,
            "video_input_mode": config.video_input_mode,
            "mode": mode,
        }
    )


def _video_data_fingerprint(video_data: Any) -> str:
    parts: List[Any] = []
    for item in video_data or []:
        if isinstance(item, tuple) and len(item) == 2:
            array, metadata = item
            parts.append(
                {
                    "shape": getattr(array, "shape", None),
                    "dtype": str(getattr(array, "dtype", "")),
                    "metadata": metadata,
                }
            )
    return stable_hash(parts)


def _actual_sample_fps(video_data: Any) -> Optional[float]:
    try:
        _frames, metadata = video_data[0]
        original_fps = float(metadata.get("fps", 0.0))
        frame_indices = metadata.get("frames_indices") or []
        total_frames = int(metadata.get("total_num_frames") or 0)
        if original_fps > 0 and frame_indices and total_frames > 0:
            return len(frame_indices) / total_frames * original_fps
    except Exception:
        return None
    return None


def _load_video_data_encoder():
    try:
        from verl.utils.reward_score.video_caption.video_utils import video_data_to_video_base64

        return video_data_to_video_base64
    except Exception:
        _add_video_caption_to_path()
        from video_utils import video_data_to_video_base64                

        return video_data_to_video_base64


def _load_frame_extractor():
    try:
        from verl.utils.reward_score.video_caption.video_utils import extract_frames_to_base64

        return extract_frames_to_base64
    except Exception:
        _add_video_caption_to_path()
        from video_utils import extract_frames_to_base64                

        return extract_frames_to_base64


def _load_numpy_frames_encoder():
    try:
        from verl.utils.reward_score.video_caption.video_utils import numpy_frames_to_base64

        return numpy_frames_to_base64
    except Exception:
        _add_video_caption_to_path()
        from video_utils import numpy_frames_to_base64                

        return numpy_frames_to_base64


def _add_video_caption_to_path() -> None:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    video_caption_dir = os.path.abspath(os.path.join(module_dir, os.pardir, "video_caption"))
    if video_caption_dir not in sys.path:
        sys.path.insert(0, video_caption_dir)


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
