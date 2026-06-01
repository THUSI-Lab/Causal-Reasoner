from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

from azure_openai_client import build_request_payload_input, call_model_api, create_api_client, ENDPOINT, MODEL, DEPLOYMENT, API_PROFILE

import numpy as np

try:
    import cv2
except ImportError:                    
    cv2 = None


logger = logging.getLogger("four_stage")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


DEFAULT_MAX_FRAMES = 50
MAX_API_IMAGES_PER_REQUEST = 50
VIDEO_EXTS = (".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm")


def format_duration(seconds: float) -> str:

    try:
        total = float(seconds)
    except Exception:
        return str(seconds)
    if total < 0:
        total = 0.0
    if total < 60:
        return f"{total:.1f}s"
    total_int = int(round(total))
    s = total_int % 60
    m = (total_int // 60) % 60
    h = (total_int // 3600) % 24
    d = total_int // 86400
    parts: List[str] = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return "".join(parts)


@dataclass
class ApiConfig:
    api_key: str = os.environ.get("API_KEY", "")
    api_base_url: str = os.environ.get("API_BASE_URL", ENDPOINT)
    model_provider_id: str = os.environ.get("MODEL_PROVIDER_ID", "azure")
    model_name: str = os.environ.get("MODEL_NAME", MODEL)
    max_tokens: int = int(os.environ.get("MAX_TOKENS", "32000"))
    temperature: float = float(os.environ.get("TEMPERATURE", "0.8"))
    embed_index_on_api_images: bool = os.environ.get("EMBED_INDEX_ON_API_IMAGES", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    verbose: bool = os.environ.get("VERBOSE_LOGGING", "0").strip().lower() in {"1", "true", "yes", "y"}
    api_call_retries: int = int(os.environ.get("API_CALL_RETRIES", "3"))
    api_call_retry_backoff_sec: float = float(os.environ.get("API_CALL_RETRY_BACKOFF_SEC", "1.0"))
    api_call_timeout_sec: float = float(os.environ.get("API_CALL_TIMEOUT_SEC", "420"))


def add_api_cli_args(parser: Any, *, include_no_embed_index: bool = True) -> None:
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""), help="API key (ignored for Azure AD auth; kept for CLI compatibility).")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("API_BASE_URL", ENDPOINT),
        help="Azure endpoint, or set env: API_BASE_URL/AZURE_OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("MODEL_PROVIDER_ID", "azure"),
        help="Provider argument kept for CLI compatibility; ignored by Azure adapter.",
    )
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", MODEL), help="Model name (deployment name).")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "32000")), help="Max tokens.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", "0.8")),
        help="Sampling temperature (lower is usually more stable).",
    )
    parser.add_argument(
        "--api-call-retries",
        type=int,
        default=int(os.environ.get("API_CALL_RETRIES", "3")),
        help="Retries for a single model call (transport/service errors).",
    )
    parser.add_argument(
        "--api-call-retry-backoff-sec",
        type=float,
        default=float(os.environ.get("API_CALL_RETRY_BACKOFF_SEC", "1.0")),
        help="Backoff seconds for model-call retries (exponential).",
    )
    parser.add_argument(
        "--api-call-timeout-sec",
        type=float,
        default=float(os.environ.get("API_CALL_TIMEOUT_SEC", "420")),
        help="Timeout seconds for a single model call.",
    )
    if include_no_embed_index:
        parser.add_argument(
            "--no-embed-index",
            action="store_true",
            help="Do not draw 1-based frame indices onto images (will use text labels instead).",
        )
    parser.add_argument("--verbose", action="store_true", help="Log raw model outputs.")


def api_config_from_args(args: Any) -> ApiConfig:
    return ApiConfig(
        api_key=getattr(args, "api_key"),
        api_base_url=getattr(args, "api_base"),
        model_provider_id=getattr(args, "provider"),
        model_name=getattr(args, "model"),
        max_tokens=int(getattr(args, "max_tokens")),
        temperature=float(getattr(args, "temperature")),
        api_call_retries=int(getattr(args, "api_call_retries")),
        api_call_retry_backoff_sec=float(getattr(args, "api_call_retry_backoff_sec")),
        api_call_timeout_sec=float(getattr(args, "api_call_timeout_sec")),
        embed_index_on_api_images=not bool(getattr(args, "no_embed_index", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


@dataclass
class SamplingConfig:
    max_frames: int = DEFAULT_MAX_FRAMES
    resize_dimension: Optional[Tuple[int, int]] = None
    jpeg_quality: int = 95


def _parse_resize_dimension(value: Any) -> Optional[Tuple[int, int]]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    m = re.fullmatch(r"(\d+)\s*[x,]\s*(\d+)", text)
    if not m:
        raise ValueError(f"Invalid resize dimension: {value!r} (expected WIDTHxHEIGHT)")
    width = int(m.group(1))
    height = int(m.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"Resize dimensions must be positive: {value!r}")
    return (width, height)


def add_sampling_cli_args(parser: Any, *, default_max_frames: int = 100, default_jpeg_quality: int = 95) -> None:
    parser.add_argument("--max-frames", type=int, default=int(default_max_frames), help="Max frames sampled per pool.")
    parser.add_argument("--jpeg-quality", type=int, default=int(default_jpeg_quality), help="JPEG quality (1-100).")
    parser.add_argument(
        "--resize-dimension",
        default=os.environ.get("SAMPLING_RESIZE_DIM", ""),
        help="Optional resize for sampled frames before JPEG encode, format WIDTHxHEIGHT (e.g. 768x768).",
    )


def sampling_config_from_args(args: Any) -> SamplingConfig:
    return SamplingConfig(
        max_frames=int(getattr(args, "max_frames")),
        resize_dimension=_parse_resize_dimension(getattr(args, "resize_dimension", "")),
        jpeg_quality=int(getattr(args, "jpeg_quality")),
    )


def default_output_root() -> str:
    return os.path.join(os.path.dirname(__file__), "causal_spafa_plan_dataset_long")


def video_id_from_path(video_path: str) -> str:
    base = os.path.basename(video_path)
    name, _ = os.path.splitext(base)
    return name


def sanitize_filename(text: str, *, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", (text or "")).strip().lower()
    text = re.sub(r"[-\s]+", "_", text)
    text = text or "unnamed"


    if len(text) > max_len:
        truncated = text[:max_len]

        last_sep = truncated.rfind("_")
        if last_sep > max_len // 2:
            truncated = truncated[:last_sep]
        text = truncated.rstrip("_")
    return text or "unnamed"


def _truncate_utf8_to_max_bytes(text: str, max_bytes: int) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    if max_bytes <= 0:
        return ""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    raw = raw[:max_bytes]
    while raw:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raw = raw[:-1]
    return ""


def stage3_step_folder_basename(step_id: int, step_goal: str, *, max_bytes: int = 220, hash_len: int = 10) -> str:

    prefix = f"{int(step_id):02d}_"
    slug = sanitize_filename(step_goal)
    base = prefix + slug
    if len(base.encode("utf-8")) <= int(max_bytes):
        return base

    digest = hashlib.sha1(str(step_goal or "").encode("utf-8")).hexdigest()[: int(hash_len)]
    suffix = f"__{digest}"
    budget = int(max_bytes) - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
    truncated = _truncate_utf8_to_max_bytes(slug, budget).rstrip("_")
    if not truncated:
        truncated = "unnamed"
    out = prefix + truncated + suffix
    if len(out.encode("utf-8")) <= int(max_bytes):
        return out


    budget2 = int(max_bytes) - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
    truncated2 = _truncate_utf8_to_max_bytes("unnamed", budget2) or "unnamed"
    return prefix + truncated2 + suffix


_PLACEHOLDER_STRINGS = {
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "unspecified",
    "tbd",
    "todo",
    "-",
    "...",
}


def _is_placeholder_str(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = re.sub(r"\s+", " ", s.strip().lower())
    return t in _PLACEHOLDER_STRINGS


_FRAME_REF_RE = re.compile(r"\b(frame|image|img|picture)\s*\d+\b", re.IGNORECASE)
_FRAME_REF_ORDINAL_RE = re.compile(
    r"\b(initial|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|beginning|ending)\s+"
    r"(frame|image|img|picture)\b",
    re.IGNORECASE,
)


def _contains_frame_ref(text: Any) -> bool:
    s = str(text or "")
    return bool(_FRAME_REF_RE.search(s) or _FRAME_REF_ORDINAL_RE.search(s))


_TIME_REF_RE = re.compile(
    r"(?:"

    r"\bt\s*=\s*\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|ms|msec|milliseconds?)\b"
    r"|"

    r"\b\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|ms|msec|milliseconds?)\b"
    r"|"

    r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b"
    r")",
    re.IGNORECASE,
)


def _contains_time_ref(text: Any) -> bool:
    s = str(text or "")
    return bool(_TIME_REF_RE.search(s))


def _text_dedupe_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _dedupe_keep_order(items: List[str], *, key_fn: Optional[Callable[[str], str]] = None) -> List[str]:
    if not items:
        return []
    if key_fn is None:
        key_fn = lambda x: x              
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        k = key_fn(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def initialize_api_client(cfg: ApiConfig) -> Any:
    try:
        return create_api_client(cfg.api_base_url, cfg.api_key)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Install with: pip install openai azure-identity")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {e}")
        return None


def extract_json_from_response(response_text: str) -> str:
    if not isinstance(response_text, str):
        raise ValueError("Response was not a string.")

    match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text)
    if match:
        return match.group(1).strip()

    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response_text[start : end + 1].strip()

    raise ValueError("Could not find a valid JSON object in the response.")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_UNDERSCORE_SKIP_KEYS = frozenset({
    "clip_relpath", "video_id", "source_video",
    "keyframe_image_path", "start_image_relpath", "end_image_relpath",
})


def strip_underscores_from_values(
    obj: Any,
    _skip_keys: frozenset = _UNDERSCORE_SKIP_KEYS,
    _parent_key: str = "",
) -> Any:

    if isinstance(obj, dict):
        return {
            k: (v if k in _skip_keys
                 else strip_underscores_from_values(v, _skip_keys, _parent_key=k))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [strip_underscores_from_values(v, _skip_keys, _parent_key) for v in obj]
    if isinstance(obj, str):
        return obj.replace("_", " ")
    return obj


def write_json(path: str, data: Any) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=dir_path or ".",
            prefix=os.path.basename(path) + ".tmp.",
            delete=False,
        ) as f:
            tmp_path = f.name
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def write_text(path: str, text: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=dir_path or ".",
            prefix=os.path.basename(path) + ".tmp.",
            delete=False,
        ) as f:
            tmp_path = f.name
            f.write(text or "")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


class OutputDirCollisionError(RuntimeError):
    pass


def same_source_video(path_a: str, path_b: str) -> bool:

    a = str(path_a or "").strip()
    b = str(path_b or "").strip()
    if not a or not b:
        return False

    try:
        if os.path.abspath(a) == os.path.abspath(b):
            return True
    except Exception:
        pass
    try:
        if os.path.realpath(a) == os.path.realpath(b):
            return True
    except Exception:
        pass
    try:
        if os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b):
            return True
    except Exception:
        pass
    return False


def ensure_video_out_dir_safe(video_out: str, video_path: str) -> None:

    if not os.path.exists(video_out):
        return
    try:
        entries = [x for x in os.listdir(video_out) if x not in {".", ".."}]
    except Exception:
        return
    if not entries:
        return

    run_summary_path = os.path.join(video_out, "run_summary.json")
    if os.path.exists(run_summary_path):
        try:
            rs = read_json(run_summary_path)
        except Exception:
            return
        src = rs.get("source_video")
        if isinstance(src, str) and src.strip() and same_source_video(video_path, src):
            return

        try:
            update_run_summary(run_summary_path, {"source_video": os.path.abspath(video_path)})
        except Exception:
            pass
        return


    return


def build_retry_prefix(errors: List[str], prev_output: str) -> str:
    err_text = "\n".join(f"- {e}" for e in (errors or [])[:50])
    prev = (prev_output or "")[:12000]
    return (
        "Your previous output was invalid and failed strict validation.\n"
        "Fix ALL errors and return ONLY the corrected strict JSON.\n\n"
        f"Validation errors:\n{err_text}\n\n"
        "Previous output (for reference; correct it):\n"
        f"{prev}\n\n"
    )


def collect_videos(input_dir: str, exts: Tuple[str, ...]) -> List[str]:

    paths: List[str] = []
    seen_dirs: set[str] = set()
    seen_files: set[str] = set()
    walk_errors: List[str] = []

    def _onerror(e: OSError) -> None:
        walk_errors.append(f"{type(e).__name__}: {e}")

    for root, dirnames, filenames in os.walk(input_dir, followlinks=True, onerror=_onerror):

        real_root = os.path.realpath(root)
        if real_root in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(real_root)



        chosen: Dict[str, Tuple[bool, str]] = {}
        for d in dirnames:
            if str(d).startswith("."):
                continue
            full = os.path.join(root, d)
            try:
                real_d = os.path.realpath(full)
            except Exception:
                continue
            if real_d in seen_dirs:
                continue
            is_link = bool(os.path.islink(full))
            rank = (is_link, str(d))
            if real_d not in chosen or rank < chosen[real_d]:
                chosen[real_d] = rank
        dirnames[:] = [rank[1] for rank in chosen.values()]
        dirnames.sort()
        for name in sorted(filenames):
            if not str(name).lower().endswith(exts):
                continue
            p = os.path.join(root, name)
            if os.path.isfile(p):

                try:
                    real_p = os.path.realpath(p)
                except Exception:
                    real_p = p
                if real_p in seen_files:
                    continue
                seen_files.add(real_p)
                paths.append(p)

    if walk_errors:
        sample = " | ".join(walk_errors[:3])
        more = "" if len(walk_errors) <= 3 else f" | ... (+{len(walk_errors) - 3})"
        logger.warning(
            "[collect_videos] Filesystem errors while scanning "
            f"{os.path.abspath(input_dir)} (errors={len(walk_errors)}): {sample}{more}"
        )
    return paths


def can_open_video(video_path: str) -> bool:

    if not isinstance(video_path, str) or not video_path:
        return False
    if not os.path.exists(video_path):
        return False
    try:
        if os.path.getsize(video_path) <= 0:
            return False
    except Exception:
        return False

    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return False

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if total_frames <= 0 or fps <= 0:
                return False
            return True
        except Exception:
            return False
        finally:
            try:
                cap.release()
            except Exception:
                pass


    lower = video_path.lower()
    if lower.endswith((".mp4", ".mov", ".m4v")):
        try:
            size = int(os.path.getsize(video_path))
            chunk = min(2 * 1024 * 1024, size)
            with open(video_path, "rb") as f:
                head = f.read(chunk)
                tail = b""
                if size > chunk:
                    try:
                        f.seek(max(0, size - chunk), os.SEEK_SET)
                        tail = f.read(chunk)
                    except Exception:
                        tail = b""
            blob = head + tail
            if b"ftyp" not in blob:
                return False
            if b"moov" not in blob:
                return False
            return True
        except Exception:
            return False


    return True


def sample_video_to_frames(
    video_path: str,
    sampling: SamplingConfig,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required to sample frames.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_dimensions = (width, height)
        if total_frames <= 0 or fps <= 0:
            raise RuntimeError(f"Video has invalid metadata (frames={total_frames}, fps={fps}).")

        if sampling.max_frames <= 0:
            raise ValueError(f"max_frames must be positive (got {sampling.max_frames})")
        if sampling.max_frames == 1:
            indices = [0]
        else:

            denom = float(sampling.max_frames - 1)
            indices = [int(round(i * (total_frames - 1) / denom)) for i in range(sampling.max_frames)]
        indices = [min(max(0, idx), total_frames - 1) for idx in indices]
        frames: List[Dict[str, Any]] = []
        last_good: Optional[Dict[str, Any]] = None
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_good is None:
                    continue



                frames.append({**last_good})
                continue


            if np.mean(frame) < 15:
                if last_good is None:
                    continue
                frames.append({**last_good})
                continue

            if sampling.resize_dimension:
                frame = cv2.resize(frame, sampling.resize_dimension)
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(sampling.jpeg_quality)])
            if not ok2:
                if last_good is None:
                    continue
                frames.append({**last_good})
                continue

            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            last_good = {"base64": b64, "timestamp_sec": float(frame_idx) / fps, "original_frame_index": int(frame_idx)}
            frames.append({**last_good})

        if len(frames) != sampling.max_frames and frames:

            while len(frames) < sampling.max_frames:
                frames.append({**frames[-1]})
            frames = frames[: sampling.max_frames]

        if len(frames) != sampling.max_frames:
            raise RuntimeError(f"Failed to sample {sampling.max_frames} frames from {video_path} (got {len(frames)}).")
        return frames, original_dimensions
    finally:
        cap.release()


def sample_frames_around_timestamp(
    video_path: str,
    center_sec: float,
    window_sec: float = 2.0,
    num_dense_frames: int = 20,
    jpeg_quality: int = 95,
    resize_dimension: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Dict[str, Any]], List[float]]:

    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required to sample frames.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0 or fps <= 0:
            raise RuntimeError(f"Video has invalid metadata (frames={total_frames}, fps={fps}).")
        duration = total_frames / fps

        start_sec = max(0.0, center_sec - window_sec)
        end_sec = min(duration, center_sec + window_sec)
        if end_sec <= start_sec:
            end_sec = min(duration, start_sec + 0.5)

        start_frame = int(start_sec * fps)
        end_frame = min(int(end_sec * fps), total_frames - 1)
        if end_frame <= start_frame:
            end_frame = min(start_frame + 1, total_frames - 1)

        if num_dense_frames <= 1:
            indices = [start_frame]
        else:
            denom = float(num_dense_frames - 1)
            indices = [
                int(round(start_frame + i * (end_frame - start_frame) / denom))
                for i in range(num_dense_frames)
            ]
        indices = [min(max(0, idx), total_frames - 1) for idx in indices]

        frames: List[Dict[str, Any]] = []
        timestamps: List[float] = []
        for seq, frame_idx in enumerate(indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if np.mean(frame) < 15:
                continue
            ts = frame_idx / fps
            if resize_dimension:
                frame = cv2.resize(frame, resize_dimension, interpolation=cv2.INTER_AREA)
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            frames.append({
                "frame_index_1based": seq,
                "base64": b64,
                "timestamp_sec": round(ts, 4),
            })
            timestamps.append(ts)
        return frames, timestamps
    finally:
        cap.release()


def save_sampled_frames_jpegs(frames: List[Dict[str, Any]], output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    rel_paths: List[str] = []
    for i, fr in enumerate(frames, start=1):
        ts = float(fr.get("timestamp_sec", 0.0))
        name = f"sample_{i:03d}_ts_{ts:.2f}s.jpg"
        path = os.path.join(output_dir, name)
        data = base64.b64decode(fr["base64"]) if isinstance(fr.get("base64"), str) else None
        if not data:
            raise RuntimeError(f"Missing base64 for sampled frame {i}.")
        with open(path, "wb") as f:
            f.write(data)
        rel_paths.append(name)
    return rel_paths


def write_frame_manifest(
    frames: List[Dict[str, Any]],
    sampled_frames_dir: str,
    manifest_path: str,
) -> Dict[str, Any]:
    manifest_dir = os.path.dirname(manifest_path)
    os.makedirs(manifest_dir, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames, start=1):
        ts = float(fr.get("timestamp_sec", 0.0))
        name = f"sample_{i:03d}_ts_{ts:.2f}s.jpg"
        abs_img = os.path.join(sampled_frames_dir, name)
        rel_img = os.path.relpath(abs_img, manifest_dir)
        entries.append(
            {
                "frame_index_1based": i,
                "timestamp_sec": ts,
                "original_frame_index": int(fr.get("original_frame_index", -1)),
                "image_relpath": rel_img,
            }
        )

    manifest = {
        "num_frames": len(entries),
        "note": "frame_index_1based is the 1-based index used in prompts and model outputs for this frame pool.",
        "frames": entries,
    }
    write_json(manifest_path, manifest)
    return manifest


def load_frames_from_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    manifest = read_json(manifest_path)
    base_dir = os.path.dirname(manifest_path)
    frames: List[Dict[str, Any]] = []
    for entry in manifest.get("frames", []):
        rel = entry.get("image_relpath")
        if not rel:
            continue
        img_path = os.path.join(base_dir, rel)
        with open(img_path, "rb") as f:
            b = f.read()
        frames.append(
            {
                "base64": base64.b64encode(b).decode("utf-8"),
                "timestamp_sec": float(entry.get("timestamp_sec", 0.0)),
                "original_frame_index": int(entry.get("original_frame_index", -1)),
            }
        )
    if len(frames) != int(manifest.get("num_frames", len(frames))):
        logger.warning("Manifest frame count mismatch; continuing with loaded frames.")
    return frames


def build_index_manifest_text(frames: List[Dict[str, Any]]) -> str:
    lines = ["Frame Index Manifest (1-based):"]
    for i, fr in enumerate(frames, start=1):
        ts = float(fr.get("timestamp_sec", 0.0))
        lines.append(f"- Frame {i}: t={ts:.2f}s")
    return "\n".join(lines)


def _overlay_index_on_base64_image(b64_img: str, index_1based: int) -> str:
    if cv2 is None:
        return b64_img
    try:
        import numpy as np

        data = base64.b64decode(b64_img)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return b64_img


        text = f"Frame {index_1based:02d}"
        cv2.putText(img, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return b64_img
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return b64_img


def _decode_base64_image(b64_img: str) -> Any:
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required to pack API images.")
    import numpy as np

    data = base64.b64decode(b64_img)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode a sampled frame for API packing.")
    return img


def _encode_image_to_base64(img: Any, *, jpeg_quality: int = 95) -> str:
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required to pack API images.")
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode a packed API panel image.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _pack_frame_group_to_panel(
    frame_group: List[Dict[str, Any]],
    *,
    start_index_1based: int,
    embed_index: bool,
) -> str:
    if not frame_group:
        raise ValueError("frame_group must be non-empty.")

    if len(frame_group) == 1:
        b64 = frame_group[0].get("base64")
        if not isinstance(b64, str) or not b64:
            raise RuntimeError(f"Missing base64 for API-packed frame {start_index_1based}.")
        if embed_index:
            return _overlay_index_on_base64_image(b64, start_index_1based)
        return b64

    import numpy as np

    decoded: List[Any] = []
    for offset, fr in enumerate(frame_group):
        idx = start_index_1based + offset
        b64 = fr.get("base64")
        if not isinstance(b64, str) or not b64:
            raise RuntimeError(f"Missing base64 for API-packed frame {idx}.")
        if embed_index:
            b64 = _overlay_index_on_base64_image(b64, idx)
        decoded.append(_decode_base64_image(b64))

    tile_h = max(int(img.shape[0]) for img in decoded)
    tile_w = max(int(img.shape[1]) for img in decoded)
    gap_px = 8
    panel_w = tile_w * len(decoded) + gap_px * (len(decoded) - 1)
    panel = np.full((tile_h, panel_w, 3), 255, dtype=np.uint8)

    x = 0
    for img in decoded:
        if tuple(img.shape[:2]) != (tile_h, tile_w):
            img = cv2.resize(img, (tile_w, tile_h))
        panel[:, x : x + tile_w] = img
        x += tile_w + gap_px

    return _encode_image_to_base64(panel)


def _format_frame_group_label(
    frame_indices_1based: List[int],
    *,
    label_prefix: str,
    label_numbers: bool,
) -> str:
    if not label_numbers:
        return str(label_prefix)
    if not frame_indices_1based:
        return str(label_prefix)
    if len(frame_indices_1based) == 1:
        return f"{label_prefix} {frame_indices_1based[0]}"

    plural = str(label_prefix)
    if not plural.endswith("s"):
        plural += "s"
    start = int(frame_indices_1based[0])
    end = int(frame_indices_1based[-1])
    contiguous = frame_indices_1based == list(range(start, end + 1))
    if contiguous:
        return f"{plural} {start}-{end} (left-to-right in the next image)"
    joined = ", ".join(str(int(x)) for x in frame_indices_1based)
    return f"{plural} {joined} (left-to-right in the next image)"


def build_api_content(
    frames: List[Dict[str, Any]],
    embed_index: bool,
    *,
    include_manifest: bool = True,
    include_frame_labels: bool = True,
    label_prefix: str = "Frame",
    label_numbers: bool = True,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if include_manifest:
        content.append({"type": "text", "text": build_index_manifest_text(frames)})

    if len(frames) > MAX_API_IMAGES_PER_REQUEST:
        chunk_size = max(1, (len(frames) + MAX_API_IMAGES_PER_REQUEST - 1) // MAX_API_IMAGES_PER_REQUEST)
        frame_groups = [frames[i : i + chunk_size] for i in range(0, len(frames), chunk_size)]
        logger.info(
            f"[api] Packing {len(frames)} sampled frames into {len(frame_groups)} panel images "
            f"(chunk_size={chunk_size}, limit={MAX_API_IMAGES_PER_REQUEST})"
        )
        content.append(
            {
                "type": "text",
                "text": (
                    "API image packing note: some provided images are multi-frame panels to stay within the "
                    f"service limit of {MAX_API_IMAGES_PER_REQUEST} images per request. Within each panel, "
                    "subframes are ordered left-to-right in chronological order."
                ),
            }
        )
        for start in range(0, len(frames), chunk_size):
            group = frames[start : start + chunk_size]
            indices = list(range(start + 1, start + 1 + len(group)))
            if include_frame_labels:
                content.append(
                    {
                        "type": "text",
                        "text": _format_frame_group_label(
                            indices,
                            label_prefix=label_prefix,
                            label_numbers=label_numbers,
                        ),
                    }
                )
            b64 = _pack_frame_group_to_panel(
                group,
                start_index_1based=start + 1,
                embed_index=embed_index,
            )
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return content

    for i, fr in enumerate(frames, start=1):
        b64 = fr.get("base64")
        if embed_index and isinstance(b64, str):
            b64 = _overlay_index_on_base64_image(b64, i)
        if include_frame_labels:
            if label_numbers:
                content.append({"type": "text", "text": f"{label_prefix} {i}"})
            else:
                content.append({"type": "text", "text": str(label_prefix)})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def cut_video_segment_ffmpeg(
    ffmpeg_bin: str,
    src_video: str,
    start_sec: float,
    end_sec: float,
    dst_video: str,
    overwrite: bool,
    *,
    mode: str = "reencode",
    seek_slop_sec: float = 1.0,
    crf: int = 18,
    preset: str = "veryfast",
    keep_audio: bool = False,
) -> None:
    duration = float(end_sec) - float(start_sec)
    if duration <= 0:
        raise ValueError(f"Non-positive clip duration: start={start_sec}, end={end_sec}")

    out_dir = os.path.dirname(dst_video)
    os.makedirs(out_dir, exist_ok=True)



    if not overwrite and os.path.exists(dst_video):
        raise FileExistsError(dst_video)
    suffix = os.path.splitext(dst_video)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(prefix=".tmp_cut_", suffix=suffix, dir=out_dir, delete=False)
    tmp_path = tmp.name
    tmp.close()





    mode = (mode or "").strip().lower()
    if mode not in {"copy", "reencode"}:
        raise ValueError(f"Unknown cut mode: {mode} (expected 'copy' or 'reencode')")

    if mode == "copy":
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            src_video,
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            tmp_path,
        ]
    else:
        pre = max(0.0, float(start_sec) - float(seek_slop_sec))
        post = float(start_sec) - pre
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{pre:.3f}",
            "-i",
            src_video,
            "-ss",
            f"{post:.3f}",
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        if keep_audio:

            cmd += ["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd.append(tmp_path)


    cmd.insert(1, "-y")
    logger.info("[ffmpeg] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) <= 0:
            raise RuntimeError(f"ffmpeg produced an empty clip: {tmp_path}")
        os.replace(tmp_path, dst_video)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"ffmpeg binary not found: '{ffmpeg_bin}'. Install ffmpeg or pass a valid path via --ffmpeg-bin."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed (exit={e.returncode}). Command: " + " ".join(cmd)) from e
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def update_run_summary(path: str, updates: Dict[str, Any]) -> None:
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            data = read_json(path)
        except Exception:
            data = {}
    data.update(updates)
    write_json(path, data)


def normalize_draft_plan(plan: Any) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []

    if not isinstance(plan, dict):
        warnings.append(f"Top-level JSON must be an object; got {type(plan).__name__}.")
        plan = {}

    def _norm_str(v: Any) -> str:
        s = str(v).strip() if v is not None else ""
        return "" if _is_placeholder_str(s) else s

    def _norm_identifier(v: Any) -> str:
        s = _norm_str(v)
        if not s:
            return ""
        ident = sanitize_filename(s)
        return "" if ident == "unnamed" else ident

    def _norm_identifier_list(v: Any) -> List[str]:
        if isinstance(v, str):
            one = _norm_identifier(v)
            return [one] if one else []
        if not isinstance(v, list):
            return []
        out_list: List[str] = []
        for x in v:
            s = _norm_identifier(x)
            if s:
                out_list.append(s)
        return _dedupe_keep_order(out_list)

    def _parse_bool(v: Any) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            t = v.strip().lower()
            if t in {"true", "t", "yes", "y", "1"}:
                return True
            if t in {"false", "f", "no", "n", "0"}:
                return False
        if isinstance(v, (int, float)) and v in {0, 1}:
            return bool(v)
        return None

    _LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•])\s+")
    _LEADING_NUMBER_RE = re.compile(r"^\s*\d+\s*[\.\)、\)]\s*")

    def _normalize_statement_list(text: str) -> List[str]:

        raw = (text or "").strip()
        if not raw:
            return []


        raw = raw.replace("\\n", "\n")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            lines = [raw]

        normalized: List[str] = []
        for line in lines:
            line = _LEADING_LIST_MARKER_RE.sub("", line)
            line = _LEADING_NUMBER_RE.sub("", line)
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            if "\n" in line:
                line = " ".join([x.strip() for x in line.splitlines() if x.strip()])
            if not line.endswith("."):
                line = f"{line}."
            normalized.append(line)
        return _dedupe_keep_order(normalized, key_fn=_text_dedupe_key)

    def _spatial_item_to_statement(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            relation = _norm_identifier(item.get("relation", "")) or "unspecified_relation"
            objects = item.get("objects", [])
            if isinstance(objects, str):
                objects_list = _norm_identifier_list([objects])
            else:
                objects_list = _norm_identifier_list(objects) if isinstance(objects, list) else []
            truth = _parse_bool(item.get("truth", True))
            if truth is None:
                truth = True
            if objects_list:
                objs = ", ".join(objects_list)
                return f"Relation '{relation}' {'holds' if truth else 'does not hold'} between {objs}."
            return f"Relation '{relation}' {'holds' if truth else 'does not hold'}."
        return str(item or "").strip()

    def _affordance_item_to_statement(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            object_name = _norm_identifier(item.get("object_name", "")) or "unspecified_object"
            affordance_types = item.get("affordance_types", [])
            if isinstance(affordance_types, str):
                affordance_list = _norm_identifier_list([affordance_types])
            else:
                affordance_list = _norm_identifier_list(affordance_types) if isinstance(affordance_types, list) else []
            reasons = _norm_str(item.get("reasons", ""))
            aff = ", ".join(affordance_list) if affordance_list else "unspecified_affordance"
            if reasons:
                return f"The object {object_name} has affordance/state {aff}. {reasons}"
            return f"The object {object_name} has affordance/state {aff}."
        return str(item or "").strip()

    def _normalize_causal_text(v: Any, *, kind: str) -> List[str]:

        if isinstance(v, list):
            out_list: List[str] = []
            for item in v:
                if isinstance(item, str):
                    out_list.extend(_normalize_statement_list(item))
                    continue
                text = _spatial_item_to_statement(item) if kind == "spatial" else _affordance_item_to_statement(item)
                out_list.extend(_normalize_statement_list(text))
            return _dedupe_keep_order(out_list, key_fn=_text_dedupe_key)
        if isinstance(v, dict):
            text = _spatial_item_to_statement(v) if kind == "spatial" else _affordance_item_to_statement(v)
            return _normalize_statement_list(text)
        return _normalize_statement_list(_norm_str(v))

    def _norm_causal_chain(v: Any) -> Dict[str, Any]:
        d = v if isinstance(v, dict) else {}
        return {
            "agent": _norm_identifier(d.get("agent", "")),
            "action": _norm_str(d.get("action", "")),
            "patient": _norm_identifier(d.get("patient", "")),
            "causal_precondition_on_spatial": _normalize_causal_text(d.get("causal_precondition_on_spatial"), kind="spatial"),
            "causal_precondition_on_affordance": _normalize_causal_text(
                d.get("causal_precondition_on_affordance"), kind="affordance"
            ),
            "causal_effect_on_spatial": _normalize_causal_text(d.get("causal_effect_on_spatial"), kind="spatial"),
            "causal_effect_on_affordance": _normalize_causal_text(d.get("causal_effect_on_affordance"), kind="affordance"),
        }

    out: Dict[str, Any] = {
        "high_level_goal": _norm_str(plan.get("high_level_goal", "")),
        "steps": [],
    }
    if not out["high_level_goal"]:
        warnings.append("Missing/empty high_level_goal.")

    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        steps = []
        warnings.append("Top-level 'steps' is not a list; replaced with empty list.")


    non_object_steps = sum(1 for s in steps if not isinstance(s, dict))
    if non_object_steps:
        warnings.append(f"Found {non_object_steps} non-object step entries; they will be skipped.")

    dict_steps = [s for s in steps if isinstance(s, dict)]
    parsed_step_ids: List[int] = []
    step_ids_ok = bool(dict_steps)
    for s in dict_steps:
        try:
            sid = int(s.get("step_id"))
        except Exception:
            step_ids_ok = False
            break
        if sid <= 0:
            step_ids_ok = False
            break
        parsed_step_ids.append(sid)
    if (
        step_ids_ok
        and len(set(parsed_step_ids)) == len(parsed_step_ids)
        and set(parsed_step_ids) == set(range(1, len(parsed_step_ids) + 1))
    ):
        if parsed_step_ids != sorted(parsed_step_ids):
            warnings.append("Reordered steps by provided step_id (1..N) to preserve chronology.")
        steps = sorted(dict_steps, key=lambda x: int(x.get("step_id")))

    seen_goals: Dict[str, int] = {}
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            warnings.append(f"Step #{idx} is not an object; skipped.")
            continue
        for forbidden in ("critical_frames", "frame_index", "interaction", "keyframe_image_path"):
            if forbidden in step:
                warnings.append(f"Removed unexpected '{forbidden}' in step_id={step.get('step_id')}.")
                step.pop(forbidden, None)

        step_goal = _norm_str(step.get("step_goal", ""))
        if not step_goal:
            step_goal = f"unnamed_step_{idx:02d}"
            warnings.append(f"Empty step_goal at step #{idx}; replaced with '{step_goal}'.")
        if step_goal in seen_goals:
            warnings.append(f"Duplicate step_goal detected: '{step_goal}' (first at step {seen_goals[step_goal]}).")
        else:
            seen_goals[step_goal] = idx

        cc = _norm_causal_chain(step.get("causal_chain"))
        if not cc.get("agent") or not cc.get("action") or not cc.get("patient"):
            warnings.append(f"Missing/empty causal_chain agent/action/patient at step #{idx}.")
        if not cc.get("causal_precondition_on_spatial"):
            warnings.append(f"Missing/empty causal_chain.causal_precondition_on_spatial at step #{idx}.")
        if not cc.get("causal_precondition_on_affordance"):
            warnings.append(f"Missing/empty causal_chain.causal_precondition_on_affordance at step #{idx}.")
        if not cc.get("causal_effect_on_spatial"):
            warnings.append(f"Missing/empty causal_chain.causal_effect_on_spatial at step #{idx}.")
        if not cc.get("causal_effect_on_affordance"):
            warnings.append(f"Missing/empty causal_chain.causal_effect_on_affordance at step #{idx}.")

        counterfactual_q = _norm_str(step.get("counterfactual_challenge_question", ""))
        if not counterfactual_q:
            warnings.append(f"Missing/empty counterfactual_challenge_question at step #{idx}.")

        expected_outcome = _norm_str(step.get("expected_challenge_outcome", ""))
        if not expected_outcome:
            warnings.append(f"Missing/empty expected_challenge_outcome at step #{idx}.")

        fr = step.get("failure_reflecting")
        if not isinstance(fr, dict):
            fr = {"reason": "", "recovery_strategy": ""}
        fr_reason = _norm_str(fr.get("reason", ""))
        fr_recovery = _norm_str(fr.get("recovery_strategy", ""))
        if not fr_reason:
            warnings.append(f"Missing/empty failure_reflecting.reason at step #{idx}.")
        if not fr_recovery:
            warnings.append(f"Missing/empty failure_reflecting.recovery_strategy at step #{idx}.")

        out["steps"].append(
            {

                "step_id": idx,
                "step_goal": step_goal,
                "rationale": _norm_str(step.get("rationale", "")),
                "causal_chain": cc,
                "counterfactual_challenge_question": counterfactual_q,
                "expected_challenge_outcome": expected_outcome,
                "failure_reflecting": {"reason": fr_reason, "recovery_strategy": fr_recovery},
            }
        )

    if not out["steps"]:
        warnings.append("Draft contains 0 usable steps after normalization.")
    return out, warnings


def estimate_min_positive_delta_sec(timestamps: List[float]) -> float:
    uniq = sorted({float(x) for x in (timestamps or [])})
    deltas = [b - a for a, b in zip(uniq, uniq[1:]) if b > a]
    if not deltas:
        return 0.1

    return max(min(deltas), 0.05)


def validate_stage2_localization(
    draft_plan: Dict[str, Any],
    localization: Any,
    num_frames: int,
    *,
    frame_timestamps: Optional[List[float]] = None,
) -> Tuple[bool, List[str], Dict[int, Dict[str, int]]]:
    errors: List[str] = []

    def _is_strict_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    steps = draft_plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return False, ["Draft plan has no steps."], {}

    if not isinstance(localization, dict):
        return False, ["Localization output must be a JSON object with a 'steps' list."], {}
    extra_top = sorted(set(localization.keys()) - {"steps"})
    if extra_top:
        errors.append(f"Localization output contains extra top-level keys (not allowed): {extra_top}")

    loc_steps = localization.get("steps", [])
    if not isinstance(loc_steps, list):
        return False, ["Localization JSON missing 'steps' list."], {}

    step_ids: List[int] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        try:
            step_ids.append(int(s.get("step_id")))
        except Exception:
            continue
    if not step_ids:
        return False, ["Draft plan contains no valid step_id values."], {}
    if len(set(step_ids)) != len(step_ids):
        errors.append("Draft plan has duplicate step_id values (unexpected).")

    expected_ids = set(step_ids)
    ordered_ids = sorted(step_ids)


    loc_sids_in_list: List[int] = []
    for obj in loc_steps:
        if not isinstance(obj, dict):
            continue
        raw_sid = obj.get("step_id")
        if _is_strict_int(raw_sid):
            loc_sids_in_list.append(int(raw_sid))
    if loc_sids_in_list and loc_sids_in_list != ordered_ids:
        errors.append("Localization 'steps' entries must be in ascending step_id order (match the draft order).")

    allowed_entry_keys = {"step_id", "start_frame_index", "end_frame_index", "independence"}
    by_id: Dict[int, Dict[str, int]] = {}
    for i, obj in enumerate(loc_steps):
        if not isinstance(obj, dict):
            errors.append(f"localization.steps[{i}] is not an object.")
            continue
        extra = sorted(set(obj.keys()) - allowed_entry_keys)
        if extra:
            errors.append(f"step_id={obj.get('step_id')} contains extra keys (not allowed): {extra}")
        raw_sid = obj.get("step_id")
        if not _is_strict_int(raw_sid):
            errors.append(f"localization.steps[{i}].step_id missing/invalid (expected int).")
            continue
        sid = int(raw_sid)
        if sid not in expected_ids:
            errors.append(f"Unexpected step_id in localization output: {sid}")
            continue
        if sid in by_id:
            errors.append(f"Duplicate localization entries for step_id={sid}")
            continue
        raw_s = obj.get("start_frame_index")
        raw_e = obj.get("end_frame_index")
        if not _is_strict_int(raw_s) or not _is_strict_int(raw_e):
            errors.append(f"step_id={sid} start_frame_index/end_frame_index missing/invalid (expected int).")
            continue
        s = int(raw_s)
        e = int(raw_e)
        by_id[sid] = {"start_frame_index": s, "end_frame_index": e}


        if sid == 1:
            if "independence" in obj:
                obj.pop("independence")                                       
        else:
            indep_val = obj.get("independence")
            if indep_val is None:
                errors.append(f"step_id={sid} missing required 'independence' field (expected 'yes' or 'no').")
            elif str(indep_val).strip().lower() not in ("yes", "no"):
                errors.append(f"step_id={sid} 'independence' must be 'yes' or 'no' (got '{indep_val}').")
            else:
                by_id[sid]["independence"] = str(indep_val).strip().lower()


    for sid in step_ids:
        if sid not in by_id:
            errors.append(f"Missing localization for step_id={sid}")


    ordered = ordered_ids
    prev_end: Optional[int] = None
    prev_end_ts: Optional[float] = None
    for pos, sid in enumerate(ordered):
        seg = by_id.get(sid)
        if not seg:
            continue
        s = seg["start_frame_index"]
        e = seg["end_frame_index"]
        if s < 1 or s > num_frames:
            errors.append(f"step_id={sid} start_frame_index out of range: {s}")
        if e < 1 or e > num_frames + 1:
            errors.append(f"step_id={sid} end_frame_index out of range: {e}")
        if not (s < e):
            errors.append(f"step_id={sid} requires start_frame_index < end_frame_index (got {s}, {e})")
        is_first = pos == 0
        is_last = pos == len(ordered) - 1
        if is_first and s != 1:
            errors.append(f"HARD full-video coverage violated: first step start_frame_index must be 1 (got {s})")
        if prev_end is not None and not is_first and s != prev_end:
            errors.append(
                f"Contiguity constraint violated: step_id={sid} requires start_frame_index == prev_end (got {s} vs {prev_end})"
            )
        if not is_last and e == num_frames + 1:
            errors.append(
                f"Only the LAST step may use end_frame_index={num_frames + 1} (got step_id={sid} with end_frame_index={e})"
            )
        if is_last and e != num_frames + 1:
            errors.append(f"HARD full-video coverage violated: last step end_frame_index must be {num_frames + 1} (got {e})")

        if frame_timestamps and 1 <= s <= len(frame_timestamps) and 1 <= e <= len(frame_timestamps) + 1:
            s_ts = float(frame_timestamps[s - 1])
            if e == len(frame_timestamps) + 1:
                e_ts = float(frame_timestamps[-1]) + estimate_min_positive_delta_sec(frame_timestamps)
            else:
                e_ts = float(frame_timestamps[e - 1])
            if not (s_ts < e_ts):
                if abs(s_ts - e_ts) < 1e-9:
                    errors.append(
                        f"step_id={sid} selected indices map to identical timestamps (got {s_ts:.2f}, {e_ts:.2f}); "
                        "this often happens when sampled frames are duplicates. Choose a larger end_frame_index that shows clear progress."
                    )
                else:
                    errors.append(f"step_id={sid} requires start_sec < end_sec (got {s_ts:.2f}, {e_ts:.2f})")
            if prev_end_ts is not None and s_ts < prev_end_ts:
                errors.append(
                    f"Monotonic constraint violated in seconds: step_id={sid} start_sec {s_ts:.2f} < prev_end_sec {prev_end_ts:.2f}"
                )
            prev_end_ts = e_ts
        prev_end = e




    _MIN_STEP_SPAN = 2
    for sid in ordered:
        seg = by_id.get(sid)
        if not seg:
            continue
        span = seg["end_frame_index"] - seg["start_frame_index"]
        if span < _MIN_STEP_SPAN:
            errors.append(
                f"step_id={sid} span is only {span} frame(s) "
                f"([{seg['start_frame_index']}, {seg['end_frame_index']})); "
                f"minimum is {_MIN_STEP_SPAN}. Merge with adjacent step or widen."
            )





    _MAX_SPAN_FRACTION = 0.60
    if len(ordered) > 1:
        for sid in ordered:
            seg = by_id.get(sid)
            if not seg:
                continue
            span = seg["end_frame_index"] - seg["start_frame_index"]
            if span > num_frames * _MAX_SPAN_FRACTION:
                errors.append(
                    f"step_id={sid} span is {span} frames "
                    f"({span * 100 // num_frames}% of video); "
                    f"exceeds {int(_MAX_SPAN_FRACTION * 100)}% balance threshold. "
                    f"Split this step or redistribute boundaries."
                )

    return len(errors) == 0, errors, by_id


def normalize_stage3_step_output(
    step_json: Dict[str, Any],
    expected_step_id: int,
    expected_step_goal: str,
    num_frames: int,
    *,
    frame_timestamps: Optional[List[float]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(step_json, dict):
        return None, ["Stage 3 output is not an object."]

    def _is_strict_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    allowed_top_keys = {
        "step_id",
        "step_goal",
        "independence",
        "detail_independence",
        "rationale",
        "causal_chain",
        "counterfactual_challenge_question",
        "expected_challenge_outcome",
        "failure_reflecting",
        "critical_frames",
    }
    extra_top = sorted(set(step_json.keys()) - allowed_top_keys)
    if extra_top:
        errors.append(f"Unexpected top-level keys (not allowed): {extra_top}")

    def _norm_str(v: Any) -> str:
        s = str(v).strip() if v is not None else ""
        return "" if _is_placeholder_str(s) else s

    def _norm_identifier(v: Any) -> str:
        s = _norm_str(v)
        if not s:
            return ""
        ident = sanitize_filename(s)
        return "" if ident == "unnamed" else ident

    def _norm_identifier_list(v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        out_list: List[str] = []
        for x in v:
            s = _norm_identifier(x)
            if s:
                out_list.append(s)
        return _dedupe_keep_order(out_list)

    def _canon_text(v: Any) -> str:
        return re.sub(r"\s+", " ", str(v or "").strip())

    def _parse_bool(v: Any) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            t = v.strip().lower()
            if t in {"true", "t", "yes", "y", "1"}:
                return True
            if t in {"false", "f", "no", "n", "0"}:
                return False
        if isinstance(v, (int, float)) and v in {0, 1}:
            return bool(v)
        return None

    allowed_step_cc_keys = {
        "agent",
        "action",
        "patient",
        "causal_precondition_on_spatial",
        "causal_precondition_on_affordance",
        "causal_effect_on_spatial",
        "causal_effect_on_affordance",
    }
    allowed_frame_cc_keys = {
        "causal_precondition_on_spatial",
        "causal_precondition_on_affordance",
        "causal_effect_on_spatial",
        "causal_effect_on_affordance",

        "agent",
        "action",
        "patient",
    }
    allowed_interaction_keys = {
        "patient",
        "affordance_type",
        "mechanism",

        "description",                             
        "hotspot",
        "tools",
        "materials",
    }

    _LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•])\s+")
    _LEADING_NUMBER_RE = re.compile(r"^\s*\d+\s*[\.\)、\)]\s*")

    def _normalize_statement_list(text: str) -> List[str]:

        raw = (text or "").strip()
        if not raw:
            return []

        raw = raw.replace("\\n", "\n")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            lines = [raw]
        normalized: List[str] = []
        for line in lines:
            line = _LEADING_LIST_MARKER_RE.sub("", line)
            line = _LEADING_NUMBER_RE.sub("", line)
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            if "\n" in line:
                line = " ".join([x.strip() for x in line.splitlines() if x.strip()])
            if not line.endswith("."):
                line = f"{line}."
            normalized.append(line)
        return _dedupe_keep_order(normalized, key_fn=_text_dedupe_key)

    def _spatial_item_to_statement(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            relation = str(item.get("relation", "")).strip()
            objects = item.get("objects", [])
            if not isinstance(objects, list):
                objects = []
            objects = [str(o).strip() for o in objects if str(o).strip()]
            truth = _parse_bool(item.get("truth", True))
            if truth is None:
                truth = True
            rel = relation or "unspecified_relation"
            if objects:
                objs = ", ".join(objects)
                return f"Relation '{rel}' {'holds' if truth else 'does not hold'} between {objs}."
            return f"Relation '{rel}' {'holds' if truth else 'does not hold'}."
        return str(item or "").strip()

    def _affordance_item_to_statement(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            object_name = str(item.get("object_name", "")).strip()
            affordance_types = item.get("affordance_types", [])
            if not isinstance(affordance_types, list):
                affordance_types = []
            affordance_types = [str(a).strip() for a in affordance_types if str(a).strip()]
            reasons = str(item.get("reasons", "")).strip()
            obj = object_name or "unspecified_object"
            aff = ", ".join(affordance_types) if affordance_types else "unspecified_affordance"
            if reasons:
                return f"The object {obj} has affordance/state {aff}. {reasons}"
            return f"The object {obj} has affordance/state {aff}."
        return str(item or "").strip()

    def _normalize_causal_text(v: Any, *, label: str, kind: str) -> List[str]:

        if isinstance(v, list):
            out_list: List[str] = []
            for item in v:
                if isinstance(item, str):
                    out_list.extend(_normalize_statement_list(item))
                    continue
                text = _spatial_item_to_statement(item) if kind == "spatial" else _affordance_item_to_statement(item)
                out_list.extend(_normalize_statement_list(text))
            normalized = _dedupe_keep_order(out_list, key_fn=_text_dedupe_key)
        elif isinstance(v, dict):
            text = _spatial_item_to_statement(v) if kind == "spatial" else _affordance_item_to_statement(v)
            normalized = _normalize_statement_list(text)
        else:
            normalized = _normalize_statement_list(_norm_str(v))

        if not normalized:
            errors.append(f"{label} is empty (expected a non-empty list of sentence strings).")
            return []
        for s in normalized:
            if _contains_frame_ref(s):
                errors.append(f"{label} must not reference frame/image indices.")
                break
        for s in normalized:
            if _contains_time_ref(s):
                errors.append(f"{label} must not reference timestamps/durations.")
                break
        return normalized

    def _norm_step_causal_chain(v: Any, *, label: str) -> Dict[str, Any]:
        d = v if isinstance(v, dict) else {}
        if isinstance(v, dict):
            extra_cc = sorted(set(d.keys()) - allowed_step_cc_keys)
            if extra_cc:
                errors.append(f"{label}.causal_chain contains extra keys (not allowed): {extra_cc}")

        agent_raw = d.get("agent", "")
        action_raw = d.get("action", "")
        patient_raw = d.get("patient", "")
        if _contains_frame_ref(agent_raw) or _contains_frame_ref(action_raw) or _contains_frame_ref(patient_raw):
            errors.append(f"{label}.causal_chain agent/action/patient must not reference frame/image indices.")
        if _contains_time_ref(agent_raw) or _contains_time_ref(action_raw) or _contains_time_ref(patient_raw):
            errors.append(f"{label}.causal_chain agent/action/patient must not reference timestamps/durations.")

        cc = {
            "agent": _norm_identifier(agent_raw),
            "action": _norm_str(action_raw),
            "patient": _norm_identifier(patient_raw),
            "causal_precondition_on_spatial": _normalize_causal_text(
                d.get("causal_precondition_on_spatial"),
                label=f"{label}.causal_chain.causal_precondition_on_spatial",
                kind="spatial",
            ),
            "causal_precondition_on_affordance": _normalize_causal_text(
                d.get("causal_precondition_on_affordance"),
                label=f"{label}.causal_chain.causal_precondition_on_affordance",
                kind="affordance",
            ),
            "causal_effect_on_spatial": _normalize_causal_text(
                d.get("causal_effect_on_spatial"),
                label=f"{label}.causal_chain.causal_effect_on_spatial",
                kind="spatial",
            ),
            "causal_effect_on_affordance": _normalize_causal_text(
                d.get("causal_effect_on_affordance"),
                label=f"{label}.causal_chain.causal_effect_on_affordance",
                kind="affordance",
            ),
        }
        if not cc["agent"] or not cc["action"] or not cc["patient"]:
            errors.append(f"{label}.causal_chain must include non-empty agent/action/patient.")
        return cc

    def _norm_frame_causal_chain(v: Any, *, label: str) -> Dict[str, Any]:
        d = v if isinstance(v, dict) else {}
        if isinstance(v, dict):
            extra_cc = sorted(set(d.keys()) - allowed_frame_cc_keys)
            if extra_cc:
                errors.append(f"{label}.causal_chain contains extra keys (not allowed): {extra_cc}")
        return {
            "causal_precondition_on_spatial": _normalize_causal_text(
                d.get("causal_precondition_on_spatial"),
                label=f"{label}.causal_chain.causal_precondition_on_spatial",
                kind="spatial",
            ),
            "causal_precondition_on_affordance": _normalize_causal_text(
                d.get("causal_precondition_on_affordance"),
                label=f"{label}.causal_chain.causal_precondition_on_affordance",
                kind="affordance",
            ),
            "causal_effect_on_spatial": _normalize_causal_text(
                d.get("causal_effect_on_spatial"),
                label=f"{label}.causal_chain.causal_effect_on_spatial",
                kind="spatial",
            ),
            "causal_effect_on_affordance": _normalize_causal_text(
                d.get("causal_effect_on_affordance"),
                label=f"{label}.causal_chain.causal_effect_on_affordance",
                kind="affordance",
            ),
        }

    def _norm_interaction(v: Any, *, label: str) -> Dict[str, str]:
        d = v if isinstance(v, dict) else {}
        if isinstance(v, dict):
            extra_inter = sorted(set(d.keys()) - allowed_interaction_keys)
            if extra_inter:
                errors.append(f"{label}.interaction contains extra keys (not allowed): {extra_inter}")
        hotspot = d.get("hotspot", {}) if isinstance(d.get("hotspot"), dict) else {}

        patient_raw = d.get("patient") or d.get("description") or hotspot.get("patient") or hotspot.get("description") or ""
        aff_raw = d.get("affordance_type") or hotspot.get("affordance_type") or ""
        mech_raw = d.get("mechanism") or hotspot.get("mechanism") or ""

        if _contains_frame_ref(patient_raw) or _contains_frame_ref(aff_raw) or _contains_frame_ref(mech_raw):
            errors.append(f"{label}.interaction must not reference frame/image indices.")
        if _contains_time_ref(patient_raw) or _contains_time_ref(aff_raw) or _contains_time_ref(mech_raw):
            errors.append(f"{label}.interaction must not reference timestamps/durations.")

        patient = _norm_str(patient_raw)
        aff = _norm_identifier(aff_raw)
        mech = _norm_str(mech_raw)
        if not patient or not aff or not mech:
            errors.append(f"{label}.interaction must include non-empty patient/affordance_type/mechanism.")
        return {"patient": patient, "affordance_type": aff, "mechanism": mech}

    sid = step_json.get("step_id")
    sid_int: Optional[int] = int(sid) if _is_strict_int(sid) else None
    if sid_int is None:
        errors.append("step_id missing/invalid (expected int).")
    if sid_int != expected_step_id:
        errors.append(f"step_id mismatch: expected {expected_step_id}, got {sid}")

    goal = _canon_text(_norm_str(step_json.get("step_goal", "")))
    if not goal:
        errors.append("Missing/empty step_goal.")
    if _contains_frame_ref(goal):
        errors.append("step_goal must not reference frame/image indices.")
    if _contains_time_ref(goal):
        errors.append("step_goal must not reference timestamps/durations.")

    raw_independence = step_json.get("independence")
    independence = ""
    if raw_independence is not None:
        independence = _norm_str(raw_independence).lower()
        if independence not in {"yes", "no"}:
            errors.append("independence must be exactly 'yes' or 'no' when present.")
    raw_detail_independence = step_json.get("detail_independence")
    detail_independence = ""
    if raw_detail_independence is not None:
        detail_independence = _norm_str(raw_detail_independence)
        if _contains_frame_ref(detail_independence):
            errors.append("detail_independence must not reference frame/image indices.")
        if _contains_time_ref(detail_independence):
            errors.append("detail_independence must not reference timestamps/durations.")
        if independence == "no" and detail_independence:
            errors.append("detail_independence must be empty when independence is 'no'.")
        if independence == "yes" and not detail_independence:
            errors.append("detail_independence must be non-empty when independence is 'yes'.")

    rationale = _norm_str(step_json.get("rationale", ""))
    if not rationale:
        errors.append("Missing/empty rationale.")
    if _contains_frame_ref(rationale):
        errors.append("rationale must not reference frame/image indices.")
    if _contains_time_ref(rationale):
        errors.append("rationale must not reference timestamps/durations.")

    counterfactual_q = _norm_str(step_json.get("counterfactual_challenge_question", ""))
    if not counterfactual_q:
        errors.append("Missing/empty counterfactual_challenge_question.")
    if counterfactual_q and not any(counterfactual_q.lstrip().lower().startswith(p) for p in ("what if", "what would", "suppose", "imagine")):
        errors.append("counterfactual_challenge_question must start with 'What if/What would/Suppose/Imagine ...?'.")
    if _contains_frame_ref(counterfactual_q):
        errors.append("counterfactual_challenge_question must not reference frame/image indices.")
    if _contains_time_ref(counterfactual_q):
        errors.append("counterfactual_challenge_question must not reference timestamps/durations.")

    expected_outcome = _norm_str(step_json.get("expected_challenge_outcome", ""))
    if not expected_outcome:
        errors.append("Missing/empty expected_challenge_outcome.")
    if _contains_frame_ref(expected_outcome):
        errors.append("expected_challenge_outcome must not reference frame/image indices.")
    if _contains_time_ref(expected_outcome):
        errors.append("expected_challenge_outcome must not reference timestamps/durations.")

    fr_raw = step_json.get("failure_reflecting")
    if not isinstance(fr_raw, dict):
        fr_raw = {}
    extra_fr = sorted(set(fr_raw.keys()) - {"reason", "recovery_strategy"})
    if extra_fr:
        errors.append(f"failure_reflecting contains extra keys (not allowed): {extra_fr}")
    fr_reason = _norm_str(fr_raw.get("reason", ""))
    fr_recovery = _norm_str(fr_raw.get("recovery_strategy", ""))
    if not fr_reason:
        errors.append("Missing/empty failure_reflecting.reason.")
    if not fr_recovery:
        errors.append("Missing/empty failure_reflecting.recovery_strategy.")
    if _contains_frame_ref(fr_reason) or _contains_frame_ref(fr_recovery):
        errors.append("failure_reflecting fields must not reference frame/image indices.")
    if _contains_time_ref(fr_reason) or _contains_time_ref(fr_recovery):
        errors.append("failure_reflecting fields must not reference timestamps/durations.")

    step_cc = _norm_step_causal_chain(step_json.get("causal_chain", {}), label="step")

    cfs = step_json.get("critical_frames")
    if not isinstance(cfs, list):
        errors.append("Missing 'critical_frames' list.")
        cfs = []
    if len(cfs) != 2:
        errors.append(f"critical_frames must have length exactly 2 (got {len(cfs)}).")

    normalized_cfs: List[Dict[str, Any]] = []
    prev_idx = -1
    prev_ts: Optional[float] = None
    for i, cf in enumerate(cfs):
        if not isinstance(cf, dict):
            errors.append(f"critical_frames[{i}] is not an object.")
            continue

        extra_cf = sorted(set(cf.keys()) - {"frame_index", "action_state_change_description", "causal_chain", "interaction"})
        if extra_cf:
            errors.append(f"critical_frames[{i}] contains extra keys (not allowed): {extra_cf}")

        raw_fi = cf.get("frame_index")
        if not _is_strict_int(raw_fi):
            errors.append(f"critical_frames[{i}].frame_index missing/invalid (expected int).")
            continue
        fi = int(raw_fi)
        if fi < 1 or fi > int(num_frames):
            errors.append(f"critical_frames[{i}].frame_index out of range: {fi}")
        if fi <= prev_idx:
            errors.append("critical_frames indices must be strictly increasing within a step.")
        prev_idx = fi

        if frame_timestamps and 1 <= fi <= len(frame_timestamps):




            ts = float(frame_timestamps[fi - 1])
            if prev_ts is not None and ts < prev_ts - 1e-9:
                errors.append("critical_frames timestamps must be non-decreasing within a step.")
            prev_ts = ts

        desc = _norm_str(cf.get("action_state_change_description", ""))
        if not desc:
            errors.append(f"critical_frames[{i}].action_state_change_description is empty.")
        if _contains_frame_ref(desc):
            errors.append(f"critical_frames[{i}].action_state_change_description must not reference frame/image indices.")
        if _contains_time_ref(desc):
            errors.append(f"critical_frames[{i}].action_state_change_description must not reference timestamps/durations.")

        cf_cc = _norm_frame_causal_chain(cf.get("causal_chain", {}), label=f"critical_frames[{i}]")

        interaction_raw = cf.get("interaction")
        if not isinstance(interaction_raw, dict):
            errors.append(f"critical_frames[{i}].interaction missing/invalid (expected an object).")
            interaction_raw = {}
        interaction_norm = _norm_interaction(interaction_raw, label=f"critical_frames[{i}]")

        normalized_cfs.append(
            {
                "frame_index": fi,
                "action_state_change_description": desc,
                "causal_chain": cf_cc,
                "interaction": interaction_norm,
            }
        )

    normalized = {
        "step_id": expected_step_id,
        "step_goal": goal,
        "rationale": rationale,
        "causal_chain": step_cc,
        "counterfactual_challenge_question": counterfactual_q,
        "expected_challenge_outcome": expected_outcome,
        "failure_reflecting": {"reason": fr_reason, "recovery_strategy": fr_recovery},
        "critical_frames": normalized_cfs,
    }
    if independence:
        normalized["independence"] = independence
    if raw_detail_independence is not None or independence == "yes":
        normalized["detail_independence"] = detail_independence

    if errors:
        return None, errors
    return normalized, []


def normalize_high_level_goal_text(value: Any) -> Tuple[Optional[str], List[str]]:
    errors: List[str] = []
    s = re.sub(r"\s+", " ", str(value or "").strip())
    if not s or _is_placeholder_str(s):
        errors.append("high_level_goal is missing/empty (or a placeholder).")
        return None, errors
    if _contains_frame_ref(s):
        errors.append("high_level_goal must not reference frame/image indices.")
    if _contains_time_ref(s):
        errors.append("high_level_goal must not reference timestamps/durations.")
    if errors:
        return None, errors
    return s, []


def save_keyframe_images_from_manifest(
    manifest_path: str,
    frame_indices_1based: List[int],
    output_dir: str,
) -> Dict[int, str]:
    manifest = read_json(manifest_path)
    base_dir = os.path.dirname(manifest_path)

    by_idx: Dict[int, Dict[str, Any]] = {}
    for entry in manifest.get("frames", []):
        if not isinstance(entry, dict):
            continue
        try:
            idx1 = int(entry.get("frame_index_1based"))
        except Exception:
            continue
        by_idx[idx1] = entry

    os.makedirs(output_dir, exist_ok=True)
    out: Dict[int, str] = {}
    for idx1 in frame_indices_1based:
        if idx1 not in by_idx:
            raise ValueError(f"frame_index not found in manifest: {idx1}")
        entry = by_idx[idx1]
        rel = entry.get("image_relpath")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"Manifest entry missing image_relpath for frame_index={idx1}")
        src = os.path.join(base_dir, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        ts = float(entry.get("timestamp_sec", 0.0))
        name = f"frame_{idx1:03d}_ts_{ts:.2f}s.jpg"
        dst = os.path.abspath(os.path.join(output_dir, name))
        shutil.copyfile(src, dst)
        out[idx1] = dst
    return out


def get_run_summary_schema_fingerprint(run_summary_path: str) -> Optional[str]:
    if not os.path.exists(run_summary_path):
        return None
    try:
        rs = read_json(run_summary_path)
    except Exception:
        return None
    raw = rs.get("schema_fingerprint")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


@lru_cache(maxsize=1)
def four_stage_schema_fingerprint() -> str:
    root = os.path.dirname(__file__)
    names = ["four_stage_prompt_templates.py", "single_step_prompt_templates.py"]
    h = hashlib.sha256()
    found = False
    for name in names:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        found = True
        h.update(name.encode("utf-8"))
        with open(path, "rb") as f:
            h.update(f.read())
    if not found:
        return "sha256:unknown"
    return "sha256:" + h.hexdigest()


def guard_schema_fingerprint(
    run_summary_path: str,
    video_out: str,
    *,
    stage: str,
    overwrite: bool,
    allow_unfingerprinted_resume: bool,
    will_resume: bool,
) -> str:
    current_fp = four_stage_schema_fingerprint()
    existing_fp = get_run_summary_schema_fingerprint(run_summary_path)
    if existing_fp is not None and existing_fp != current_fp and not overwrite:
        raise RuntimeError(
            f"Refusing to run {stage}: schema_fingerprint mismatch.\n"
            f"- video_out: {os.path.abspath(video_out)}\n"
            f"- existing: {existing_fp}\n"
            f"- current : {current_fp}\n"
            "Use --overwrite to regenerate outputs, or use a different --output-root."
        )
    if will_resume and existing_fp is None and not allow_unfingerprinted_resume:
        raise RuntimeError(
            f"Refusing to resume {stage} outputs without schema_fingerprint in run_summary.json.\n"
            f"- video_out: {os.path.abspath(video_out)}\n"
            "Re-run with --overwrite, or pass --allow-unfingerprinted-resume to bypass this check."
        )
    return current_fp


def call_chat_completion(client: Any, cfg: ApiConfig, messages: List[Dict[str, Any]], max_tokens: int) -> Tuple[str, Dict[str, int]]:

    max_attempts = int(getattr(cfg, "api_call_retries", 1) or 1)
    max_attempts = max(1, max_attempts)
    backoff_sec = float(getattr(cfg, "api_call_retry_backoff_sec", 1.0) or 0.0)
    timeout_sec = float(getattr(cfg, "api_call_timeout_sec", 420.0) or 420.0)

    start = time.time()
    payload_input = build_request_payload_input(messages)
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for attempt in range(1, max_attempts + 1):
        try:
            content, usage = call_model_api(
                client,
                model_name=cfg.model_name,
                payload_input=payload_input,
                timeout_sec=timeout_sec,
                temperature=cfg.temperature,
                max_tokens=max_tokens if max_tokens and int(max_tokens) > 0 else None,
                reasoning_effort="low",
            )
            break
        except Exception as e:
            if attempt >= max_attempts:
                raise RuntimeError(f"Model call failed after {max_attempts} attempts: {e}") from e
            sleep_sec = min(max(0.0, backoff_sec) * (2 ** (attempt - 1)), 8.0)
            logger.warning(f"Model call error (attempt {attempt}/{max_attempts}): {e}; retrying in {sleep_sec:.1f}s")
            time.sleep(sleep_sec)

    end = time.time()
    logger.info(f"Model call finished in {end - start:.2f}s (tokens: in={usage.get('prompt_tokens', 0)}, out={usage.get('completion_tokens', 0)})")
    if cfg.verbose:
        logger.info("Raw model output:\n" + content)
    return content, usage





MODEL_PRICING = {
    "Doubao-Seed-2.0-pro":  (0.0032, 0.016),
    "Doubao-Seed-2.0-lite": (0.0006, 0.0036),
    "Doubao-Seed-1.8":      (0.0008, 0.002),
    "gpt-5.4":              (0.0, 0.0),                                     
}


def calculate_cost_yuan(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:

    pricing = MODEL_PRICING.get(model_name)
    if not pricing:
        return 0.0
    input_price, output_price = pricing
    cost = (prompt_tokens / 1000.0) * input_price + (completion_tokens / 1000.0) * output_price
    return round(cost, 6)


def compute_cost_summary_from_run_summary(run_summary_path: str) -> Dict[str, Any]:

    try:
        summary = read_json(run_summary_path)
    except Exception:
        return {}

    model_name = ""
    api_config = summary.get("api_config", {})
    if isinstance(api_config, dict):
        model_name = str(api_config.get("model_name", ""))

    total_prompt = 0
    total_completion = 0
    total_calls = 0
    per_stage = {}

    for stage_key in ("stage1", "stage2", "stage3", "stage4"):
        stage_data = summary.get(stage_key, {})
        if not isinstance(stage_data, dict):
            continue
        tu = stage_data.get("token_usage", {})
        if not isinstance(tu, dict):
            continue
        pt = int(tu.get("prompt_tokens", 0) or 0)
        ct = int(tu.get("completion_tokens", 0) or 0)
        calls = int(tu.get("api_calls", 0) or 0)
        cost = calculate_cost_yuan(model_name, pt, ct)
        per_stage[stage_key] = {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "api_calls": calls,
            "cost_yuan": cost,
        }
        total_prompt += pt
        total_completion += ct
        total_calls += calls

    total_cost = calculate_cost_yuan(model_name, total_prompt, total_completion)

    return {
        "model_name": model_name,
        "per_stage": per_stage,
        "total": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "api_calls": total_calls,
            "cost_yuan": total_cost,
        },
    }


def print_cost_summary(run_summary_path: str) -> None:

    cs = compute_cost_summary_from_run_summary(run_summary_path)
    if not cs or not cs.get("total"):
        return

    model = cs.get("model_name", "unknown")
    total = cs["total"]
    per_stage = cs.get("per_stage", {})

    lines = [
        f"===== Cost Summary (model: {model}) =====",
    ]
    for sk in ("stage1", "stage2", "stage3", "stage4"):
        sd = per_stage.get(sk)
        if sd:
            lines.append(
                f"  {sk}: {sd['api_calls']} calls, "
                f"in={sd['prompt_tokens']:,} out={sd['completion_tokens']:,} tokens, "
                f"cost={sd['cost_yuan']:.4f} CNY"
            )
    lines.append(
        f"  TOTAL: {total['api_calls']} calls, "
        f"in={total['prompt_tokens']:,} out={total['completion_tokens']:,} tokens "
        f"(total={total['total_tokens']:,}), "
        f"cost={total['cost_yuan']:.4f} CNY"
    )
    lines.append("=" * 50)

    for line in lines:
        logger.info(line)






def build_cumulative_prefix_videos(
    video_out: str,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict:

    if logger is None:
        logger = logging.getLogger("four_stage")

    result = {"built": 0, "skipped": 0, "failed": 0}

    seg_path = os.path.join(video_out, "stage2", "step_segments.json")
    if not os.path.isfile(seg_path):
        logger.warning(f"[prefix] step_segments.json not found: {seg_path}")
        return result

    with open(seg_path, "r", encoding="utf-8") as f:
        seg_data = json.load(f)

    segments = seg_data.get("segments", [])
    if not segments:
        logger.warning(f"[prefix] no segments in {seg_path}")
        return result


    ordered_clips: List[Tuple[int, str]] = []
    for seg in sorted(segments, key=lambda s: int(s.get("step_id", 0))):
        sid = int(seg.get("step_id", 0))
        clip_rel = seg.get("clip_relpath", "")
        if not clip_rel:
            logger.warning(f"[prefix] step {sid}: no clip_relpath")
            return result
        clip_abs = os.path.join(video_out, "stage2", clip_rel)
        if not os.path.isfile(clip_abs):
            logger.warning(f"[prefix] step {sid}: clip not found: {clip_abs}")
            return result
        ordered_clips.append((sid, clip_abs))

    if not ordered_clips:
        return result

    prefix_dir = os.path.join(video_out, "cumulative_last_frame_segments")
    os.makedirs(prefix_dir, exist_ok=True)

    for end_idx in range(len(ordered_clips)):
        end_sid = ordered_clips[end_idx][0]
        dst = os.path.join(prefix_dir, f"segment_start_to_step{end_sid:02d}_last.mp4")

        if os.path.isfile(dst) and not overwrite:
            result["skipped"] += 1
            continue

        srcs = [clip for _, clip in ordered_clips[: end_idx + 1]]

        ok = _concat_or_copy_clips(ffmpeg_bin, srcs, dst, overwrite)
        if ok:
            result["built"] += 1
        else:
            result["failed"] += 1
            logger.warning(f"[prefix] failed to build {os.path.basename(dst)}")

    logger.info(
        f"[prefix] done: built={result['built']} skipped={result['skipped']} "
        f"failed={result['failed']} → {prefix_dir}"
    )
    return result


def _concat_or_copy_clips(
    ffmpeg_bin: str,
    srcs: List[str],
    dst: str,
    overwrite: bool,
) -> bool:

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_dst = dst + ".tmp.mp4"
    ow_flag = "-y" if overwrite else "-n"

    try:

        if os.path.exists(tmp_dst):
            os.remove(tmp_dst)

        if len(srcs) == 1:

            cmd = [
                ffmpeg_bin, ow_flag, "-hide_banner", "-loglevel", "error",
                "-i", srcs[0], "-c", "copy",
                "-avoid_negative_ts", "make_zero", tmp_dst,
            ]
            subprocess.run(cmd, check=True, timeout=300)
            os.replace(tmp_dst, dst)
            return True


        concat_txt = dst + ".concat.txt"
        try:
            with open(concat_txt, "w", encoding="utf-8") as f:
                for src in srcs:
                    ap = os.path.abspath(src).replace("\\", "/").replace("'", "\\'")
                    f.write(f"file '{ap}'\n")


            copy_cmd = [
                ffmpeg_bin, ow_flag, "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", concat_txt,
                "-c", "copy", "-avoid_negative_ts", "make_zero", tmp_dst,
            ]
            try:
                if os.path.exists(tmp_dst):
                    os.remove(tmp_dst)
                subprocess.run(copy_cmd, check=True, timeout=300)
                os.replace(tmp_dst, dst)
                return True
            except Exception:
                pass


            reencode_cmd = [
                ffmpeg_bin, ow_flag, "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", concat_txt,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart", tmp_dst,
            ]
            if os.path.exists(tmp_dst):
                os.remove(tmp_dst)
            subprocess.run(reencode_cmd, check=True, timeout=600)
            os.replace(tmp_dst, dst)
            return True
        finally:
            try:
                os.remove(concat_txt)
            except OSError:
                pass
    except Exception:
        return False
    finally:
        try:
            if os.path.exists(tmp_dst):
                os.remove(tmp_dst)
        except OSError:
            pass
