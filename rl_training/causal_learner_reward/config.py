

from dataclasses import dataclass, field
import os
from typing import Any, Dict, Mapping, Optional


SUPPORTED_TASKS = (
    "Task_01",
    "Task_02",
    "Task_06",
    "Task_07",
    "Task_18",
    "Task_19",
    "Task_20",
)




DEFAULT_ALPHA_TABLE: Dict[str, float] = {
    "Task_01": 0.15,
    "Task_02": 0.0,
    "Task_06": 0.0,
    "Task_07": 0.0,
    "Task_18": 0.25,
    "Task_19": 0.0,
    "Task_20": 0.0,
}

RULE_DISABLED_TASKS = {"Task_19"}

DEFAULT_FAILURE_SCORE: float = -1.0
DEFAULT_PARTIAL_MATCH_THRESHOLD: float = 0.3
DEFAULT_RUBRIC_ID = "strict_multimodal_rubric"

DEFAULT_FACT_REWARD_WEIGHTS: Dict[str, float] = {
    "fact_recall": 0.70,
    "answer_similarity": 0.20,
    "patient": 0.10,
}

DEFAULT_TASK18_WEIGHTS: Dict[str, float] = {
    "flaw_step": 0.15,
    "flaw_type": 0.20,
    "repair_coverage": 0.50,
    "anti_copy": 0.15,
}

DEFAULT_TASK20_WEIGHTS: Dict[str, float] = {
    "recovery_coverage": 0.45,
    "action_signal": 0.20,
    "novelty": 0.20,
    "failure_addressing": 0.15,
}


@dataclass
class JudgeAPIConfig:


    mode: str = "vllm"
    base_url: str = "http://localhost:8002/v1"
    model_name: str = "Qwen3-8B-Instruct"
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    endpoints_file: Optional[str] = None
    endpoint_reload_interval: float = 5.0
    max_concurrent_judge_requests_per_endpoint: int = 4
    endpoint_failure_threshold: int = 3
    endpoint_cooldown_seconds: float = 30.0
    max_retries: int = 3
    timeout: float = 60.0
    http_connect_timeout: float = 10.0
    http_read_timeout: Optional[float] = None
    http_write_timeout: float = 30.0
    http_pool_timeout: float = 5.0
    http_max_connections: int = 32
    http_max_keepalive_connections: int = 16
    http_trust_env: Optional[bool] = None
    openai_sdk_max_retries: int = 0
    retry_backoff_base: float = 1.0
    retry_backoff_max: float = 8.0
    retry_jitter: float = 0.2
    temperature: float = 0.0
    max_tokens: int = 512
    judge_chunk_size: int = 16
    max_concurrent_judge_requests: int = 4
    video_transport: str = "native_video"
    strict_video_input: bool = True
    openai_native_video: bool = False
    openai_use_responses_api: bool = True
    native_video_content_type: str = "input_video"
    allow_file_url: bool = False
    enable_video_cache: bool = True
    enable_judge_result_cache: bool = True
    fps: float = 1.0
    max_frames_for_judge: int = 8
    video_input_mode: str = "native_video"
    request_timeout_buffer: float = 30.0

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any] | "JudgeAPIConfig"]) -> Optional["JudgeAPIConfig"]:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        data = dict(value)
        if "model" in data and "model_name" not in data:
            data["model_name"] = data.pop("model")
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})

    @property
    def resolved_api_key(self) -> str:
        default_api_key = "EMPTY" if self.mode == "vllm" else ""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or default_api_key
        return os.environ.get("OPENAI_API_KEY") or default_api_key

    def validate(self) -> None:
        if self.mode not in {"vllm", "openai"}:
            raise ValueError(f"Unsupported judge mode: {self.mode}")
        if self.video_transport not in {"native_video", "frame_fallback"}:
            raise ValueError(f"Unsupported video_transport: {self.video_transport}")
        if self.judge_chunk_size <= 0:
            raise ValueError("judge_chunk_size must be positive")
        if self.max_concurrent_judge_requests <= 0:
            raise ValueError("max_concurrent_judge_requests must be positive")
        if self.max_concurrent_judge_requests_per_endpoint <= 0:
            raise ValueError("max_concurrent_judge_requests_per_endpoint must be positive")
        if self.endpoint_reload_interval < 0:
            raise ValueError("endpoint_reload_interval must be non-negative")
        if self.endpoint_failure_threshold <= 0:
            raise ValueError("endpoint_failure_threshold must be positive")
        if self.endpoint_cooldown_seconds < 0:
            raise ValueError("endpoint_cooldown_seconds must be non-negative")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.http_connect_timeout <= 0:
            raise ValueError("http_connect_timeout must be positive")
        if self.effective_http_read_timeout <= 0:
            raise ValueError("http_read_timeout must be positive")
        if self.http_write_timeout <= 0:
            raise ValueError("http_write_timeout must be positive")
        if self.http_pool_timeout <= 0:
            raise ValueError("http_pool_timeout must be positive")
        if self.http_max_connections <= 0:
            raise ValueError("http_max_connections must be positive")
        if self.http_max_keepalive_connections < 0:
            raise ValueError("http_max_keepalive_connections must be non-negative")
        if self.openai_sdk_max_retries < 0:
            raise ValueError("openai_sdk_max_retries must be non-negative")
        if self.retry_backoff_base < 0:
            raise ValueError("retry_backoff_base must be non-negative")
        if self.retry_backoff_max < 0:
            raise ValueError("retry_backoff_max must be non-negative")
        if self.retry_jitter < 0:
            raise ValueError("retry_jitter must be non-negative")
        if self.mode == "openai" and self.video_transport == "native_video":
            if self.strict_video_input and not self.openai_native_video:
                raise ValueError(
                    "unsupported_native_video_input: official OpenAI vision APIs do not currently expose "
                    "a confirmed native video-understanding input path for this reward. Set "
                    "video_transport='frame_fallback' explicitly, or set openai_native_video=true only for "
                    "a provider/model that supports native video."
                )

    @property
    def effective_http_read_timeout(self) -> float:
        return float(self.timeout if self.http_read_timeout is None else self.http_read_timeout)

    @property
    def effective_http_trust_env(self) -> bool:
        if self.http_trust_env is not None:
            return bool(self.http_trust_env)
        return self.mode == "openai"


@dataclass
class CausalLearnerRewardConfig:


    phase: int = 1
    alpha_table: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ALPHA_TABLE))
    fact_reward_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FACT_REWARD_WEIGHTS))
    task18_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TASK18_WEIGHTS))
    task20_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TASK20_WEIGHTS))
    failure_score: float = DEFAULT_FAILURE_SCORE
    judge_api_config: Optional[JudgeAPIConfig] = None
    rubric_id: str = DEFAULT_RUBRIC_ID
