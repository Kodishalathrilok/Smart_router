"""Application settings, loaded from environment variables and `.env`.

Settings are constructed once at startup and passed explicitly to the components
that need them; nothing reads configuration from global state.
"""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLES = ("cheap", "mid", "strong", "judge")
OPTIONAL_ROLES = ("judge_fallback", "second_opinion")  # enabled only when their PROVIDER/MODEL are set


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid. Never silently worked around."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    mock_mode: bool = False

    groq_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")
    nvidia_api_key: SecretStr = SecretStr("")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    cheap_provider: str = "groq"
    cheap_model: str = ""
    mid_provider: str = "groq"
    mid_model: str = ""
    strong_provider: str = "groq"
    strong_model: str = ""
    judge_provider: str = "gemini"
    judge_model: str = ""
    judge_fallback_provider: str = ""
    judge_fallback_model: str = ""
    second_opinion_provider: str = ""  # dedicated second-opinion judge; empty -> the other configured judge
    second_opinion_model: str = ""
    models_config: Path = Path("config/models.json")
    validate_models: bool = True

    quality_threshold: float = Field(0.78, ge=0.0, le=1.0)
    judge_confidence_min: float = Field(0.7, ge=0.0, le=1.0)  # below this, a judge acceptance gets a second opinion
    second_opinion: bool = True  # ask the other configured judge before escalating (SmartRouter mode only)
    verify_execute_code: bool = False  # run the answer's own >>> examples (executes model-written code)
    code_check_timeout_s: float = Field(10.0, gt=0)
    second_opinion_timeout_s: float = Field(20.0, gt=0)  # wall-clock budget for a whole second opinion, retries included
    second_opinion_max_retries: int = Field(1, ge=0, le=3)  # transient errors only
    second_opinion_max_tokens: int = Field(800, ge=64)
    max_escalations: int = Field(2, ge=0, le=5)
    cheap_max_difficulty: float = Field(0.35, ge=0.0, le=1.0)
    mid_max_difficulty: float = Field(0.70, ge=0.0, le=1.0)

    generation_temperature: float = Field(0.2, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(1024, ge=16)
    judge_max_tokens: int = Field(1024, ge=64)

    request_timeout_s: float = Field(60.0, gt=0)
    max_retries: int = Field(3, ge=0, le=6)
    retry_base_delay_s: float = Field(1.0, ge=0)
    retry_max_delay_s: float = Field(30.0, ge=0)
    max_concurrent_requests: int = Field(1, ge=1, le=16)
    max_query_chars: int = Field(8000, ge=1, le=100_000)
    max_inflight_queries: int = Field(4, ge=1, le=64)  # POST /api/query beyond this gets 429 (each query can fan out to many paid calls)
    request_deadline_s: float = Field(300.0, gt=0)  # wall-clock cap for one POST /api/query, retries and escalations included
    # Hard cap on provider calls of every kind (generation, judge, fallback judge, second opinion, retries) per query.
    # A clean 3-attempt run needs 3 generations + 3 judges + up to 3 second opinions = 9.
    max_provider_calls_per_query: int = Field(16, ge=2, le=200)
    stats_window: int = Field(2000, ge=10, le=100_000)  # /api/stats summarizes the most recent N queries

    log_dir: Path = Path("data/logs")
    benchmark_dir: Path = Path("data/benchmarks")
    results_dir: Path = Path("benchmark/results")
    host: str = "127.0.0.1"
    # Host headers the API answers to. Blocks DNS-rebinding: a web page that re-points its own domain at
    # 127.0.0.1 would otherwise be "same origin" with this unauthenticated API. Comma-separated; "*" disables.
    allowed_hosts: str = "127.0.0.1,localhost,[::1]"
    port: int = 8000

    @field_validator("models_config", "log_dir", "benchmark_dir", "results_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Relative paths are anchored at the project root, not the current directory."""
        return value if value.is_absolute() else PROJECT_ROOT / value

    @model_validator(mode="after")
    def _check_thresholds(self) -> "Settings":
        if self.cheap_max_difficulty >= self.mid_max_difficulty:
            raise ValueError("CHEAP_MAX_DIFFICULTY must be lower than MID_MAX_DIFFICULTY")
        if self.verify_execute_code and not self.is_loopback_host:
            raise ValueError("VERIFY_EXECUTE_CODE runs model-written code with no OS sandbox; refusing to enable it on a non-loopback HOST")
        return self

    @property
    def is_loopback_host(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")

    @property
    def allowed_host_list(self) -> list[str]:
        return [h.strip() for h in self.allowed_hosts.split(",") if h.strip()]

    def role_enabled(self, role: str) -> bool:
        """Required roles are always enabled; optional roles only when configured."""
        return role in ROLES or bool(getattr(self, f"{role}_provider") or getattr(self, f"{role}_model"))

    def role(self, role: str) -> tuple[str, str]:
        """Return (provider, model_id) configured for a role. Mock mode routes every role to the mock provider."""
        provider, model = getattr(self, f"{role}_provider"), getattr(self, f"{role}_model")
        if self.mock_mode:
            return "mock", model or f"mock-{role}"
        return provider, model

    def redact(self, text: str) -> str:
        """Remove API key values from text before it is logged or returned."""
        for secret in (self.groq_api_key, self.gemini_api_key, self.nvidia_api_key):
            value = secret.get_secret_value()
            if value:
                text = text.replace(value, "***")
        return text
