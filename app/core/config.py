"""Application settings, loaded from environment variables and `.env`.

Settings are constructed once at startup and passed explicitly to the components
that need them; nothing reads configuration from global state.
"""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLES = ("cheap", "mid", "strong", "judge")
OPTIONAL_ROLES = ("judge_fallback",)  # enabled only when JUDGE_FALLBACK_PROVIDER/MODEL are set


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
    models_config: Path = Path("config/models.json")
    validate_models: bool = True

    quality_threshold: float = Field(0.78, ge=0.0, le=1.0)
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

    log_dir: Path = Path("data/logs")
    benchmark_dir: Path = Path("data/benchmarks")
    results_dir: Path = Path("benchmark/results")
    host: str = "127.0.0.1"
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
        return self

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
