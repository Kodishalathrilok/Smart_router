"""Build provider instances from settings. The only place that knows concrete provider classes."""

from collections.abc import Iterable

from app.core.analyzer import HeuristicAnalyzer
from app.core.config import ConfigError, Settings
from app.models.schemas import ModelSpec
from app.providers.base import LLMProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.groq_provider import GroqProvider
from app.providers.mock_provider import MockProvider


def build_providers(settings: Settings, specs: Iterable[ModelSpec]) -> dict[str, LLMProvider]:
    """Instantiate exactly the providers the configured models need. Missing keys are a ConfigError."""
    specs = list(specs)
    if settings.mock_mode:
        analyzer = HeuristicAnalyzer()
        return {"mock": MockProvider({s.id: s.capability_score for s in specs}, lambda q: analyzer.analyze(q).difficulty)}

    providers: dict[str, LLMProvider] = {}
    for name in sorted({s.provider for s in specs}):
        roles = ", ".join(s.role.upper() for s in specs if s.provider == name)
        if name == "groq":
            providers[name] = GroqProvider(_key(settings.groq_api_key, "GROQ_API_KEY", roles), base_url=settings.groq_base_url, timeout=settings.request_timeout_s)
        elif name == "gemini":
            providers[name] = GeminiProvider(_key(settings.gemini_api_key, "GEMINI_API_KEY", roles), base_url=settings.gemini_base_url, timeout=settings.request_timeout_s)
        else:
            raise ConfigError(f"Unknown provider '{name}' configured for {roles}. Supported: groq, gemini.")
    return providers


def _key(secret, env_name: str, roles: str) -> str:
    value = secret.get_secret_value()
    if not value:
        raise ConfigError(f"{env_name} is not set but is required by {roles}. Set it in .env or enable MOCK_MODE=true.")
    return value
