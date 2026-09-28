"""Build provider instances from settings. The only place that knows concrete provider classes."""

from collections.abc import Iterable

from app.core.analyzer import HeuristicAnalyzer
from app.core.config import ConfigError, Settings
from app.models.schemas import ModelSpec
from app.providers.base import LLMProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.mock_provider import MockProvider
from app.providers.openai_compatible import OpenAICompatibleProvider

# provider name -> (settings key attribute, env var name, settings base-url attribute, token field)
OPENAI_COMPATIBLE = {
    "groq": ("groq_api_key", "GROQ_API_KEY", "groq_base_url", "max_completion_tokens"),
    "nvidia": ("nvidia_api_key", "NVIDIA_API_KEY", "nvidia_base_url", "max_tokens"),
}
SUPPORTED = ", ".join([*OPENAI_COMPATIBLE, "gemini"])


def build_providers(settings: Settings, specs: Iterable[ModelSpec]) -> dict[str, LLMProvider]:
    """Instantiate exactly the providers the configured models need. Missing keys are a ConfigError."""
    specs = list(specs)
    if settings.mock_mode:
        analyzer = HeuristicAnalyzer()
        return {"mock": MockProvider({s.id: s.capability_score for s in specs}, lambda q: analyzer.analyze(q).difficulty)}

    providers: dict[str, LLMProvider] = {}
    for name in sorted({s.provider for s in specs}):
        roles = ", ".join(s.role.upper() for s in specs if s.provider == name)
        if name in OPENAI_COMPATIBLE:
            key_attr, env_name, url_attr, token_param = OPENAI_COMPATIBLE[name]
            providers[name] = OpenAICompatibleProvider(
                name,
                _key(getattr(settings, key_attr), env_name, roles),
                base_url=getattr(settings, url_attr),
                timeout=settings.request_timeout_s,
                token_param=token_param,
            )
        elif name == "gemini":
            providers[name] = GeminiProvider(_key(settings.gemini_api_key, "GEMINI_API_KEY", roles), base_url=settings.gemini_base_url, timeout=settings.request_timeout_s)
        else:
            raise ConfigError(f"Unknown provider '{name}' configured for {roles}. Supported: {SUPPORTED}.")
    return providers


def _key(secret, env_name: str, roles: str) -> str:
    value = secret.get_secret_value()
    if not value:
        raise ConfigError(f"{env_name} is not set but is required by {roles}. Set it in .env or enable MOCK_MODE=true.")
    return value
