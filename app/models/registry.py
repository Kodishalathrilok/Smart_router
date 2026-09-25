"""Model registry: which concrete model fills each role, plus its metadata.

IDs/providers come from settings (.env); metadata from config/models.json.
Validation fails loudly — a misconfigured model is never swapped for another.
"""

import json
from collections.abc import Mapping

from app.core.config import ROLES, ConfigError, Settings
from app.models.schemas import ModelSpec, Role
from app.providers.base import LLMProvider, ProviderError
from app.router.schemas import Tier


class ModelRegistry:
    def __init__(self, specs: Mapping[Role, ModelSpec]) -> None:
        missing = set(ROLES) - set(specs)
        if missing:
            raise ConfigError(f"Registry is missing roles: {sorted(missing)}")
        self._specs = dict(specs)

    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelRegistry":
        try:
            profiles = json.loads(settings.models_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Cannot read model metadata from {settings.models_config}: {exc}") from None
        specs = {}
        for role in ROLES:
            provider, model = settings.role(role)
            if not model:
                raise ConfigError(f"{role.upper()}_MODEL is not set. Set it in .env (or enable MOCK_MODE=true).")
            if role not in profiles:
                raise ConfigError(f"{settings.models_config} has no entry for role '{role}'.")
            specs[role] = ModelSpec(id=model, provider=provider, role=role, **profiles[role])
        return cls(specs)

    def for_tier(self, tier: Tier) -> ModelSpec:
        return self._specs[tier]

    @property
    def judge(self) -> ModelSpec:
        return self._specs["judge"]

    def all(self) -> list[ModelSpec]:
        return [self._specs[r] for r in ROLES]

    async def validate(self, providers: Mapping[str, LLMProvider]) -> None:
        """Check every configured model is served by its provider. Raises ConfigError otherwise."""
        for provider_name in sorted({s.provider for s in self.all()}):
            if provider_name not in providers:
                raise ConfigError(f"No provider instance for '{provider_name}'.")
            try:
                available = set(await providers[provider_name].list_models())
            except ProviderError as exc:
                raise ConfigError(f"Could not list models from {provider_name} ({exc.error_type}). Check the API key and network.") from None
            for spec in self.all():
                if spec.provider == provider_name and spec.id not in available:
                    raise ConfigError(
                        f"{spec.role.upper()}_MODEL='{spec.id}' is not available from {provider_name}. "
                        "Fix the model ID in .env; no automatic substitution is performed."
                    )
