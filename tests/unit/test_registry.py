import pytest

from app.core.config import ConfigError
from app.models.registry import ModelRegistry
from app.providers.base import AuthenticationError
from app.providers.factory import build_providers
from tests.conftest import FakeProvider, make_settings


def test_mock_registry_has_all_roles(registry):
    assert [s.role for s in registry.all()] == ["cheap", "mid", "strong", "judge"]
    assert registry.for_tier("cheap").id == "mock-cheap"
    assert registry.for_tier("strong").normalized_cost == 10.0
    assert all(s.provider == "mock" for s in registry.all())


def test_live_mode_requires_model_ids(tmp_path):
    with pytest.raises(ConfigError, match="CHEAP_MODEL is not set"):
        ModelRegistry.from_settings(make_settings(tmp_path, mock_mode=False))


def test_models_come_from_config(tmp_path):
    s = make_settings(tmp_path, mock_mode=False, cheap_model="a", mid_model="b", strong_model="c", judge_model="j", judge_provider="groq")
    reg = ModelRegistry.from_settings(s)
    assert [x.id for x in reg.all()] == ["a", "b", "c", "j"]
    assert reg.judge.provider == "groq"


def test_missing_api_key_is_config_error(tmp_path):
    s = make_settings(tmp_path, mock_mode=False, cheap_model="a", mid_model="b", strong_model="c", judge_model="j")
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        build_providers(s, ModelRegistry.from_settings(s).all())


def test_unknown_provider_is_config_error(tmp_path):
    s = make_settings(tmp_path, mock_mode=False, cheap_provider="nope", cheap_model="a", mid_model="b", strong_model="c", judge_model="j",
                      groq_api_key="k", gemini_api_key="k")
    with pytest.raises(ConfigError, match="Unknown provider 'nope'"):
        build_providers(s, ModelRegistry.from_settings(s).all())


async def test_validate_rejects_unavailable_model_without_substitution(registry):
    provider = FakeProvider({"mock-cheap": "", "mock-mid": "", "mock-judge": ""})  # strong missing
    with pytest.raises(ConfigError, match="STRONG_MODEL='mock-strong' is not available"):
        await registry.validate({"mock": provider})


async def test_validate_passes_and_reports_listing_errors(registry):
    ok = FakeProvider({s.id: "" for s in registry.all()})
    await registry.validate({"mock": ok})

    class Broken(FakeProvider):
        async def list_models(self):
            raise AuthenticationError("bad", provider="mock")

    with pytest.raises(ConfigError, match="AuthenticationError"):
        await registry.validate({"mock": Broken({})})
