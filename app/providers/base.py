"""Provider abstraction: one interface, normalized responses, typed errors.

Everything above this layer (executor, judge, pipeline) sees only `LLMProvider`,
`LLMResponse` and `ProviderError` subclasses, never provider-specific payloads.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import httpx
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Provider-independent completion result."""

    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0  # filled in by the executor, which owns timing
    finish_reason: str | None = None


class ProviderError(Exception):
    """Base class for provider failures. `transient` errors are safe to retry."""

    transient: ClassVar[bool] = False

    def __init__(self, message: str, *, provider: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code

    @property
    def error_type(self) -> str:
        return type(self).__name__


class AuthenticationError(ProviderError):
    """Invalid or missing API key."""


class ModelUnavailableError(ProviderError):
    """The requested model ID does not exist or is not accessible."""


class BadRequestError(ProviderError):
    """The provider rejected the request (4xx other than auth/404/429)."""


class EmptyResponseError(ProviderError):
    """The provider returned no usable text (empty completion, safety block, ...)."""


class RateLimitError(ProviderError):
    transient = True

    def __init__(self, message: str, *, provider: str, status_code: int | None = 429, retry_after: float | None = None) -> None:
        super().__init__(message, provider=provider, status_code=status_code)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    transient = True


class ProviderServerError(ProviderError):
    """5xx or malformed provider response."""

    transient = True


class NetworkError(ProviderError):
    transient = True


class LLMProvider(ABC):
    """Common interface for every model provider."""

    name: str

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return a completion. `json_mode` asks the provider for structured JSON output where supported."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return model IDs this provider currently serves (used for startup validation)."""

    async def aclose(self) -> None:
        """Release network resources."""


class HTTPProvider(LLMProvider):
    """Shared httpx plumbing: timeouts, network errors, and HTTP status → typed error mapping."""

    def __init__(self, *, base_url: str, headers: dict[str, str], timeout: float, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout, transport=transport)

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(f"{self.name} request timed out", provider=self.name) from None
        except httpx.TransportError as exc:
            raise NetworkError(f"{self.name} network error: {type(exc).__name__}", provider=self.name) from None
        if response.status_code >= 400:
            raise self._error_for(response)
        try:
            return response.json()
        except ValueError:
            raise ProviderServerError(f"{self.name} returned a non-JSON response", provider=self.name) from None

    def _error_for(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        detail = response.text[:300]
        message = f"{self.name} HTTP {status}: {detail}"
        if status in (401, 403) or (status == 400 and "api key" in detail.lower()):
            return AuthenticationError(f"{self.name} rejected the API key (HTTP {status})", provider=self.name, status_code=status)
        if status == 404:
            return ModelUnavailableError(message, provider=self.name, status_code=status)
        if status == 429 and "request too large" in detail.lower():
            # The request itself exceeds a per-minute limit (e.g. max_tokens > OTPM); retrying can never succeed.
            return BadRequestError(message, provider=self.name, status_code=status)
        if status == 429:
            return RateLimitError(message, provider=self.name, retry_after=_retry_after(response))
        if status == 408:
            return ProviderTimeoutError(message, provider=self.name, status_code=status)
        if status >= 500:
            return ProviderServerError(message, provider=self.name, status_code=status)
        return BadRequestError(message, provider=self.name, status_code=status)

    async def aclose(self) -> None:
        await self._client.aclose()


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None
