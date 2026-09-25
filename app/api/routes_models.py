"""GET /api/models: the configured registry (never includes API keys)."""

from fastapi import APIRouter, Depends

from app.api.deps import get_runtime
from app.core.bootstrap import Runtime

router = APIRouter(prefix="/api")


@router.get("/models")
async def models(runtime: Runtime = Depends(get_runtime)) -> dict:
    s = runtime.settings
    return {
        "mock_mode": s.mock_mode,
        "models": [spec.model_dump() for spec in runtime.registry.all()],
        "routing": {"cheap_max_difficulty": s.cheap_max_difficulty, "mid_max_difficulty": s.mid_max_difficulty},
        "quality_threshold": s.quality_threshold,
        "max_escalations": s.max_escalations,
    }
