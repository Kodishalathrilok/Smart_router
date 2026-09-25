"""POST /api/query and GET /api/query/{query_id}."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_runtime
from app.core.bootstrap import Runtime
from app.schemas.common import QueryRecord, QueryRequest, QueryResponse

router = APIRouter(prefix="/api")


@router.post("/query", response_model=QueryResponse)
async def submit_query(body: QueryRequest, runtime: Runtime = Depends(get_runtime)) -> QueryResponse:
    """Run the SmartRouter pipeline. Provider failures are reported via final_status, not HTTP 5xx."""
    limit = runtime.settings.max_query_chars
    if len(body.query) > limit:
        raise HTTPException(status_code=413, detail=f"query exceeds MAX_QUERY_CHARS ({limit})")
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank")
    record = await runtime.pipeline.run(body.query, mode="smart")
    return QueryResponse.from_record(record)


@router.get("/query/{query_id}", response_model=QueryRecord)
async def get_query(query_id: str, runtime: Runtime = Depends(get_runtime)) -> QueryRecord:
    record = runtime.queries.get(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown query_id {query_id}")
    return record
