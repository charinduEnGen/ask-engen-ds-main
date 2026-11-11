import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI

import src.services.analytics  # noqa: F401 — triggers self-registration in src.services.registry

from src.routes.dashboard import router as dashboard_router
from src.routes.glossary import router as glossary_router
from src.routes.knowledge import router as knowledge_router
from src.routes.llm import router as llm_router
from src.routes.predict import router as predict_router
from src.schemas.user_request import QueryResponse, UserRequest
from src.schemas.query_stages import EnrichRequest, SqlStageResponse
from src.services.analytics.pipeline import run_enrich_stage, run_sql_stage
from src.core.session import memory as session_memory
from src.services.registry import get_all_routers, get_pipeline
from src.utils.logging_utils import log_timestamp

logger = logging.getLogger("analytics_engine")

_SESSION_PURGE_INTERVAL_SECONDS = 60 * 60


async def _purge_expired_sessions_loop() -> None:
    while True:
        try:
            await asyncio.sleep(_SESSION_PURGE_INTERVAL_SECONDS)
            deleted = await asyncio.to_thread(session_memory.purge_expired_sessions)
            print(f"[{log_timestamp()}][SESSION_PURGE] swept data/sessions/ -> deleted {deleted} expired file(s)")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Session purge sweep failed: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pid = os.getpid()
    print(f"[{log_timestamp()}][STARTUP][pid={pid}] ready — projects load lazily on first ingest/query")
    purge_task = asyncio.create_task(_purge_expired_sessions_loop())
    yield
    purge_task.cancel()
    try:
        await purge_task
    except asyncio.CancelledError:
        pass
    print(f"[{log_timestamp()}][SHUTDOWN][pid={pid}] worker stopping")


app = FastAPI(title="Ask Engen Analytics Engine", lifespan=lifespan)

app.include_router(dashboard_router)
app.include_router(glossary_router)
app.include_router(knowledge_router)
app.include_router(llm_router)
app.include_router(predict_router)

for _svc_router in get_all_routers():
    app.include_router(_svc_router)


@app.post("/query/{project}", response_model=QueryResponse)
async def query_by_project(project: str, user_request: UserRequest, background_tasks: BackgroundTasks) -> QueryResponse:
    if not user_request.mode:
        user_request.mode = project
    pipeline_fn = get_pipeline(user_request.mode) or get_pipeline("analytics")
    return await pipeline_fn(user_request, background_tasks)


@app.post("/query", response_model=QueryResponse)
async def query_default(user_request: UserRequest, background_tasks: BackgroundTasks) -> QueryResponse:
    pipeline_fn = get_pipeline(user_request.mode or "analytics") or get_pipeline("analytics")
    return await pipeline_fn(user_request, background_tasks)


@app.post("/query/{project}/sql", response_model=SqlStageResponse)
async def query_sql_stage(project: str, user_request: UserRequest, background_tasks: BackgroundTasks) -> SqlStageResponse:
    if not user_request.mode:
        user_request.mode = project
    return await run_sql_stage(user_request, background_tasks)


@app.post("/query/{project}/enrich", response_model=QueryResponse)
async def query_enrich_stage(project: str, req: EnrichRequest, background_tasks: BackgroundTasks) -> QueryResponse:
    return await run_enrich_stage(req, background_tasks)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    print(
        f"[{log_timestamp()}][STARTUP] launching uvicorn with workers=1"
    )
    uvicorn.run("main:app", host="0.0.0.0", port=5005, workers=1)
