"""Analytics service — mode="analytics".

Registers the pipeline and combined router with src.services.registry at
import time so main.py only needs to import this package.
"""

from fastapi import APIRouter

from src.services.analytics.routes.datasets import router as _datasets_router
from src.services.analytics.routes.db_connect import router as _db_connect_router

# Combined router for all analytics-specific routes
router = APIRouter()
router.include_router(_datasets_router)
router.include_router(_db_connect_router)

from src.services.analytics.pipeline import run_query_pipeline  # noqa: E402
from src.services.registry import register  # noqa: E402

register("analytics", run_query_pipeline, router)
