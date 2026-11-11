"""Prediction / forecasting service — stub for the Prediction service tier.

The forecasting service is not yet active. Routes here return a structured
'coming_soon' response so the orchestrator and frontend can handle it cleanly.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/{project}")
async def predict(project: str) -> dict:
    return {
        "status": "coming_soon",
        "message": (
            "The Prediction service is under development. "
            "Forecasting and predictive models will be available in a future release."
        ),
    }


@router.get("/health")
async def predict_health() -> dict:
    return {"status": "coming_soon"}
