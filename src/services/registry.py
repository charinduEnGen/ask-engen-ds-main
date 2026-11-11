"""Service registry — maps mode strings to their pipeline entry points and
FastAPI routers. Each service registers itself at import time via register().
main.py calls get_all_routers() and routes queries via get_pipeline().
"""

from typing import Callable, Optional

from fastapi import APIRouter

_PIPELINE_REGISTRY: dict[str, Callable] = {}
_ROUTER_REGISTRY: dict[str, APIRouter] = {}


def register(mode: str, pipeline_fn: Callable, router: Optional[APIRouter] = None) -> None:
    """Called once per service at import time."""
    _PIPELINE_REGISTRY[mode] = pipeline_fn
    if router is not None:
        _ROUTER_REGISTRY[mode] = router


def get_pipeline(mode: str) -> Optional[Callable]:
    """Returns the pipeline entry point for `mode`, or None if not registered."""
    return _PIPELINE_REGISTRY.get(mode)


def get_all_routers() -> list[APIRouter]:
    return list(_ROUTER_REGISTRY.values())


def registered_modes() -> list[str]:
    return list(_PIPELINE_REGISTRY.keys())
