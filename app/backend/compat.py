"""Compatibility helpers for dependency version differences."""

from __future__ import annotations

from typing import Any


def model_to_dict(model: Any, **kwargs) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)
