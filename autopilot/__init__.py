"""Autopilot backend package exposing the FastAPI application factory."""

from .api import create_app

__all__ = ["create_app"]
