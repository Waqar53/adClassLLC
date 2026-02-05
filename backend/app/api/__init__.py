"""API routes package."""

from app.api.routes import (
    health,
    creative,
    roas,
    churn,
    attribution,
    audience,
    clients,
    campaigns,
)

__all__ = [
    "health",
    "creative",
    "roas",
    "churn",
    "attribution",
    "audience",
    "clients",
    "campaigns",
]
