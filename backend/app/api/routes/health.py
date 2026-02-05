"""
Health Check API Routes

Provides system health and status endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_db
from app.core.cache import get_redis
from app.config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


@router.get("/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """
    Detailed health check with database and cache status.
    """
    health = {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "components": {}
    }
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        health["components"]["database"] = {"status": "healthy"}
    except Exception as e:
        health["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health["status"] = "degraded"
    
    # Check Redis
    try:
        redis = get_redis()
        await redis.ping()
        health["components"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health["status"] = "degraded"
    
    return health


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """
    Kubernetes readiness probe.
    Returns 200 only if all dependencies are ready.
    """
    try:
        await db.execute(text("SELECT 1"))
        redis = get_redis()
        await redis.ping()
        return {"ready": True}
    except Exception:
        return {"ready": False}


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"alive": True}
