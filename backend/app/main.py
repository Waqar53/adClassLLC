"""
FastAPI Main Application

Entry point for the AdClass AI Platform API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
import structlog

from app.config import settings
from app.core.database import init_db, close_db
from app.core.cache import init_redis, close_redis
from app.api.routes import (
    health,
    creative,
    roas,
    churn,
    attribution,
    audience,
    clients,
    campaigns,
    webhooks,
    mlops,
    alerts,
    dashboard,
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting AdClass AI Platform", version=settings.APP_VERSION)
    
    await init_db()
    logger.info("Database connection established")
    
    await init_redis()
    logger.info("Redis connection established")
    
    # TODO: Initialize Kafka consumers
    # TODO: Load ML models
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AdClass AI Platform")
    
    await close_db()
    await close_redis()
    
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## AdClass AI Platform API
    
    The AI System That Turns Ad Data Into Predictable Revenue Growth.
    
    ### Features
    
    * **Creative Performance Predictor** - Predict ad performance before launch
    * **Real-Time ROAS Optimizer** - Dynamic budget allocation
    * **Churn Prediction** - Client health scoring and early warning
    * **Multi-Touch Attribution** - Shapley value-based attribution
    * **Audience Intelligence** - AI-powered audience segmentation
    
    ### Authentication
    
    All endpoints require JWT Bearer token authentication except health checks.
    """,
    version=settings.APP_VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================
# REGISTER API ROUTES
# ===========================================

# Health & Status
app.include_router(
    health.router,
    prefix=f"{settings.API_V1_PREFIX}/health",
    tags=["Health"]
)

# Core Resources
app.include_router(
    clients.router,
    prefix=f"{settings.API_V1_PREFIX}/clients",
    tags=["Clients"]
)

app.include_router(
    campaigns.router,
    prefix=f"{settings.API_V1_PREFIX}/campaigns",
    tags=["Campaigns"]
)

# AI/ML Modules
app.include_router(
    creative.router,
    prefix=f"{settings.API_V1_PREFIX}/creative",
    tags=["Module 1: Creative Predictor"]
)

app.include_router(
    roas.router,
    prefix=f"{settings.API_V1_PREFIX}/roas",
    tags=["Module 2: ROAS Optimizer"]
)

app.include_router(
    churn.router,
    prefix=f"{settings.API_V1_PREFIX}/churn",
    tags=["Module 3: Churn Prediction"]
)

app.include_router(
    attribution.router,
    prefix=f"{settings.API_V1_PREFIX}/attribution",
    tags=["Module 4: Attribution"]
)

app.include_router(
    audience.router,
    prefix=f"{settings.API_V1_PREFIX}/audience",
    tags=["Module 5: Audience Intelligence"]
)

# Webhooks & Integration
app.include_router(
    webhooks.router,
    prefix=f"{settings.API_V1_PREFIX}/webhooks",
    tags=["Webhooks"]
)

# MLOps & Model Management
app.include_router(
    mlops.router,
    prefix=f"{settings.API_V1_PREFIX}/mlops",
    tags=["MLOps"]
)

# Alerts & Notifications
app.include_router(
    alerts.router,
    prefix=f"{settings.API_V1_PREFIX}/alerts",
    tags=["Alerts"]
)

# Dashboard Summary
app.include_router(
    dashboard.router,
    prefix=f"{settings.API_V1_PREFIX}/dashboard",
    tags=["Dashboard"]
)


# ===========================================
# ROOT ENDPOINT
# ===========================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
