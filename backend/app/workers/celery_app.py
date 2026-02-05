"""
Celery Task Definitions

Background tasks for data processing and ML operations.
"""

from celery import Celery, Task
from datetime import datetime, timedelta
from typing import Optional
import json

from app.config import settings


# Initialize Celery
celery_app = Celery(
    "adclass_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"]
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3300,  # 55 min soft limit
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=86400,  # Results expire in 24h
)

# Scheduled tasks (beat)
celery_app.conf.beat_schedule = {
    "sync-meta-campaigns": {
        "task": "app.workers.tasks.sync_platform_data",
        "schedule": timedelta(minutes=30),
        "args": ("meta",),
    },
    "sync-google-campaigns": {
        "task": "app.workers.tasks.sync_platform_data",
        "schedule": timedelta(minutes=30),
        "args": ("google",),
    },
    "sync-tiktok-campaigns": {
        "task": "app.workers.tasks.sync_platform_data",
        "schedule": timedelta(minutes=30),
        "args": ("tiktok",),
    },
    "run-roas-optimization": {
        "task": "app.workers.tasks.run_roas_optimization",
        "schedule": timedelta(minutes=30),
    },
    "update-client-health": {
        "task": "app.workers.tasks.update_client_health",
        "schedule": timedelta(hours=1),
    },
    "compute-attribution": {
        "task": "app.workers.tasks.compute_attribution_reports",
        "schedule": timedelta(hours=6),
    },
    "refresh-audience-segments": {
        "task": "app.workers.tasks.refresh_audience_segments",
        "schedule": timedelta(hours=24),
    },
}


class BaseTask(Task):
    """Base task class with error handling."""
    
    abstract = True
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        print(f"Task {self.name} [{task_id}] failed: {exc}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        print(f"Task {self.name} [{task_id}] succeeded")
