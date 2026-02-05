"""
MLOps API Routes

Model management, experiments, and A/B testing endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

from app.services.mlops import (
    get_mlflow_client,
    get_model_monitor,
    get_ab_test_manager,
    ModelType,
    ModelStage
)

router = APIRouter(prefix="/mlops", tags=["MLOps"])


class ModelRegistrationRequest(BaseModel):
    model_type: str
    artifact_path: str
    metrics: dict
    parameters: dict
    description: str = ""
    tags: dict = {}


class StageTransitionRequest(BaseModel):
    model_type: str
    version: str
    stage: str


class ABTestRequest(BaseModel):
    name: str
    model_type: str
    control_version: str
    treatment_version: str
    traffic_split: float = 0.5


class ABTestOutcomeRequest(BaseModel):
    test_id: str
    user_id: str
    metric_name: str
    value: float


# Model Registry
@router.get("/models")
async def list_models(
    model_type: Optional[str] = None,
    stage: Optional[str] = None
):
    """List registered models."""
    client = get_mlflow_client()
    
    models = {}
    if model_type:
        try:
            mt = ModelType(model_type)
            stage_filter = ModelStage(stage) if stage else None
            models[model_type] = [
                {
                    "model_id": m.model_id,
                    "version": m.version,
                    "stage": m.stage.value,
                    "created_at": m.created_at.isoformat(),
                    "metrics": m.metrics,
                    "description": m.description
                }
                for m in client.list_model_versions(mt, stage_filter)
            ]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid model type")
    else:
        for mt in ModelType:
            models[mt.value] = [
                {
                    "model_id": m.model_id,
                    "version": m.version,
                    "stage": m.stage.value,
                    "created_at": m.created_at.isoformat(),
                    "metrics": m.metrics
                }
                for m in client.list_model_versions(mt)
            ]
    
    return {"models": models}


@router.get("/models/{model_type}/production")
async def get_production_model(model_type: str):
    """Get production model for a type."""
    client = get_mlflow_client()
    
    try:
        mt = ModelType(model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    model = client.get_production_model(mt)
    if not model:
        raise HTTPException(status_code=404, detail="No production model found")
    
    return {
        "model_id": model.model_id,
        "version": model.version,
        "stage": model.stage.value,
        "created_at": model.created_at.isoformat(),
        "metrics": model.metrics,
        "parameters": model.parameters,
        "artifact_path": model.artifact_path
    }


@router.post("/models/register")
async def register_model(request: ModelRegistrationRequest):
    """Register a new model version."""
    client = get_mlflow_client()
    
    try:
        mt = ModelType(request.model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    model = client.register_model(
        model_type=mt,
        model_artifact_path=request.artifact_path,
        metrics=request.metrics,
        parameters=request.parameters,
        description=request.description,
        tags=request.tags
    )
    
    return {
        "model_id": model.model_id,
        "version": model.version,
        "stage": model.stage.value
    }


@router.post("/models/transition")
async def transition_model_stage(request: StageTransitionRequest):
    """Transition model to new stage."""
    client = get_mlflow_client()
    
    try:
        mt = ModelType(request.model_type)
        stage = ModelStage(request.stage)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model type or stage")
    
    success = client.transition_model_stage(mt, request.version, stage)
    if not success:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    return {"status": "success", "new_stage": stage.value}


# Monitoring
@router.get("/monitoring/{model_type}")
async def get_model_metrics(model_type: str):
    """Get current model monitoring metrics."""
    monitor = get_model_monitor()
    
    try:
        mt = ModelType(model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    metrics = monitor.compute_metrics(mt)
    drift = monitor.detect_drift(mt)
    
    return {
        "model_type": model_type,
        "metrics": metrics,
        "drift": drift,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/monitoring/{model_type}/log")
async def log_prediction(
    model_type: str,
    prediction_id: str,
    features: dict,
    prediction: float
):
    """Log a prediction for monitoring."""
    monitor = get_model_monitor()
    
    try:
        mt = ModelType(model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    monitor.log_prediction(mt, prediction_id, features, prediction)
    
    return {"status": "logged"}


# A/B Testing
@router.post("/ab-tests")
async def create_ab_test(request: ABTestRequest):
    """Create a new A/B test."""
    manager = get_ab_test_manager()
    
    try:
        mt = ModelType(request.model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    test_id = manager.create_test(
        name=request.name,
        model_type=mt,
        control_version=request.control_version,
        treatment_version=request.treatment_version,
        traffic_split=request.traffic_split
    )
    
    return {"test_id": test_id, "status": "running"}


@router.get("/ab-tests")
async def list_ab_tests():
    """List all A/B tests."""
    manager = get_ab_test_manager()
    
    return {
        "tests": [
            {
                "test_id": t.test_id,
                "name": t.name,
                "model_type": t.model_type.value,
                "status": t.status,
                "started_at": t.started_at.isoformat(),
                "traffic_split": t.traffic_split
            }
            for t in manager.tests.values()
        ]
    }


@router.get("/ab-tests/{test_id}")
async def get_ab_test(test_id: str):
    """Get A/B test details and results."""
    manager = get_ab_test_manager()
    
    if test_id not in manager.tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test = manager.tests[test_id]
    analysis = manager.analyze_test(test_id)
    
    return {
        "test": {
            "test_id": test.test_id,
            "name": test.name,
            "model_type": test.model_type.value,
            "control_version": test.control_version,
            "treatment_version": test.treatment_version,
            "traffic_split": test.traffic_split,
            "status": test.status,
            "started_at": test.started_at.isoformat()
        },
        "analysis": analysis
    }


@router.post("/ab-tests/{test_id}/assign")
async def get_variant(test_id: str, user_id: str):
    """Get variant assignment for a user."""
    manager = get_ab_test_manager()
    
    if test_id not in manager.tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    variant = manager.get_variant(test_id, user_id)
    
    return {"test_id": test_id, "user_id": user_id, "variant": variant}


@router.post("/ab-tests/{test_id}/outcome")
async def record_outcome(test_id: str, request: ABTestOutcomeRequest):
    """Record an outcome for A/B test."""
    manager = get_ab_test_manager()
    
    if test_id not in manager.tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    manager.record_outcome(
        test_id=test_id,
        user_id=request.user_id,
        metric_name=request.metric_name,
        value=request.value
    )
    
    return {"status": "recorded"}


@router.post("/ab-tests/{test_id}/end")
async def end_ab_test(test_id: str):
    """End an A/B test."""
    manager = get_ab_test_manager()
    
    if test_id not in manager.tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    manager.end_test(test_id)
    
    return {"status": "ended", "final_analysis": manager.analyze_test(test_id)}
