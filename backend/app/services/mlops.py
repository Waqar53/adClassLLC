"""
MLOps Infrastructure

MLflow integration, model registry, A/B testing, and monitoring.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path
import numpy as np


class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelType(str, Enum):
    CREATIVE_PREDICTOR = "creative_predictor"
    ROAS_OPTIMIZER = "roas_optimizer"
    CHURN_MODEL = "churn_model"
    ATTRIBUTION = "attribution"
    AUDIENCE = "audience"


@dataclass
class ModelVersion:
    """A versioned model in the registry."""
    model_id: str
    model_type: ModelType
    version: str
    stage: ModelStage
    created_at: datetime
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    artifact_path: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Experiment:
    """MLflow experiment."""
    experiment_id: str
    name: str
    artifact_location: str
    created_at: datetime
    runs: List["ExperimentRun"] = field(default_factory=list)


@dataclass
class ExperimentRun:
    """A single experiment run."""
    run_id: str
    experiment_id: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    artifacts: List[str]
    tags: Dict[str, str]


class MLflowClient:
    """
    MLflow client for experiment tracking and model registry.
    
    In production: Connects to actual MLflow server.
    """
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri
        self.experiments: Dict[str, Experiment] = {}
        self.models: Dict[str, List[ModelVersion]] = {}
        self.active_runs: Dict[str, ExperimentRun] = {}
    
    # Experiment Tracking
    def create_experiment(self, name: str, artifact_location: str = "") -> str:
        """Create a new experiment."""
        experiment_id = hashlib.md5(name.encode()).hexdigest()[:12]
        
        self.experiments[experiment_id] = Experiment(
            experiment_id=experiment_id,
            name=name,
            artifact_location=artifact_location or f"mlflow-artifacts/{name}",
            created_at=datetime.now()
        )
        
        return experiment_id
    
    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get experiment by name."""
        for exp in self.experiments.values():
            if exp.name == name:
                return exp
        return None
    
    def start_run(
        self,
        experiment_id: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start a new experiment run."""
        run_id = hashlib.md5(f"{experiment_id}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            status="RUNNING",
            started_at=datetime.now(),
            ended_at=None,
            metrics={},
            parameters={},
            artifacts=[],
            tags=tags or {}
        )
        
        if run_name:
            run.tags["mlflow.runName"] = run_name
        
        self.active_runs[run_id] = run
        
        if experiment_id in self.experiments:
            self.experiments[experiment_id].runs.append(run)
        
        return run_id
    
    def log_param(self, run_id: str, key: str, value: Any):
        """Log a parameter to a run."""
        if run_id in self.active_runs:
            self.active_runs[run_id].parameters[key] = value
    
    def log_params(self, run_id: str, params: Dict[str, Any]):
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(run_id, key, value)
    
    def log_metric(self, run_id: str, key: str, value: float, step: Optional[int] = None):
        """Log a metric to a run."""
        if run_id in self.active_runs:
            self.active_runs[run_id].metrics[key] = value
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float]):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(run_id, key, value)
    
    def log_artifact(self, run_id: str, local_path: str, artifact_path: str = ""):
        """Log an artifact (file) to a run."""
        if run_id in self.active_runs:
            self.active_runs[run_id].artifacts.append(f"{artifact_path}/{Path(local_path).name}")
    
    def end_run(self, run_id: str, status: str = "FINISHED"):
        """End a run."""
        if run_id in self.active_runs:
            self.active_runs[run_id].status = status
            self.active_runs[run_id].ended_at = datetime.now()
    
    # Model Registry
    def register_model(
        self,
        model_type: ModelType,
        model_artifact_path: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        description: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> ModelVersion:
        """Register a new model version."""
        model_name = model_type.value
        
        # Get next version
        existing = self.models.get(model_name, [])
        version = f"v{len(existing) + 1}"
        
        model_version = ModelVersion(
            model_id=hashlib.md5(f"{model_name}{version}".encode()).hexdigest()[:12],
            model_type=model_type,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime.now(),
            metrics=metrics,
            parameters=parameters,
            artifact_path=model_artifact_path,
            description=description,
            tags=tags or {}
        )
        
        if model_name not in self.models:
            self.models[model_name] = []
        self.models[model_name].append(model_version)
        
        return model_version
    
    def get_model_version(self, model_type: ModelType, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        model_name = model_type.value
        for mv in self.models.get(model_name, []):
            if mv.version == version:
                return mv
        return None
    
    def get_production_model(self, model_type: ModelType) -> Optional[ModelVersion]:
        """Get the production model for a type."""
        model_name = model_type.value
        for mv in self.models.get(model_name, []):
            if mv.stage == ModelStage.PRODUCTION:
                return mv
        return None
    
    def transition_model_stage(
        self,
        model_type: ModelType,
        version: str,
        stage: ModelStage
    ) -> bool:
        """Transition a model to a new stage."""
        model = self.get_model_version(model_type, version)
        if model:
            # If promoting to production, demote current production
            if stage == ModelStage.PRODUCTION:
                current_prod = self.get_production_model(model_type)
                if current_prod and current_prod.version != version:
                    current_prod.stage = ModelStage.ARCHIVED
            
            model.stage = stage
            return True
        return False
    
    def list_model_versions(
        self,
        model_type: ModelType,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersion]:
        """List model versions with optional stage filter."""
        model_name = model_type.value
        versions = self.models.get(model_name, [])
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        return sorted(versions, key=lambda x: x.created_at, reverse=True)


class ModelMonitor:
    """
    Monitor model performance and detect drift.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions: Dict[str, List[Dict[str, Any]]] = {}
        self.actuals: Dict[str, List[Dict[str, Any]]] = {}
        self.metrics_history: Dict[str, List[Dict[str, float]]] = {}
    
    def log_prediction(
        self,
        model_type: ModelType,
        prediction_id: str,
        features: Dict[str, Any],
        prediction: Any,
        timestamp: Optional[datetime] = None
    ):
        """Log a prediction for monitoring."""
        model_name = model_type.value
        
        if model_name not in self.predictions:
            self.predictions[model_name] = []
        
        self.predictions[model_name].append({
            "id": prediction_id,
            "features": features,
            "prediction": prediction,
            "timestamp": timestamp or datetime.now()
        })
        
        # Keep window size
        if len(self.predictions[model_name]) > self.window_size:
            self.predictions[model_name] = self.predictions[model_name][-self.window_size:]
    
    def log_actual(
        self,
        model_type: ModelType,
        prediction_id: str,
        actual: Any
    ):
        """Log actual value for a prediction."""
        model_name = model_type.value
        
        if model_name not in self.actuals:
            self.actuals[model_name] = []
        
        self.actuals[model_name].append({
            "id": prediction_id,
            "actual": actual,
            "timestamp": datetime.now()
        })
    
    def compute_metrics(self, model_type: ModelType) -> Dict[str, float]:
        """Compute current model metrics."""
        model_name = model_type.value
        
        predictions = self.predictions.get(model_name, [])
        actuals = self.actuals.get(model_name, [])
        
        if not predictions or not actuals:
            return {}
        
        # Match predictions with actuals
        actual_map = {a["id"]: a["actual"] for a in actuals}
        matched = [(p["prediction"], actual_map[p["id"]]) 
                   for p in predictions if p["id"] in actual_map]
        
        if not matched:
            return {}
        
        preds = np.array([m[0] for m in matched])
        acts = np.array([m[1] for m in matched])
        
        # Calculate metrics
        mae = float(np.mean(np.abs(preds - acts)))
        mse = float(np.mean((preds - acts) ** 2))
        rmse = float(np.sqrt(mse))
        
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "sample_count": len(matched)
        }
    
    def detect_drift(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Detect data and concept drift.
        
        Returns drift indicators and alerts.
        """
        model_name = model_type.value
        predictions = self.predictions.get(model_name, [])
        
        if len(predictions) < 100:
            return {"drift_detected": False, "message": "Insufficient data"}
        
        # Split into recent vs historical
        midpoint = len(predictions) // 2
        historical = predictions[:midpoint]
        recent = predictions[midpoint:]
        
        # Compare feature distributions (simplified)
        historical_features = [p["features"] for p in historical]
        recent_features = [p["features"] for p in recent]
        
        # Compare prediction distributions
        hist_preds = [p["prediction"] for p in historical]
        recent_preds = [p["prediction"] for p in recent]
        
        hist_mean = np.mean(hist_preds)
        recent_mean = np.mean(recent_preds)
        
        drift_score = abs(recent_mean - hist_mean) / (hist_mean + 1e-6)
        
        return {
            "drift_detected": drift_score > 0.15,
            "drift_score": float(drift_score),
            "historical_mean": float(hist_mean),
            "recent_mean": float(recent_mean),
            "sample_count": len(predictions),
            "alert": drift_score > 0.15,
        }


class ABTestManager:
    """
    A/B testing for model versions.
    """
    
    @dataclass
    class ABTest:
        test_id: str
        name: str
        model_type: ModelType
        control_version: str
        treatment_version: str
        traffic_split: float  # 0-1, fraction for treatment
        started_at: datetime
        ended_at: Optional[datetime]
        status: str  # running, completed, stopped
        results: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self):
        self.tests: Dict[str, ABTestManager.ABTest] = {}
        self.assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {test_id: variant}
    
    def create_test(
        self,
        name: str,
        model_type: ModelType,
        control_version: str,
        treatment_version: str,
        traffic_split: float = 0.5
    ) -> str:
        """Create a new A/B test."""
        test_id = hashlib.md5(f"{name}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        self.tests[test_id] = self.ABTest(
            test_id=test_id,
            name=name,
            model_type=model_type,
            control_version=control_version,
            treatment_version=treatment_version,
            traffic_split=traffic_split,
            started_at=datetime.now(),
            ended_at=None,
            status="running"
        )
        
        return test_id
    
    def get_variant(self, test_id: str, user_id: str) -> str:
        """Get the variant for a user in a test."""
        if test_id not in self.tests or self.tests[test_id].status != "running":
            return "control"
        
        # Check existing assignment
        if user_id in self.assignments and test_id in self.assignments[user_id]:
            return self.assignments[user_id][test_id]
        
        # Assign variant
        test = self.tests[test_id]
        variant = "treatment" if np.random.random() < test.traffic_split else "control"
        
        if user_id not in self.assignments:
            self.assignments[user_id] = {}
        self.assignments[user_id][test_id] = variant
        
        return variant
    
    def record_outcome(
        self,
        test_id: str,
        user_id: str,
        metric_name: str,
        value: float
    ):
        """Record an outcome for a test."""
        if test_id not in self.tests:
            return
        
        test = self.tests[test_id]
        variant = self.assignments.get(user_id, {}).get(test_id)
        
        if not variant:
            return
        
        if "outcomes" not in test.results:
            test.results["outcomes"] = {"control": [], "treatment": []}
        
        test.results["outcomes"][variant].append({
            "user_id": user_id,
            "metric": metric_name,
            "value": value
        })
    
    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        if test_id not in self.tests:
            return {"error": "Test not found"}
        
        test = self.tests[test_id]
        outcomes = test.results.get("outcomes", {"control": [], "treatment": []})
        
        control_values = [o["value"] for o in outcomes["control"]]
        treatment_values = [o["value"] for o in outcomes["treatment"]]
        
        if not control_values or not treatment_values:
            return {"error": "Insufficient data"}
        
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        lift = (treatment_mean - control_mean) / (control_mean + 1e-6)
        
        # Simplified statistical significance (would use scipy in production)
        control_std = np.std(control_values)
        treatment_std = np.std(treatment_values)
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        
        z_score = abs(treatment_mean - control_mean) / (pooled_std + 1e-6)
        significant = z_score > 1.96  # 95% confidence
        
        return {
            "test_id": test_id,
            "control_mean": float(control_mean),
            "treatment_mean": float(treatment_mean),
            "lift": float(lift),
            "lift_percent": float(lift * 100),
            "significant": significant,
            "z_score": float(z_score),
            "control_samples": len(control_values),
            "treatment_samples": len(treatment_values),
            "winner": "treatment" if significant and lift > 0 else ("control" if significant and lift < 0 else "inconclusive")
        }
    
    def end_test(self, test_id: str):
        """End an A/B test."""
        if test_id in self.tests:
            self.tests[test_id].status = "completed"
            self.tests[test_id].ended_at = datetime.now()


class ModelRetrainer:
    """
    Automated model retraining.
    """
    
    def __init__(
        self,
        mlflow_client: MLflowClient,
        monitor: ModelMonitor
    ):
        self.mlflow = mlflow_client
        self.monitor = monitor
        self.schedules: Dict[ModelType, timedelta] = {}
        self.last_retrain: Dict[ModelType, datetime] = {}
    
    def set_schedule(self, model_type: ModelType, interval: timedelta):
        """Set retraining schedule for a model type."""
        self.schedules[model_type] = interval
    
    def should_retrain(self, model_type: ModelType) -> Tuple[bool, str]:
        """Check if model should be retrained."""
        # Check drift
        drift_result = self.monitor.detect_drift(model_type)
        if drift_result.get("drift_detected"):
            return True, "drift_detected"
        
        # Check schedule
        schedule = self.schedules.get(model_type)
        last = self.last_retrain.get(model_type)
        
        if schedule and last:
            if datetime.now() - last > schedule:
                return True, "scheduled"
        
        # Check performance degradation
        metrics = self.monitor.compute_metrics(model_type)
        if metrics.get("rmse", 0) > 0.5:  # Threshold
            return True, "performance_degradation"
        
        return False, "no_action_needed"
    
    async def trigger_retrain(
        self,
        model_type: ModelType,
        train_function: Callable,
        data: Any
    ) -> Optional[ModelVersion]:
        """Trigger model retraining."""
        experiment_name = f"{model_type.value}_retrain"
        
        # Get or create experiment
        experiment = self.mlflow.get_experiment(experiment_name)
        if not experiment:
            experiment_id = self.mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Start run
        run_id = self.mlflow.start_run(
            experiment_id,
            run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        try:
            # Train model
            model, metrics, params = await train_function(data)
            
            # Log to MLflow
            self.mlflow.log_params(run_id, params)
            self.mlflow.log_metrics(run_id, metrics)
            
            # Register model
            model_version = self.mlflow.register_model(
                model_type=model_type,
                model_artifact_path=f"models/{model_type.value}/{datetime.now().strftime('%Y%m%d')}",
                metrics=metrics,
                parameters=params,
                description=f"Retrained on {datetime.now().isoformat()}"
            )
            
            self.last_retrain[model_type] = datetime.now()
            self.mlflow.end_run(run_id, "FINISHED")
            
            return model_version
            
        except Exception as e:
            self.mlflow.end_run(run_id, "FAILED")
            raise


# Singleton instances
_mlflow_client: Optional[MLflowClient] = None
_model_monitor: Optional[ModelMonitor] = None
_ab_test_manager: Optional[ABTestManager] = None


def get_mlflow_client() -> MLflowClient:
    global _mlflow_client
    if _mlflow_client is None:
        _mlflow_client = MLflowClient()
    return _mlflow_client


def get_model_monitor() -> ModelMonitor:
    global _model_monitor
    if _model_monitor is None:
        _model_monitor = ModelMonitor()
    return _model_monitor


def get_ab_test_manager() -> ABTestManager:
    global _ab_test_manager
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    return _ab_test_manager


# Helper type alias
from typing import Tuple
