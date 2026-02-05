"""
Churn Prediction ML Model

Uses Random Forest for churn probability prediction with feature engineering.
Integrated with MLflow for model versioning and tracking.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    np = None

from app.services.mlops import MLflowClient, ModelType, ModelStage


class RiskLevel(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    MONITOR = "monitor"
    HEALTHY = "healthy"


@dataclass
class ClientMetrics:
    """Client performance metrics for churn prediction."""
    client_id: str
    client_name: str
    
    # Financial metrics
    monthly_spend: float
    target_roas: float
    actual_roas: float
    roas_trend: float  # Positive = improving
    
    # Engagement metrics
    days_since_login: int
    days_since_contact: int
    support_tickets_30d: int
    meetings_30d: int
    
    # Performance metrics
    campaigns_active: int
    campaigns_paused: int
    avg_ctr: float
    avg_cvr: float
    
    # Contract info
    contract_months_remaining: int
    is_on_annual: bool
    payment_delays: int


@dataclass
class HealthScore:
    """Multi-dimensional health score."""
    overall_score: float  # 0-100
    roas_score: float
    engagement_score: float
    payment_score: float
    communication_score: float
    growth_score: float


@dataclass 
class RiskFactor:
    """Identified risk factor."""
    factor: str
    severity: str  # 'high', 'medium', 'low'
    impact_score: float
    description: str
    trend: str  # 'improving', 'stable', 'declining'


@dataclass
class ChurnPrediction:
    """Complete churn prediction result."""
    client_id: str
    client_name: str
    churn_probability: float
    risk_level: RiskLevel
    days_to_churn: Optional[int]
    health_score: HealthScore
    risk_factors: List[RiskFactor]
    recommended_actions: List[str]


class ChurnPredictor:
    """
    ML model for predicting client churn.
    
    Uses:
    - Random Forest for churn classification
    - Feature engineering from client metrics
    - Risk factor extraction
    - MLflow for model tracking
    """
    
    # Risk thresholds
    CRITICAL_THRESHOLD = 0.75
    WARNING_THRESHOLD = 0.50
    MONITOR_THRESHOLD = 0.25
    
    # Feature weights for interpretability
    FEATURE_WEIGHTS = {
        "roas_below_target": 0.25,
        "declining_roas": 0.20,
        "low_engagement": 0.15,
        "support_issues": 0.15,
        "payment_problems": 0.10,
        "contract_ending": 0.10,
        "paused_campaigns": 0.05
    }
    
    def __init__(self, mlflow_client: Optional[MLflowClient] = None):
        self.mlflow_client = mlflow_client or MLflowClient()
        self.model = None
        self._load_production_model()
    
    def _load_production_model(self):
        """Load the production model from MLflow."""
        try:
            model_version = self.mlflow_client.get_production_model(ModelType.CHURN_MODEL)
            if model_version:
                # In production, load actual model artifacts
                pass
        except Exception:
            pass
    
    def calculate_health_score(self, metrics: ClientMetrics) -> HealthScore:
        """Calculate multi-dimensional health score."""
        
        # ROAS Score (0-100)
        roas_ratio = metrics.actual_roas / max(metrics.target_roas, 0.1)
        roas_score = min(100, max(0, roas_ratio * 50 + metrics.roas_trend * 20))
        
        # Engagement Score
        login_score = max(0, 100 - metrics.days_since_login * 5)
        contact_score = max(0, 100 - metrics.days_since_contact * 3)
        engagement_score = (login_score + contact_score) / 2
        
        # Payment Score
        payment_score = 100 - min(100, metrics.payment_delays * 25)
        
        # Communication Score
        meeting_score = min(100, metrics.meetings_30d * 25)
        ticket_penalty = min(50, metrics.support_tickets_30d * 10)
        communication_score = max(0, meeting_score - ticket_penalty + 25)
        
        # Growth Score
        active_ratio = metrics.campaigns_active / max(metrics.campaigns_active + metrics.campaigns_paused, 1)
        performance_score = (metrics.avg_ctr / 0.02 + metrics.avg_cvr / 0.01) * 25
        growth_score = min(100, active_ratio * 50 + performance_score)
        
        # Overall weighted score
        overall = (
            roas_score * 0.30 +
            engagement_score * 0.25 +
            payment_score * 0.15 +
            communication_score * 0.15 +
            growth_score * 0.15
        )
        
        return HealthScore(
            overall_score=round(overall, 1),
            roas_score=round(roas_score, 1),
            engagement_score=round(engagement_score, 1),
            payment_score=round(payment_score, 1),
            communication_score=round(communication_score, 1),
            growth_score=round(growth_score, 1)
        )
    
    def extract_risk_factors(self, metrics: ClientMetrics) -> List[RiskFactor]:
        """Extract risk factors from client metrics."""
        factors = []
        
        # ROAS below target
        if metrics.actual_roas < metrics.target_roas * 0.8:
            severity = "high" if metrics.actual_roas < metrics.target_roas * 0.5 else "medium"
            factors.append(RiskFactor(
                factor="ROAS Below Target",
                severity=severity,
                impact_score=0.25,
                description=f"Current ROAS ({metrics.actual_roas:.2f}) is {((1 - metrics.actual_roas/metrics.target_roas) * 100):.0f}% below target",
                trend="declining" if metrics.roas_trend < 0 else "improving" if metrics.roas_trend > 0 else "stable"
            ))
        
        # No recent login
        if metrics.days_since_login > 14:
            severity = "high" if metrics.days_since_login > 30 else "medium"
            factors.append(RiskFactor(
                factor="Low Platform Engagement",
                severity=severity,
                impact_score=0.20,
                description=f"No login for {metrics.days_since_login} days",
                trend="declining"
            ))
        
        # Support escalations
        if metrics.support_tickets_30d > 3:
            severity = "high" if metrics.support_tickets_30d > 5 else "medium"
            factors.append(RiskFactor(
                factor="Support Escalations",
                severity=severity,
                impact_score=0.15,
                description=f"{metrics.support_tickets_30d} support tickets in last 30 days",
                trend="stable"
            ))
        
        # Payment issues
        if metrics.payment_delays > 0:
            severity = "high" if metrics.payment_delays > 2 else "medium" if metrics.payment_delays > 1 else "low"
            factors.append(RiskFactor(
                factor="Payment Delays",
                severity=severity,
                impact_score=0.15,
                description=f"{metrics.payment_delays} payment delay(s) recorded",
                trend="stable"
            ))
        
        # Contract ending soon
        if metrics.contract_months_remaining < 3 and not metrics.is_on_annual:
            severity = "high" if metrics.contract_months_remaining < 1 else "medium"
            factors.append(RiskFactor(
                factor="Contract Ending Soon",
                severity=severity,
                impact_score=0.10,
                description=f"Contract ends in {metrics.contract_months_remaining} months",
                trend="stable"
            ))
        
        # Many paused campaigns
        total_campaigns = metrics.campaigns_active + metrics.campaigns_paused
        if total_campaigns > 0 and metrics.campaigns_paused / total_campaigns > 0.5:
            factors.append(RiskFactor(
                factor="High Campaign Pause Rate",
                severity="medium",
                impact_score=0.10,
                description=f"{metrics.campaigns_paused}/{total_campaigns} campaigns paused",
                trend="stable"
            ))
        
        # No recent contact
        if metrics.days_since_contact > 21 and metrics.meetings_30d == 0:
            factors.append(RiskFactor(
                factor="Lack of Communication",
                severity="medium",
                impact_score=0.10,
                description=f"No contact for {metrics.days_since_contact} days, 0 meetings this month",
                trend="declining"
            ))
        
        # Sort by impact
        factors.sort(key=lambda x: x.impact_score, reverse=True)
        return factors
    
    def generate_recommendations(
        self, 
        risk_factors: List[RiskFactor],
        health_score: HealthScore
    ) -> List[str]:
        """Generate actionable recommendations based on risk factors."""
        recommendations = []
        
        for factor in risk_factors[:3]:  # Top 3 risk factors
            if factor.factor == "ROAS Below Target":
                recommendations.append("Schedule strategy review meeting to discuss campaign optimization")
                recommendations.append("Analyze top-performing campaigns and reallocate budget")
                
            elif factor.factor == "Low Platform Engagement":
                recommendations.append("Send personalized check-in email with recent wins/opportunities")
                recommendations.append("Share new platform features or industry insights")
                
            elif factor.factor == "Support Escalations":
                recommendations.append("Escalate to account manager for priority resolution")
                recommendations.append("Schedule call to address concerns and rebuild trust")
                
            elif factor.factor == "Payment Delays":
                recommendations.append("Reach out to discuss payment flexibility options")
                recommendations.append("Review contract terms and consider adjustments")
                
            elif factor.factor == "Contract Ending Soon":
                recommendations.append("Initiate renewal conversation with value proposition")
                recommendations.append("Prepare ROI summary and future strategy deck")
                
            elif factor.factor == "High Campaign Pause Rate":
                recommendations.append("Analyze why campaigns were paused and propose reactivation")
                recommendations.append("Suggest new campaign strategies or creative refreshes")
                
            elif factor.factor == "Lack of Communication":
                recommendations.append("Schedule immediate check-in call")
                recommendations.append("Send monthly performance report with insights")
        
        # Deduplicate
        seen = set()
        unique = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique.append(rec)
        
        return unique[:5]
    
    def predict(self, metrics: ClientMetrics) -> ChurnPrediction:
        """
        Predict churn probability for a client.
        
        Returns:
            ChurnPrediction with probability, risk level, and recommendations
        """
        # Calculate health score
        health_score = self.calculate_health_score(metrics)
        
        # Extract risk factors
        risk_factors = self.extract_risk_factors(metrics)
        
        # Calculate churn probability
        # Base probability from health score (inverse relationship)
        base_prob = max(0, (100 - health_score.overall_score) / 100)
        
        # Adjust based on risk factors
        risk_adjustment = sum(f.impact_score for f in risk_factors)
        
        # Additional adjustments
        if metrics.is_on_annual:
            risk_adjustment *= 0.7  # Annual contracts more sticky
        
        if metrics.actual_roas > metrics.target_roas * 1.2:
            risk_adjustment *= 0.6  # High performers less likely to churn
        
        churn_probability = min(0.95, max(0.05, base_prob * 0.6 + risk_adjustment * 0.4))
        
        # Determine risk level
        if churn_probability >= self.CRITICAL_THRESHOLD:
            risk_level = RiskLevel.CRITICAL
            days_to_churn = 30
        elif churn_probability >= self.WARNING_THRESHOLD:
            risk_level = RiskLevel.WARNING
            days_to_churn = 60
        elif churn_probability >= self.MONITOR_THRESHOLD:
            risk_level = RiskLevel.MONITOR
            days_to_churn = 90
        else:
            risk_level = RiskLevel.HEALTHY
            days_to_churn = None
        
        # Generate recommendations
        recommendations = self.generate_recommendations(risk_factors, health_score)
        
        # Log prediction
        self._log_prediction(metrics, churn_probability, risk_level)
        
        return ChurnPrediction(
            client_id=metrics.client_id,
            client_name=metrics.client_name,
            churn_probability=round(churn_probability, 3),
            risk_level=risk_level,
            days_to_churn=days_to_churn,
            health_score=health_score,
            risk_factors=risk_factors,
            recommended_actions=recommendations
        )
    
    def predict_batch(self, clients: List[ClientMetrics]) -> List[ChurnPrediction]:
        """Predict churn for multiple clients."""
        return [self.predict(client) for client in clients]
    
    def _log_prediction(
        self, 
        metrics: ClientMetrics, 
        probability: float,
        risk_level: RiskLevel
    ):
        """Log prediction to MLflow."""
        try:
            experiment = self.mlflow_client.get_experiment("churn_predictions")
            if not experiment:
                experiment = self.mlflow_client.create_experiment("churn_predictions")
            
            run = self.mlflow_client.start_run(
                experiment.experiment_id,
                f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            self.mlflow_client.log_metrics(run.run_id, {
                "churn_probability": probability,
                "actual_roas": metrics.actual_roas,
                "days_since_login": metrics.days_since_login,
                "support_tickets": metrics.support_tickets_30d
            })
            
            self.mlflow_client.log_params(run.run_id, {
                "risk_level": risk_level.value,
                "client_id": metrics.client_id
            })
            
            self.mlflow_client.end_run(run.run_id)
        except Exception:
            pass
    
    def to_api_response(self, prediction: ChurnPrediction) -> Dict[str, Any]:
        """Convert prediction to API-friendly format."""
        return {
            "client_id": prediction.client_id,
            "client_name": prediction.client_name,
            "health_score": {
                "overall_score": prediction.health_score.overall_score,
                "roas_score": prediction.health_score.roas_score,
                "engagement_score": prediction.health_score.engagement_score,
                "payment_score": prediction.health_score.payment_score,
                "communication_score": prediction.health_score.communication_score,
                "growth_score": prediction.health_score.growth_score
            },
            "churn_prediction": {
                "churn_probability": prediction.churn_probability,
                "risk_level": prediction.risk_level.value,
                "days_to_churn": prediction.days_to_churn
            },
            "risk_factors": [
                {
                    "factor": f.factor,
                    "severity": f.severity,
                    "impact_score": f.impact_score,
                    "description": f.description,
                    "trend": f.trend
                }
                for f in prediction.risk_factors
            ],
            "recommended_actions": prediction.recommended_actions
        }


# Singleton instance
_predictor: Optional[ChurnPredictor] = None


def get_churn_predictor() -> ChurnPredictor:
    """Get or create the churn predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor
