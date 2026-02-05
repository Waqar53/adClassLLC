"""
Churn Prediction & Client Health ML Models

XGBoost/LightGBM ensemble with survival analysis for churn prediction.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime, date
# Note: In production, import actual libraries:
# import xgboost as xgb
# import lightgbm as lgb
# from lifelines import CoxPHFitter


@dataclass
class ClientFeatures:
    """Features for churn prediction."""
    client_id: str
    
    # ROAS metrics
    current_roas: float
    target_roas: float
    roas_7d_avg: float
    roas_30d_avg: float
    roas_trend: float  # Slope
    
    # Engagement
    dashboard_logins_7d: int
    dashboard_logins_30d: int
    meeting_attendance_rate: float
    email_open_rate: float
    
    # Communication
    avg_response_time_hours: float
    last_contact_days: int
    support_tickets_30d: int
    ticket_sentiment_avg: float
    
    # Financial
    monthly_spend: float
    revenue_mom_change: float
    payment_delay_avg_days: float
    invoices_outstanding: int
    
    # Contract
    contract_days_remaining: int
    is_month_to_month: bool
    tenure_months: int


@dataclass
class HealthScoreComponents:
    """Breakdown of health score."""
    overall_score: int
    roas_score: int
    engagement_score: int
    payment_score: int
    communication_score: int
    growth_score: int


@dataclass
class ChurnPredictionResult:
    """Churn prediction result."""
    churn_probability: float
    risk_level: str  # critical, warning, monitor, healthy
    predicted_churn_date: Optional[date]
    days_to_churn: Optional[int]
    survival_curve: List[Tuple[int, float]]  # (days, probability)
    confidence: float


@dataclass
class RiskFactor:
    """Individual risk factor."""
    factor: str
    severity: str
    impact: float
    description: str
    trend: str


class XGBoostChurnModel:
    """
    XGBoost classifier for churn prediction.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            'roas_ratio', 'roas_trend', 'login_frequency', 'engagement_trend',
            'response_time', 'payment_health', 'revenue_growth', 'tenure',
            'contract_remaining', 'support_sentiment'
        ]
        self.feature_importances = {}
        
    def prepare_features(self, client: ClientFeatures) -> np.ndarray:
        """Convert client features to model input."""
        features = np.array([
            client.current_roas / client.target_roas if client.target_roas > 0 else 1.0,
            client.roas_trend,
            client.dashboard_logins_30d / 30,  # Normalize to daily
            (client.dashboard_logins_7d / 7) - (client.dashboard_logins_30d / 30),
            1.0 / (1.0 + client.avg_response_time_hours / 24),  # Lower is better
            1.0 / (1.0 + client.payment_delay_avg_days / 30),
            client.revenue_mom_change,
            np.log1p(client.tenure_months),
            client.contract_days_remaining / 365,
            client.ticket_sentiment_avg if client.support_tickets_30d > 0 else 0.5
        ])
        return features.reshape(1, -1)
    
    def predict_proba(self, features: np.ndarray) -> float:
        """
        Predict churn probability.
        
        Note: In production, this uses trained XGBoost model.
        Here we simulate with a heuristic.
        """
        # Simulated prediction based on feature heuristics
        roas_ratio = features[0, 0]
        login_freq = features[0, 2]
        response_health = features[0, 4]
        payment_health = features[0, 5]
        revenue_growth = features[0, 6]
        
        # Simple weighted score (would be ML model in production)
        score = (
            0.3 * (1 - min(1, roas_ratio)) +
            0.2 * (1 - min(1, login_freq * 5)) +
            0.15 * (1 - response_health) +
            0.15 * (1 - payment_health) +
            0.2 * max(0, -revenue_growth)
        )
        
        return min(0.95, max(0.05, score))


class LightGBMChurnModel:
    """
    LightGBM classifier for churn prediction.
    """
    
    def __init__(self):
        self.model = None
        
    def predict_proba(self, features: np.ndarray) -> float:
        """Predict churn probability."""
        # Simulated - would use trained LightGBM
        base_score = features.mean() * 0.5
        return min(0.95, max(0.05, base_score))


class SurvivalAnalyzer:
    """
    Cox Proportional Hazards model for time-to-churn estimation.
    """
    
    def __init__(self):
        self.baseline_survival = None
        self.hazard_ratios = {}
        
    def predict_survival(
        self,
        features: np.ndarray,
        time_points: List[int] = [30, 60, 90, 120, 180]
    ) -> List[Tuple[int, float]]:
        """
        Predict survival probability at given time points.
        
        Returns list of (days, survival_probability) tuples.
        """
        # Simulated survival curve
        # In production, uses fitted CoxPH model
        
        base_risk = features.mean()
        survival = []
        
        for days in time_points:
            # Exponential decay with risk adjustment
            prob = np.exp(-0.005 * days * (1 + base_risk))
            survival.append((days, float(prob)))
        
        return survival
    
    def predict_median_survival(self, features: np.ndarray) -> Optional[int]:
        """Predict median survival time (50% probability threshold)."""
        survival = self.predict_survival(features, list(range(30, 365, 30)))
        
        for days, prob in survival:
            if prob < 0.5:
                return days
        
        return None  # Survival > 365 days


class ChurnEnsemble:
    """
    Ensemble of XGBoost and LightGBM for churn prediction.
    """
    
    def __init__(self, xgb_weight: float = 0.5):
        self.xgb = XGBoostChurnModel()
        self.lgb = LightGBMChurnModel()
        self.xgb_weight = xgb_weight
        self.lgb_weight = 1 - xgb_weight
        
    def predict(self, features: np.ndarray) -> float:
        """Ensemble prediction."""
        xgb_prob = self.xgb.predict_proba(features)
        lgb_prob = self.lgb.predict_proba(features)
        
        return self.xgb_weight * xgb_prob + self.lgb_weight * lgb_prob


class ClientHealthScorer:
    """
    Composite health scoring system.
    """
    
    WEIGHTS = {
        'roas': 0.25,
        'engagement': 0.20,
        'payment': 0.15,
        'communication': 0.20,
        'growth': 0.20
    }
    
    def calculate_score(self, client: ClientFeatures) -> HealthScoreComponents:
        """Calculate health score components."""
        
        # ROAS Score (0-100)
        if client.target_roas > 0:
            roas_ratio = client.current_roas / client.target_roas
            roas_score = min(100, int(roas_ratio * 100))
        else:
            roas_score = 50
        
        # Engagement Score
        logins_score = min(100, client.dashboard_logins_30d * 5)
        meeting_score = int(client.meeting_attendance_rate * 100)
        engagement_score = int(0.6 * logins_score + 0.4 * meeting_score)
        
        # Payment Score
        delay_penalty = min(100, client.payment_delay_avg_days * 10)
        payment_score = max(0, 100 - delay_penalty)
        
        # Communication Score
        response_penalty = min(100, client.avg_response_time_hours * 2)
        recency_penalty = min(50, client.last_contact_days * 2)
        communication_score = max(0, 100 - response_penalty - recency_penalty)
        
        # Growth Score
        growth_pct = client.revenue_mom_change * 100
        growth_score = min(100, max(0, 50 + growth_pct * 2))
        
        # Overall weighted score
        overall = int(
            self.WEIGHTS['roas'] * roas_score +
            self.WEIGHTS['engagement'] * engagement_score +
            self.WEIGHTS['payment'] * payment_score +
            self.WEIGHTS['communication'] * communication_score +
            self.WEIGHTS['growth'] * growth_score
        )
        
        return HealthScoreComponents(
            overall_score=overall,
            roas_score=roas_score,
            engagement_score=engagement_score,
            payment_score=payment_score,
            communication_score=communication_score,
            growth_score=growth_score
        )


class ChurnPredictorService:
    """
    High-level service for churn prediction and health scoring.
    """
    
    RISK_THRESHOLDS = {
        'critical': 0.7,
        'warning': 0.5,
        'monitor': 0.3
    }
    
    def __init__(self):
        self.ensemble = ChurnEnsemble()
        self.survival = SurvivalAnalyzer()
        self.health_scorer = ClientHealthScorer()
        
    def predict(self, client: ClientFeatures) -> Tuple[ChurnPredictionResult, HealthScoreComponents, List[RiskFactor]]:
        """
        Full churn prediction with health scoring.
        
        Returns:
            - Churn prediction result
            - Health score components
            - Risk factors
        """
        # Prepare features
        features = self.ensemble.xgb.prepare_features(client)
        
        # Get churn probability
        churn_prob = self.ensemble.predict(features)
        
        # Get survival curve
        survival_curve = self.survival.predict_survival(features)
        median_days = self.survival.predict_median_survival(features)
        
        # Determine risk level
        if churn_prob >= self.RISK_THRESHOLDS['critical']:
            risk_level = 'critical'
        elif churn_prob >= self.RISK_THRESHOLDS['warning']:
            risk_level = 'warning'
        elif churn_prob >= self.RISK_THRESHOLDS['monitor']:
            risk_level = 'monitor'
        else:
            risk_level = 'healthy'
        
        # Calculate health score
        health = self.health_scorer.calculate_score(client)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(client, health)
        
        # Build prediction result
        prediction = ChurnPredictionResult(
            churn_probability=churn_prob,
            risk_level=risk_level,
            predicted_churn_date=date.today().replace(day=1) if median_days else None,
            days_to_churn=median_days,
            survival_curve=survival_curve,
            confidence=0.85  # Would be calibrated in production
        )
        
        return prediction, health, risk_factors
    
    def _identify_risk_factors(
        self,
        client: ClientFeatures,
        health: HealthScoreComponents
    ) -> List[RiskFactor]:
        """Identify key risk factors for a client."""
        factors = []
        
        if health.roas_score < 70:
            factors.append(RiskFactor(
                factor="Below Target ROAS",
                severity="high" if health.roas_score < 50 else "medium",
                impact=0.25,
                description=f"ROAS at {client.current_roas:.2f}x vs target {client.target_roas:.2f}x",
                trend="declining" if client.roas_trend < 0 else "stable"
            ))
        
        if health.engagement_score < 60:
            factors.append(RiskFactor(
                factor="Low Engagement",
                severity="medium",
                impact=0.20,
                description=f"Only {client.dashboard_logins_30d} dashboard logins in 30 days",
                trend="declining"
            ))
        
        if health.communication_score < 60:
            factors.append(RiskFactor(
                factor="Communication Gap",
                severity="medium",
                impact=0.20,
                description=f"Average response time: {client.avg_response_time_hours:.0f} hours",
                trend="declining"
            ))
        
        if health.payment_score < 80:
            factors.append(RiskFactor(
                factor="Payment Delays",
                severity="medium",
                impact=0.15,
                description=f"Average payment delay: {client.payment_delay_avg_days:.0f} days",
                trend="stable"
            ))
        
        if client.revenue_mom_change < -0.1:
            factors.append(RiskFactor(
                factor="Revenue Decline",
                severity="high",
                impact=0.20,
                description=f"Revenue down {abs(client.revenue_mom_change)*100:.0f}% month-over-month",
                trend="declining"
            ))
        
        return sorted(factors, key=lambda x: x.impact, reverse=True)


# Singleton service
_churn_service: Optional[ChurnPredictorService] = None


def get_churn_predictor() -> ChurnPredictorService:
    """Get or create the churn predictor service."""
    global _churn_service
    if _churn_service is None:
        _churn_service = ChurnPredictorService()
    return _churn_service
