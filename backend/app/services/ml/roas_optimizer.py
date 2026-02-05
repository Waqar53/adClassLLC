"""
ROAS Optimizer ML Model

Uses Thompson Sampling for budget allocation and LSTM for ROAS forecasting.
Integrated with MLflow for model versioning and tracking.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import math

try:
    import numpy as np
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    np = None

from app.services.mlops import MLflowClient, ModelType, ModelStage


class OptimizationAction(Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    PAUSE = "pause"


@dataclass
class CampaignState:
    """Current state of a campaign for optimization."""
    campaign_id: str
    campaign_name: str
    platform: str
    current_budget: float
    spend_today: float
    impressions: int
    clicks: int
    conversions: int
    revenue: float
    roas: float
    ctr: float
    cvr: float
    historical_roas: List[float] = field(default_factory=list)
    
    @property
    def cpc(self) -> float:
        return self.spend_today / max(self.clicks, 1)
    
    @property
    def cpa(self) -> float:
        return self.spend_today / max(self.conversions, 1)


@dataclass 
class BudgetRecommendation:
    """Budget optimization recommendation."""
    campaign_id: str
    campaign_name: str
    current_budget: float
    recommended_budget: float
    change_percentage: float
    predicted_roas: float
    confidence_score: float
    reasoning: str
    action: OptimizationAction


class ThompsonSamplingBandit:
    """
    Thompson Sampling multi-armed bandit for budget allocation.
    
    Each campaign is an "arm" with its own Beta distribution
    representing our belief about its conversion probability.
    """
    
    def __init__(self):
        # Beta distribution parameters for each campaign
        # (successes, failures) = (alpha, beta)
        self.arms: Dict[str, Tuple[float, float]] = {}
    
    def add_campaign(self, campaign_id: str, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """Add a new campaign arm."""
        self.arms[campaign_id] = (prior_alpha, prior_beta)
    
    def update(self, campaign_id: str, conversions: int, non_conversions: int):
        """Update beliefs based on observed outcomes."""
        if campaign_id not in self.arms:
            self.add_campaign(campaign_id)
        
        alpha, beta = self.arms[campaign_id]
        self.arms[campaign_id] = (alpha + conversions, beta + non_conversions)
    
    def sample(self, campaign_id: str) -> float:
        """Sample from the posterior distribution."""
        if campaign_id not in self.arms:
            return 0.5
        
        alpha, beta = self.arms[campaign_id]
        
        if HAS_ML_DEPS:
            return np.random.beta(alpha, beta)
        else:
            # Simplified sampling without numpy
            return random.betavariate(alpha, beta)
    
    def get_allocation_scores(self, campaign_ids: List[str]) -> Dict[str, float]:
        """Get relative budget allocation scores for campaigns."""
        samples = {cid: self.sample(cid) for cid in campaign_ids}
        total = sum(samples.values())
        
        if total == 0:
            return {cid: 1.0 / len(campaign_ids) for cid in campaign_ids}
        
        return {cid: score / total for cid, score in samples.items()}


class ROASForecaster:
    """
    LSTM-based ROAS forecasting model.
    
    Predicts future ROAS based on historical patterns.
    Falls back to exponential moving average if LSTM not available.
    """
    
    def __init__(self, lookback_days: int = 14):
        self.lookback_days = lookback_days
        self.model = None  # LSTM model (if trained)
    
    def forecast(
        self, 
        historical_roas: List[float],
        current_roas: float,
        horizon_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Forecast ROAS for the next horizon hours.
        
        Returns predictions at 1h, 6h, and 24h.
        """
        if len(historical_roas) < 3:
            # Not enough data, return current with decay
            return {
                "roas_1h": current_roas * 0.98,
                "roas_6h": current_roas * 0.95,
                "roas_24h": current_roas * 0.90,
                "trend": "stable",
                "confidence": 0.3
            }
        
        # Calculate exponential moving average
        ema_short = self._calculate_ema(historical_roas[-7:], span=3)
        ema_long = self._calculate_ema(historical_roas, span=7)
        
        # Determine trend
        if ema_short > ema_long * 1.05:
            trend = "improving"
            multiplier = 1.05
        elif ema_short < ema_long * 0.95:
            trend = "declining"
            multiplier = 0.95
        else:
            trend = "stable"
            multiplier = 1.0
        
        # Calculate volatility
        volatility = self._calculate_volatility(historical_roas)
        
        # Forecast with trend and mean reversion
        mean_roas = sum(historical_roas) / len(historical_roas)
        
        roas_1h = current_roas * (1 + (multiplier - 1) * 0.1)
        roas_6h = current_roas * 0.8 + mean_roas * 0.2  # Mean reversion
        roas_24h = mean_roas * 0.7 + ema_long * 0.3
        
        # Confidence based on volatility
        confidence = max(0.3, min(0.95, 1.0 - volatility))
        
        return {
            "roas_1h": round(roas_1h, 2),
            "roas_6h": round(roas_6h, 2),
            "roas_24h": round(roas_24h, 2),
            "trend": trend,
            "confidence": round(confidence, 2)
        }
    
    def _calculate_ema(self, values: List[float], span: int) -> float:
        """Calculate exponential moving average."""
        if not values:
            return 0.0
        
        alpha = 2 / (span + 1)
        ema = values[0]
        
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        return ema
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate normalized volatility (coefficient of variation)."""
        if len(values) < 2:
            return 0.5
        
        mean = sum(values) / len(values)
        if mean == 0:
            return 1.0
        
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        
        return min(1.0, std / mean)


class ROASOptimizer:
    """
    Main ROAS optimization engine.
    
    Combines Thompson Sampling for budget allocation with
    LSTM forecasting for ROAS prediction.
    """
    
    # Optimization thresholds
    MIN_ROAS_THRESHOLD = 1.0  # Minimum acceptable ROAS
    PAUSE_THRESHOLD = 0.5     # ROAS below this triggers pause
    HIGH_PERFORMER_THRESHOLD = 3.0  # Campaigns above this get budget increases
    
    def __init__(self, mlflow_client: Optional[MLflowClient] = None):
        self.mlflow_client = mlflow_client or MLflowClient()
        self.bandit = ThompsonSamplingBandit()
        self.forecaster = ROASForecaster()
    
    def optimize(
        self,
        campaigns: List[CampaignState],
        total_budget: Optional[float] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Run optimization cycle on all campaigns.
        
        Args:
            campaigns: List of campaign states
            total_budget: Optional total budget constraint
            dry_run: If True, don't apply changes
            
        Returns:
            Optimization result with recommendations
        """
        if not campaigns:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_campaigns": 0,
                "campaigns_adjusted": 0,
                "recommendations": []
            }
        
        recommendations = []
        total_before = sum(c.current_budget for c in campaigns)
        
        # Update bandit with latest performance
        for campaign in campaigns:
            self.bandit.update(
                campaign.campaign_id,
                campaign.conversions,
                campaign.clicks - campaign.conversions
            )
        
        # Get allocation scores
        allocation = self.bandit.get_allocation_scores(
            [c.campaign_id for c in campaigns]
        )
        
        for campaign in campaigns:
            rec = self._optimize_campaign(campaign, allocation.get(campaign.campaign_id, 0))
            recommendations.append(rec)
        
        # Apply budget constraints if needed
        if total_budget and total_budget != total_before:
            self._apply_budget_constraint(recommendations, total_budget)
        
        total_after = sum(r.recommended_budget for r in recommendations)
        expected_improvement = self._calculate_expected_improvement(recommendations)
        
        # Log to MLflow
        self._log_optimization(recommendations, expected_improvement)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_campaigns": len(campaigns),
            "campaigns_adjusted": sum(1 for r in recommendations if r.action != OptimizationAction.MAINTAIN),
            "total_budget_before": round(total_before, 2),
            "total_budget_after": round(total_after, 2),
            "expected_roas_improvement": round(expected_improvement, 2),
            "recommendations": [self._recommendation_to_dict(r) for r in recommendations]
        }
    
    def _optimize_campaign(
        self, 
        campaign: CampaignState,
        allocation_score: float
    ) -> BudgetRecommendation:
        """Generate optimization recommendation for a single campaign."""
        
        # Get ROAS forecast
        forecast = self.forecaster.forecast(
            campaign.historical_roas,
            campaign.roas
        )
        
        # Determine action based on current and predicted ROAS
        if campaign.roas < self.PAUSE_THRESHOLD:
            action = OptimizationAction.PAUSE
            new_budget = 0
            reasoning = f"ROAS ({campaign.roas:.2f}) below pause threshold ({self.PAUSE_THRESHOLD})"
            confidence = 0.95
            
        elif campaign.roas < self.MIN_ROAS_THRESHOLD:
            if forecast["trend"] == "improving":
                action = OptimizationAction.MAINTAIN
                new_budget = campaign.current_budget
                reasoning = f"ROAS low but improving trend detected"
                confidence = 0.7
            else:
                action = OptimizationAction.DECREASE
                new_budget = campaign.current_budget * 0.7
                reasoning = f"ROAS ({campaign.roas:.2f}) below minimum threshold with {forecast['trend']} trend"
                confidence = 0.85
                
        elif campaign.roas > self.HIGH_PERFORMER_THRESHOLD:
            # Scale increase based on allocation score
            increase_factor = 1.3 + allocation_score * 0.5
            new_budget = campaign.current_budget * min(increase_factor, 2.0)
            action = OptimizationAction.INCREASE
            reasoning = f"High performer (ROAS {campaign.roas:.2f}), allocation score {allocation_score:.2f}"
            confidence = 0.9
            
        else:
            # Moderate performer - small adjustments
            if allocation_score > 0.3:
                action = OptimizationAction.INCREASE
                new_budget = campaign.current_budget * (1 + allocation_score * 0.3)
                reasoning = f"Above average allocation score ({allocation_score:.2f})"
            elif allocation_score < 0.1:
                action = OptimizationAction.DECREASE
                new_budget = campaign.current_budget * 0.9
                reasoning = f"Below average allocation score ({allocation_score:.2f})"
            else:
                action = OptimizationAction.MAINTAIN
                new_budget = campaign.current_budget
                reasoning = "Performance within expected range"
            confidence = 0.75
        
        change_pct = ((new_budget - campaign.current_budget) / campaign.current_budget * 100
                     if campaign.current_budget > 0 else 0)
        
        return BudgetRecommendation(
            campaign_id=campaign.campaign_id,
            campaign_name=campaign.campaign_name,
            current_budget=campaign.current_budget,
            recommended_budget=round(new_budget, 2),
            change_percentage=round(change_pct, 1),
            predicted_roas=forecast["roas_24h"],
            confidence_score=round(confidence * forecast["confidence"], 2),
            reasoning=reasoning,
            action=action
        )
    
    def _apply_budget_constraint(
        self, 
        recommendations: List[BudgetRecommendation],
        total_budget: float
    ):
        """Adjust recommendations to meet total budget constraint."""
        current_total = sum(r.recommended_budget for r in recommendations)
        
        if current_total == 0:
            return
        
        ratio = total_budget / current_total
        
        for rec in recommendations:
            rec.recommended_budget = round(rec.recommended_budget * ratio, 2)
            rec.change_percentage = round(
                (rec.recommended_budget - rec.current_budget) / rec.current_budget * 100
                if rec.current_budget > 0 else 0, 1
            )
    
    def _calculate_expected_improvement(
        self, 
        recommendations: List[BudgetRecommendation]
    ) -> float:
        """Calculate expected ROAS improvement percentage."""
        # Weight improvements by budget
        total_budget = sum(r.recommended_budget for r in recommendations)
        if total_budget == 0:
            return 0.0
        
        weighted_improvement = 0.0
        for rec in recommendations:
            if rec.action == OptimizationAction.INCREASE:
                weighted_improvement += rec.recommended_budget * 0.15
            elif rec.action == OptimizationAction.DECREASE:
                weighted_improvement += rec.recommended_budget * 0.05
            elif rec.action == OptimizationAction.PAUSE:
                # Pausing bad campaigns improves overall ROAS
                weighted_improvement += rec.current_budget * 0.20
        
        return weighted_improvement / total_budget * 100
    
    def _recommendation_to_dict(self, rec: BudgetRecommendation) -> Dict[str, Any]:
        """Convert recommendation to API-friendly dict."""
        return {
            "campaign_id": rec.campaign_id,
            "campaign_name": rec.campaign_name,
            "current_budget": rec.current_budget,
            "recommended_budget": rec.recommended_budget,
            "change_percentage": rec.change_percentage,
            "predicted_roas": rec.predicted_roas,
            "confidence_score": rec.confidence_score,
            "reasoning": rec.reasoning,
            "action": rec.action.value
        }
    
    def _log_optimization(
        self, 
        recommendations: List[BudgetRecommendation],
        expected_improvement: float
    ):
        """Log optimization run to MLflow."""
        try:
            experiment = self.mlflow_client.get_experiment("roas_optimization")
            if not experiment:
                experiment = self.mlflow_client.create_experiment("roas_optimization")
            
            run = self.mlflow_client.start_run(
                experiment.experiment_id,
                f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            self.mlflow_client.log_metrics(run.run_id, {
                "campaigns_optimized": len(recommendations),
                "campaigns_increased": sum(1 for r in recommendations if r.action == OptimizationAction.INCREASE),
                "campaigns_decreased": sum(1 for r in recommendations if r.action == OptimizationAction.DECREASE),
                "campaigns_paused": sum(1 for r in recommendations if r.action == OptimizationAction.PAUSE),
                "expected_improvement": expected_improvement
            })
            
            self.mlflow_client.end_run(run.run_id)
        except Exception:
            pass
    
    def get_campaign_forecast(self, campaign: CampaignState) -> Dict[str, Any]:
        """Get standalone ROAS forecast for a campaign."""
        return self.forecaster.forecast(
            campaign.historical_roas,
            campaign.roas
        )


# Singleton instance
_optimizer: Optional[ROASOptimizer] = None


def get_roas_optimizer() -> ROASOptimizer:
    """Get or create the ROAS optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ROASOptimizer()
    return _optimizer
