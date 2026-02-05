"""
Churn Prediction & Client Health API Routes

Module 3: Predict client churn and provide health scoring.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime, date, timedelta
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

router = APIRouter()


# ===========================================
# REQUEST/RESPONSE MODELS
# ===========================================

class HealthScore(BaseModel):
    """Client health score breakdown."""
    overall_score: int = Field(..., ge=0, le=100)
    roas_score: int = Field(..., ge=0, le=100)
    engagement_score: int = Field(..., ge=0, le=100)
    payment_score: int = Field(..., ge=0, le=100)
    communication_score: int = Field(..., ge=0, le=100)
    growth_score: int = Field(..., ge=0, le=100)


class ChurnPrediction(BaseModel):
    """Churn prediction result."""
    churn_probability: float = Field(..., ge=0, le=1)
    predicted_churn_date: Optional[date] = None
    days_to_churn: Optional[int] = None
    risk_level: str  # critical, warning, monitor, healthy
    confidence: float


class RiskFactor(BaseModel):
    """Individual risk factor."""
    factor: str
    severity: str  # high, medium, low
    impact_score: float
    description: str
    trend: str  # improving, stable, declining


class ClientHealthReport(BaseModel):
    """Complete client health report."""
    client_id: UUID
    client_name: str
    health_score: HealthScore
    churn_prediction: ChurnPrediction
    risk_factors: List[RiskFactor]
    recommended_actions: List[str]
    last_updated: datetime


class ChurnAlert(BaseModel):
    """Churn alert notification."""
    id: UUID
    client_id: UUID
    client_name: str
    alert_type: str  # critical, warning, monitor
    health_score: int
    churn_probability: float
    triggered_at: datetime
    risk_factors: List[str]
    recommended_actions: List[str]
    status: str  # open, acknowledged, resolved


class InteractionLog(BaseModel):
    """Client interaction record."""
    interaction_type: str
    occurred_at: datetime
    response_time_minutes: Optional[int] = None
    sentiment_score: Optional[float] = None
    notes: Optional[str] = None


# ===========================================
# API ENDPOINTS
# ===========================================

@router.get("/health/{client_id}", response_model=ClientHealthReport)
async def get_client_health(
    client_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive health report for a client.
    
    Includes health scores, churn prediction, and risk factors.
    """
    from app.services.ml.churn_predictor import get_churn_predictor, ClientMetrics
    
    predictor = get_churn_predictor()
    
    # Create client metrics (would normally come from database)
    metrics = ClientMetrics(
        client_id=str(client_id),
        client_name="Acme Corp",
        monthly_spend=25000.0,
        target_roas=4.0,
        actual_roas=3.2,
        roas_trend=-0.2,
        days_since_login=12,
        days_since_contact=8,
        support_tickets_30d=2,
        meetings_30d=1,
        campaigns_active=5,
        campaigns_paused=2,
        avg_ctr=0.022,
        avg_cvr=0.015,
        contract_months_remaining=4,
        is_on_annual=False,
        payment_delays=0
    )
    
    # Run ML prediction
    prediction = predictor.predict(metrics)
    
    # Convert to API response format
    report = ClientHealthReport(
        client_id=client_id,
        client_name=prediction.client_name,
        health_score=HealthScore(
            overall_score=int(prediction.health_score.overall_score),
            roas_score=int(prediction.health_score.roas_score),
            engagement_score=int(prediction.health_score.engagement_score),
            payment_score=int(prediction.health_score.payment_score),
            communication_score=int(prediction.health_score.communication_score),
            growth_score=int(prediction.health_score.growth_score)
        ),
        churn_prediction=ChurnPrediction(
            churn_probability=prediction.churn_probability,
            predicted_churn_date=date.today() + timedelta(days=prediction.days_to_churn) if prediction.days_to_churn else None,
            days_to_churn=prediction.days_to_churn,
            risk_level=prediction.risk_level.value,
            confidence=0.82
        ),
        risk_factors=[
            RiskFactor(
                factor=rf.factor,
                severity=rf.severity,
                impact_score=rf.impact_score,
                description=rf.description,
                trend=rf.trend
            )
            for rf in prediction.risk_factors
        ],
        recommended_actions=prediction.recommended_actions,
        last_updated=datetime.utcnow()
    )
    
    return report


@router.get("/health", response_model=List[ClientHealthReport])
async def get_all_client_health(
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    min_score: Optional[int] = Query(None, ge=0, le=100),
    max_score: Optional[int] = Query(None, ge=0, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    Get health reports for all clients.
    
    Can filter by risk level (critical, warning, monitor, healthy)
    or by score range.
    """
    # TODO: Query all clients and filter
    return []


@router.get("/alerts", response_model=List[ChurnAlert])
async def get_churn_alerts(
    status: str = Query("open", description="Filter: open, acknowledged, resolved, all"),
    alert_type: Optional[str] = Query(None, description="Filter: critical, warning, monitor"),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db)
):
    """
    Get active churn alerts.
    
    Returns clients that need attention based on health thresholds.
    """
    # TODO: Query churn_alerts table
    alerts = [
        ChurnAlert(
            id=UUID("12345678-1234-1234-1234-123456789001"),
            client_id=UUID("12345678-1234-1234-1234-123456789101"),
            client_name="Acme Corp",
            alert_type="warning",
            health_score=65,
            churn_probability=0.35,
            triggered_at=datetime(2026, 2, 3, 14, 30),
            risk_factors=[
                "Low dashboard engagement",
                "Slow response times"
            ],
            recommended_actions=[
                "Schedule strategy review call",
                "Send personalized performance report"
            ],
            status="open"
        )
    ]
    
    return alerts


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: UUID,
    notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Mark an alert as acknowledged."""
    return {
        "alert_id": str(alert_id),
        "status": "acknowledged",
        "acknowledged_at": datetime.utcnow().isoformat()
    }


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: UUID,
    resolution_notes: str,
    db: AsyncSession = Depends(get_db)
):
    """Mark an alert as resolved."""
    return {
        "alert_id": str(alert_id),
        "status": "resolved",
        "resolved_at": datetime.utcnow().isoformat(),
        "notes": resolution_notes
    }


@router.post("/interactions/{client_id}")
async def log_interaction(
    client_id: UUID,
    interaction: InteractionLog,
    db: AsyncSession = Depends(get_db)
):
    """
    Log a client interaction.
    
    Used to track engagement for health scoring.
    """
    # TODO: Insert into client_interactions table
    return {
        "status": "logged",
        "client_id": str(client_id),
        "interaction_type": interaction.interaction_type,
        "occurred_at": interaction.occurred_at.isoformat()
    }


@router.get("/trends/{client_id}")
async def get_health_trends(
    client_id: UUID,
    days: int = Query(90, ge=7, le=365),
    db: AsyncSession = Depends(get_db)
):
    """
    Get health score trends over time.
    
    Returns daily or weekly snapshots for trend analysis.
    """
    # TODO: Query client_health_snapshots table
    trends = {
        "client_id": str(client_id),
        "period_days": days,
        "data_points": [
            {"date": "2026-01-01", "health_score": 78, "churn_probability": 0.15},
            {"date": "2026-01-15", "health_score": 72, "churn_probability": 0.22},
            {"date": "2026-02-01", "health_score": 65, "churn_probability": 0.35}
        ],
        "trend_direction": "declining",
        "avg_change_per_week": -2.5
    }
    
    return trends


@router.get("/at-risk")
async def get_at_risk_clients(
    threshold: int = Query(50, description="Health score threshold"),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    Get list of at-risk clients below health threshold.
    
    Sorted by churn probability (highest first).
    """
    # TODO: Query and sort clients
    at_risk = [
        {
            "client_id": "12345678-1234-1234-1234-123456789101",
            "client_name": "Acme Corp",
            "health_score": 65,
            "churn_probability": 0.35,
            "monthly_revenue": 25000,
            "days_to_renewal": 45
        }
    ]
    
    return {"at_risk_clients": at_risk, "total": len(at_risk)}
