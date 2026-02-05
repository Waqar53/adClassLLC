"""
Celery Task Implementations

All background tasks for the AdClass platform.
"""

from celery import shared_task
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import json


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================
# PLATFORM DATA SYNC TASKS
# ===========================================

@shared_task(bind=True, max_retries=3)
def sync_platform_data(self, platform: str, client_id: Optional[str] = None):
    """
    Sync campaign data from ad platforms.
    
    Args:
        platform: meta, google, or tiktok
        client_id: Optional specific client to sync
    """
    from app.services import get_meta_service, get_google_ads_service, get_tiktok_ads_service
    
    try:
        if platform == "meta":
            service = get_meta_service()
            # In production: fetch all accounts and sync
            result = run_async(service.sync_all("act_123456789", client_id or "demo"))
        elif platform == "google":
            service = get_google_ads_service()
            result = run_async(service.sync_all("1234567890", client_id or "demo"))
        elif platform == "tiktok":
            service = get_tiktok_ads_service()
            result = run_async(service.sync_all("7123456789012345678", client_id or "demo"))
        else:
            return {"error": f"Unknown platform: {platform}"}
        
        return {
            "status": "success",
            "platform": platform,
            "synced": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as exc:
        self.retry(countdown=60 * (self.request.retries + 1), exc=exc)


@shared_task
def sync_all_platforms():
    """Sync all platforms for all clients."""
    results = {}
    for platform in ["meta", "google", "tiktok"]:
        sync_platform_data.delay(platform)
        results[platform] = "queued"
    return results


# ===========================================
# ROAS OPTIMIZATION TASKS
# ===========================================

@shared_task(bind=True)
def run_roas_optimization(self, client_id: Optional[str] = None, dry_run: bool = False):
    """
    Run ROAS optimization cycle.
    
    Args:
        client_id: Optional specific client
        dry_run: If True, don't apply changes
    """
    from app.models.ml.roas_optimizer import get_roas_optimizer, CampaignState
    import numpy as np
    
    optimizer = get_roas_optimizer()
    
    # In production: Fetch real campaign data from database
    mock_campaigns = [
        CampaignState(
            campaign_id="camp_1",
            current_budget=500.0,
            spend_today=350.0,
            impressions=45000,
            clicks=1125,
            conversions=67,
            revenue=4020.0,
            ctr=0.025,
            cvr=0.060,
            roas=11.49,
            hour_of_day=datetime.now().hour,
            day_of_week=datetime.now().weekday()
        ),
        CampaignState(
            campaign_id="camp_2",
            current_budget=300.0,
            spend_today=280.0,
            impressions=32000,
            clicks=640,
            conversions=12,
            revenue=720.0,
            ctr=0.020,
            cvr=0.019,
            roas=2.57,
            hour_of_day=datetime.now().hour,
            day_of_week=datetime.now().weekday()
        ),
    ]
    
    # Run optimization
    decisions = optimizer.optimize(mock_campaigns, {}, dry_run=dry_run)
    
    results = []
    for decision in decisions:
        results.append({
            "campaign_id": decision.campaign_id,
            "current_budget": decision.current_budget,
            "recommended_budget": decision.recommended_budget,
            "action": decision.action,
            "predicted_roas": decision.predicted_roas,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning
        })
        
        # In production: Apply budget changes via API
        if not dry_run and decision.action != "maintain":
            apply_budget_change.delay(
                decision.campaign_id,
                decision.recommended_budget,
                decision.reasoning
            )
    
    return {
        "status": "success",
        "campaigns_analyzed": len(mock_campaigns),
        "decisions": results,
        "dry_run": dry_run,
        "timestamp": datetime.utcnow().isoformat()
    }


@shared_task
def apply_budget_change(campaign_id: str, new_budget: float, reason: str):
    """Apply budget change to a campaign."""
    # In production: Call platform API to update budget
    return {
        "campaign_id": campaign_id,
        "new_budget": new_budget,
        "reason": reason,
        "applied_at": datetime.utcnow().isoformat()
    }


# ===========================================
# CLIENT HEALTH TASKS
# ===========================================

@shared_task
def update_client_health(client_id: Optional[str] = None):
    """
    Update client health scores and churn predictions.
    """
    from app.models.ml.churn_model import get_churn_predictor, ClientFeatures
    
    predictor = get_churn_predictor()
    
    # In production: Fetch real client data
    mock_clients = [
        ClientFeatures(
            client_id="client_1",
            current_roas=4.5,
            target_roas=4.0,
            roas_7d_avg=4.2,
            roas_30d_avg=4.0,
            roas_trend=0.05,
            dashboard_logins_7d=12,
            dashboard_logins_30d=45,
            meeting_attendance_rate=0.85,
            email_open_rate=0.65,
            avg_response_time_hours=4.5,
            last_contact_days=3,
            support_tickets_30d=2,
            ticket_sentiment_avg=0.7,
            monthly_spend=25000.0,
            revenue_mom_change=0.08,
            payment_delay_avg_days=2.0,
            invoices_outstanding=0,
            contract_days_remaining=180,
            is_month_to_month=False,
            tenure_months=24
        ),
        ClientFeatures(
            client_id="client_2",
            current_roas=1.8,
            target_roas=3.0,
            roas_7d_avg=1.5,
            roas_30d_avg=2.2,
            roas_trend=-0.15,
            dashboard_logins_7d=2,
            dashboard_logins_30d=8,
            meeting_attendance_rate=0.40,
            email_open_rate=0.25,
            avg_response_time_hours=48.0,
            last_contact_days=21,
            support_tickets_30d=5,
            ticket_sentiment_avg=0.3,
            monthly_spend=8000.0,
            revenue_mom_change=-0.15,
            payment_delay_avg_days=12.0,
            invoices_outstanding=2,
            contract_days_remaining=45,
            is_month_to_month=True,
            tenure_months=6
        ),
    ]
    
    results = []
    alerts = []
    
    for client in mock_clients:
        prediction, health, risk_factors = predictor.predict(client)
        
        results.append({
            "client_id": client.client_id,
            "health_score": health.overall_score,
            "churn_probability": prediction.churn_probability,
            "risk_level": prediction.risk_level,
            "risk_factors": [f.factor for f in risk_factors[:3]]
        })
        
        # Create alerts for at-risk clients
        if prediction.risk_level in ["critical", "warning"]:
            alerts.append({
                "client_id": client.client_id,
                "risk_level": prediction.risk_level,
                "health_score": health.overall_score,
                "churn_probability": prediction.churn_probability
            })
            # Trigger alert notification
            send_churn_alert.delay(client.client_id, prediction.risk_level)
    
    return {
        "status": "success",
        "clients_analyzed": len(mock_clients),
        "results": results,
        "alerts_created": len(alerts),
        "timestamp": datetime.utcnow().isoformat()
    }


@shared_task
def send_churn_alert(client_id: str, risk_level: str):
    """Send alert for at-risk client."""
    # In production: Send email/Slack/webhook notification
    return {
        "client_id": client_id,
        "risk_level": risk_level,
        "alert_sent_at": datetime.utcnow().isoformat(),
        "channels": ["email", "slack"]
    }


# ===========================================
# ATTRIBUTION TASKS  
# ===========================================

@shared_task
def compute_attribution_reports(client_id: Optional[str] = None):
    """
    Compute attribution reports for all clients.
    """
    from app.models.ml.attribution_model import get_attribution_engine, CustomerJourney, Touchpoint
    from datetime import datetime, timedelta
    
    engine = get_attribution_engine()
    
    # In production: Fetch real journey data from database
    mock_journeys = [
        CustomerJourney(
            customer_id="cust_1",
            touchpoints=[
                Touchpoint("tp1", "cust_1", "meta", "camp_1", datetime.now() - timedelta(days=7), "impression"),
                Touchpoint("tp2", "cust_1", "google", "camp_2", datetime.now() - timedelta(days=5), "click"),
                Touchpoint("tp3", "cust_1", "meta", "camp_1", datetime.now() - timedelta(days=2), "click"),
                Touchpoint("tp4", "cust_1", "email", None, datetime.now() - timedelta(days=1), "conversion", 150.0),
            ],
            conversion_value=150.0,
            converted=True
        ),
        CustomerJourney(
            customer_id="cust_2",
            touchpoints=[
                Touchpoint("tp5", "cust_2", "tiktok", "camp_3", datetime.now() - timedelta(days=3), "click"),
                Touchpoint("tp6", "cust_2", "meta", "camp_1", datetime.now() - timedelta(days=1), "conversion", 85.0),
            ],
            conversion_value=85.0,
            converted=True
        ),
    ]
    
    # Compute attribution
    attribution = engine.compute_attribution(mock_journeys)
    
    results = []
    for channel, attr in attribution.items():
        results.append({
            "channel": channel,
            "shapley_value": round(attr.shapley_value, 4),
            "markov_value": round(attr.markov_value, 4),
            "first_touch": round(attr.first_touch, 4),
            "last_touch": round(attr.last_touch, 4),
            "linear": round(attr.linear, 4),
            "time_decay": round(attr.time_decay, 4)
        })
    
    return {
        "status": "success",
        "journeys_analyzed": len(mock_journeys),
        "attribution": results,
        "total_conversions": sum(1 for j in mock_journeys if j.converted),
        "total_value": sum(j.conversion_value for j in mock_journeys),
        "timestamp": datetime.utcnow().isoformat()
    }


# ===========================================
# AUDIENCE TASKS
# ===========================================

@shared_task
def refresh_audience_segments(client_id: Optional[str] = None):
    """
    Refresh audience segments for clients.
    """
    from app.models.ml.audience_model import get_audience_service, AudienceMember
    import random
    
    service = get_audience_service()
    
    # In production: Fetch real customer data
    mock_members = [
        AudienceMember(
            customer_id=f"cust_{i}",
            features={
                "ltv": random.uniform(50, 500),
                "purchase_frequency": random.uniform(1, 12),
                "avg_order_value": random.uniform(25, 150),
                "days_since_purchase": random.randint(1, 90),
                "email_engagement": random.uniform(0, 1),
                "website_visits_30d": random.randint(1, 50),
                "cart_abandonment_rate": random.uniform(0, 0.5),
                "product_views_30d": random.randint(5, 100),
            }
        )
        for i in range(100)
    ]
    
    # Run segmentation
    result = service.segment_audience(mock_members, method="kmeans", n_segments=5)
    
    segments_info = []
    for segment in result["segments"]:
        segments_info.append({
            "segment_id": segment.segment_id,
            "name": segment.name,
            "size": segment.size,
            "characteristics": segment.characteristics[:3],
            "avg_value": round(segment.avg_value, 2)
        })
    
    return {
        "status": "success",
        "total_members": result["total_members"],
        "segments": segments_info,
        "timestamp": datetime.utcnow().isoformat()
    }


@shared_task
def create_lookalike_audience(
    client_id: str,
    seed_segment_id: str,
    target_size: int = 1000,
    expansion: float = 2.0
):
    """
    Create lookalike audience from seed segment.
    """
    from app.models.ml.audience_model import get_audience_service, AudienceMember
    import random
    
    service = get_audience_service()
    
    # Mock seed and candidate data
    seed = [
        AudienceMember(
            customer_id=f"seed_{i}",
            features={
                "ltv": random.uniform(200, 500),
                "purchase_frequency": random.uniform(6, 12),
                "avg_order_value": random.uniform(100, 200),
            }
        )
        for i in range(50)
    ]
    
    candidates = [
        AudienceMember(
            customer_id=f"cand_{i}",
            features={
                "ltv": random.uniform(50, 400),
                "purchase_frequency": random.uniform(1, 10),
                "avg_order_value": random.uniform(25, 175),
            }
        )
        for i in range(500)
    ]
    
    result = service.create_lookalike(seed, candidates, target_size, expansion)
    
    return {
        "status": "success",
        "client_id": client_id,
        "seed_size": result["seed_size"],
        "lookalike_size": result["lookalike_size"],
        "avg_similarity": round(result["avg_similarity"], 3),
        "timestamp": datetime.utcnow().isoformat()
    }


@shared_task
def sync_audience_to_platform(segment_id: str, platform: str, client_id: str):
    """
    Sync audience segment to ad platform.
    """
    from app.services import get_meta_service, get_google_ads_service, get_tiktok_ads_service
    
    # In production: Create custom audience and upload users
    if platform == "meta":
        service = get_meta_service()
        # Would call service.create_custom_audience and add_users_to_audience
    elif platform == "google":
        service = get_google_ads_service()
    elif platform == "tiktok":
        service = get_tiktok_ads_service()
    
    return {
        "status": "success",
        "segment_id": segment_id,
        "platform": platform,
        "synced_at": datetime.utcnow().isoformat()
    }


# ===========================================
# CREATIVE ANALYSIS TASKS
# ===========================================

@shared_task
def analyze_creative_batch(creative_ids: List[str]):
    """
    Batch analyze creatives for performance prediction.
    """
    from app.models.ml.creative_predictor import get_creative_predictor
    
    predictor = get_creative_predictor()
    
    results = []
    for creative_id in creative_ids:
        # In production: Fetch creative data from database
        prediction = predictor.predict_text_only(
            headline=f"Amazing Product {creative_id}",
            body_text="Shop now and save big!",
            cta_type="SHOP_NOW",
            platform="meta"
        )
        
        results.append({
            "creative_id": creative_id,
            "predicted_ctr": prediction["predicted_ctr"],
            "predicted_cvr": prediction["predicted_cvr"],
            "quality_score": prediction["overall_quality_score"],
            "recommendations": prediction["recommendations"][:3]
        })
    
    return {
        "status": "success",
        "creatives_analyzed": len(creative_ids),
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


# ===========================================
# ML MODEL TASKS
# ===========================================

@shared_task
def retrain_model(model_name: str, dataset_path: str):
    """
    Retrain a ML model with new data.
    """
    # In production: Load data, train model, version with MLflow
    return {
        "status": "queued",
        "model_name": model_name,
        "dataset": dataset_path,
        "queued_at": datetime.utcnow().isoformat()
    }


@shared_task
def compute_feature_store_batch(entity_type: str, entity_ids: List[str]):
    """
    Batch compute features for entities.
    """
    from app.services.feature_store import get_feature_store
    
    store = get_feature_store()
    
    results = {}
    for entity_id in entity_ids:
        if entity_type == "campaign":
            # Mock metrics
            features = store.compute_campaign_features(
                entity_id,
                {
                    "spend": [100, 150, 120, 180, 200, 160, 140],
                    "revenue": [500, 750, 600, 900, 1000, 800, 700],
                    "clicks": [50, 75, 60, 90, 100, 80, 70],
                    "impressions": [2000, 3000, 2400, 3600, 4000, 3200, 2800],
                    "conversions": [5, 8, 6, 9, 10, 8, 7]
                }
            )
            results[entity_id] = features
    
    return {
        "status": "success",
        "entity_type": entity_type,
        "entities_processed": len(entity_ids),
        "timestamp": datetime.utcnow().isoformat()
    }
