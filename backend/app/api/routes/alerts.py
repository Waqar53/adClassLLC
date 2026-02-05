"""
Alert API Routes

Alert management and notification configuration.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

from app.services.alert_system import (
    get_alert_manager,
    AlertSeverity,
    AlertType,
    AlertChannel
)

router = APIRouter(prefix="/alerts", tags=["Alerts"])


class NotificationConfigRequest(BaseModel):
    email_recipients: List[str] = []
    slack_channel: str = "alerts"
    sms_recipients: List[str] = []
    webhook_url: str = ""


class AlertCheckRequest(BaseModel):
    context: dict
    alert_type: Optional[str] = None


@router.get("/")
async def list_alerts(
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    resolved: bool = False,
    limit: int = Query(50, le=200)
):
    """List alerts with optional filters."""
    manager = get_alert_manager()
    
    sev = AlertSeverity(severity) if severity else None
    at = AlertType(alert_type) if alert_type else None
    
    if resolved:
        alerts = [a for a in manager.alerts.values() if a.resolved_at is not None]
    else:
        alerts = manager.get_active_alerts(severity=sev, alert_type=at)
    
    alerts = sorted(alerts, key=lambda x: x.created_at, reverse=True)[:limit]
    
    return {
        "alerts": [
            {
                "alert_id": a.alert_id,
                "type": a.alert_type.value,
                "severity": a.severity.value,
                "title": a.title,
                "message": a.message,
                "created_at": a.created_at.isoformat(),
                "acknowledged_at": a.acknowledged_at.isoformat() if a.acknowledged_at else None,
                "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                "acknowledged_by": a.acknowledged_by
            }
            for a in alerts
        ],
        "total": len(alerts)
    }


@router.get("/stats")
async def alert_stats():
    """Get alert statistics."""
    manager = get_alert_manager()
    
    active = manager.get_active_alerts()
    
    stats = {
        "total_active": len(active),
        "by_severity": {},
        "by_type": {}
    }
    
    for sev in AlertSeverity:
        count = len([a for a in active if a.severity == sev])
        if count > 0:
            stats["by_severity"][sev.value] = count
    
    for at in AlertType:
        count = len([a for a in active if a.alert_type == at])
        if count > 0:
            stats["by_type"][at.value] = count
    
    return stats


@router.get("/{alert_id}")
async def get_alert(alert_id: str):
    """Get alert details."""
    manager = get_alert_manager()
    
    if alert_id not in manager.alerts:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    a = manager.alerts[alert_id]
    
    return {
        "alert_id": a.alert_id,
        "type": a.alert_type.value,
        "severity": a.severity.value,
        "title": a.title,
        "message": a.message,
        "data": a.data,
        "created_at": a.created_at.isoformat(),
        "acknowledged_at": a.acknowledged_at.isoformat() if a.acknowledged_at else None,
        "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
        "acknowledged_by": a.acknowledged_by,
        "channels_sent": [c.value for c in a.channels_sent]
    }


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user_id: str):
    """Acknowledge an alert."""
    manager = get_alert_manager()
    
    success = manager.acknowledge_alert(alert_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {"status": "acknowledged", "acknowledged_by": user_id}


@router.post("/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    manager = get_alert_manager()
    
    success = manager.resolve_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {"status": "resolved"}


@router.post("/check")
async def check_alerts(
    request: AlertCheckRequest,
    background_tasks: BackgroundTasks
):
    """Check conditions and trigger alerts if needed."""
    manager = get_alert_manager()
    
    at = AlertType(request.alert_type) if request.alert_type else None
    
    alerts = await manager.check_and_alert(request.context, at)
    
    return {
        "triggered_count": len(alerts),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "type": a.alert_type.value,
                "severity": a.severity.value,
                "title": a.title
            }
            for a in alerts
        ]
    }


@router.post("/config")
async def configure_notifications(request: NotificationConfigRequest):
    """Configure notification settings."""
    manager = get_alert_manager()
    
    manager.configure_notifications({
        "email_recipients": request.email_recipients,
        "slack_channel": request.slack_channel,
        "sms_recipients": request.sms_recipients,
        "webhook_url": request.webhook_url
    })
    
    return {"status": "configured"}


@router.get("/config")
async def get_notification_config():
    """Get current notification configuration."""
    manager = get_alert_manager()
    
    return manager.notification_config


@router.get("/rules")
async def list_alert_rules():
    """List configured alert rules."""
    manager = get_alert_manager()
    
    return {
        "rules": [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "alert_type": r.alert_type.value,
                "severity": r.severity.value,
                "channels": [c.value for c in r.channels],
                "cooldown_minutes": r.cooldown_minutes,
                "enabled": r.enabled,
                "last_triggered": r.last_triggered.isoformat() if r.last_triggered else None
            }
            for r in manager.rules.values()
        ]
    }


@router.post("/rules/{rule_id}/toggle")
async def toggle_rule(rule_id: str, enabled: bool):
    """Enable or disable an alert rule."""
    manager = get_alert_manager()
    
    if rule_id not in manager.rules:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    manager.rules[rule_id].enabled = enabled
    
    return {"status": "updated", "enabled": enabled}
