"""
Alert System

Automated alerting for churn, performance anomalies, and system events.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    # Client Health
    CHURN_RISK = "churn_risk"
    HEALTH_DECLINE = "health_decline"
    PAYMENT_ISSUE = "payment_issue"
    CONTRACT_EXPIRING = "contract_expiring"
    
    # Performance
    ROAS_DROP = "roas_drop"
    BUDGET_DEPLETED = "budget_depleted"
    CAMPAIGN_UNDERPERFORMING = "campaign_underperforming"
    ABNORMAL_SPEND = "abnormal_spend"
    
    # System
    MODEL_DRIFT = "model_drift"
    SYNC_FAILURE = "sync_failure"
    API_ERROR = "api_error"
    HIGH_LATENCY = "high_latency"


class AlertChannel(str, Enum):
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


@dataclass
class Alert:
    """An alert instance."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any]
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    channels_sent: List[AlertChannel] = field(default_factory=list)


@dataclass
class AlertRule:
    """Rule for triggering alerts."""
    rule_id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]
    channels: List[AlertChannel]
    cooldown_minutes: int = 60
    enabled: bool = True
    last_triggered: Optional[datetime] = None


class AlertNotifier:
    """
    Send alerts through various channels.
    """
    
    async def send_email(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        html: bool = True
    ):
        """Send email alert."""
        # In production: Use SendGrid, AWS SES, etc.
        print(f"[EMAIL] To: {recipients}, Subject: {subject}")
    
    async def send_slack(
        self,
        channel: str,
        message: str,
        attachments: Optional[List[Dict]] = None
    ):
        """Send Slack alert."""
        # In production: Use Slack API
        print(f"[SLACK] #{channel}: {message}")
    
    async def send_sms(
        self,
        phone_numbers: List[str],
        message: str
    ):
        """Send SMS alert."""
        # In production: Use Twilio
        print(f"[SMS] To: {phone_numbers}, Message: {message}")
    
    async def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any]
    ):
        """Send webhook alert."""
        # In production: Use httpx/aiohttp
        print(f"[WEBHOOK] {url}: {json.dumps(payload)}")
    
    async def send(
        self,
        alert: Alert,
        channels: List[AlertChannel],
        config: Dict[str, Any]
    ):
        """Send alert through specified channels."""
        for channel in channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self.send_email(
                        recipients=config.get("email_recipients", []),
                        subject=f"[{alert.severity.value.upper()}] {alert.title}",
                        body=self._format_email(alert)
                    )
                elif channel == AlertChannel.SLACK:
                    await self.send_slack(
                        channel=config.get("slack_channel", "alerts"),
                        message=self._format_slack(alert)
                    )
                elif channel == AlertChannel.SMS:
                    await self.send_sms(
                        phone_numbers=config.get("sms_recipients", []),
                        message=f"{alert.severity.value.upper()}: {alert.title}"
                    )
                elif channel == AlertChannel.WEBHOOK:
                    await self.send_webhook(
                        url=config.get("webhook_url", ""),
                        payload=self._format_webhook(alert)
                    )
                
                alert.channels_sent.append(channel)
                
            except Exception as e:
                print(f"Failed to send alert via {channel}: {e}")
    
    def _format_email(self, alert: Alert) -> str:
        return f"""
        <h2>{alert.title}</h2>
        <p><strong>Severity:</strong> {alert.severity.value}</p>
        <p><strong>Type:</strong> {alert.alert_type.value}</p>
        <p>{alert.message}</p>
        <hr>
        <pre>{json.dumps(alert.data, indent=2)}</pre>
        <p><small>Alert ID: {alert.alert_id}</small></p>
        """
    
    def _format_slack(self, alert: Alert) -> str:
        severity_emoji = {
            AlertSeverity.LOW: "ℹ️",
            AlertSeverity.MEDIUM: "⚠️",
            AlertSeverity.HIGH: "🚨",
            AlertSeverity.CRITICAL: "🔴"
        }
        return f"{severity_emoji.get(alert.severity, '📢')} *{alert.title}*\n{alert.message}"
    
    def _format_webhook(self, alert: Alert) -> Dict[str, Any]:
        return {
            "alert_id": alert.alert_id,
            "type": alert.alert_type.value,
            "severity": alert.severity.value,
            "title": alert.title,
            "message": alert.message,
            "data": alert.data,
            "timestamp": alert.created_at.isoformat()
        }


class AlertManager:
    """
    Main alert management service.
    """
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.notifier = AlertNotifier()
        self.notification_config: Dict[str, Any] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Set up default alert rules."""
        # Churn risk alert
        self.add_rule(AlertRule(
            rule_id="churn_critical",
            name="Critical Churn Risk",
            alert_type=AlertType.CHURN_RISK,
            severity=AlertSeverity.CRITICAL,
            condition=lambda d: d.get("churn_probability", 0) >= 0.8,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown_minutes=240
        ))
        
        self.add_rule(AlertRule(
            rule_id="churn_high",
            name="High Churn Risk",
            alert_type=AlertType.CHURN_RISK,
            severity=AlertSeverity.HIGH,
            condition=lambda d: 0.6 <= d.get("churn_probability", 0) < 0.8,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown_minutes=1440  # 24 hours
        ))
        
        # ROAS drop alert
        self.add_rule(AlertRule(
            rule_id="roas_drop",
            name="ROAS Drop Alert",
            alert_type=AlertType.ROAS_DROP,
            severity=AlertSeverity.HIGH,
            condition=lambda d: d.get("roas_change", 0) < -0.3,  # 30% drop
            channels=[AlertChannel.SLACK],
            cooldown_minutes=60
        ))
        
        # Budget depleted
        self.add_rule(AlertRule(
            rule_id="budget_depleted",
            name="Budget Depleted",
            alert_type=AlertType.BUDGET_DEPLETED,
            severity=AlertSeverity.MEDIUM,
            condition=lambda d: d.get("budget_remaining_percent", 100) < 10,
            channels=[AlertChannel.SLACK, AlertChannel.IN_APP],
            cooldown_minutes=60
        ))
        
        # Model drift
        self.add_rule(AlertRule(
            rule_id="model_drift",
            name="Model Drift Detected",
            alert_type=AlertType.MODEL_DRIFT,
            severity=AlertSeverity.HIGH,
            condition=lambda d: d.get("drift_score", 0) > 0.15,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown_minutes=1440
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
    
    def configure_notifications(self, config: Dict[str, Any]):
        """Configure notification settings."""
        self.notification_config = config
    
    async def check_and_alert(
        self,
        context: Dict[str, Any],
        alert_type: Optional[AlertType] = None
    ) -> List[Alert]:
        """
        Check conditions and create alerts if needed.
        
        Args:
            context: Data to check against rules
            alert_type: Optional filter for specific alert type
        """
        triggered_alerts = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if alert_type and rule.alert_type != alert_type:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                cooldown = timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() - rule.last_triggered < cooldown:
                    continue
            
            # Check condition
            try:
                if rule.condition(context):
                    alert = await self._create_and_send_alert(rule, context)
                    triggered_alerts.append(alert)
                    rule.last_triggered = datetime.now()
            except Exception as e:
                print(f"Error checking rule {rule.rule_id}: {e}")
        
        return triggered_alerts
    
    async def _create_and_send_alert(
        self,
        rule: AlertRule,
        context: Dict[str, Any]
    ) -> Alert:
        """Create an alert and send notifications."""
        import hashlib
        
        alert_id = hashlib.md5(
            f"{rule.rule_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Build alert message
        title, message = self._build_message(rule, context)
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=title,
            message=message,
            data=context,
            created_at=datetime.now()
        )
        
        self.alerts[alert_id] = alert
        
        # Send notifications
        await self.notifier.send(alert, rule.channels, self.notification_config)
        
        return alert
    
    def _build_message(
        self,
        rule: AlertRule,
        context: Dict[str, Any]
    ) -> tuple:
        """Build alert title and message."""
        templates = {
            AlertType.CHURN_RISK: (
                "Churn Risk: {client_name}",
                "Client {client_name} has {churn_probability:.0%} churn probability. Risk factors: {risk_factors}"
            ),
            AlertType.ROAS_DROP: (
                "ROAS Drop: {campaign_name}",
                "Campaign {campaign_name} ROAS dropped by {roas_change:.0%}. Current: {current_roas:.2f}x"
            ),
            AlertType.BUDGET_DEPLETED: (
                "Budget Low: {campaign_name}",
                "Campaign {campaign_name} has only {budget_remaining_percent:.0f}% budget remaining."
            ),
            AlertType.MODEL_DRIFT: (
                "Model Drift: {model_name}",
                "Model {model_name} shows drift score of {drift_score:.2f}. Consider retraining."
            )
        }
        
        template = templates.get(rule.alert_type, (rule.name, str(context)))
        
        try:
            title = template[0].format(**context)
            message = template[1].format(**context)
        except KeyError:
            title = rule.name
            message = json.dumps(context)
        
        return title, message
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged_at = datetime.now()
            self.alerts[alert_id].acknowledged_by = user_id
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved_at = datetime.now()
            return True
        return False
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None
    ) -> List[Alert]:
        """Get active (unresolved) alerts."""
        alerts = [a for a in self.alerts.values() if a.resolved_at is None]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)


# Singleton
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
