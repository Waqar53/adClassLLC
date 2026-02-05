"""
Intervention Recommendations Engine

AI-driven recommendations for preventing churn and improving client health.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np


class InterventionType(str, Enum):
    OUTREACH = "outreach"
    DISCOUNT = "discount"
    OPTIMIZATION = "optimization"
    TRAINING = "training"
    ESCALATION = "escalation"
    REACTIVATION = "reactivation"


class InterventionPriority(str, Enum):
    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InterventionChannel(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    MEETING = "meeting"
    IN_APP = "in_app"
    AUTOMATED = "automated"


@dataclass
class Intervention:
    """A recommended intervention."""
    id: str
    type: InterventionType
    priority: InterventionPriority
    channel: InterventionChannel
    title: str
    description: str
    action_items: List[str]
    expected_impact: Dict[str, float]  # metric -> expected change
    confidence: float
    due_within_days: int
    owner: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, skipped
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class ClientContext:
    """Context for generating interventions."""
    client_id: str
    client_name: str
    health_score: float
    churn_probability: float
    roas: float
    target_roas: float
    monthly_spend: float
    tenure_months: int
    last_contact_days: int
    engagement_score: float
    recent_issues: List[str]
    contract_end_days: Optional[int] = None
    payment_status: str = "current"


class InterventionEngine:
    """
    Generate AI-driven intervention recommendations.
    """
    
    def __init__(self):
        self.intervention_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load intervention templates."""
        return {
            "executive_outreach": {
                "type": InterventionType.OUTREACH,
                "title": "Executive Business Review",
                "description": "Schedule executive-level call to discuss account strategy and value.",
                "action_items": [
                    "Review account history and prepare insights",
                    "Prepare ROI analysis deck",
                    "Schedule call with C-level stakeholder",
                    "Present value proposition and roadmap"
                ],
                "expected_impact": {"health_score": 15, "churn_probability": -0.20},
                "due_days": 3
            },
            "performance_optimization": {
                "type": InterventionType.OPTIMIZATION,
                "title": "Campaign Optimization Session",
                "description": "Deep-dive optimization to improve underperforming campaigns.",
                "action_items": [
                    "Analyze top 5 underperforming campaigns",
                    "Identify creative and targeting issues",
                    "Implement optimizations",
                    "Set up enhanced monitoring"
                ],
                "expected_impact": {"roas": 0.5, "health_score": 10},
                "due_days": 5
            },
            "retention_discount": {
                "type": InterventionType.DISCOUNT,
                "title": "Loyalty Discount Offer",
                "description": "Offer strategic discount to retain high-value client.",
                "action_items": [
                    "Calculate client lifetime value",
                    "Determine discount tier (10-25%)",
                    "Prepare contract amendment",
                    "Present offer with value framing"
                ],
                "expected_impact": {"churn_probability": -0.30, "health_score": 20},
                "due_days": 7
            },
            "training_session": {
                "type": InterventionType.TRAINING,
                "title": "Platform Training & Best Practices",
                "description": "Comprehensive training to improve client's platform utilization.",
                "action_items": [
                    "Assess current platform usage",
                    "Identify feature adoption gaps",
                    "Create customized training plan",
                    "Deliver training and provide resources"
                ],
                "expected_impact": {"engagement_score": 25, "health_score": 8},
                "due_days": 14
            },
            "manager_escalation": {
                "type": InterventionType.ESCALATION,
                "title": "Account Manager Escalation",
                "description": "Escalate to senior account manager for strategic review.",
                "action_items": [
                    "Document all account issues",
                    "Brief senior AM on situation",
                    "Schedule joint client call",
                    "Develop recovery action plan"
                ],
                "expected_impact": {"health_score": 12, "churn_probability": -0.15},
                "due_days": 2
            },
            "quick_win_campaign": {
                "type": InterventionType.OPTIMIZATION,
                "title": "Quick Win Campaign Setup",
                "description": "Launch proven high-ROAS campaign to demonstrate value.",
                "action_items": [
                    "Identify best performing templates",
                    "Create campaign with proven targeting",
                    "Set conservative initial budget",
                    "Monitor closely for first 7 days"
                ],
                "expected_impact": {"roas": 1.0, "health_score": 15},
                "due_days": 7
            },
            "reactivation_campaign": {
                "type": InterventionType.REACTIVATION,
                "title": "Client Reactivation Campaign",
                "description": "Multi-touch reactivation for dormant client.",
                "action_items": [
                    "Research recent company news/changes",
                    "Prepare personalized reactivation offer",
                    "Execute 5-touch outreach sequence",
                    "Track and follow up on engagement"
                ],
                "expected_impact": {"health_score": 30},
                "due_days": 21
            }
        }
    
    def generate_interventions(
        self,
        context: ClientContext
    ) -> List[Intervention]:
        """
        Generate recommended interventions for a client.
        
        Uses rule-based + ML scoring to recommend optimal interventions.
        """
        interventions = []
        
        # Critical churn risk - immediate action
        if context.churn_probability >= 0.7:
            interventions.append(self._create_intervention(
                "executive_outreach",
                context,
                priority=InterventionPriority.IMMEDIATE,
                confidence=0.92
            ))
            interventions.append(self._create_intervention(
                "manager_escalation",
                context,
                priority=InterventionPriority.IMMEDIATE,
                confidence=0.88
            ))
            
            # Add discount for high-value clients
            if context.monthly_spend > 10000:
                interventions.append(self._create_intervention(
                    "retention_discount",
                    context,
                    priority=InterventionPriority.HIGH,
                    confidence=0.75
                ))
        
        # High churn risk
        elif context.churn_probability >= 0.5:
            interventions.append(self._create_intervention(
                "executive_outreach",
                context,
                priority=InterventionPriority.HIGH,
                confidence=0.85
            ))
        
        # Poor ROAS
        if context.roas < context.target_roas * 0.8:
            interventions.append(self._create_intervention(
                "performance_optimization",
                context,
                priority=InterventionPriority.HIGH,
                confidence=0.90
            ))
            
            # Add quick win for struggling accounts
            if context.roas < context.target_roas * 0.5:
                interventions.append(self._create_intervention(
                    "quick_win_campaign",
                    context,
                    priority=InterventionPriority.HIGH,
                    confidence=0.82
                ))
        
        # Low engagement
        if context.engagement_score < 40:
            interventions.append(self._create_intervention(
                "training_session",
                context,
                priority=InterventionPriority.MEDIUM,
                confidence=0.78
            ))
        
        # No recent contact
        if context.last_contact_days > 14:
            if context.health_score < 60:
                interventions.append(self._create_intervention(
                    "executive_outreach",
                    context,
                    priority=InterventionPriority.MEDIUM,
                    confidence=0.72
                ))
        
        # Contract expiring
        if context.contract_end_days and context.contract_end_days < 30:
            interventions.append(self._create_intervention(
                "executive_outreach",
                context,
                priority=InterventionPriority.HIGH,
                confidence=0.88
            ))
        
        # Dormant client
        if context.last_contact_days > 30 and context.health_score < 30:
            interventions.append(self._create_intervention(
                "reactivation_campaign",
                context,
                priority=InterventionPriority.MEDIUM,
                confidence=0.65
            ))
        
        # Deduplicate and rank
        interventions = self._deduplicate_interventions(interventions)
        interventions = self._rank_interventions(interventions, context)
        
        return interventions[:5]  # Return top 5 recommendations
    
    def _create_intervention(
        self,
        template_name: str,
        context: ClientContext,
        priority: InterventionPriority,
        confidence: float
    ) -> Intervention:
        """Create intervention from template."""
        import hashlib
        
        template = self.intervention_templates[template_name]
        
        intervention_id = hashlib.md5(
            f"{context.client_id}{template_name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        return Intervention(
            id=intervention_id,
            type=template["type"],
            priority=priority,
            channel=self._determine_channel(template["type"], context),
            title=template["title"],
            description=template["description"].replace(
                "{client_name}", context.client_name
            ),
            action_items=template["action_items"],
            expected_impact=template["expected_impact"],
            confidence=confidence,
            due_within_days=template["due_days"]
        )
    
    def _determine_channel(
        self,
        intervention_type: InterventionType,
        context: ClientContext
    ) -> InterventionChannel:
        """Determine best channel for intervention."""
        # High-value clients get personal outreach
        if context.monthly_spend > 20000:
            if intervention_type in [InterventionType.OUTREACH, InterventionType.ESCALATION]:
                return InterventionChannel.MEETING
            return InterventionChannel.PHONE
        
        # Critical situations need phone/meeting
        if context.churn_probability > 0.7:
            return InterventionChannel.PHONE
        
        # Trainings via meeting
        if intervention_type == InterventionType.TRAINING:
            return InterventionChannel.MEETING
        
        # Default to email
        return InterventionChannel.EMAIL
    
    def _deduplicate_interventions(
        self,
        interventions: List[Intervention]
    ) -> List[Intervention]:
        """Remove duplicate intervention types."""
        seen_types = set()
        unique = []
        
        for intervention in interventions:
            key = f"{intervention.type.value}_{intervention.title}"
            if key not in seen_types:
                seen_types.add(key)
                unique.append(intervention)
        
        return unique
    
    def _rank_interventions(
        self,
        interventions: List[Intervention],
        context: ClientContext
    ) -> List[Intervention]:
        """Rank interventions by expected impact and confidence."""
        def score(intervention: Intervention) -> float:
            # Priority score
            priority_score = {
                InterventionPriority.IMMEDIATE: 4,
                InterventionPriority.HIGH: 3,
                InterventionPriority.MEDIUM: 2,
                InterventionPriority.LOW: 1
            }[intervention.priority]
            
            # Impact score
            impact_score = sum(abs(v) for v in intervention.expected_impact.values())
            
            # Confidence
            confidence_score = intervention.confidence
            
            return priority_score * 2 + impact_score * 0.5 + confidence_score * 3
        
        return sorted(interventions, key=score, reverse=True)
    
    def estimate_impact(
        self,
        interventions: List[Intervention],
        context: ClientContext
    ) -> Dict[str, Any]:
        """Estimate combined impact of interventions."""
        combined_impact = {}
        
        for intervention in interventions:
            for metric, change in intervention.expected_impact.items():
                if metric not in combined_impact:
                    combined_impact[metric] = 0
                combined_impact[metric] += change
        
        # Estimate new values
        predicted = {
            "current_health_score": context.health_score,
            "predicted_health_score": min(100, context.health_score + combined_impact.get("health_score", 0)),
            "current_churn_probability": context.churn_probability,
            "predicted_churn_probability": max(0, context.churn_probability + combined_impact.get("churn_probability", 0)),
            "current_roas": context.roas,
            "predicted_roas": context.roas + combined_impact.get("roas", 0),
            "interventions_count": len(interventions),
            "estimated_effort_hours": sum(
                4 if i.priority == InterventionPriority.IMMEDIATE else
                3 if i.priority == InterventionPriority.HIGH else 2
                for i in interventions
            )
        }
        
        return predicted


class InterventionTracker:
    """Track intervention execution and outcomes."""
    
    def __init__(self):
        self.interventions: Dict[str, Intervention] = {}
        self.outcomes: Dict[str, Dict[str, Any]] = {}
    
    def assign_intervention(
        self,
        intervention: Intervention,
        owner: str
    ):
        """Assign intervention to owner."""
        intervention.owner = owner
        intervention.status = "in_progress"
        self.interventions[intervention.id] = intervention
    
    def complete_intervention(
        self,
        intervention_id: str,
        outcome: Dict[str, Any]
    ):
        """Mark intervention as completed with outcome."""
        if intervention_id in self.interventions:
            self.interventions[intervention_id].status = "completed"
            self.interventions[intervention_id].completed_at = datetime.now()
            self.outcomes[intervention_id] = outcome
    
    def get_effectiveness(
        self,
        intervention_type: Optional[InterventionType] = None
    ) -> Dict[str, Any]:
        """Calculate intervention effectiveness."""
        completed = [i for i in self.interventions.values() if i.status == "completed"]
        
        if intervention_type:
            completed = [i for i in completed if i.type == intervention_type]
        
        if not completed:
            return {"sample_size": 0}
        
        # Calculate success rate
        successful = 0
        for i in completed:
            outcome = self.outcomes.get(i.id, {})
            if outcome.get("churn_prevented") or outcome.get("health_improved"):
                successful += 1
        
        return {
            "sample_size": len(completed),
            "success_rate": successful / len(completed) if completed else 0,
            "avg_completion_time_days": np.mean([
                (i.completed_at - i.created_at).days
                for i in completed if i.completed_at
            ]) if completed else 0
        }


# Singletons
_intervention_engine: Optional[InterventionEngine] = None
_intervention_tracker: Optional[InterventionTracker] = None


def get_intervention_engine() -> InterventionEngine:
    global _intervention_engine
    if _intervention_engine is None:
        _intervention_engine = InterventionEngine()
    return _intervention_engine


def get_intervention_tracker() -> InterventionTracker:
    global _intervention_tracker
    if _intervention_tracker is None:
        _intervention_tracker = InterventionTracker()
    return _intervention_tracker
