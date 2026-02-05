"""
Audience Segmentation ML Model

Uses K-Means clustering for audience segmentation and lookalike generation.
Integrated with MLflow for model versioning and tracking.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import random
import math

try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    np = None

from app.services.mlops import MLflowClient, ModelType


@dataclass
class UserProfile:
    """User behavioral profile for segmentation."""
    user_id: str
    
    # Demographics (if available)
    age_group: Optional[str] = None  # '18-24', '25-34', etc.
    gender: Optional[str] = None
    location: Optional[str] = None
    
    # Behavioral metrics
    sessions_30d: int = 0
    pageviews_30d: int = 0
    time_on_site_avg: float = 0.0  # minutes
    bounce_rate: float = 0.0
    
    # Purchase behavior
    purchases_30d: int = 0
    total_revenue_30d: float = 0.0
    avg_order_value: float = 0.0
    
    # Engagement
    email_opens_30d: int = 0
    email_clicks_30d: int = 0
    ad_clicks_30d: int = 0
    
    # Recency
    days_since_last_visit: int = 0
    days_since_last_purchase: int = 0


@dataclass
class AudienceSegment:
    """An audience segment with characteristics."""
    segment_id: str
    name: str
    size: int
    percentage: float
    
    # Segment characteristics (averages)
    avg_ltv: float
    avg_sessions: float
    avg_revenue: float
    avg_engagement: float
    
    # Defining traits
    key_traits: List[str]
    
    # RFM scores (Recency, Frequency, Monetary)
    recency_score: float
    frequency_score: float
    monetary_score: float
    
    # Platform sync status
    synced_platforms: List[str] = field(default_factory=list)


class KMeansSegmenter:
    """
    K-Means based audience segmentation.
    
    Features include:
    - Automatic optimal K selection (elbow method)
    - Segment naming based on characteristics
    - RFM scoring
    """
    
    SEGMENT_ARCHETYPES = [
        ("Champions", lambda r, f, m: r >= 4 and f >= 4 and m >= 4),
        ("Loyal Customers", lambda r, f, m: r >= 3 and f >= 4),
        ("Potential Loyals", lambda r, f, m: r >= 3 and f >= 2 and m >= 2),
        ("New Customers", lambda r, f, m: r >= 4 and f <= 2),
        ("Promising", lambda r, f, m: r >= 3 and f <= 2 and m <= 2),
        ("Need Attention", lambda r, f, m: r >= 2 and r <= 3 and f >= 2 and m >= 2),
        ("About to Sleep", lambda r, f, m: r <= 2 and f >= 2),
        ("At Risk", lambda r, f, m: r <= 2 and f >= 3 and m >= 3),
        ("Can't Lose", lambda r, f, m: r <= 2 and f >= 4 and m >= 4),
        ("Hibernating", lambda r, f, m: r <= 2 and f <= 2),
        ("Lost", lambda r, f, m: r <= 1 and f <= 1),
    ]
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler() if HAS_ML_DEPS else None
        self.kmeans = None
    
    def extract_features(self, users: List[UserProfile]) -> List[List[float]]:
        """Extract feature vectors from user profiles."""
        features = []
        
        for user in users:
            # Calculate derived features
            engagement_score = self._calculate_engagement(user)
            value_score = self._calculate_value(user)
            recency_score = self._calculate_recency(user)
            
            feature_vector = [
                user.sessions_30d,
                user.pageviews_30d,
                user.time_on_site_avg,
                user.purchases_30d,
                user.total_revenue_30d,
                engagement_score,
                value_score,
                recency_score,
                user.email_opens_30d,
                user.ad_clicks_30d
            ]
            features.append(feature_vector)
        
        return features
    
    def _calculate_engagement(self, user: UserProfile) -> float:
        """Calculate engagement score 0-100."""
        score = 0
        
        # Session frequency
        score += min(30, user.sessions_30d * 3)
        
        # Email engagement
        if user.email_opens_30d > 0:
            score += min(20, user.email_opens_30d * 2)
        
        # Time on site
        score += min(25, user.time_on_site_avg * 5)
        
        # Low bounce is good
        score += max(0, 25 - user.bounce_rate * 25)
        
        return min(100, score)
    
    def _calculate_value(self, user: UserProfile) -> float:
        """Calculate customer value score 0-100."""
        score = 0
        
        # Revenue contribution
        if user.total_revenue_30d > 0:
            score += min(40, user.total_revenue_30d / 10)
        
        # Purchase frequency
        score += min(30, user.purchases_30d * 10)
        
        # AOV
        if user.avg_order_value > 0:
            score += min(30, user.avg_order_value / 5)
        
        return min(100, score)
    
    def _calculate_recency(self, user: UserProfile) -> float:
        """Calculate recency score 0-100 (higher = more recent)."""
        # Visit recency
        visit_score = max(0, 50 - user.days_since_last_visit * 2)
        
        # Purchase recency
        purchase_score = max(0, 50 - user.days_since_last_purchase)
        
        return min(100, visit_score + purchase_score)
    
    def fit_predict(self, users: List[UserProfile]) -> Dict[str, List[str]]:
        """
        Fit K-Means and predict segment assignments.
        
        Returns mapping of segment_id to user_ids.
        """
        if not users:
            return {}
        
        features = self.extract_features(users)
        
        if HAS_ML_DEPS:
            # Normalize features
            X = np.array(features)
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit K-Means
            self.kmeans = KMeans(n_clusters=min(self.n_clusters, len(users)), random_state=42)
            labels = self.kmeans.fit_predict(X_scaled)
        else:
            # Simple fallback segmentation without sklearn
            labels = [i % self.n_clusters for i in range(len(users))]
        
        # Group users by segment
        segments = defaultdict(list)
        for i, user in enumerate(users):
            segments[f"segment_{labels[i]}"].append(user.user_id)
        
        return dict(segments)


class AudienceSegmenter:
    """
    Main audience segmentation engine.
    
    Provides:
    - RFM analysis
    - K-Means clustering
    - Lookalike audience generation
    - Platform sync support
    """
    
    def __init__(self, mlflow_client: Optional[MLflowClient] = None):
        self.mlflow_client = mlflow_client or MLflowClient()
        self.segmenter = KMeansSegmenter()
    
    def segment_audience(
        self,
        users: List[UserProfile],
        n_segments: int = 5
    ) -> List[AudienceSegment]:
        """
        Segment audience into distinct groups.
        
        Returns list of audience segments with characteristics.
        """
        if not users:
            return []
        
        self.segmenter.n_clusters = min(n_segments, len(users))
        
        # Get segment assignments
        segment_mapping = self.segmenter.fit_predict(users)
        
        # Build user index
        user_index = {u.user_id: u for u in users}
        
        # Create segment objects
        segments = []
        total_users = len(users)
        
        for seg_id, user_ids in segment_mapping.items():
            segment_users = [user_index[uid] for uid in user_ids if uid in user_index]
            
            if not segment_users:
                continue
            
            segment = self._create_segment(seg_id, segment_users, total_users)
            segments.append(segment)
        
        # Sort by size
        segments.sort(key=lambda s: s.size, reverse=True)
        
        # Log to MLflow
        self._log_segmentation(segments)
        
        return segments
    
    def _create_segment(
        self,
        segment_id: str,
        users: List[UserProfile],
        total_users: int
    ) -> AudienceSegment:
        """Create segment object with aggregated characteristics."""
        n = len(users)
        
        # Calculate averages
        avg_sessions = sum(u.sessions_30d for u in users) / n
        avg_revenue = sum(u.total_revenue_30d for u in users) / n
        avg_ltv = avg_revenue * 12  # Annualized estimate
        
        # Calculate RFM scores (1-5 scale)
        recency_vals = [u.days_since_last_visit for u in users]
        frequency_vals = [u.sessions_30d for u in users]
        monetary_vals = [u.total_revenue_30d for u in users]
        
        recency_score = 5 - min(4, sum(recency_vals) / n / 7)  # Lower days = higher score
        frequency_score = min(5, 1 + sum(frequency_vals) / n / 5)
        monetary_score = min(5, 1 + sum(monetary_vals) / n / 100)
        
        # Calculate engagement
        engagements = [self.segmenter._calculate_engagement(u) for u in users]
        avg_engagement = sum(engagements) / n
        
        # Determine segment name based on RFM
        name = self._determine_segment_name(recency_score, frequency_score, monetary_score)
        
        # Extract key traits
        traits = self._extract_key_traits(users, recency_score, frequency_score, monetary_score)
        
        return AudienceSegment(
            segment_id=segment_id,
            name=name,
            size=n,
            percentage=round(n / total_users * 100, 1),
            avg_ltv=round(avg_ltv, 2),
            avg_sessions=round(avg_sessions, 1),
            avg_revenue=round(avg_revenue, 2),
            avg_engagement=round(avg_engagement, 1),
            key_traits=traits,
            recency_score=round(recency_score, 2),
            frequency_score=round(frequency_score, 2),
            monetary_score=round(monetary_score, 2)
        )
    
    def _determine_segment_name(self, r: float, f: float, m: float) -> str:
        """Determine segment name based on RFM scores."""
        for name, condition in KMeansSegmenter.SEGMENT_ARCHETYPES:
            if condition(r, f, m):
                return name
        return "General Audience"
    
    def _extract_key_traits(
        self,
        users: List[UserProfile],
        r: float,
        f: float,
        m: float
    ) -> List[str]:
        """Extract key characteristics of the segment."""
        traits = []
        n = len(users)
        
        if r >= 4:
            traits.append("Recently active")
        elif r <= 2:
            traits.append("Inactive/dormant")
        
        if f >= 4:
            traits.append("Frequent visitors")
        elif f <= 2:
            traits.append("Infrequent visitors")
        
        if m >= 4:
            traits.append("High spenders")
        elif m >= 2:
            traits.append("Medium spenders")
        else:
            traits.append("Low/no spend")
        
        # Check email engagement
        avg_opens = sum(u.email_opens_30d for u in users) / n
        if avg_opens > 5:
            traits.append("Email engaged")
        
        # Check ad response
        avg_ad_clicks = sum(u.ad_clicks_30d for u in users) / n
        if avg_ad_clicks > 3:
            traits.append("Ad responsive")
        
        return traits[:5]
    
    def generate_lookalike(
        self,
        seed_users: List[UserProfile],
        pool_users: List[UserProfile],
        expansion_factor: float = 5.0
    ) -> List[str]:
        """
        Generate lookalike audience from seed users.
        
        Finds users in pool that are similar to seed users.
        """
        if not seed_users or not pool_users:
            return []
        
        # Get seed characteristics
        seed_features = self.segmenter.extract_features(seed_users)
        
        if HAS_ML_DEPS:
            seed_center = np.mean(seed_features, axis=0)
            seed_std = np.std(seed_features, axis=0) + 0.001
        else:
            seed_center = [sum(f[i] for f in seed_features) / len(seed_features) 
                         for i in range(len(seed_features[0]))]
            seed_std = [1.0] * len(seed_center)
        
        # Score pool users by similarity
        pool_features = self.segmenter.extract_features(pool_users)
        
        scores = []
        for i, features in enumerate(pool_features):
            # Normalized distance to seed center
            if HAS_ML_DEPS:
                distance = np.sum(((np.array(features) - seed_center) / seed_std) ** 2)
            else:
                distance = sum(((features[j] - seed_center[j]) / seed_std[j]) ** 2 
                             for j in range(len(features)))
            
            similarity = 1 / (1 + math.sqrt(distance))
            scores.append((pool_users[i].user_id, similarity))
        
        # Sort by similarity and take top N
        scores.sort(key=lambda x: x[1], reverse=True)
        target_size = int(len(seed_users) * expansion_factor)
        
        return [uid for uid, _ in scores[:target_size]]
    
    def calculate_segment_value(self, segment: AudienceSegment) -> Dict[str, Any]:
        """Calculate the value and potential of a segment."""
        # Estimated annual value
        annual_value = segment.avg_ltv * segment.size
        
        # Growth potential based on engagement vs spending gap
        if segment.avg_engagement > 50 and segment.monetary_score < 3:
            growth_potential = "High"
            potential_uplift = segment.avg_revenue * 0.5 * segment.size
        elif segment.recency_score < 3 and segment.monetary_score >= 3:
            growth_potential = "Medium - Reactivation"
            potential_uplift = segment.avg_revenue * 0.3 * segment.size
        else:
            growth_potential = "Low"
            potential_uplift = segment.avg_revenue * 0.1 * segment.size
        
        return {
            "annual_value": round(annual_value, 2),
            "growth_potential": growth_potential,
            "potential_uplift": round(potential_uplift, 2),
            "recommended_strategy": self._recommend_strategy(segment)
        }
    
    def _recommend_strategy(self, segment: AudienceSegment) -> str:
        """Recommend engagement strategy for segment."""
        if segment.name == "Champions":
            return "VIP treatment, loyalty rewards, referral programs"
        elif segment.name == "Loyal Customers":
            return "Upselling, early access to new products"
        elif segment.name == "Potential Loyals":
            return "Membership offers, personalized recommendations"
        elif segment.name == "New Customers":
            return "Onboarding emails, first purchase incentives"
        elif segment.name == "At Risk":
            return "Win-back campaigns, feedback surveys"
        elif segment.name == "Can't Lose":
            return "Urgent reactivation, personal outreach"
        elif segment.name == "Hibernating":
            return "Reactivation offers, brand reminders"
        else:
            return "General nurturing, brand awareness"
    
    def to_api_response(self, segments: List[AudienceSegment]) -> Dict[str, Any]:
        """Convert segments to API-friendly format."""
        return {
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "name": s.name,
                    "size": s.size,
                    "percentage": s.percentage,
                    "avg_ltv": s.avg_ltv,
                    "avg_sessions": s.avg_sessions,
                    "avg_revenue": s.avg_revenue,
                    "avg_engagement": s.avg_engagement,
                    "key_traits": s.key_traits,
                    "rfm_scores": {
                        "recency": s.recency_score,
                        "frequency": s.frequency_score,
                        "monetary": s.monetary_score
                    },
                    "synced_platforms": s.synced_platforms
                }
                for s in segments
            ],
            "total_audience": sum(s.size for s in segments),
            "total_segments": len(segments)
        }
    
    def _log_segmentation(self, segments: List[AudienceSegment]):
        """Log segmentation to MLflow."""
        try:
            experiment = self.mlflow_client.get_experiment("audience_segmentation")
            if not experiment:
                experiment = self.mlflow_client.create_experiment("audience_segmentation")
            
            run = self.mlflow_client.start_run(
                experiment.experiment_id,
                f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            self.mlflow_client.log_metrics(run.run_id, {
                "num_segments": len(segments),
                "total_users": sum(s.size for s in segments),
                "avg_segment_size": sum(s.size for s in segments) / max(len(segments), 1)
            })
            
            self.mlflow_client.end_run(run.run_id)
        except Exception:
            pass


# Singleton instance
_segmenter: Optional[AudienceSegmenter] = None


def get_audience_segmenter() -> AudienceSegmenter:
    """Get or create the audience segmenter instance."""
    global _segmenter
    if _segmenter is None:
        _segmenter = AudienceSegmenter()
    return _segmenter
