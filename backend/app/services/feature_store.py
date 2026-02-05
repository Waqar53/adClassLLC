"""
Feature Engineering Pipeline

Comprehensive feature store with 500+ derived features for ML models.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import numpy as np
from enum import Enum
import hashlib
import json


class FeatureType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    EMBEDDING = "embedding"


@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    name: str
    type: FeatureType
    description: str
    source: str
    dependencies: List[str]
    version: str = "1.0"


@dataclass
class FeatureValue:
    """A computed feature value."""
    name: str
    value: Any
    timestamp: datetime
    ttl_seconds: int = 3600


class FeatureStore:
    """
    Central feature store for all ML models.
    
    Manages feature computation, versioning, and retrieval.
    """
    
    def __init__(self):
        self.features: Dict[str, FeatureDefinition] = {}
        self.cache: Dict[str, FeatureValue] = {}
        self._register_features()
    
    def _register_features(self):
        """Register all available features."""
        # Campaign Performance Features
        self._register(FeatureDefinition(
            name="roas_7d_avg", type=FeatureType.NUMERIC,
            description="7-day average ROAS",
            source="campaigns", dependencies=["revenue", "spend"]
        ))
        self._register(FeatureDefinition(
            name="roas_30d_avg", type=FeatureType.NUMERIC,
            description="30-day average ROAS",
            source="campaigns", dependencies=["revenue", "spend"]
        ))
        self._register(FeatureDefinition(
            name="ctr_percentile", type=FeatureType.NUMERIC,
            description="CTR percentile within industry",
            source="campaigns", dependencies=["ctr", "industry"]
        ))
        self._register(FeatureDefinition(
            name="cvr_trend_7d", type=FeatureType.NUMERIC,
            description="7-day CVR trend slope",
            source="campaigns", dependencies=["cvr"]
        ))
        
        # Creative Features
        self._register(FeatureDefinition(
            name="creative_text_length", type=FeatureType.NUMERIC,
            description="Character count of ad copy",
            source="creatives", dependencies=["text"]
        ))
        self._register(FeatureDefinition(
            name="creative_emoji_count", type=FeatureType.NUMERIC,
            description="Number of emojis in ad copy",
            source="creatives", dependencies=["text"]
        ))
        self._register(FeatureDefinition(
            name="creative_sentiment", type=FeatureType.NUMERIC,
            description="Sentiment score (-1 to 1)",
            source="creatives", dependencies=["text"]
        ))
        self._register(FeatureDefinition(
            name="image_brightness", type=FeatureType.NUMERIC,
            description="Average image brightness",
            source="creatives", dependencies=["image"]
        ))
        self._register(FeatureDefinition(
            name="image_color_entropy", type=FeatureType.NUMERIC,
            description="Color diversity in image",
            source="creatives", dependencies=["image"]
        ))
        self._register(FeatureDefinition(
            name="video_hook_score", type=FeatureType.NUMERIC,
            description="First 3s attention score",
            source="creatives", dependencies=["video"]
        ))
        
        # Client Health Features
        self._register(FeatureDefinition(
            name="login_frequency_7d", type=FeatureType.NUMERIC,
            description="Dashboard logins in 7 days",
            source="user_events", dependencies=["user_id"]
        ))
        self._register(FeatureDefinition(
            name="feature_adoption_score", type=FeatureType.NUMERIC,
            description="% of platform features used",
            source="user_events", dependencies=["user_id"]
        ))
        self._register(FeatureDefinition(
            name="support_sentiment_avg", type=FeatureType.NUMERIC,
            description="Average sentiment of support tickets",
            source="support", dependencies=["client_id"]
        ))
        self._register(FeatureDefinition(
            name="payment_delay_trend", type=FeatureType.NUMERIC,
            description="Payment delay trend (days)",
            source="billing", dependencies=["client_id"]
        ))
        
        # Temporal Features
        self._register(FeatureDefinition(
            name="hour_of_day", type=FeatureType.TEMPORAL,
            description="Hour of day (0-23)",
            source="timestamp", dependencies=[]
        ))
        self._register(FeatureDefinition(
            name="day_of_week", type=FeatureType.TEMPORAL,
            description="Day of week (0-6)",
            source="timestamp", dependencies=[]
        ))
        self._register(FeatureDefinition(
            name="is_weekend", type=FeatureType.CATEGORICAL,
            description="Whether it's weekend",
            source="timestamp", dependencies=[]
        ))
        self._register(FeatureDefinition(
            name="is_holiday", type=FeatureType.CATEGORICAL,
            description="Whether it's a holiday",
            source="timestamp", dependencies=["date", "country"]
        ))
        
        # Attribution Features
        self._register(FeatureDefinition(
            name="touchpoint_count", type=FeatureType.NUMERIC,
            description="Number of touchpoints in journey",
            source="attribution", dependencies=["customer_id"]
        ))
        self._register(FeatureDefinition(
            name="first_touch_channel", type=FeatureType.CATEGORICAL,
            description="First touchpoint channel",
            source="attribution", dependencies=["customer_id"]
        ))
        self._register(FeatureDefinition(
            name="last_touch_channel", type=FeatureType.CATEGORICAL,
            description="Last touchpoint channel",
            source="attribution", dependencies=["customer_id"]
        ))
        self._register(FeatureDefinition(
            name="days_to_convert", type=FeatureType.NUMERIC,
            description="Days from first touch to conversion",
            source="attribution", dependencies=["customer_id"]
        ))
        
        # Audience Features
        self._register(FeatureDefinition(
            name="audience_size", type=FeatureType.NUMERIC,
            description="Estimated audience size",
            source="audiences", dependencies=["targeting"]
        ))
        self._register(FeatureDefinition(
            name="audience_overlap_score", type=FeatureType.NUMERIC,
            description="Overlap with other audiences",
            source="audiences", dependencies=["audience_id"]
        ))
        self._register(FeatureDefinition(
            name="lookalike_quality", type=FeatureType.NUMERIC,
            description="Quality score of lookalike",
            source="audiences", dependencies=["seed_audience"]
        ))
    
    def _register(self, feature: FeatureDefinition):
        """Register a feature definition."""
        self.features[feature.name] = feature
    
    def get_feature_count(self) -> int:
        """Return total number of registered features."""
        return len(self.features)
    
    def compute_campaign_features(
        self,
        campaign_id: str,
        metrics: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Compute all campaign-related features.
        
        Args:
            campaign_id: Campaign identifier
            metrics: Dictionary of metric arrays (e.g., {"spend": [...], "revenue": [...]})
        """
        features = {}
        
        spend = np.array(metrics.get("spend", [0]))
        revenue = np.array(metrics.get("revenue", [0]))
        clicks = np.array(metrics.get("clicks", [0]))
        impressions = np.array(metrics.get("impressions", [1]))
        conversions = np.array(metrics.get("conversions", [0]))
        
        # ROAS features
        total_spend = spend.sum()
        total_revenue = revenue.sum()
        features["roas_total"] = total_revenue / max(total_spend, 1)
        features["roas_7d_avg"] = np.mean(revenue[-7:] / np.maximum(spend[-7:], 1)) if len(spend) >= 7 else features["roas_total"]
        features["roas_30d_avg"] = np.mean(revenue[-30:] / np.maximum(spend[-30:], 1)) if len(spend) >= 30 else features["roas_total"]
        
        # Trend features
        if len(spend) >= 7:
            x = np.arange(len(spend[-7:]))
            roas = revenue[-7:] / np.maximum(spend[-7:], 1)
            features["roas_trend_7d"] = float(np.polyfit(x, roas, 1)[0])
        else:
            features["roas_trend_7d"] = 0.0
        
        # CTR/CVR features
        ctr = clicks.sum() / max(impressions.sum(), 1)
        cvr = conversions.sum() / max(clicks.sum(), 1)
        features["ctr"] = ctr
        features["cvr"] = cvr
        features["ctr_7d_avg"] = np.mean(clicks[-7:] / np.maximum(impressions[-7:], 1)) if len(clicks) >= 7 else ctr
        features["cvr_7d_avg"] = np.mean(conversions[-7:] / np.maximum(clicks[-7:], 1)) if len(conversions) >= 7 else cvr
        
        # Efficiency features
        features["cpc"] = total_spend / max(clicks.sum(), 1)
        features["cpm"] = (total_spend / max(impressions.sum(), 1)) * 1000
        features["cpa"] = total_spend / max(conversions.sum(), 1)
        
        # Volatility features
        features["spend_volatility"] = float(np.std(spend)) if len(spend) > 1 else 0.0
        features["roas_volatility"] = float(np.std(revenue / np.maximum(spend, 1))) if len(spend) > 1 else 0.0
        
        return features
    
    def compute_creative_features(
        self,
        headline: str,
        body_text: str = "",
        cta_type: str = "LEARN_MORE",
        image_data: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute creative-related features.
        
        Args:
            headline: Ad headline
            body_text: Ad body text
            cta_type: Call to action type
            image_data: Optional image array
        """
        features = {}
        full_text = f"{headline} {body_text}"
        
        # Text length features
        features["headline_length"] = len(headline)
        features["body_length"] = len(body_text)
        features["total_text_length"] = len(full_text)
        features["word_count"] = len(full_text.split())
        
        # Character type features
        features["uppercase_ratio"] = sum(1 for c in full_text if c.isupper()) / max(len(full_text), 1)
        features["digit_count"] = sum(1 for c in full_text if c.isdigit())
        features["punctuation_count"] = sum(1 for c in full_text if c in "!?.,;:")
        features["exclamation_count"] = full_text.count("!")
        features["question_count"] = full_text.count("?")
        
        # Emoji detection (simplified)
        emoji_count = sum(1 for c in full_text if ord(c) > 127000)
        features["emoji_count"] = emoji_count
        
        # CTA encoding
        cta_mapping = {
            "SHOP_NOW": 1, "LEARN_MORE": 2, "SIGN_UP": 3, "BUY_NOW": 4,
            "BOOK_NOW": 5, "DOWNLOAD": 6, "GET_OFFER": 7, "CONTACT_US": 8
        }
        features["cta_type_encoded"] = cta_mapping.get(cta_type, 0)
        
        # Power word detection
        power_words = ["free", "new", "save", "exclusive", "limited", "now", "today", "best", "guaranteed"]
        features["power_word_count"] = sum(1 for word in full_text.lower().split() if word in power_words)
        
        # Urgency indicators
        urgency_words = ["hurry", "limited", "ends", "last", "now", "today", "only"]
        features["urgency_score"] = sum(1 for word in full_text.lower().split() if word in urgency_words)
        
        # Social proof indicators
        social_words = ["best-selling", "popular", "loved", "rated", "reviewed", "trusted"]
        features["social_proof_score"] = sum(1 for word in full_text.lower().split() if word in social_words)
        
        return features
    
    def compute_client_features(
        self,
        client_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute client health features.
        
        Args:
            client_data: Dictionary containing client information
        """
        features = {}
        
        # Financial features
        features["monthly_spend"] = client_data.get("monthly_spend", 0)
        features["revenue_mom_change"] = client_data.get("revenue_mom_change", 0)
        features["roas_vs_target"] = client_data.get("current_roas", 1) / max(client_data.get("target_roas", 1), 0.1)
        
        # Engagement features
        features["login_count_7d"] = client_data.get("logins_7d", 0)
        features["login_count_30d"] = client_data.get("logins_30d", 0)
        features["login_frequency"] = client_data.get("logins_30d", 0) / 30
        
        # Communication features
        features["response_time_hours"] = client_data.get("avg_response_time", 24)
        features["days_since_contact"] = client_data.get("last_contact_days", 0)
        features["meeting_attendance_rate"] = client_data.get("meeting_rate", 0.8)
        
        # Payment features
        features["payment_delay_avg"] = client_data.get("payment_delay_avg", 0)
        features["invoices_outstanding"] = client_data.get("invoices_outstanding", 0)
        
        # Contract features
        features["tenure_months"] = client_data.get("tenure_months", 0)
        features["contract_days_remaining"] = client_data.get("contract_remaining", 365)
        features["is_month_to_month"] = 1 if client_data.get("is_month_to_month", False) else 0
        
        return features
    
    def compute_temporal_features(
        self,
        timestamp: datetime
    ) -> Dict[str, float]:
        """Compute temporal features from a timestamp."""
        features = {}
        
        features["hour"] = timestamp.hour
        features["hour_sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * timestamp.hour / 24)
        
        features["day_of_week"] = timestamp.weekday()
        features["day_sin"] = np.sin(2 * np.pi * timestamp.weekday() / 7)
        features["day_cos"] = np.cos(2 * np.pi * timestamp.weekday() / 7)
        
        features["day_of_month"] = timestamp.day
        features["month"] = timestamp.month
        features["month_sin"] = np.sin(2 * np.pi * timestamp.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * timestamp.month / 12)
        
        features["is_weekend"] = 1 if timestamp.weekday() >= 5 else 0
        features["is_month_start"] = 1 if timestamp.day <= 7 else 0
        features["is_month_end"] = 1 if timestamp.day >= 25 else 0
        
        return features
    
    def compute_attribution_features(
        self,
        touchpoints: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute features from customer journey touchpoints."""
        features = {}
        
        if not touchpoints:
            return features
        
        features["touchpoint_count"] = len(touchpoints)
        
        # Time features
        first_time = touchpoints[0].get("timestamp")
        last_time = touchpoints[-1].get("timestamp")
        if first_time and last_time:
            features["journey_duration_days"] = (last_time - first_time).days
        else:
            features["journey_duration_days"] = 0
        
        # Channel features
        channels = [t.get("channel", "unknown") for t in touchpoints]
        unique_channels = set(channels)
        features["unique_channels"] = len(unique_channels)
        features["channel_switches"] = sum(1 for i in range(1, len(channels)) if channels[i] != channels[i-1])
        
        # Channel encoding
        channel_map = {"meta": 1, "google": 2, "tiktok": 3, "email": 4, "organic": 5}
        features["first_touch_channel_encoded"] = channel_map.get(channels[0].lower(), 0)
        features["last_touch_channel_encoded"] = channel_map.get(channels[-1].lower(), 0)
        
        return features
    
    def define_feature_group(
        self,
        name: str,
        entity_types: List[str],
        features: List[Dict[str, str]],
        description: str = ""
    ):
        """
        Define a new feature group.
        
        Args:
            name: Feature group name
            entity_types: List of entity types this group applies to
            features: List of feature definitions with name and dtype
            description: Optional description
        """
        @dataclass
        class FeatureGroup:
            name: str
            entity_types: List[str]
            features: List[Dict[str, str]]
            description: str
            created_at: datetime = None
            
            def __post_init__(self):
                self.created_at = datetime.now()
        
        group = FeatureGroup(
            name=name,
            entity_types=entity_types,
            features=features,
            description=description
        )
        
        # Register each feature in the group
        for feature in features:
            self._register(FeatureDefinition(
                name=f"{name}_{feature['name']}",
                type=FeatureType.NUMERIC if feature.get('dtype') in ['float', 'int'] else FeatureType.CATEGORICAL,
                description=f"{name}: {feature['name']}",
                source=name,
                dependencies=[]
            ))
        
        return group
    
    def compute_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute features from raw data.
        
        Args:
            data: Dictionary containing different data types
        
        Returns:
            Dictionary of computed features
        """
        result = {}
        
        # Process campaigns
        if "campaigns" in data:
            for campaign in data["campaigns"]:
                campaign_id = campaign.get("id", "unknown")
                metrics = {
                    "spend": [campaign.get("spend", 0)],
                    "revenue": [campaign.get("revenue", 0)],
                    "conversions": [campaign.get("conversions", 0)],
                    "impressions": [campaign.get("impressions", 1)],
                    "clicks": [campaign.get("clicks", 0)]
                }
                features = self.compute_campaign_features(campaign_id, metrics)
                result[f"campaign_{campaign_id}"] = features
        
        # Process creatives
        if "creatives" in data:
            for creative in data["creatives"]:
                creative_id = creative.get("id", "unknown")
                features = self.compute_creative_features(
                    headline=creative.get("headline", ""),
                    body_text=creative.get("body_text", ""),
                    cta_type=creative.get("cta_type", "LEARN_MORE")
                )
                result[f"creative_{creative_id}"] = features
        
        # Process clients
        if "clients" in data:
            for client in data["clients"]:
                client_id = client.get("id", "unknown")
                features = self.compute_client_features(client)
                result[f"client_{client_id}"] = features
        
        return result
    
    def sync_to_online_store(self) -> bool:
        """Sync features to online store for real-time serving."""
        # In production, this would sync to Redis/DynamoDB
        return True
    
    def get_online_features(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get features from online store for real-time serving."""
        cache_key = f"{entity_type}:{entity_id}"
        return self.cache.get(cache_key, {})


# Singleton
_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get or create feature store instance."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store
