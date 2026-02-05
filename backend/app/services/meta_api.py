"""
Meta Marketing API Integration Service

Handles data ingestion from Facebook/Instagram Ads.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import asyncio
import hashlib
import json

from app.config import settings
from app.core.cache import cache

# In production: from facebook_business.api import FacebookAdsApi
# from facebook_business.adobjects.adaccount import AdAccount


@dataclass
class MetaCampaign:
    """Meta campaign data structure."""
    campaign_id: str
    name: str
    objective: str
    status: str
    daily_budget: float
    lifetime_budget: Optional[float]
    start_time: Optional[datetime]
    stop_time: Optional[datetime]
    buying_type: str


@dataclass
class MetaAdSet:
    """Meta ad set data structure."""
    adset_id: str
    campaign_id: str
    name: str
    status: str
    targeting: Dict[str, Any]
    daily_budget: float
    optimization_goal: str
    billing_event: str


@dataclass
class MetaAd:
    """Meta ad data structure."""
    ad_id: str
    adset_id: str
    name: str
    status: str
    creative_id: str
    tracking_specs: Optional[Dict]


@dataclass
class MetaInsights:
    """Meta insights/metrics data."""
    date: date
    impressions: int
    reach: int
    clicks: int
    unique_clicks: int
    spend: float
    conversions: int
    conversion_value: float
    ctr: float
    cpc: float
    cpm: float


class MetaMarketingService:
    """
    Service for Meta (Facebook/Instagram) Marketing API integration.
    
    Handles:
    - Campaign, Ad Set, Ad data fetching
    - Performance insights retrieval
    - Audience management
    - Creative analysis
    """
    
    API_VERSION = "v18.0"
    BASE_URL = "https://graph.facebook.com"
    
    # Fields to fetch for each entity
    CAMPAIGN_FIELDS = [
        "id", "name", "objective", "status", "daily_budget", 
        "lifetime_budget", "start_time", "stop_time", "buying_type"
    ]
    
    ADSET_FIELDS = [
        "id", "campaign_id", "name", "status", "targeting",
        "daily_budget", "optimization_goal", "billing_event"
    ]
    
    AD_FIELDS = [
        "id", "adset_id", "name", "status", "creative", "tracking_specs"
    ]
    
    INSIGHTS_FIELDS = [
        "impressions", "reach", "clicks", "unique_clicks", "spend",
        "actions", "action_values", "ctr", "cpc", "cpm",
        "video_p25_watched_actions", "video_p50_watched_actions",
        "video_p75_watched_actions", "video_p100_watched_actions"
    ]
    
    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token or settings.META_ACCESS_TOKEN
        self.app_id = settings.META_APP_ID
        self.app_secret = settings.META_APP_SECRET
        self._initialized = False
    
    def _init_api(self):
        """Initialize Facebook Ads API."""
        if not self._initialized and self.access_token:
            # In production:
            # FacebookAdsApi.init(
            #     app_id=self.app_id,
            #     app_secret=self.app_secret,
            #     access_token=self.access_token
            # )
            self._initialized = True
    
    async def get_ad_accounts(self, user_id: str = "me") -> List[Dict]:
        """Get all ad accounts for a user."""
        self._init_api()
        
        # Mock response for demo
        # In production: user = User(fbid=user_id).get_ad_accounts()
        return [
            {
                "id": "act_123456789",
                "name": "AdClass Demo Account",
                "currency": "USD",
                "timezone": "America/New_York",
                "account_status": 1
            }
        ]
    
    async def get_campaigns(
        self,
        account_id: str,
        status_filter: Optional[List[str]] = None
    ) -> List[MetaCampaign]:
        """
        Fetch all campaigns for an ad account.
        
        Args:
            account_id: Meta ad account ID (with act_ prefix)
            status_filter: Optional list of statuses to filter by
        """
        self._init_api()
        
        # Check cache first
        cache_key = f"meta:campaigns:{account_id}"
        cached = await cache.get(cache_key)
        if cached:
            return [MetaCampaign(**c) for c in cached]
        
        # Mock response for demo
        campaigns = [
            MetaCampaign(
                campaign_id="123456789",
                name="Summer Sale - Conversions",
                objective="CONVERSIONS",
                status="ACTIVE",
                daily_budget=500.0,
                lifetime_budget=None,
                start_time=datetime(2026, 1, 1),
                stop_time=None,
                buying_type="AUCTION"
            ),
            MetaCampaign(
                campaign_id="123456790",
                name="Brand Awareness Q1",
                objective="BRAND_AWARENESS",
                status="ACTIVE",
                daily_budget=300.0,
                lifetime_budget=None,
                start_time=datetime(2026, 1, 15),
                stop_time=datetime(2026, 3, 31),
                buying_type="AUCTION"
            ),
            MetaCampaign(
                campaign_id="123456791",
                name="Retargeting - Website Visitors",
                objective="CONVERSIONS",
                status="ACTIVE",
                daily_budget=200.0,
                lifetime_budget=None,
                start_time=datetime(2026, 1, 1),
                stop_time=None,
                buying_type="AUCTION"
            )
        ]
        
        # Cache for 5 minutes
        await cache.set(cache_key, [c.__dict__ for c in campaigns], ttl=300)
        
        return campaigns
    
    async def get_adsets(
        self,
        campaign_id: str
    ) -> List[MetaAdSet]:
        """Fetch all ad sets for a campaign."""
        self._init_api()
        
        # Mock response
        return [
            MetaAdSet(
                adset_id="234567890",
                campaign_id=campaign_id,
                name="US - 25-44 - Interests",
                status="ACTIVE",
                targeting={
                    "age_min": 25,
                    "age_max": 44,
                    "geo_locations": {"countries": ["US"]},
                    "interests": [{"id": "123", "name": "Shopping"}]
                },
                daily_budget=250.0,
                optimization_goal="OFFSITE_CONVERSIONS",
                billing_event="IMPRESSIONS"
            )
        ]
    
    async def get_ads(
        self,
        adset_id: str
    ) -> List[MetaAd]:
        """Fetch all ads for an ad set."""
        self._init_api()
        
        # Mock response
        return [
            MetaAd(
                ad_id="345678901",
                adset_id=adset_id,
                name="Creative A - Image",
                status="ACTIVE",
                creative_id="456789012",
                tracking_specs=None
            )
        ]
    
    async def get_insights(
        self,
        entity_id: str,
        entity_type: str = "campaign",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        breakdown: Optional[List[str]] = None
    ) -> List[MetaInsights]:
        """
        Fetch performance insights for a campaign/adset/ad.
        
        Args:
            entity_id: Campaign, AdSet, or Ad ID
            entity_type: Type of entity (campaign, adset, ad)
            start_date: Start date for insights
            end_date: End date for insights
            breakdown: Optional breakdowns (age, gender, placement, etc.)
        """
        self._init_api()
        
        if not start_date:
            start_date = date.today() - timedelta(days=7)
        if not end_date:
            end_date = date.today()
        
        # Mock insights data
        insights = []
        current = start_date
        while current <= end_date:
            insights.append(MetaInsights(
                date=current,
                impressions=45000 + (hash(str(current)) % 10000),
                reach=38000 + (hash(str(current)) % 8000),
                clicks=1125 + (hash(str(current)) % 500),
                unique_clicks=980 + (hash(str(current)) % 400),
                spend=342.50 + (hash(str(current)) % 100),
                conversions=67 + (hash(str(current)) % 30),
                conversion_value=4020.0 + (hash(str(current)) % 1000),
                ctr=0.025,
                cpc=0.30,
                cpm=7.61
            ))
            current += timedelta(days=1)
        
        return insights
    
    async def get_creative_details(
        self,
        creative_id: str
    ) -> Dict[str, Any]:
        """Fetch creative details including image/video URLs."""
        self._init_api()
        
        # Mock response
        return {
            "id": creative_id,
            "name": "Summer Sale Creative",
            "title": "50% Off Everything!",
            "body": "Shop now and save big on our entire collection.",
            "call_to_action_type": "SHOP_NOW",
            "image_url": "https://example.com/creative.jpg",
            "thumbnail_url": "https://example.com/thumb.jpg",
            "object_story_spec": {
                "page_id": "123456789",
                "link_data": {
                    "link": "https://example.com/shop",
                    "message": "Shop now and save big!"
                }
            }
        }
    
    async def create_custom_audience(
        self,
        account_id: str,
        name: str,
        description: str,
        customer_file_source: str = "USER_PROVIDED_ONLY"
    ) -> Dict[str, str]:
        """Create a custom audience for targeting."""
        self._init_api()
        
        # Mock response
        audience_id = f"audience_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        return {
            "id": audience_id,
            "name": name,
            "description": description,
            "approximate_count": 0
        }
    
    async def add_users_to_audience(
        self,
        audience_id: str,
        users: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Add users to a custom audience.
        
        Users should be hashed (SHA-256) email or phone.
        """
        self._init_api()
        
        # Mock response
        return {
            "audience_id": audience_id,
            "num_received": len(users),
            "num_invalid_entries": 0,
            "session_id": "session_123"
        }
    
    async def sync_all(
        self,
        account_id: str,
        client_id: str
    ) -> Dict[str, int]:
        """
        Full sync of all campaign data for an account.
        
        Returns counts of synced entities.
        """
        campaigns = await self.get_campaigns(account_id)
        
        total_adsets = 0
        total_ads = 0
        
        for campaign in campaigns:
            adsets = await self.get_adsets(campaign.campaign_id)
            total_adsets += len(adsets)
            
            for adset in adsets:
                ads = await self.get_ads(adset.adset_id)
                total_ads += len(ads)
        
        return {
            "campaigns": len(campaigns),
            "adsets": total_adsets,
            "ads": total_ads
        }


# Singleton instance
_meta_service: Optional[MetaMarketingService] = None


def get_meta_service() -> MetaMarketingService:
    """Get or create Meta Marketing service instance."""
    global _meta_service
    if _meta_service is None:
        _meta_service = MetaMarketingService()
    return _meta_service
