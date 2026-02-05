"""
TikTok Ads API Integration Service

Handles data ingestion from TikTok Ads.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass

from app.config import settings
from app.core.cache import cache


@dataclass
class TikTokCampaign:
    """TikTok campaign data structure."""
    campaign_id: str
    campaign_name: str
    objective_type: str
    status: str
    budget: float
    budget_mode: str  # BUDGET_MODE_DAY or BUDGET_MODE_TOTAL


@dataclass
class TikTokAdGroup:
    """TikTok ad group data structure."""
    adgroup_id: str
    campaign_id: str
    adgroup_name: str
    status: str
    budget: float
    optimization_goal: str
    placement_type: str
    audience: Dict[str, Any]


@dataclass
class TikTokAd:
    """TikTok ad data structure."""
    ad_id: str
    adgroup_id: str
    ad_name: str
    status: str
    ad_format: str
    video_id: Optional[str]
    image_ids: List[str]
    landing_page_url: str


@dataclass
class TikTokMetrics:
    """TikTok ads metrics."""
    date: date
    impressions: int
    clicks: int
    reach: int
    spend: float
    conversions: int
    conversion_rate: float
    ctr: float
    cpc: float
    cpm: float
    video_views: int
    video_watched_2s: int
    video_watched_6s: int
    likes: int
    comments: int
    shares: int


class TikTokAdsService:
    """
    Service for TikTok Ads API integration.
    
    Handles:
    - Campaign, Ad Group, Ad data fetching
    - Performance metrics retrieval
    - Custom audience management
    - Creative assets
    """
    
    BASE_URL = "https://business-api.tiktok.com/open_api/v1.3"
    
    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        access_token: Optional[str] = None
    ):
        self.app_id = app_id or settings.TIKTOK_APP_ID
        self.app_secret = app_secret or settings.TIKTOK_APP_SECRET
        self.access_token = access_token or settings.TIKTOK_ACCESS_TOKEN
    
    async def get_advertiser_accounts(self) -> List[Dict[str, str]]:
        """Get all advertiser accounts."""
        # Mock response
        return [
            {
                "advertiser_id": "7123456789012345678",
                "advertiser_name": "AdClass Demo - TikTok",
                "currency": "USD",
                "timezone": "America/New_York"
            }
        ]
    
    async def get_campaigns(
        self,
        advertiser_id: str,
        status_filter: Optional[List[str]] = None
    ) -> List[TikTokCampaign]:
        """Fetch all campaigns for an advertiser."""
        cache_key = f"tiktok:campaigns:{advertiser_id}"
        cached = await cache.get(cache_key)
        if cached:
            return [TikTokCampaign(**c) for c in cached]
        
        # Mock response
        campaigns = [
            TikTokCampaign(
                campaign_id="1800000000000001",
                campaign_name="Viral Video - Gen Z",
                objective_type="CONVERSIONS",
                status="ENABLE",
                budget=500.0,
                budget_mode="BUDGET_MODE_DAY"
            ),
            TikTokCampaign(
                campaign_id="1800000000000002",
                campaign_name="Brand Challenge",
                objective_type="REACH",
                status="ENABLE",
                budget=1000.0,
                budget_mode="BUDGET_MODE_DAY"
            ),
            TikTokCampaign(
                campaign_id="1800000000000003",
                campaign_name="Product Launch - Spark Ads",
                objective_type="TRAFFIC",
                status="ENABLE",
                budget=300.0,
                budget_mode="BUDGET_MODE_DAY"
            )
        ]
        
        await cache.set(cache_key, [c.__dict__ for c in campaigns], ttl=300)
        return campaigns
    
    async def get_adgroups(
        self,
        advertiser_id: str,
        campaign_id: str
    ) -> List[TikTokAdGroup]:
        """Fetch all ad groups for a campaign."""
        # Mock response
        return [
            TikTokAdGroup(
                adgroup_id="1800100000000001",
                campaign_id=campaign_id,
                adgroup_name="US - 18-24 - Fashion Interest",
                status="ENABLE",
                budget=250.0,
                optimization_goal="CONVERT",
                placement_type="PLACEMENT_TYPE_AUTOMATIC",
                audience={
                    "age_groups": ["AGE_18_24"],
                    "genders": ["GENDER_FEMALE"],
                    "interests": [{"interest_id": "123", "interest_name": "Fashion"}]
                }
            )
        ]
    
    async def get_ads(
        self,
        advertiser_id: str,
        adgroup_id: str
    ) -> List[TikTokAd]:
        """Fetch all ads for an ad group."""
        # Mock response
        return [
            TikTokAd(
                ad_id="1800200000000001",
                adgroup_id=adgroup_id,
                ad_name="Trend Dance Creative",
                status="ENABLE",
                ad_format="SINGLE_VIDEO",
                video_id="v12345678901234567890",
                image_ids=[],
                landing_page_url="https://example.com/tiktok-landing"
            )
        ]
    
    async def get_metrics(
        self,
        advertiser_id: str,
        campaign_ids: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        metrics: Optional[List[str]] = None
    ) -> List[TikTokMetrics]:
        """
        Fetch performance metrics for campaigns.
        
        Args:
            advertiser_id: TikTok advertiser ID
            campaign_ids: List of campaign IDs
            start_date: Start date
            end_date: End date
            metrics: Specific metrics to fetch
        """
        if not start_date:
            start_date = date.today() - timedelta(days=7)
        if not end_date:
            end_date = date.today()
        
        # Mock metrics
        result = []
        current = start_date
        while current <= end_date:
            seed = hash(f"{campaign_ids[0] if campaign_ids else ''}{current}")
            result.append(TikTokMetrics(
                date=current,
                impressions=85000 + (seed % 20000),
                clicks=2550 + (seed % 800),
                reach=72000 + (seed % 15000),
                spend=425.00 + (seed % 100),
                conversions=95 + (seed % 40),
                conversion_rate=0.037,
                ctr=0.030,
                cpc=0.17,
                cpm=5.00,
                video_views=68000 + (seed % 12000),
                video_watched_2s=54400 + (seed % 10000),
                video_watched_6s=34000 + (seed % 8000),
                likes=1250 + (seed % 500),
                comments=180 + (seed % 80),
                shares=340 + (seed % 150)
            ))
            current += timedelta(days=1)
        
        return result
    
    async def get_video_insights(
        self,
        advertiser_id: str,
        video_id: str
    ) -> Dict[str, Any]:
        """Get detailed insights for a video creative."""
        # Mock response
        return {
            "video_id": video_id,
            "duration": 15,
            "format": "9:16",
            "avg_watch_time": 8.5,
            "completion_rate": 0.42,
            "hook_rate": 0.78,  # First 2 seconds retention
            "engagement_rate": 0.065,
            "share_rate": 0.004,
            "trending_score": 72
        }
    
    async def create_custom_audience(
        self,
        advertiser_id: str,
        name: str,
        audience_type: str = "CUSTOMER_FILE"
    ) -> str:
        """Create a custom audience."""
        # Mock response
        return f"{hash(name) % 1000000000}"
    
    async def get_trending_hashtags(
        self,
        advertiser_id: str,
        country: str = "US"
    ) -> List[Dict[str, Any]]:
        """Get trending hashtags for content ideas."""
        # Mock response
        return [
            {"hashtag": "#fyp", "views": 500000000, "trend": "stable"},
            {"hashtag": "#fashion", "views": 85000000, "trend": "rising"},
            {"hashtag": "#shopping", "views": 42000000, "trend": "rising"},
            {"hashtag": "#sale", "views": 28000000, "trend": "rising"}
        ]
    
    async def sync_all(
        self,
        advertiser_id: str,
        client_id: str
    ) -> Dict[str, int]:
        """Full sync of all campaign data."""
        campaigns = await self.get_campaigns(advertiser_id)
        
        total_adgroups = 0
        total_ads = 0
        
        for campaign in campaigns:
            adgroups = await self.get_adgroups(advertiser_id, campaign.campaign_id)
            total_adgroups += len(adgroups)
            
            for ag in adgroups:
                ads = await self.get_ads(advertiser_id, ag.adgroup_id)
                total_ads += len(ads)
        
        return {
            "campaigns": len(campaigns),
            "adgroups": total_adgroups,
            "ads": total_ads
        }


# Singleton
_tiktok_service: Optional[TikTokAdsService] = None


def get_tiktok_ads_service() -> TikTokAdsService:
    """Get or create TikTok Ads service instance."""
    global _tiktok_service
    if _tiktok_service is None:
        _tiktok_service = TikTokAdsService()
    return _tiktok_service
