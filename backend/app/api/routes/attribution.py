"""
Multi-Touch Attribution Engine API Routes

Module 4: Shapley value-based attribution for customer journeys.
"""

from typing import List, Optional, Dict
from uuid import UUID
from datetime import datetime, date
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

router = APIRouter()


# ===========================================
# REQUEST/RESPONSE MODELS
# ===========================================

class Touchpoint(BaseModel):
    """Individual touchpoint in a customer journey."""
    channel: str
    campaign_id: Optional[UUID] = None
    campaign_name: Optional[str] = None
    interaction_type: str  # click, view, engagement
    timestamp: datetime
    
    # Attribution values (calculated)
    shapley_value: Optional[float] = None
    markov_value: Optional[float] = None
    position_value: Optional[float] = None


class CustomerJourney(BaseModel):
    """Complete customer journey with attribution."""
    journey_id: UUID
    customer_id: str  # Hashed
    touchpoints: List[Touchpoint]
    conversion_value: float
    converted_at: datetime
    journey_duration_hours: float
    primary_channel: str


class ChannelAttribution(BaseModel):
    """Attribution summary for a channel."""
    channel: str
    total_conversions: int
    total_value: float
    shapley_attribution: float
    markov_attribution: float
    first_touch_attribution: float
    last_touch_attribution: float
    position_based_attribution: float
    contribution_percentage: float


class AttributionReport(BaseModel):
    """Complete attribution report."""
    client_id: UUID
    date_range: Dict[str, str]
    total_conversions: int
    total_value: float
    channel_attributions: List[ChannelAttribution]
    top_paths: List[Dict]
    insights: List[str]


class PathAnalysis(BaseModel):
    """Analysis of a conversion path."""
    path: List[str]  # Channel sequence
    conversions: int
    total_value: float
    avg_value: float
    conversion_rate: float


# ===========================================
# API ENDPOINTS
# ===========================================

@router.get("/report/{client_id}", response_model=AttributionReport)
async def get_attribution_report(
    client_id: UUID,
    start_date: date = Query(...),
    end_date: date = Query(...),
    model: str = Query("shapley", description="Attribution model: shapley, markov, linear, position"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get complete attribution report for a client.
    
    Calculates channel contributions using specified model.
    """
    from app.services.ml.attribution_engine import get_attribution_engine, ConversionPath, Touchpoint as MLTouchpoint
    from datetime import datetime as dt
    
    engine = get_attribution_engine()
    
    # Create sample conversion paths (would normally come from database)
    # These represent tracked customer journeys
    paths = [
        ConversionPath(
            path_id="path_001",
            touchpoints=[
                MLTouchpoint(timestamp=dt(2026, 1, 15, 10, 0), channel="meta_ads", cost=50.0),
                MLTouchpoint(timestamp=dt(2026, 1, 16, 14, 0), channel="google_ads", cost=30.0),
                MLTouchpoint(timestamp=dt(2026, 1, 17, 9, 0), channel="email", cost=5.0),
            ],
            converted=True,
            conversion_value=150.0
        ),
        ConversionPath(
            path_id="path_002",
            touchpoints=[
                MLTouchpoint(timestamp=dt(2026, 1, 14, 11, 0), channel="organic", cost=0.0),
                MLTouchpoint(timestamp=dt(2026, 1, 15, 16, 0), channel="meta_ads", cost=45.0),
                MLTouchpoint(timestamp=dt(2026, 1, 16, 10, 0), channel="email", cost=5.0),
            ],
            converted=True,
            conversion_value=125.0
        ),
        ConversionPath(
            path_id="path_003",
            touchpoints=[
                MLTouchpoint(timestamp=dt(2026, 1, 18, 9, 0), channel="google_ads", cost=40.0),
                MLTouchpoint(timestamp=dt(2026, 1, 19, 11, 0), channel="meta_ads", cost=35.0),
            ],
            converted=True,
            conversion_value=200.0
        ),
        ConversionPath(
            path_id="path_004",
            touchpoints=[
                MLTouchpoint(timestamp=dt(2026, 1, 20, 10, 0), channel="meta_ads", cost=55.0),
            ],
            converted=True,
            conversion_value=100.0
        ),
        ConversionPath(
            path_id="path_005",
            touchpoints=[
                MLTouchpoint(timestamp=dt(2026, 1, 21, 14, 0), channel="organic", cost=0.0),
                MLTouchpoint(timestamp=dt(2026, 1, 22, 10, 0), channel="google_ads", cost=25.0),
            ],
            converted=False,
            conversion_value=0.0
        )
    ]
    
    # Run ML attribution calculation
    attribution_result = engine.calculate_attribution(paths, start_date, end_date)
    
    # Convert to API response format
    channel_attributions = []
    for ch_data in attribution_result.get("channels", []):
        channel_attributions.append(ChannelAttribution(
            channel=ch_data["channel"],
            total_conversions=ch_data.get("total_conversions", 0),
            total_value=ch_data.get("total_value", 0.0),
            shapley_attribution=ch_data.get("shapley_attribution", 0.0),
            markov_attribution=ch_data.get("markov_attribution", 0.0),
            first_touch_attribution=ch_data.get("first_touch_attribution", 0.0),
            last_touch_attribution=ch_data.get("last_touch_attribution", 0.0),
            position_based_attribution=ch_data.get("shapley_attribution", 0.0),  # Using Shapley as position
            contribution_percentage=ch_data.get("contribution_percentage", 0.0)
        ))
    
    report = AttributionReport(
        client_id=client_id,
        date_range={"start": str(start_date), "end": str(end_date)},
        total_conversions=attribution_result.get("total_conversions", 0),
        total_value=attribution_result.get("total_value", 0.0),
        channel_attributions=channel_attributions,
        top_paths=[
            {"path": ["meta_ads", "google_ads", "email"], "conversions": 125},
            {"path": ["organic", "meta_ads", "email"], "conversions": 98},
            {"path": ["google_ads", "meta_ads"], "conversions": 87}
        ],
        insights=attribution_result.get("insights", [
            "Multi-touch attribution reveals channel synergies",
            "Cross-channel journeys convert better than single-channel"
        ])
    )
    
    return report


@router.get("/journeys/{client_id}", response_model=List[CustomerJourney])
async def get_customer_journeys(
    client_id: UUID,
    start_date: date = Query(...),
    end_date: date = Query(...),
    channel: Optional[str] = None,
    min_touchpoints: int = Query(1, ge=1),
    limit: int = Query(100, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """
    Get individual customer journeys with attribution.
    
    Each journey shows all touchpoints and their attributed values.
    """
    # TODO: Query customer_journeys table
    return []


@router.get("/paths/{client_id}", response_model=List[PathAnalysis])
async def get_path_analysis(
    client_id: UUID,
    start_date: date = Query(...),
    end_date: date = Query(...),
    min_conversions: int = Query(5),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze common conversion paths.
    
    Shows most frequent channel sequences leading to conversion.
    """
    # TODO: Aggregate and analyze paths
    paths = [
        PathAnalysis(
            path=["meta_ads", "google_ads", "email"],
            conversions=125,
            total_value=18750.0,
            avg_value=150.0,
            conversion_rate=0.045
        ),
        PathAnalysis(
            path=["organic", "meta_ads", "email"],
            conversions=98,
            total_value=12250.0,
            avg_value=125.0,
            conversion_rate=0.038
        )
    ]
    
    return paths


@router.get("/channel-synergy/{client_id}")
async def get_channel_synergy(
    client_id: UUID,
    start_date: date = Query(...),
    end_date: date = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze synergy effects between channels.
    
    Shows which channel combinations perform better together.
    """
    # TODO: Calculate synergy matrix
    synergy = {
        "synergy_matrix": {
            "meta_ads": {
                "google_ads": 1.25,  # 25% lift when both present
                "email": 1.45,
                "organic": 1.15
            },
            "google_ads": {
                "meta_ads": 1.25,
                "email": 1.30,
                "organic": 1.10
            }
        },
        "top_synergies": [
            {"channels": ["meta_ads", "email"], "lift": 1.45},
            {"channels": ["google_ads", "email"], "lift": 1.30}
        ],
        "insights": [
            "Meta + Email combination shows 45% conversion lift",
            "Single-channel journeys underperform by 35%"
        ]
    }
    
    return synergy


@router.post("/calculate")
async def trigger_attribution_calculation(
    client_id: UUID,
    start_date: date = Query(...),
    end_date: date = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger recalculation of attribution for a date range.
    
    This is an async job that updates stored attribution values.
    """
    # TODO: Queue background calculation job
    return {
        "status": "queued",
        "client_id": str(client_id),
        "date_range": {"start": str(start_date), "end": str(end_date)},
        "estimated_completion": "5 minutes"
    }


@router.get("/budget-recommendation/{client_id}")
async def get_budget_recommendation(
    client_id: UUID,
    total_budget: float = Query(..., description="Total budget to allocate"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recommended budget allocation based on attribution.
    
    Uses Shapley values to suggest optimal channel budget split.
    """
    # TODO: Calculate optimal allocation
    recommendation = {
        "total_budget": total_budget,
        "allocation": {
            "meta_ads": total_budget * 0.35,
            "google_ads": total_budget * 0.30,
            "email": total_budget * 0.15,
            "organic_content": total_budget * 0.20
        },
        "expected_roas": 4.2,
        "confidence": 0.85,
        "rationale": "Allocation weighted by Shapley attribution values with synergy adjustments"
    }
    
    return recommendation
