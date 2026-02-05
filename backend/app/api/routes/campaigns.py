"""
Campaigns API Routes

CRUD operations for campaign management and sync.
"""

from typing import List, Optional
from uuid import UUID
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

router = APIRouter()


class CampaignResponse(BaseModel):
    """Campaign response model."""
    id: UUID
    ad_account_id: UUID
    platform: str
    platform_campaign_id: str
    name: str
    objective: Optional[str] = None
    status: str = "active"
    daily_budget: Optional[float] = None
    lifetime_budget: Optional[float] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    predicted_roas: Optional[float] = None
    risk_score: Optional[float] = None


class CampaignListResponse(BaseModel):
    """Paginated campaign list."""
    items: List[CampaignResponse]
    total: int
    page: int
    page_size: int


class SyncRequest(BaseModel):
    """Request to sync campaigns from ad platform."""
    ad_account_id: UUID
    full_sync: bool = False


@router.get("/", response_model=CampaignListResponse)
async def list_campaigns(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    client_id: Optional[UUID] = None,
    platform: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all campaigns with pagination and filtering."""
    # TODO: Query database
    return CampaignListResponse(items=[], total=0, page=page, page_size=page_size)


@router.get("/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get campaign by ID."""
    # TODO: Query database
    raise HTTPException(status_code=404, detail="Campaign not found")


@router.post("/sync")
async def sync_campaigns(
    request: SyncRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Sync campaigns from ad platform.
    
    Fetches latest campaign data from Meta/Google/TikTok.
    """
    # TODO: Queue sync job
    return {
        "status": "queued",
        "ad_account_id": str(request.ad_account_id),
        "full_sync": request.full_sync
    }


@router.get("/{campaign_id}/metrics")
async def get_campaign_metrics(
    campaign_id: UUID,
    start_date: date = Query(...),
    end_date: date = Query(...),
    granularity: str = Query("daily", description="hourly, daily, weekly"),
    db: AsyncSession = Depends(get_db)
):
    """Get performance metrics for a campaign."""
    # TODO: Query performance_metrics table
    return {
        "campaign_id": str(campaign_id),
        "date_range": {"start": str(start_date), "end": str(end_date)},
        "metrics": [],
        "summary": {
            "impressions": 0,
            "clicks": 0,
            "conversions": 0,
            "spend": 0,
            "revenue": 0,
            "roas": 0
        }
    }


@router.get("/{campaign_id}/creatives")
async def get_campaign_creatives(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get all creatives for a campaign."""
    # TODO: Query ad_creatives table
    return {"campaign_id": str(campaign_id), "creatives": []}
