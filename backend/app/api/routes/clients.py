"""
Clients API Routes

CRUD operations for client management.
"""

from typing import List, Optional
from uuid import UUID
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

router = APIRouter()


class ClientCreate(BaseModel):
    """Create client request."""
    name: str = Field(..., min_length=1, max_length=255)
    industry: Optional[str] = None
    website: Optional[str] = None
    monthly_budget: Optional[float] = None
    contract_start_date: Optional[date] = None
    contract_end_date: Optional[date] = None
    primary_contact_email: Optional[EmailStr] = None
    slack_channel_id: Optional[str] = None
    notes: Optional[str] = None


class ClientUpdate(BaseModel):
    """Update client request."""
    name: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[str] = None
    monthly_budget: Optional[float] = None
    contract_start_date: Optional[date] = None
    contract_end_date: Optional[date] = None
    primary_contact_email: Optional[EmailStr] = None
    slack_channel_id: Optional[str] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None


class ClientResponse(BaseModel):
    """Client response model."""
    id: UUID
    name: str
    industry: Optional[str] = None
    website: Optional[str] = None
    monthly_budget: Optional[float] = None
    contract_start_date: Optional[date] = None
    contract_end_date: Optional[date] = None
    health_score: int = 50
    churn_probability: Optional[float] = None
    is_active: bool = True


class ClientListResponse(BaseModel):
    """Paginated client list."""
    items: List[ClientResponse]
    total: int
    page: int
    page_size: int


@router.get("/", response_model=ClientListResponse)
async def list_clients(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    is_active: Optional[bool] = None,
    industry: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all clients with pagination and filtering."""
    # TODO: Query database
    return ClientListResponse(items=[], total=0, page=page, page_size=page_size)


@router.post("/", response_model=ClientResponse)
async def create_client(
    client: ClientCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new client."""
    # TODO: Insert into database
    return ClientResponse(
        id=UUID("12345678-1234-1234-1234-123456789001"),
        name=client.name,
        industry=client.industry,
        website=client.website,
        monthly_budget=client.monthly_budget,
        contract_start_date=client.contract_start_date,
        contract_end_date=client.contract_end_date
    )


@router.get("/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get client by ID."""
    # TODO: Query database
    raise HTTPException(status_code=404, detail="Client not found")


@router.put("/{client_id}", response_model=ClientResponse)
async def update_client(
    client_id: UUID,
    client: ClientUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a client."""
    # TODO: Update in database
    raise HTTPException(status_code=404, detail="Client not found")


@router.delete("/{client_id}")
async def delete_client(
    client_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete a client (soft delete)."""
    # TODO: Soft delete in database
    return {"status": "deleted", "client_id": str(client_id)}
