"""
Webhook API Routes

Handle incoming webhooks from ad platforms and payment systems.
"""

from fastapi import APIRouter, Request, Header, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any
from datetime import datetime

from app.services.webhook_handler import (
    get_webhook_handler,
    WebhookSource,
    WebhookEventType
)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


@router.post("/meta")
async def meta_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None)
):
    """
    Handle Meta (Facebook/Instagram) webhooks.
    
    Processes:
    - Campaign status changes
    - Ad rejections
    - Conversion events
    """
    payload = await request.body()
    signature = x_hub_signature_256 or ""
    
    handler = get_webhook_handler()
    result = await handler.receive(
        source=WebhookSource.META,
        payload=payload,
        signature=signature
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=401, detail=result["message"])
    
    return result


@router.get("/meta")
async def meta_webhook_verify(
    hub_mode: str = None,
    hub_verify_token: str = None,
    hub_challenge: str = None
):
    """Meta webhook verification endpoint."""
    verify_token = "adclass_meta_verify_token"  # From config in production
    
    if hub_mode == "subscribe" and hub_verify_token == verify_token:
        return int(hub_challenge)
    
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/google")
async def google_webhook(
    request: Request,
    x_goog_signature: Optional[str] = Header(None)
):
    """
    Handle Google Ads webhooks.
    
    Processes:
    - Change events
    - Conversion imports
    """
    payload = await request.body()
    signature = x_goog_signature or ""
    
    handler = get_webhook_handler()
    result = await handler.receive(
        source=WebhookSource.GOOGLE,
        payload=payload,
        signature=signature
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=401, detail=result["message"])
    
    return result


@router.post("/tiktok")
async def tiktok_webhook(
    request: Request,
    x_tiktok_signature: Optional[str] = Header(None),
    x_tiktok_timestamp: Optional[str] = Header(None)
):
    """
    Handle TikTok Ads webhooks.
    
    Processes:
    - Campaign events
    - Video performance updates
    """
    payload = await request.body()
    signature = x_tiktok_signature or ""
    timestamp = x_tiktok_timestamp or ""
    
    handler = get_webhook_handler()
    result = await handler.receive(
        source=WebhookSource.TIKTOK,
        payload=payload,
        signature=signature,
        timestamp=timestamp
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=401, detail=result["message"])
    
    return result


@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None)
):
    """
    Handle Stripe payment webhooks.
    
    Processes:
    - Payment succeeded/failed
    - Subscription changes
    """
    payload = await request.body()
    signature = stripe_signature or ""
    
    handler = get_webhook_handler()
    result = await handler.receive(
        source=WebhookSource.STRIPE,
        payload=payload,
        signature=signature
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=401, detail=result["message"])
    
    return result


@router.post("/hubspot")
async def hubspot_webhook(
    request: Request,
    x_hubspot_signature_v3: Optional[str] = Header(None)
):
    """
    Handle HubSpot CRM webhooks.
    
    Processes:
    - Contact updates
    - Deal changes
    """
    payload = await request.body()
    signature = x_hubspot_signature_v3 or ""
    
    handler = get_webhook_handler()
    result = await handler.receive(
        source=WebhookSource.HUBSPOT,
        payload=payload,
        signature=signature
    )
    
    return result


@router.get("/status")
async def webhook_status():
    """Get webhook processing status."""
    handler = get_webhook_handler()
    
    return {
        "active": True,
        "processed_count": len(handler.processor.processed_events),
        "queue_size": handler.processor.event_queue.qsize(),
        "supported_sources": [s.value for s in WebhookSource]
    }
