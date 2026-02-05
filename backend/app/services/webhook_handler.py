"""
Webhook Handler Service

Real-time webhook processing for ad platform events.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import json
import asyncio
from collections import defaultdict


class WebhookSource(str, Enum):
    META = "meta"
    GOOGLE = "google"
    TIKTOK = "tiktok"
    STRIPE = "stripe"
    HUBSPOT = "hubspot"
    SALESFORCE = "salesforce"


class WebhookEventType(str, Enum):
    # Ad Platform Events
    CAMPAIGN_CREATED = "campaign.created"
    CAMPAIGN_UPDATED = "campaign.updated"
    CAMPAIGN_DELETED = "campaign.deleted"
    CAMPAIGN_PAUSED = "campaign.paused"
    AD_CREATED = "ad.created"
    AD_UPDATED = "ad.updated"
    AD_REJECTED = "ad.rejected"
    BUDGET_DEPLETED = "budget.depleted"
    
    # Conversion Events
    CONVERSION = "conversion"
    LEAD = "lead"
    PURCHASE = "purchase"
    
    # CRM Events
    CONTACT_CREATED = "contact.created"
    CONTACT_UPDATED = "contact.updated"
    DEAL_CREATED = "deal.created"
    DEAL_UPDATED = "deal.updated"
    
    # Billing Events
    PAYMENT_SUCCEEDED = "payment.succeeded"
    PAYMENT_FAILED = "payment.failed"
    SUBSCRIPTION_UPDATED = "subscription.updated"


@dataclass
class WebhookEvent:
    """A received webhook event."""
    event_id: str
    source: WebhookSource
    event_type: WebhookEventType
    payload: Dict[str, Any]
    received_at: datetime
    processed: bool = False
    retry_count: int = 0


class WebhookVerifier:
    """
    Webhook signature verification for different platforms.
    """
    
    def __init__(self, secrets: Dict[WebhookSource, str]):
        self.secrets = secrets
    
    def verify_meta(self, payload: bytes, signature: str) -> bool:
        """Verify Meta webhook signature."""
        secret = self.secrets.get(WebhookSource.META, "")
        expected = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature)
    
    def verify_google(self, payload: bytes, signature: str) -> bool:
        """Verify Google webhook signature."""
        # Google uses different verification depending on the service
        secret = self.secrets.get(WebhookSource.GOOGLE, "")
        expected = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
    
    def verify_tiktok(self, payload: bytes, signature: str, timestamp: str) -> bool:
        """Verify TikTok webhook signature."""
        secret = self.secrets.get(WebhookSource.TIKTOK, "")
        message = f"{timestamp}.{payload.decode()}"
        expected = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
    
    def verify_stripe(self, payload: bytes, signature: str) -> bool:
        """Verify Stripe webhook signature."""
        secret = self.secrets.get(WebhookSource.STRIPE, "")
        # Parse Stripe signature header
        parts = dict(x.split("=") for x in signature.split(","))
        timestamp = parts.get("t", "")
        v1_signature = parts.get("v1", "")
        
        message = f"{timestamp}.{payload.decode()}"
        expected = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, v1_signature)
    
    def verify(self, source: WebhookSource, payload: bytes, signature: str, **kwargs) -> bool:
        """Verify webhook based on source."""
        verifiers = {
            WebhookSource.META: self.verify_meta,
            WebhookSource.GOOGLE: self.verify_google,
            WebhookSource.STRIPE: self.verify_stripe,
        }
        
        verifier = verifiers.get(source)
        if verifier:
            if source == WebhookSource.TIKTOK:
                return self.verify_tiktok(payload, signature, kwargs.get("timestamp", ""))
            return verifier(payload, signature)
        
        return True  # Unknown source, skip verification (not recommended in production)


class WebhookProcessor:
    """
    Process and route webhook events.
    """
    
    def __init__(self):
        self.handlers: Dict[WebhookEventType, List[Callable]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processed_events: Dict[str, WebhookEvent] = {}
    
    def register_handler(self, event_type: WebhookEventType, handler: Callable):
        """Register a handler for an event type."""
        self.handlers[event_type].append(handler)
    
    async def enqueue(self, event: WebhookEvent):
        """Add event to processing queue."""
        await self.event_queue.put(event)
    
    async def process_event(self, event: WebhookEvent) -> bool:
        """Process a single webhook event."""
        handlers = self.handlers.get(event.event_type, [])
        
        if not handlers:
            # No handlers registered, mark as processed
            event.processed = True
            return True
        
        try:
            for handler in handlers:
                await handler(event)
            event.processed = True
            self.processed_events[event.event_id] = event
            return True
        except Exception as e:
            event.retry_count += 1
            if event.retry_count < 3:
                # Re-queue for retry
                await self.enqueue(event)
            return False
    
    async def process_queue(self):
        """Process all events in queue."""
        while not self.event_queue.empty():
            event = await self.event_queue.get()
            await self.process_event(event)
            self.event_queue.task_done()


class WebhookHandler:
    """
    Main webhook handling service.
    """
    
    def __init__(self, secrets: Optional[Dict[WebhookSource, str]] = None):
        self.verifier = WebhookVerifier(secrets or {})
        self.processor = WebhookProcessor()
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Set up default event handlers."""
        # Campaign events
        self.processor.register_handler(
            WebhookEventType.CAMPAIGN_UPDATED,
            self._handle_campaign_update
        )
        self.processor.register_handler(
            WebhookEventType.CAMPAIGN_PAUSED,
            self._handle_campaign_pause
        )
        self.processor.register_handler(
            WebhookEventType.BUDGET_DEPLETED,
            self._handle_budget_depleted
        )
        
        # Conversion events
        self.processor.register_handler(
            WebhookEventType.CONVERSION,
            self._handle_conversion
        )
        self.processor.register_handler(
            WebhookEventType.PURCHASE,
            self._handle_purchase
        )
        
        # Payment events
        self.processor.register_handler(
            WebhookEventType.PAYMENT_FAILED,
            self._handle_payment_failed
        )
    
    async def receive(
        self,
        source: WebhookSource,
        payload: bytes,
        signature: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Receive and validate a webhook.
        
        Returns processing result.
        """
        # Verify signature
        if not self.verifier.verify(source, payload, signature, **kwargs):
            return {"status": "error", "message": "Invalid signature"}
        
        # Parse payload
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON payload"}
        
        # Extract event type
        event_type = self._extract_event_type(source, data)
        if not event_type:
            return {"status": "ignored", "message": "Unknown event type"}
        
        # Create event
        event = WebhookEvent(
            event_id=self._generate_event_id(source, data),
            source=source,
            event_type=event_type,
            payload=data,
            received_at=datetime.now()
        )
        
        # Check for duplicate
        if event.event_id in self.processor.processed_events:
            return {"status": "duplicate", "event_id": event.event_id}
        
        # Queue for processing
        await self.processor.enqueue(event)
        
        # Process immediately (or could be done by background worker)
        await self.processor.process_event(event)
        
        return {
            "status": "processed" if event.processed else "queued",
            "event_id": event.event_id
        }
    
    def _extract_event_type(self, source: WebhookSource, data: Dict) -> Optional[WebhookEventType]:
        """Extract event type from webhook payload."""
        if source == WebhookSource.META:
            object_type = data.get("object", "")
            entry = data.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            field = changes.get("field", "")
            
            if object_type == "campaign" and field == "status":
                status = changes.get("value", {}).get("status", "")
                if status == "PAUSED":
                    return WebhookEventType.CAMPAIGN_PAUSED
                return WebhookEventType.CAMPAIGN_UPDATED
            
        elif source == WebhookSource.GOOGLE:
            event_type = data.get("eventType", "")
            if "campaign" in event_type.lower():
                return WebhookEventType.CAMPAIGN_UPDATED
        
        elif source == WebhookSource.STRIPE:
            event_type = data.get("type", "")
            mapping = {
                "payment_intent.succeeded": WebhookEventType.PAYMENT_SUCCEEDED,
                "payment_intent.payment_failed": WebhookEventType.PAYMENT_FAILED,
                "customer.subscription.updated": WebhookEventType.SUBSCRIPTION_UPDATED,
            }
            return mapping.get(event_type)
        
        elif source == WebhookSource.HUBSPOT:
            event_type = data.get("subscriptionType", "")
            if "contact" in event_type:
                return WebhookEventType.CONTACT_UPDATED
            if "deal" in event_type:
                return WebhookEventType.DEAL_UPDATED
        
        return None
    
    def _generate_event_id(self, source: WebhookSource, data: Dict) -> str:
        """Generate unique event ID."""
        # Use platform-specific ID if available
        if source == WebhookSource.META:
            return data.get("entry", [{}])[0].get("id", "")
        elif source == WebhookSource.STRIPE:
            return data.get("id", "")
        
        # Generate hash-based ID
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    # Event handlers
    async def _handle_campaign_update(self, event: WebhookEvent):
        """Handle campaign update event."""
        # In production: Update campaign in database
        print(f"Campaign updated: {event.payload}")
    
    async def _handle_campaign_pause(self, event: WebhookEvent):
        """Handle campaign pause event."""
        # In production: Update status, notify team
        print(f"Campaign paused: {event.payload}")
    
    async def _handle_budget_depleted(self, event: WebhookEvent):
        """Handle budget depleted event."""
        # In production: Trigger budget optimization, notify client
        print(f"Budget depleted: {event.payload}")
    
    async def _handle_conversion(self, event: WebhookEvent):
        """Handle conversion event."""
        # In production: Record touchpoint for attribution
        print(f"Conversion: {event.payload}")
    
    async def _handle_purchase(self, event: WebhookEvent):
        """Handle purchase event."""
        # In production: Update revenue metrics, attribution
        print(f"Purchase: {event.payload}")
    
    async def _handle_payment_failed(self, event: WebhookEvent):
        """Handle payment failure."""
        # In production: Alert account manager, update client health
        print(f"Payment failed: {event.payload}")


# Singleton
_webhook_handler: Optional[WebhookHandler] = None


def get_webhook_handler() -> WebhookHandler:
    """Get or create webhook handler instance."""
    global _webhook_handler
    if _webhook_handler is None:
        _webhook_handler = WebhookHandler()
    return _webhook_handler
