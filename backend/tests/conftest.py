"""
PyTest Configuration and Fixtures

Shared fixtures for the test suite.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient, ASGITransport
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def app():
    """Create application instance."""
    from app.main import app
    yield app


@pytest.fixture
async def client(app) -> AsyncGenerator:
    """Create async HTTP client for testing."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def sample_campaign_data():
    """Sample campaign data for tests."""
    return {
        "campaign_id": "test_camp_1",
        "name": "Test Campaign",
        "platform": "meta",
        "budget": 500.0,
        "spend": 350.0,
        "impressions": 45000,
        "clicks": 1125,
        "conversions": 67,
        "revenue": 4020.0
    }


@pytest.fixture
def sample_client_data():
    """Sample client data for tests."""
    return {
        "client_id": "test_client_1",
        "name": "Test Client Inc",
        "email": "test@example.com",
        "industry": "retail",
        "monthly_spend": 25000,
        "target_roas": 4.0
    }


@pytest.fixture
def sample_creative_data():
    """Sample creative data for tests."""
    return {
        "headline": "50% Off Everything - Today Only!",
        "body_text": "Shop now and save big on our entire collection. Limited time offer.",
        "cta_type": "SHOP_NOW",
        "platform": "meta"
    }


@pytest.fixture
def sample_journey_data():
    """Sample customer journey data for tests."""
    from datetime import datetime, timedelta
    
    now = datetime.now()
    return {
        "customer_id": "test_cust_1",
        "touchpoints": [
            {"channel": "meta", "timestamp": now - timedelta(days=7), "type": "impression"},
            {"channel": "google", "timestamp": now - timedelta(days=5), "type": "click"},
            {"channel": "email", "timestamp": now - timedelta(days=2), "type": "click"},
            {"channel": "meta", "timestamp": now - timedelta(days=1), "type": "conversion"}
        ],
        "conversion_value": 150.0,
        "converted": True
    }


@pytest.fixture
def sample_audience_members():
    """Sample audience member data for tests."""
    import numpy as np
    
    return [
        {
            "customer_id": f"cust_{i}",
            "features": {
                "ltv": float(50 + i * 10),
                "purchase_frequency": float(1 + (i % 12)),
                "avg_order_value": float(25 + i * 5),
                "days_since_purchase": int(1 + (i % 90))
            }
        }
        for i in range(50)
    ]


# Markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests that require ML models"
    )
