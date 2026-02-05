"""
Test Suite for AdClass AI Platform

Comprehensive tests for API endpoints, ML models, and services.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from datetime import datetime, date, timedelta
import numpy as np


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        """Test basic health check."""
        response = await client.get("/api/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_readiness_check(self, client: AsyncClient):
        """Test Kubernetes readiness probe."""
        response = await client.get("/api/health/ready")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_liveness_check(self, client: AsyncClient):
        """Test Kubernetes liveness probe."""
        response = await client.get("/api/health/live")
        assert response.status_code == 200


class TestCreativeEndpoints:
    """Test Creative Performance Predictor endpoints."""
    
    @pytest.mark.asyncio
    async def test_predict_creative(self, client: AsyncClient):
        """Test creative prediction."""
        response = await client.post("/api/creative/predict", json={
            "headline": "50% Off Summer Sale!",
            "body_text": "Shop now and save big on everything.",
            "cta_type": "SHOP_NOW",
            "platform": "meta"
        })
        assert response.status_code == 200
        data = response.json()
        assert "predicted_ctr" in data
        assert "predicted_cvr" in data
        assert "recommendations" in data
    
    @pytest.mark.asyncio
    async def test_batch_analyze(self, client: AsyncClient):
        """Test batch creative analysis."""
        response = await client.post("/api/creative/batch", json={
            "creatives": [
                {"headline": "Test 1", "body_text": "Body 1"},
                {"headline": "Test 2", "body_text": "Body 2"}
            ]
        })
        assert response.status_code == 200


class TestROASEndpoints:
    """Test ROAS Optimizer endpoints."""
    
    @pytest.mark.asyncio
    async def test_optimize_budgets(self, client: AsyncClient):
        """Test budget optimization."""
        response = await client.post("/api/roas/optimize", json={
            "campaigns": [
                {
                    "campaign_id": "camp_1",
                    "current_budget": 500,
                    "spend": 350,
                    "conversions": 50,
                    "revenue": 2500
                }
            ],
            "constraints": {"max_budget": 1000}
        })
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_forecast(self, client: AsyncClient):
        """Test ROAS forecasting."""
        response = await client.post("/api/roas/forecast", json={
            "campaign_id": "camp_1",
            "budget_scenario": 1000,
            "days_ahead": 7
        })
        assert response.status_code == 200


class TestChurnEndpoints:
    """Test Churn Prediction endpoints."""
    
    @pytest.mark.asyncio
    async def test_predict_churn(self, client: AsyncClient):
        """Test churn prediction."""
        response = await client.post("/api/churn/predict", json={
            "client_id": "client_1"
        })
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "risk_level" in data
    
    @pytest.mark.asyncio
    async def test_health_scores(self, client: AsyncClient):
        """Test bulk health scores."""
        response = await client.get("/api/churn/scores")
        assert response.status_code == 200


class TestAttributionEndpoints:
    """Test Attribution endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_attribution(self, client: AsyncClient):
        """Test attribution calculation."""
        response = await client.get("/api/attribution/", params={
            "start_date": "2026-01-01",
            "end_date": "2026-01-31"
        })
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_channel_performance(self, client: AsyncClient):
        """Test channel performance."""
        response = await client.get("/api/attribution/channels")
        assert response.status_code == 200


class TestAudienceEndpoints:
    """Test Audience Intelligence endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_segments(self, client: AsyncClient):
        """Test segment retrieval."""
        response = await client.get("/api/audience/segments")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_create_lookalike(self, client: AsyncClient):
        """Test lookalike creation."""
        response = await client.post("/api/audience/lookalike", json={
            "seed_segment_id": "seg_1",
            "expansion": 2.0,
            "target_size": 1000
        })
        assert response.status_code == 200


class TestCampaignEndpoints:
    """Test Campaign management endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_campaigns(self, client: AsyncClient):
        """Test campaign listing."""
        response = await client.get("/api/campaigns/")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_sync_campaigns(self, client: AsyncClient):
        """Test campaign sync."""
        response = await client.post("/api/campaigns/sync", json={
            "platform": "meta",
            "account_id": "act_123"
        })
        assert response.status_code == 200


class TestClientEndpoints:
    """Test Client management endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_clients(self, client: AsyncClient):
        """Test client listing."""
        response = await client.get("/api/clients/")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_create_client(self, client: AsyncClient):
        """Test client creation."""
        response = await client.post("/api/clients/", json={
            "name": "Test Client",
            "email": "test@example.com",
            "industry": "retail"
        })
        # 200 or 201 depending on implementation
        assert response.status_code in [200, 201]
