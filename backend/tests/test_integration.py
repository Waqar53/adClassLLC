"""
Integration Tests

End-to-end tests for the complete platform workflow.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from datetime import datetime, timedelta
import json


class TestFullWorkflow:
    """Test complete user workflows."""
    
    @pytest.mark.asyncio
    async def test_client_onboarding_flow(self, client: AsyncClient):
        """Test full client onboarding workflow."""
        # 1. Create client
        response = await client.post("/api/clients/", json={
            "name": "Integration Test Client",
            "email": "test@integration.com",
            "industry": "e-commerce"
        })
        assert response.status_code in [200, 201]
        
        # 2. Check health (new client should be healthy)
        response = await client.get("/api/churn/scores")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_campaign_analysis_flow(self, client: AsyncClient):
        """Test campaign creation and analysis."""
        # 1. Create campaign creative
        response = await client.post("/api/creative/predict", json={
            "headline": "Summer Sale - 50% Off Everything!",
            "body_text": "Shop now and save big on our entire collection.",
            "cta_type": "SHOP_NOW",
            "platform": "meta"
        })
        assert response.status_code == 200
        prediction = response.json()
        assert "predicted_ctr" in prediction
        
        # 2. Optimize budget
        response = await client.post("/api/roas/optimize", json={
            "campaigns": [{
                "campaign_id": "test_camp",
                "current_budget": 500,
                "spend": 350,
                "conversions": 50,
                "revenue": 2500
            }],
            "constraints": {"max_budget": 1000}
        })
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_attribution_analysis_flow(self, client: AsyncClient):
        """Test attribution analysis workflow."""
        # Get attribution
        response = await client.get("/api/attribution/", params={
            "start_date": "2026-01-01",
            "end_date": "2026-01-31"
        })
        assert response.status_code == 200
        
        # Get channel performance
        response = await client.get("/api/attribution/channels")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_audience_segmentation_flow(self, client: AsyncClient):
        """Test audience segmentation workflow."""
        # Get segments
        response = await client.get("/api/audience/segments")
        assert response.status_code == 200
        
        # Create lookalike
        response = await client.post("/api/audience/lookalike", json={
            "seed_segment_id": "top_customers",
            "expansion": 2.0,
            "target_size": 10000
        })
        assert response.status_code == 200


class TestMLOpsWorkflow:
    """Test MLOps integration."""
    
    @pytest.mark.asyncio
    async def test_model_lifecycle(self, client: AsyncClient):
        """Test model registration and management."""
        # List models
        response = await client.get("/api/mlops/models")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_ab_test_creation(self, client: AsyncClient):
        """Test A/B test workflow."""
        # Create test
        response = await client.post("/api/mlops/ab-tests", json={
            "name": "Integration Test",
            "model_type": "creative_predictor",
            "control_version": "v1",
            "treatment_version": "v2",
            "traffic_split": 0.5
        })
        assert response.status_code == 200
        
        # Get tests
        response = await client.get("/api/mlops/ab-tests")
        assert response.status_code == 200


class TestAlertWorkflow:
    """Test alert system."""
    
    @pytest.mark.asyncio
    async def test_alert_management(self, client: AsyncClient):
        """Test alert listing and management."""
        # Get alerts
        response = await client.get("/api/alerts/")
        assert response.status_code == 200
        
        # Get stats
        response = await client.get("/api/alerts/stats")
        assert response.status_code == 200
        
        # Get rules
        response = await client.get("/api/alerts/rules")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_alert_trigger(self, client: AsyncClient):
        """Test alert triggering."""
        response = await client.post("/api/alerts/check", json={
            "context": {
                "client_name": "Test Client",
                "churn_probability": 0.85,
                "risk_factors": ["Low ROAS", "No recent contact"]
            },
            "alert_type": "churn_risk"
        })
        assert response.status_code == 200


class TestWebhookWorkflow:
    """Test webhook endpoints."""
    
    @pytest.mark.asyncio
    async def test_webhook_status(self, client: AsyncClient):
        """Test webhook status endpoint."""
        response = await client.get("/api/webhooks/status")
        assert response.status_code == 200
        data = response.json()
        assert data["active"] == True
    
    @pytest.mark.asyncio
    async def test_meta_webhook_verify(self, client: AsyncClient):
        """Test Meta webhook verification."""
        response = await client.get("/api/webhooks/meta", params={
            "hub_mode": "subscribe",
            "hub_verify_token": "adclass_meta_verify_token",
            "hub_challenge": "12345"
        })
        assert response.status_code == 200


class TestPerformance:
    """Performance tests."""
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self, client: AsyncClient):
        """Health check should be fast."""
        import time
        
        start = time.time()
        for _ in range(10):
            await client.get("/api/health/")
        duration = time.time() - start
        
        # 10 requests should complete in under 1 second
        assert duration < 1.0
    
    @pytest.mark.asyncio
    async def test_prediction_performance(self, client: AsyncClient):
        """Prediction should be reasonably fast."""
        import time
        
        start = time.time()
        await client.post("/api/creative/predict", json={
            "headline": "Test headline",
            "body_text": "Test body",
            "cta_type": "SHOP_NOW",
            "platform": "meta"
        })
        duration = time.time() - start
        
        # Prediction should complete in under 2 seconds
        assert duration < 2.0


class TestDataIntegrity:
    """Data integrity tests."""
    
    @pytest.mark.asyncio
    async def test_consistent_predictions(self, client: AsyncClient):
        """Same input should give similar predictions."""
        payload = {
            "headline": "Consistent Test Headline",
            "body_text": "Consistent test body text.",
            "cta_type": "SHOP_NOW",
            "platform": "meta"
        }
        
        # Make multiple predictions
        results = []
        for _ in range(3):
            response = await client.post("/api/creative/predict", json=payload)
            results.append(response.json())
        
        # CTR predictions should be similar (within 20%)
        ctrs = [r["predicted_ctr"] for r in results]
        avg_ctr = sum(ctrs) / len(ctrs)
        for ctr in ctrs:
            assert abs(ctr - avg_ctr) / avg_ctr < 0.2
