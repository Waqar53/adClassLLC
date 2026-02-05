"""
Test Suite for ML Models

Unit tests for all 5 ML modules.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta


class TestCreativePredictor:
    """Test Creative Performance Predictor ML model."""
    
    def test_text_only_prediction(self):
        """Test prediction with text only."""
        from app.models.ml.creative_predictor import get_creative_predictor
        
        predictor = get_creative_predictor()
        result = predictor.predict_text_only(
            headline="50% Off Summer Sale!",
            body_text="Shop now and save big.",
            cta_type="SHOP_NOW",
            platform="meta"
        )
        
        assert "predicted_ctr" in result
        assert "predicted_cvr" in result
        assert result["predicted_ctr"] > 0
        assert result["predicted_cvr"] > 0
        assert len(result["recommendations"]) > 0
    
    def test_feature_extraction(self):
        """Test text feature extraction."""
        from app.models.ml.creative_predictor import get_creative_predictor
        
        predictor = get_creative_predictor()
        features = predictor.text_encoder.extract_features(
            "Amazing 50% Off Sale - Limited Time Only!"
        )
        
        assert "char_count" in features
        assert "word_count" in features
        assert features["word_count"] == 7
        assert features["has_numbers"] == True
        assert features["exclamation_count"] == 1


class TestROASOptimizer:
    """Test ROAS Optimizer ML model."""
    
    def test_campaign_optimization(self):
        """Test campaign budget optimization."""
        from app.models.ml.roas_optimizer import get_roas_optimizer, CampaignState
        
        optimizer = get_roas_optimizer()
        
        campaigns = [
            CampaignState(
                campaign_id="camp_1",
                current_budget=500.0,
                spend_today=350.0,
                impressions=45000,
                clicks=1125,
                conversions=67,
                revenue=4020.0,
                ctr=0.025,
                cvr=0.060,
                roas=11.49,
                hour_of_day=12,
                day_of_week=2
            )
        ]
        
        decisions = optimizer.optimize(campaigns, {})
        
        assert len(decisions) == 1
        assert decisions[0].campaign_id == "camp_1"
        assert decisions[0].action in ["increase", "decrease", "maintain", "pause"]
        assert decisions[0].confidence > 0
    
    def test_thompson_sampling(self):
        """Test Thompson Sampling bandit."""
        from app.models.ml.roas_optimizer import ThompsonSamplingBandit
        
        bandit = ThompsonSamplingBandit(n_arms=5)
        
        # Update with some rewards
        for i in range(10):
            arm = i % 5
            bandit.update(arm, i * 0.1)
        
        # Select arm
        selected = bandit.select_arm()
        assert 0 <= selected < 5
    
    def test_forecast_prediction(self):
        """Test ROAS forecasting."""
        from app.models.ml.roas_optimizer import get_roas_optimizer
        
        optimizer = get_roas_optimizer()
        
        # Generate mock historical data
        historical_roas = np.random.uniform(2.0, 6.0, 30)
        
        forecast = optimizer.forecaster.predict(
            historical_roas=historical_roas,
            forecast_days=7,
            budget_scenario=1000.0
        )
        
        assert len(forecast) == 7
        assert all(v > 0 for v in forecast)


class TestChurnPredictor:
    """Test Churn Prediction ML model."""
    
    def test_churn_prediction(self):
        """Test churn prediction."""
        from app.models.ml.churn_model import get_churn_predictor, ClientFeatures
        
        predictor = get_churn_predictor()
        
        features = ClientFeatures(
            client_id="client_1",
            current_roas=4.5,
            target_roas=4.0,
            roas_7d_avg=4.2,
            roas_30d_avg=4.0,
            roas_trend=0.05,
            dashboard_logins_7d=12,
            dashboard_logins_30d=45,
            meeting_attendance_rate=0.85,
            email_open_rate=0.65,
            avg_response_time_hours=4.5,
            last_contact_days=3,
            support_tickets_30d=2,
            ticket_sentiment_avg=0.7,
            monthly_spend=25000.0,
            revenue_mom_change=0.08,
            payment_delay_avg_days=2.0,
            invoices_outstanding=0,
            contract_days_remaining=180,
            is_month_to_month=False,
            tenure_months=24
        )
        
        prediction, health, risk_factors = predictor.predict(features)
        
        assert 0 <= prediction.churn_probability <= 1
        assert prediction.risk_level in ["healthy", "monitor", "warning", "critical"]
        assert 0 <= health.overall_score <= 100
    
    def test_health_score_calculation(self):
        """Test health score calculation."""
        from app.models.ml.churn_model import HealthScoreCalculator
        
        calculator = HealthScoreCalculator()
        
        scores = {
            "performance": 80,
            "engagement": 70,
            "communication": 85,
            "financial": 90
        }
        
        result = calculator.calculate(scores)
        
        assert 0 <= result.overall_score <= 100
        assert result.performance_score == 80
        assert result.engagement_score == 70


class TestAttributionEngine:
    """Test Attribution ML models."""
    
    def test_shapley_calculation(self):
        """Test Shapley value attribution."""
        from app.models.ml.attribution_model import ShapleyCalculator, CustomerJourney, Touchpoint
        
        calculator = ShapleyCalculator()
        
        now = datetime.now()
        journeys = [
            CustomerJourney(
                customer_id="cust_1",
                touchpoints=[
                    Touchpoint("tp1", "cust_1", "meta", None, now - timedelta(days=3), "click"),
                    Touchpoint("tp2", "cust_1", "google", None, now - timedelta(days=1), "click"),
                ],
                conversion_value=100.0,
                converted=True
            ),
            CustomerJourney(
                customer_id="cust_2",
                touchpoints=[
                    Touchpoint("tp3", "cust_2", "google", None, now - timedelta(days=2), "click"),
                ],
                conversion_value=80.0,
                converted=True
            )
        ]
        
        shapley = calculator.compute(journeys)
        
        assert "meta" in shapley or "google" in shapley
        assert sum(shapley.values()) == pytest.approx(1.0, rel=0.1)
    
    def test_markov_chain(self):
        """Test Markov Chain attribution."""
        from app.models.ml.attribution_model import MarkovChainAttributor, CustomerJourney, Touchpoint
        
        attributor = MarkovChainAttributor()
        
        now = datetime.now()
        journeys = [
            CustomerJourney(
                customer_id="cust_1",
                touchpoints=[
                    Touchpoint("tp1", "cust_1", "meta", None, now - timedelta(days=3), "click"),
                    Touchpoint("tp2", "cust_1", "email", None, now - timedelta(days=1), "click"),
                ],
                conversion_value=100.0,
                converted=True
            )
        ]
        
        attributor.fit(journeys)
        attribution = attributor.compute_attribution()
        
        assert isinstance(attribution, dict)
    
    def test_path_analysis(self):
        """Test path analysis."""
        from app.models.ml.attribution_model import PathAnalyzer, CustomerJourney, Touchpoint
        
        analyzer = PathAnalyzer()
        
        now = datetime.now()
        journeys = [
            CustomerJourney(
                customer_id="cust_1",
                touchpoints=[
                    Touchpoint("tp1", "cust_1", "meta", None, now, "click"),
                    Touchpoint("tp2", "cust_1", "google", None, now, "click"),
                ],
                conversion_value=100.0,
                converted=True
            )
        ]
        
        result = analyzer.analyze(journeys)
        
        assert "total_paths" in result
        assert "top_paths" in result


class TestAudienceIntelligence:
    """Test Audience Intelligence ML models."""
    
    def test_kmeans_clustering(self):
        """Test K-Means clustering."""
        from app.models.ml.audience_model import KMeansClusterer, AudienceMember
        
        clusterer = KMeansClusterer(n_clusters=3)
        
        members = [
            AudienceMember(
                customer_id=f"cust_{i}",
                features={
                    "ltv": np.random.uniform(50, 500),
                    "purchase_frequency": np.random.uniform(1, 12),
                    "avg_order_value": np.random.uniform(25, 150)
                }
            )
            for i in range(50)
        ]
        
        assignments = clusterer.fit(members)
        
        assert len(assignments) == 50
        assert all(0 <= a < 3 for a in assignments)
    
    def test_lookalike_generation(self):
        """Test lookalike audience generation."""
        from app.models.ml.audience_model import LookalikeGenerator, AudienceMember
        
        generator = LookalikeGenerator()
        
        seed = [
            AudienceMember(
                customer_id=f"seed_{i}",
                features={"ltv": 400 + i * 10, "frequency": 10}
            )
            for i in range(10)
        ]
        
        candidates = [
            AudienceMember(
                customer_id=f"cand_{i}",
                features={"ltv": 100 + i * 20, "frequency": i % 12}
            )
            for i in range(100)
        ]
        
        generator.fit_seed(seed)
        lookalikes = generator.generate_lookalike(candidates, size=20, expansion_factor=2.0)
        
        assert len(lookalikes) <= 20
        assert all(isinstance(item, tuple) for item in lookalikes)
    
    def test_segment_characteristics(self):
        """Test segment characteristic extraction."""
        from app.models.ml.audience_model import get_audience_service, AudienceMember
        
        service = get_audience_service()
        
        members = [
            AudienceMember(
                customer_id=f"cust_{i}",
                features={
                    "ltv": np.random.uniform(50, 500),
                    "purchase_frequency": np.random.uniform(1, 12),
                    "avg_order_value": np.random.uniform(25, 150),
                    "days_since_purchase": np.random.randint(1, 90)
                }
            )
            for i in range(30)
        ]
        
        result = service.segment_audience(members, method="kmeans", n_segments=3)
        
        assert "segments" in result
        assert len(result["segments"]) <= 3
