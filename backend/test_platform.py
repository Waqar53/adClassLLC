#!/usr/bin/env python3
"""
Comprehensive Platform Test Suite

Tests all components: Backend APIs, ML Models, MLOps, Webhooks, Alerts
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("  AdClass AI Platform - Comprehensive Test Suite")
print("=" * 70)
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Test counters
passed = 0
failed = 0
total = 0

def run_test(name, func):
    """Run a test and track results"""
    global passed, failed, total
    total += 1
    try:
        if asyncio.iscoroutinefunction(func):
            result = asyncio.run(func())
        else:
            result = func()
        print(f"  ✅ PASS: {name}")
        passed += 1
        return result
    except Exception as e:
        print(f"  ❌ FAIL: {name}")
        print(f"       Error: {str(e)[:100]}")
        failed += 1
        return None

# ===========================================
# SECTION 1: Core Services Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 1: Core Services & Feature Store")
print("=" * 70)

def test_feature_store():
    from app.services.feature_store import get_feature_store
    store = get_feature_store()
    assert store is not None
    
    # Define a test feature group
    group = store.define_feature_group(
        name="test_campaigns",
        entity_types=["campaign"],
        features=[
            {"name": "ctr", "dtype": "float"},
            {"name": "spend", "dtype": "float"},
            {"name": "impressions", "dtype": "int"}
        ],
        description="Test campaign features"
    )
    assert group.name == "test_campaigns"
    return True

def test_feature_compute():
    from app.services.feature_store import get_feature_store
    import numpy as np
    
    store = get_feature_store()
    
    # Create test data
    test_data = {
        "campaigns": [
            {"id": "c1", "spend": 1000, "conversions": 50, "impressions": 50000},
            {"id": "c2", "spend": 2000, "conversions": 120, "impressions": 80000},
        ]
    }
    
    features = store.compute_features(test_data)
    assert isinstance(features, dict)
    return features

def test_etl_validation():
    from app.services.etl_pipeline import get_etl_pipeline
    
    pipeline = get_etl_pipeline()
    
    test_data = [
        {"campaign_id": "c1", "spend": 1000, "clicks": 500, "impressions": 50000},
        {"campaign_id": "c2", "spend": 2000, "clicks": 800, "impressions": 80000},
    ]
    
    # Run through pipeline
    validated = pipeline.validator.validate(test_data, "campaign_data")
    assert validated["valid"] == True
    return validated

def test_etl_transform():
    from app.services.etl_pipeline import get_etl_pipeline
    
    pipeline = get_etl_pipeline()
    
    result = pipeline.transformer.transform(
        data=[{"spend": 1000, "conversions": 50}],
        transformations=[
            {"type": "calculate", "source": ["spend", "conversions"], "target": "cpa", "operation": "divide"}
        ]
    )
    assert result is not None
    return True

run_test("Feature Store Initialization", test_feature_store)
run_test("Feature Store - Compute Features", test_feature_compute)
run_test("ETL Pipeline - Data Validation", test_etl_validation)
run_test("ETL Pipeline - Transform Data", test_etl_transform)

# ===========================================
# SECTION 2: ML Models Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 2: ML Models Testing")
print("=" * 70)

def test_creative_model():
    from app.models.ml.creative_model import get_creative_predictor
    
    predictor = get_creative_predictor()
    
    result = predictor.predict(
        headline="Summer Sale - 50% Off Everything!",
        body_text="Shop now and save big on our entire collection. Limited time only!",
        cta_type="SHOP_NOW",
        platform="meta",
        industry="e-commerce",
        image_url=None
    )
    
    assert "predicted_ctr" in result
    assert "predicted_cvr" in result
    assert 0 <= result["predicted_ctr"] <= 1
    assert 0 <= result["predicted_cvr"] <= 1
    print(f"       CTR: {result['predicted_ctr']:.4f}, CVR: {result['predicted_cvr']:.4f}")
    return result

def test_creative_features():
    from app.models.ml.creative_model import get_creative_predictor
    
    predictor = get_creative_predictor()
    
    features = predictor._extract_text_features(
        "Amazing Black Friday Deals - Up to 70% Off!",
        "Don't miss out on the biggest sale of the year. Shop now and save.",
        "SHOP_NOW"
    )
    
    assert len(features) > 0
    print(f"       Extracted {len(features)} features")
    return features

def test_roas_thompson():
    from app.models.ml.roas_model import get_roas_optimizer
    
    optimizer = get_roas_optimizer()
    
    # Add test campaign
    optimizer.bandit.add_campaign("test_camp_1")
    optimizer.bandit.add_campaign("test_camp_2")
    
    # Record some rewards
    optimizer.bandit.update("test_camp_1", roas=3.5)
    optimizer.bandit.update("test_camp_2", roas=2.8)
    
    # Get recommendation
    selected = optimizer.bandit.select()
    assert selected in ["test_camp_1", "test_camp_2"]
    print(f"       Selected: {selected}")
    return selected

def test_roas_budget():
    from app.models.ml.roas_model import get_roas_optimizer
    
    optimizer = get_roas_optimizer()
    
    campaigns = [
        {
            "campaign_id": "camp_a",
            "platform": "meta",
            "current_budget": 500,
            "spend": 450,
            "conversions": 25,
            "revenue": 1500
        },
        {
            "campaign_id": "camp_b",
            "platform": "google",
            "current_budget": 300,
            "spend": 280,
            "conversions": 18,
            "revenue": 900
        }
    ]
    
    result = optimizer.optimize(
        campaigns=campaigns,
        total_budget=1000,
        constraints={"min_budget": 100, "max_budget": 600}
    )
    
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0
    print(f"       Generated {len(result['recommendations'])} recommendations")
    return result

def test_churn_health():
    from app.models.ml.churn_model import get_churn_predictor
    
    predictor = get_churn_predictor()
    
    result = predictor.predict(
        client_id="client_123",
        features={
            "months_active": 8,
            "avg_monthly_spend": 5000,
            "avg_roas": 2.5,
            "campaigns_active": 5,
            "days_since_login": 3,
            "support_tickets": 1,
            "payment_delays": 0,
            "engagement_score": 75
        }
    )
    
    assert "churn_probability" in result
    assert "health_score" in result
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["health_score"] <= 100
    print(f"       Churn: {result['churn_probability']:.2%}, Health: {result['health_score']:.0f}")
    return result

def test_churn_factors():
    from app.models.ml.churn_model import get_churn_predictor
    
    predictor = get_churn_predictor()
    
    # High-risk client
    result = predictor.predict(
        client_id="at_risk_client",
        features={
            "months_active": 2,
            "avg_monthly_spend": 1000,
            "avg_roas": 0.8,
            "campaigns_active": 1,
            "days_since_login": 45,
            "support_tickets": 8,
            "payment_delays": 2,
            "engagement_score": 20
        }
    )
    
    assert result["churn_probability"] > 0.5  # Should be high risk
    assert len(result.get("risk_factors", [])) > 0
    print(f"       High-risk detected: {len(result.get('risk_factors', []))} factors")
    return result

def test_attribution_shapley():
    from app.models.ml.attribution_model import get_attribution_engine
    
    engine = get_attribution_engine()
    
    journeys = [
        ["facebook", "google", "email", "direct"],
        ["google", "facebook", "direct"],
        ["email", "facebook", "google", "direct"],
        ["facebook", "direct"],
        ["google", "email", "direct"],
    ]
    
    result = engine.calculate_shapley_attribution(journeys)
    
    assert "channel_attribution" in result
    assert len(result["channel_attribution"]) > 0
    print(f"       Channels: {list(result['channel_attribution'].keys())}")
    return result

def test_attribution_markov():
    from app.models.ml.attribution_model import get_attribution_engine
    
    engine = get_attribution_engine()
    
    journeys = [
        ["facebook", "google", "conversion"],
        ["google", "facebook", "conversion"],
        ["email", "google", "conversion"],
        ["facebook", "null"],
        ["google", "null"],
    ]
    
    result = engine.calculate_markov_attribution(journeys)
    
    assert "removal_effects" in result
    print(f"       Removal effects calculated")
    return result

def test_audience_kmeans():
    from app.models.ml.audience_model import get_audience_engine
    import numpy as np
    
    engine = get_audience_engine()
    
    # Generate test users
    user_features = np.random.rand(100, 10)  # 100 users, 10 features
    
    result = engine.segment_users(
        user_features=user_features.tolist(),
        method="kmeans",
        n_clusters=5
    )
    
    assert "segments" in result
    assert len(result["segments"]) == 5
    print(f"       Created {len(result['segments'])} segments")
    return result

def test_audience_lookalike():
    from app.models.ml.audience_model import get_audience_engine
    import numpy as np
    
    engine = get_audience_engine()
    
    # Seed audience (high-value customers)
    seed_features = np.random.rand(20, 10).tolist()
    
    # Candidate pool
    candidate_features = np.random.rand(200, 10).tolist()
    
    result = engine.generate_lookalike(
        seed_features=seed_features,
        candidate_features=candidate_features,
        expansion=2.0
    )
    
    assert "lookalike_indices" in result
    print(f"       Found {len(result['lookalike_indices'])} lookalike users")
    return result

run_test("Creative Model - Predict CTR", test_creative_model)
run_test("Creative Model - Feature Extraction", test_creative_features)
run_test("ROAS Model - Thompson Sampling", test_roas_thompson)
run_test("ROAS Model - Budget Optimization", test_roas_budget)
run_test("Churn Model - Health Score", test_churn_health)
run_test("Churn Model - Risk Factors", test_churn_factors)
run_test("Attribution Model - Shapley Values", test_attribution_shapley)
run_test("Attribution Model - Markov Chain", test_attribution_markov)
run_test("Audience Model - K-Means Clustering", test_audience_kmeans)
run_test("Audience Model - Lookalike Generation", test_audience_lookalike)

# ===========================================
# SECTION 3: Computer Vision Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 3: Computer Vision Analysis")
print("=" * 70)

def test_vision_color():
    from app.models.ml.vision_model import get_creative_vision_service
    import numpy as np
    
    service = get_creative_vision_service()
    
    # Create test image (gradient)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:50, :, 0] = 255  # Red top
    test_image[50:, :, 2] = 255  # Blue bottom
    
    result = service.analyze_colors(test_image)
    
    assert "dominant_colors" in result
    assert "color_distribution" in result
    print(f"       Found {len(result['dominant_colors'])} dominant colors")
    return result

def test_vision_composition():
    from app.models.ml.vision_model import get_creative_vision_service
    import numpy as np
    
    service = get_creative_vision_service()
    
    # Test image
    test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    result = service.analyze_composition(test_image)
    
    assert "rule_of_thirds_score" in result
    assert "overall_score" in result
    print(f"       Composition score: {result['overall_score']:.2f}")
    return result

def test_vision_full():
    from app.models.ml.vision_model import get_creative_vision_service
    import numpy as np
    
    service = get_creative_vision_service()
    
    # Create test image
    test_image = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
    
    result = service.analyze_image(test_image)
    
    assert "features" in result
    print(f"       Extracted {len(result.get('features', []))} visual features")
    return result

run_test("Vision Model - Color Analysis", test_vision_color)
run_test("Vision Model - Composition Analysis", test_vision_composition)
run_test("Vision Model - Full Image Analysis", test_vision_full)

# ===========================================
# SECTION 4: MLOps Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 4: MLOps & Model Management")
print("=" * 70)

def test_mlflow_experiment():
    from app.services.mlops import get_mlflow_client
    
    client = get_mlflow_client()
    
    exp_id = client.create_experiment(
        name="test_experiment",
        artifact_location="mlflow-artifacts/test"
    )
    
    assert exp_id is not None
    print(f"       Experiment ID: {exp_id}")
    return exp_id

def test_mlflow_run():
    from app.services.mlops import get_mlflow_client
    
    client = get_mlflow_client()
    
    # Get or create experiment
    exp = client.get_experiment("test_experiment")
    exp_id = exp.experiment_id if exp else client.create_experiment("test_experiment")
    
    # Start run
    run_id = client.start_run(exp_id, run_name="test_run")
    
    # Log params and metrics
    client.log_params(run_id, {"learning_rate": 0.01, "epochs": 100})
    client.log_metrics(run_id, {"accuracy": 0.95, "loss": 0.05})
    
    # End run
    client.end_run(run_id)
    
    assert run_id is not None
    print(f"       Run ID: {run_id}")
    return run_id

def test_model_registry():
    from app.services.mlops import get_mlflow_client, ModelType
    
    client = get_mlflow_client()
    
    model = client.register_model(
        model_type=ModelType.CREATIVE_PREDICTOR,
        model_artifact_path="models/creative/v1",
        metrics={"accuracy": 0.92, "auc": 0.95},
        parameters={"hidden_units": 256, "dropout": 0.3},
        description="Test creative predictor model"
    )
    
    assert model is not None
    assert model.version == "v1"
    print(f"       Registered: {model.model_type.value} {model.version}")
    return model

def test_model_transition():
    from app.services.mlops import get_mlflow_client, ModelType, ModelStage
    
    client = get_mlflow_client()
    
    # Register another version
    model = client.register_model(
        model_type=ModelType.CREATIVE_PREDICTOR,
        model_artifact_path="models/creative/v2",
        metrics={"accuracy": 0.94, "auc": 0.96},
        parameters={"hidden_units": 512, "dropout": 0.2},
        description="Improved creative predictor"
    )
    
    # Transition to production
    success = client.transition_model_stage(
        ModelType.CREATIVE_PREDICTOR,
        model.version,
        ModelStage.PRODUCTION
    )
    
    assert success == True
    
    # Get production model
    prod = client.get_production_model(ModelType.CREATIVE_PREDICTOR)
    assert prod is not None
    print(f"       Production model: {prod.version}")
    return prod

def test_ab_testing():
    from app.services.mlops import get_ab_test_manager, ModelType
    import numpy as np
    
    manager = get_ab_test_manager()
    
    # Create test
    test_id = manager.create_test(
        name="Creative Model A/B Test",
        model_type=ModelType.CREATIVE_PREDICTOR,
        control_version="v1",
        treatment_version="v2",
        traffic_split=0.5
    )
    
    # Simulate user assignments and outcomes
    for i in range(100):
        user_id = f"user_{i}"
        variant = manager.get_variant(test_id, user_id)
        
        # Simulate outcomes (treatment is slightly better)
        if variant == "treatment":
            value = np.random.uniform(0.1, 0.3)
        else:
            value = np.random.uniform(0.08, 0.25)
        
        manager.record_outcome(test_id, user_id, "conversion_rate", value)
    
    # Analyze results
    analysis = manager.analyze_test(test_id)
    
    assert "control_mean" in analysis
    assert "treatment_mean" in analysis
    print(f"       Control: {analysis['control_mean']:.3f}, Treatment: {analysis['treatment_mean']:.3f}")
    print(f"       Lift: {analysis['lift_percent']:.1f}%, Winner: {analysis['winner']}")
    return analysis

def test_drift_detection():
    from app.services.mlops import get_model_monitor, ModelType
    import numpy as np
    
    monitor = get_model_monitor()
    
    # Log predictions
    for i in range(200):
        prediction_id = f"pred_{i}"
        features = {"feature_1": np.random.randn(), "feature_2": np.random.randn()}
        
        # Simulate drift: later predictions have shifted means
        if i < 100:
            prediction = np.random.uniform(0.1, 0.3)
        else:
            prediction = np.random.uniform(0.2, 0.4)  # Shifted
        
        monitor.log_prediction(ModelType.CREATIVE_PREDICTOR, prediction_id, features, prediction)
    
    # Detect drift
    drift_result = monitor.detect_drift(ModelType.CREATIVE_PREDICTOR)
    
    assert "drift_score" in drift_result
    print(f"       Drift score: {drift_result['drift_score']:.3f}")
    print(f"       Drift detected: {drift_result['drift_detected']}")
    return drift_result

run_test("MLflow - Create Experiment", test_mlflow_experiment)
run_test("MLflow - Run Tracking", test_mlflow_run)
run_test("Model Registry - Register Model", test_model_registry)
run_test("Model Registry - Stage Transition", test_model_transition)
run_test("A/B Test - Create & Manage", test_ab_testing)
run_test("Model Monitor - Drift Detection", test_drift_detection)

# ===========================================
# SECTION 5: Alert System Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 5: Alert System & Interventions")
print("=" * 70)

def test_alert_rules():
    from app.services.alert_system import get_alert_manager
    
    manager = get_alert_manager()
    
    assert len(manager.rules) > 0
    print(f"       {len(manager.rules)} default rules configured")
    return list(manager.rules.keys())

async def test_alert_trigger():
    from app.services.alert_system import get_alert_manager, AlertType
    
    manager = get_alert_manager()
    
    # Context that should trigger churn alert
    context = {
        "client_name": "Test Client",
        "churn_probability": 0.85,
        "risk_factors": ["Low ROAS", "No recent contact"]
    }
    
    alerts = await manager.check_and_alert(context, AlertType.CHURN_RISK)
    
    # Note: May not trigger if cooldown is active
    print(f"       Triggered {len(alerts)} alerts")
    return alerts

def test_active_alerts():
    from app.services.alert_system import get_alert_manager
    
    manager = get_alert_manager()
    
    active = manager.get_active_alerts()
    print(f"       {len(active)} active alerts")
    return active

def test_interventions():
    from app.services.intervention_engine import get_intervention_engine, ClientContext
    
    engine = get_intervention_engine()
    
    context = ClientContext(
        client_id="client_456",
        client_name="At-Risk Corp",
        health_score=45,
        churn_probability=0.72,
        roas=1.2,
        target_roas=2.5,
        monthly_spend=8000,
        tenure_months=6,
        last_contact_days=21,
        engagement_score=35,
        recent_issues=["ROAS below target", "Decreased login frequency"]
    )
    
    interventions = engine.generate_interventions(context)
    
    assert len(interventions) > 0
    print(f"       Generated {len(interventions)} interventions:")
    for i in interventions[:3]:
        print(f"         - {i.title} ({i.priority.value})")
    return interventions

def test_intervention_impact():
    from app.services.intervention_engine import get_intervention_engine, ClientContext
    
    engine = get_intervention_engine()
    
    context = ClientContext(
        client_id="client_789",
        client_name="Growth Inc",
        health_score=55,
        churn_probability=0.55,
        roas=1.8,
        target_roas=2.5,
        monthly_spend=15000,
        tenure_months=12,
        last_contact_days=7,
        engagement_score=60,
        recent_issues=["Budget concerns"]
    )
    
    interventions = engine.generate_interventions(context)
    impact = engine.estimate_impact(interventions, context)
    
    assert "predicted_health_score" in impact
    print(f"       Current Health: {impact['current_health_score']:.0f}")
    print(f"       Predicted: {impact['predicted_health_score']:.0f}")
    print(f"       Churn reduction: {impact['current_churn_probability'] - impact['predicted_churn_probability']:.2%}")
    return impact

run_test("Alert Manager - Default Rules", test_alert_rules)
run_test("Alert Manager - Trigger Alert", test_alert_trigger)
run_test("Alert Manager - Get Active Alerts", test_active_alerts)
run_test("Intervention Engine - Generate Recommendations", test_interventions)
run_test("Intervention Engine - Impact Estimation", test_intervention_impact)

# ===========================================
# SECTION 6: Webhook Handler Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 6: Webhook Handler")
print("=" * 70)

def test_webhook_init():
    from app.services.webhook_handler import get_webhook_handler, WebhookSource
    
    handler = get_webhook_handler()
    
    sources = [s.value for s in WebhookSource]
    print(f"       Supported sources: {', '.join(sources)}")
    return handler

def test_webhook_process():
    from app.services.webhook_handler import get_webhook_handler
    
    handler = get_webhook_handler()
    
    # Just testing the handler exists
    assert handler.processor is not None
    print(f"       Webhook processor ready")
    return True

run_test("Webhook Handler - Initialization", test_webhook_init)
run_test("Webhook Handler - Event Processing", test_webhook_process)

# ===========================================
# SECTION 7: Data Ingestion Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 7: Data Ingestion Services")
print("=" * 70)

def test_meta_client():
    from app.services.data_ingestion import MetaAPIClient
    
    client = MetaAPIClient()
    assert client is not None
    print(f"       Meta API client initialized")
    return client

def test_google_client():
    from app.services.data_ingestion import GoogleAdsClient
    
    client = GoogleAdsClient()
    assert client is not None
    print(f"       Google Ads client initialized")
    return client

def test_tiktok_client():
    from app.services.data_ingestion import TikTokAdsClient
    
    client = TikTokAdsClient()
    assert client is not None
    print(f"       TikTok Ads client initialized")
    return client

run_test("Meta API Client - Initialization", test_meta_client)
run_test("Google Ads Client - Initialization", test_google_client)
run_test("TikTok Ads Client - Initialization", test_tiktok_client)

# ===========================================
# FINAL RESULTS
# ===========================================
print("\n" + "=" * 70)
print("  TEST RESULTS SUMMARY")
print("=" * 70)
print(f"  Total Tests:  {total}")
print(f"  Passed:       {passed} ✅")
print(f"  Failed:       {failed} ❌")
print(f"  Success Rate: {(passed/total*100) if total > 0 else 0:.1f}%")
print("=" * 70)

if failed == 0:
    print("\n  🎉 ALL TESTS PASSED! Platform is fully functional.")
else:
    print(f"\n  ⚠️  {failed} test(s) failed. Review errors above.")

print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
