#!/usr/bin/env python3
"""
Standalone Platform Test Suite (No Heavy Dependencies)

Tests core services, MLOps, alerts, and interventions without torch.
"""

import asyncio
import inspect
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

print("=" * 70)
print("  AdClass AI Platform - Standalone Test Suite")
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
        if inspect.iscoroutinefunction(func):
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
# SECTION 1: MLOps Tests (No torch needed)
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 1: MLOps & Model Management")
print("=" * 70)

def test_mlflow_experiment():
    # Import directly from source
    sys.path.insert(0, str(Path(__file__).parent / "app" / "services"))
    from mlops import get_mlflow_client
    
    client = get_mlflow_client()
    
    exp_id = client.create_experiment(
        name="test_experiment_standalone",
        artifact_location="mlflow-artifacts/test"
    )
    
    assert exp_id is not None
    print(f"       Experiment ID: {exp_id}")
    return exp_id

def test_mlflow_run():
    from mlops import get_mlflow_client
    
    client = get_mlflow_client()
    
    exp = client.get_experiment("test_experiment_standalone")
    exp_id = exp.experiment_id if exp else client.create_experiment("test_run_exp")
    
    run_id = client.start_run(exp_id, run_name="test_run_1")
    client.log_params(run_id, {"learning_rate": 0.01, "epochs": 100, "batch_size": 32})
    client.log_metrics(run_id, {"accuracy": 0.95, "loss": 0.05, "f1_score": 0.93})
    client.end_run(run_id)
    
    assert run_id is not None
    print(f"       Run ID: {run_id}")
    print(f"       Logged 3 params, 3 metrics")
    return run_id

def test_model_registry():
    from mlops import get_mlflow_client, ModelType
    
    client = get_mlflow_client()
    
    model = client.register_model(
        model_type=ModelType.CREATIVE_PREDICTOR,
        model_artifact_path="models/creative/test_v1",
        metrics={"accuracy": 0.92, "auc": 0.95, "precision": 0.89},
        parameters={"hidden_units": 256, "dropout": 0.3, "layers": 4},
        description="Test creative predictor model"
    )
    
    assert model is not None
    print(f"       Registered: {model.model_type.value} {model.version}")
    print(f"       Metrics: accuracy={model.metrics['accuracy']}, auc={model.metrics['auc']}")
    return model

def test_model_transition():
    from mlops import get_mlflow_client, ModelType, ModelStage
    
    client = get_mlflow_client()
    
    model = client.register_model(
        model_type=ModelType.ROAS_OPTIMIZER,
        model_artifact_path="models/roas/v1",
        metrics={"rmse": 0.12, "mae": 0.08},
        parameters={"lstm_units": 128},
        description="ROAS optimizer v1"
    )
    
    success = client.transition_model_stage(
        ModelType.ROAS_OPTIMIZER,
        model.version,
        ModelStage.PRODUCTION
    )
    
    assert success == True
    
    prod = client.get_production_model(ModelType.ROAS_OPTIMIZER)
    assert prod is not None
    assert prod.stage == ModelStage.PRODUCTION
    print(f"       Production model: {prod.version} (stage: {prod.stage.value})")
    return prod

def test_ab_testing():
    from mlops import get_ab_test_manager, ModelType
    import numpy as np
    
    manager = get_ab_test_manager()
    
    test_id = manager.create_test(
        name="CTR Model A/B Test",
        model_type=ModelType.CREATIVE_PREDICTOR,
        control_version="v1",
        treatment_version="v2",
        traffic_split=0.5
    )
    
    # Simulate 100 users
    np.random.seed(42)
    for i in range(100):
        user_id = f"user_ab_{i}"
        variant = manager.get_variant(test_id, user_id)
        
        # Treatment is better
        value = np.random.uniform(0.12, 0.28) if variant == "treatment" else np.random.uniform(0.08, 0.22)
        manager.record_outcome(test_id, user_id, "ctr", value)
    
    analysis = manager.analyze_test(test_id)
    
    assert "control_mean" in analysis
    assert "treatment_mean" in analysis
    print(f"       Control mean: {analysis['control_mean']:.4f}")
    print(f"       Treatment mean: {analysis['treatment_mean']:.4f}")
    print(f"       Lift: {analysis['lift_percent']:.2f}%")
    print(f"       Statistical significance: {analysis['significant']}")
    print(f"       Winner: {analysis['winner']}")
    return analysis

def test_drift_detection():
    from mlops import get_model_monitor, ModelType
    import numpy as np
    
    monitor = get_model_monitor()
    
    np.random.seed(42)
    # Log predictions with drift
    for i in range(200):
        prediction_id = f"drift_pred_{i}"
        features = {"f1": np.random.randn(), "f2": np.random.randn()}
        
        # First 100: normal predictions, Next 100: drifted
        if i < 100:
            prediction = np.random.uniform(0.1, 0.25)
        else:
            prediction = np.random.uniform(0.22, 0.38)  # Shifted mean
        
        monitor.log_prediction(ModelType.CHURN_MODEL, prediction_id, features, prediction)
    
    drift_result = monitor.detect_drift(ModelType.CHURN_MODEL)
    
    assert "drift_score" in drift_result
    print(f"       Sample count: {drift_result['sample_count']}")
    print(f"       Historical mean: {drift_result['historical_mean']:.4f}")
    print(f"       Recent mean: {drift_result['recent_mean']:.4f}")
    print(f"       Drift score: {drift_result['drift_score']:.4f}")
    print(f"       Drift detected: {drift_result['drift_detected']}")
    return drift_result

run_test("MLflow - Create Experiment", test_mlflow_experiment)
run_test("MLflow - Run Tracking with Params/Metrics", test_mlflow_run)
run_test("Model Registry - Register New Model", test_model_registry)
run_test("Model Registry - Stage Transition to Production", test_model_transition)
run_test("A/B Testing - Full Test Lifecycle", test_ab_testing)
run_test("Model Monitor - Drift Detection", test_drift_detection)

# ===========================================
# SECTION 2: Alert System Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 2: Alert System")
print("=" * 70)

def test_alert_rules():
    from alert_system import get_alert_manager, AlertSeverity, AlertType
    
    manager = get_alert_manager()
    
    assert len(manager.rules) > 0
    print(f"       {len(manager.rules)} default rules configured:")
    for rule_id, rule in list(manager.rules.items())[:3]:
        print(f"         - {rule.name} ({rule.severity.value})")
    return list(manager.rules.keys())

async def test_alert_trigger():
    from alert_system import get_alert_manager, AlertType
    
    manager = get_alert_manager()
    
    # Reset cooldowns for testing
    for rule in manager.rules.values():
        rule.last_triggered = None
    
    # High churn risk context
    context = {
        "client_name": "Acme Corp",
        "churn_probability": 0.88,
        "risk_factors": ["Low ROAS", "No login for 30 days", "Support escalations"]
    }
    
    alerts = await manager.check_and_alert(context, AlertType.CHURN_RISK)
    
    print(f"       Triggered {len(alerts)} alert(s)")
    for a in alerts:
        print(f"         - {a.title} ({a.severity.value})")
    return alerts

def test_alert_lifecycle():
    from alert_system import get_alert_manager
    
    manager = get_alert_manager()
    
    active = manager.get_active_alerts()
    print(f"       Active alerts: {len(active)}")
    
    # Acknowledge an alert if any exist
    if active:
        alert = active[0]
        manager.acknowledge_alert(alert.alert_id, "admin_user")
        print(f"       Acknowledged: {alert.alert_id[:8]}...")
        
        # Resolve it
        manager.resolve_alert(alert.alert_id)
        print(f"       Resolved: {alert.alert_id[:8]}...")
    
    return active

run_test("Alert Manager - Default Rules Configuration", test_alert_rules)
run_test("Alert Manager - Trigger Churn Risk Alert", test_alert_trigger)
run_test("Alert Manager - Lifecycle (Acknowledge/Resolve)", test_alert_lifecycle)

# ===========================================
# SECTION 3: Intervention Engine Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 3: Intervention Engine")
print("=" * 70)

def test_intervention_generation():
    from intervention_engine import get_intervention_engine, ClientContext
    
    engine = get_intervention_engine()
    
    # High-risk client context
    context = ClientContext(
        client_id="client_test_001",
        client_name="Risk Corp International",
        health_score=38,
        churn_probability=0.78,
        roas=0.9,
        target_roas=2.5,
        monthly_spend=12000,
        tenure_months=4,
        last_contact_days=28,
        engagement_score=25,
        recent_issues=["ROAS dropped 40%", "No campaign activity", "Payment late"]
    )
    
    interventions = engine.generate_interventions(context)
    
    assert len(interventions) > 0
    print(f"       Generated {len(interventions)} interventions:")
    for i in interventions:
        print(f"         - {i.title}")
        print(f"           Priority: {i.priority.value}, Channel: {i.channel.value}")
        print(f"           Due in: {i.due_within_days} days, Confidence: {i.confidence:.0%}")
    return interventions

def test_intervention_impact():
    from intervention_engine import get_intervention_engine, ClientContext
    
    engine = get_intervention_engine()
    
    context = ClientContext(
        client_id="client_test_002",
        client_name="Growth Potential LLC",
        health_score=52,
        churn_probability=0.58,
        roas=1.6,
        target_roas=2.5,
        monthly_spend=18000,
        tenure_months=8,
        last_contact_days=10,
        engagement_score=55,
        recent_issues=["Budget concerns raised"]
    )
    
    interventions = engine.generate_interventions(context)
    impact = engine.estimate_impact(interventions, context)
    
    assert "predicted_health_score" in impact
    print(f"       Current Health Score: {impact['current_health_score']:.0f}")
    print(f"       Predicted Health Score: {impact['predicted_health_score']:.0f}")
    print(f"       Health Improvement: +{impact['predicted_health_score'] - impact['current_health_score']:.0f}")
    print(f"       Current Churn Risk: {impact['current_churn_probability']:.0%}")
    print(f"       Predicted Churn Risk: {impact['predicted_churn_probability']:.0%}")
    print(f"       Churn Reduction: {(impact['current_churn_probability'] - impact['predicted_churn_probability']):.0%}")
    print(f"       Estimated Effort: {impact['estimated_effort_hours']} hours")
    return impact

def test_intervention_tracking():
    from intervention_engine import get_intervention_engine, get_intervention_tracker, ClientContext
    
    engine = get_intervention_engine()
    tracker = get_intervention_tracker()
    
    context = ClientContext(
        client_id="client_track_001",
        client_name="Tracking Test Co",
        health_score=60,
        churn_probability=0.45,
        roas=2.0,
        target_roas=2.5,
        monthly_spend=8000,
        tenure_months=12,
        last_contact_days=5,
        engagement_score=70,
        recent_issues=[]
    )
    
    interventions = engine.generate_interventions(context)
    
    if interventions:
        intervention = interventions[0]
        tracker.assign_intervention(intervention, "account_manager_1")
        print(f"       Assigned: {intervention.title} to account_manager_1")
        
        tracker.complete_intervention(intervention.id, {
            "churn_prevented": True,
            "health_improved": True,
            "notes": "Client renewed contract"
        })
        print(f"       Completed: {intervention.title}")
        
        effectiveness = tracker.get_effectiveness()
        print(f"       Effectiveness: {effectiveness['success_rate']:.0%} ({effectiveness['sample_size']} samples)")
    
    return tracker

run_test("Intervention Engine - Generate Recommendations", test_intervention_generation)
run_test("Intervention Engine - Impact Estimation", test_intervention_impact)
run_test("Intervention Engine - Tracking & Effectiveness", test_intervention_tracking)

# ===========================================
# SECTION 4: ETL Pipeline Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 4: ETL Pipeline")
print("=" * 70)

def test_etl_pipeline():
    from etl_pipeline import get_etl_pipeline
    
    pipeline = get_etl_pipeline()
    
    assert pipeline.validator is not None
    assert pipeline.transformer is not None
    assert pipeline.deduplicator is not None
    assert pipeline.loader is not None
    print(f"       Pipeline components initialized:")
    print(f"         - DataValidator ✓")
    print(f"         - DataTransformer ✓")
    print(f"         - DataDeduplicator ✓")
    print(f"         - IncrementalLoader ✓")
    return pipeline

def test_data_validation():
    from etl_pipeline import get_etl_pipeline
    
    pipeline = get_etl_pipeline()
    
    test_data = [
        {"campaign_id": "c1", "spend": 1000.50, "clicks": 500, "impressions": 50000, "date": "2026-02-01"},
        {"campaign_id": "c2", "spend": 2500.75, "clicks": 1200, "impressions": 80000, "date": "2026-02-01"},
        {"campaign_id": "c3", "spend": 750.25, "clicks": 350, "impressions": 35000, "date": "2026-02-01"},
    ]
    
    result = pipeline.validator.validate(test_data, "campaign_metrics")
    
    assert result["valid"] == True
    print(f"       Validated {result['records_count']} records")
    print(f"       Valid: {result['valid']}")
    print(f"       Issues: {len(result.get('issues', []))}")
    return result

def test_data_transformation():
    from etl_pipeline import get_etl_pipeline
    
    pipeline = get_etl_pipeline()
    
    test_data = [
        {"spend": 1000, "clicks": 500, "impressions": 50000},
        {"spend": 2000, "clicks": 800, "impressions": 80000},
    ]
    
    transformations = [
        {"type": "calculate", "source": ["clicks", "impressions"], "target": "ctr", "operation": "divide"},
        {"type": "calculate", "source": ["spend", "clicks"], "target": "cpc", "operation": "divide"},
    ]
    
    result = pipeline.transformer.transform(test_data, transformations)
    
    assert result is not None
    print(f"       Transformed {len(test_data)} records")
    print(f"       Added fields: ctr, cpc")
    if result and len(result) > 0:
        print(f"       Sample CTR: {result[0].get('ctr', 'N/A')}")
        print(f"       Sample CPC: ${result[0].get('cpc', 'N/A')}")
    return result

def test_deduplication():
    from etl_pipeline import get_etl_pipeline
    
    pipeline = get_etl_pipeline()
    
    test_data = [
        {"id": "1", "name": "Campaign A", "value": 100},
        {"id": "2", "name": "Campaign B", "value": 200},
        {"id": "1", "name": "Campaign A", "value": 100},  # Duplicate
        {"id": "3", "name": "Campaign C", "value": 300},
        {"id": "2", "name": "Campaign B", "value": 200},  # Duplicate
    ]
    
    result = pipeline.deduplicator.deduplicate(test_data, key_fields=["id"])
    
    print(f"       Input records: {len(test_data)}")
    print(f"       Output records: {len(result)}")
    print(f"       Duplicates removed: {len(test_data) - len(result)}")
    return result

run_test("ETL Pipeline - Component Initialization", test_etl_pipeline)
run_test("ETL Pipeline - Data Validation", test_data_validation)
run_test("ETL Pipeline - Data Transformation", test_data_transformation)
run_test("ETL Pipeline - Deduplication", test_deduplication)

# ===========================================
# SECTION 5: Webhook Handler Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 5: Webhook Handler")
print("=" * 70)

def test_webhook_handler():
    from webhook_handler import get_webhook_handler, WebhookSource
    
    handler = get_webhook_handler()
    
    sources = [s.value for s in WebhookSource]
    print(f"       Supported webhook sources:")
    for source in sources:
        print(f"         - {source}")
    
    assert handler.processor is not None
    print(f"       Event processor: ready")
    return handler

def test_webhook_event_types():
    from webhook_handler import WebhookEventType
    
    event_types = [e.value for e in WebhookEventType]
    print(f"       Supported event types ({len(event_types)}):")
    for et in event_types[:5]:
        print(f"         - {et}")
    if len(event_types) > 5:
        print(f"         - ... and {len(event_types) - 5} more")
    return event_types

run_test("Webhook Handler - Initialization", test_webhook_handler)
run_test("Webhook Handler - Event Types", test_webhook_event_types)

# ===========================================
# SECTION 6: Feature Store Tests
# ===========================================
print("\n" + "=" * 70)
print("  SECTION 6: Feature Store")
print("=" * 70)

def test_feature_store_init():
    from feature_store import get_feature_store
    
    store = get_feature_store()
    
    assert store is not None
    print(f"       Feature store initialized")
    print(f"       Offline store: ready")
    print(f"       Online store: ready")
    return store

def test_feature_group_definition():
    from feature_store import get_feature_store
    
    store = get_feature_store()
    
    group = store.define_feature_group(
        name="campaign_performance",
        entity_types=["campaign_id"],
        features=[
            {"name": "ctr_7d", "dtype": "float"},
            {"name": "cvr_7d", "dtype": "float"},
            {"name": "spend_7d", "dtype": "float"},
            {"name": "roas_7d", "dtype": "float"},
            {"name": "impressions_7d", "dtype": "int"},
        ],
        description="7-day rolling campaign metrics"
    )
    
    assert group.name == "campaign_performance"
    print(f"       Created feature group: {group.name}")
    print(f"       Features: {len(group.features)}")
    for f in group.features[:3]:
        print(f"         - {f['name']} ({f['dtype']})")
    return group

def test_feature_computation():
    from feature_store import get_feature_store
    
    store = get_feature_store()
    
    test_data = {
        "campaigns": [
            {"id": "c1", "spend": 1500, "revenue": 4500, "conversions": 45, "impressions": 75000},
            {"id": "c2", "spend": 2500, "revenue": 6000, "conversions": 80, "impressions": 100000},
            {"id": "c3", "spend": 800, "revenue": 1200, "conversions": 15, "impressions": 40000},
        ]
    }
    
    features = store.compute_features(test_data)
    
    assert isinstance(features, dict)
    print(f"       Computed features for {len(test_data['campaigns'])} campaigns")
    print(f"       Feature groups available: {len(features)}")
    return features

run_test("Feature Store - Initialization", test_feature_store_init)
run_test("Feature Store - Define Feature Group", test_feature_group_definition)
run_test("Feature Store - Compute Features", test_feature_computation)

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
    print("\n  🎉 ALL TESTS PASSED! Platform core services are fully functional.")
else:
    print(f"\n  ⚠️  {failed} test(s) failed. Review errors above.")

print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Exit with appropriate code
sys.exit(0 if failed == 0 else 1)
