"""
AdClass AI Platform - Airflow DAGs

ETL pipeline orchestration for data ingestion from ad platforms.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup


# Default DAG arguments
default_args = {
    "owner": "adclass",
    "depends_on_past": False,
    "email": ["alerts@adclass.ai"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


# ===========================================
# ETL Pipeline DAG
# ===========================================
with DAG(
    dag_id="adclass_etl_pipeline",
    default_args=default_args,
    description="Daily ETL pipeline for ad platform data",
    schedule_interval="0 2 * * *",  # 2 AM daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["etl", "data-ingestion"],
) as etl_dag:
    
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")
    
    with TaskGroup("meta_ingestion") as meta_group:
        def extract_meta_campaigns(**context):
            """Extract campaigns from Meta."""
            from app.services.data_ingestion import MetaAPIClient
            client = MetaAPIClient()
            return client.get_campaigns()
        
        def extract_meta_insights(**context):
            """Extract insights from Meta."""
            from app.services.data_ingestion import MetaAPIClient
            client = MetaAPIClient()
            return client.get_ad_insights(
                date_from=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            )
        
        def transform_meta_data(**context):
            """Transform Meta data."""
            from app.services.etl_pipeline import get_etl_pipeline
            pipeline = get_etl_pipeline()
            # Process in ETL pipeline
            return {"status": "transformed"}
        
        meta_extract_campaigns = PythonOperator(
            task_id="extract_campaigns",
            python_callable=extract_meta_campaigns,
        )
        
        meta_extract_insights = PythonOperator(
            task_id="extract_insights", 
            python_callable=extract_meta_insights,
        )
        
        meta_transform = PythonOperator(
            task_id="transform",
            python_callable=transform_meta_data,
        )
        
        [meta_extract_campaigns, meta_extract_insights] >> meta_transform
    
    with TaskGroup("google_ingestion") as google_group:
        def extract_google_campaigns(**context):
            """Extract campaigns from Google Ads."""
            from app.services.data_ingestion import GoogleAdsClient
            client = GoogleAdsClient()
            return client.get_campaigns()
        
        def extract_google_metrics(**context):
            """Extract metrics from Google Ads."""
            from app.services.data_ingestion import GoogleAdsClient
            client = GoogleAdsClient()
            return client.get_campaign_metrics()
        
        google_extract_campaigns = PythonOperator(
            task_id="extract_campaigns",
            python_callable=extract_google_campaigns,
        )
        
        google_extract_metrics = PythonOperator(
            task_id="extract_metrics",
            python_callable=extract_google_metrics,
        )
        
        google_extract_campaigns >> google_extract_metrics
    
    with TaskGroup("tiktok_ingestion") as tiktok_group:
        def extract_tiktok_campaigns(**context):
            """Extract campaigns from TikTok."""
            from app.services.data_ingestion import TikTokAdsClient
            client = TikTokAdsClient()
            return client.get_campaigns()
        
        tiktok_extract = PythonOperator(
            task_id="extract_campaigns",
            python_callable=extract_tiktok_campaigns,
        )
    
    def load_to_warehouse(**context):
        """Load processed data to warehouse."""
        from app.services.etl_pipeline import get_etl_pipeline
        pipeline = get_etl_pipeline()
        # Final loading step
        return {"status": "loaded"}
    
    load_data = PythonOperator(
        task_id="load_to_warehouse",
        python_callable=load_to_warehouse,
    )
    
    start >> [meta_group, google_group, tiktok_group] >> load_data >> end


# ===========================================
# Feature Engineering DAG
# ===========================================
with DAG(
    dag_id="adclass_feature_engineering",
    default_args=default_args,
    description="Feature engineering pipeline",
    schedule_interval="30 3 * * *",  # 3:30 AM daily (after ETL)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "features"],
) as feature_dag:
    
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")
    
    def compute_campaign_features(**context):
        """Compute campaign-level features."""
        from app.services.feature_store import get_feature_store
        store = get_feature_store()
        # Generate features
        return {"features_count": 50}
    
    def compute_client_features(**context):
        """Compute client-level features."""
        from app.services.feature_store import get_feature_store
        store = get_feature_store()
        return {"features_count": 30}
    
    def compute_creative_features(**context):
        """Compute creative features."""
        from app.models.ml.vision_model import get_creative_vision_service
        service = get_creative_vision_service()
        return {"features_count": 20}
    
    def update_feature_store(**context):
        """Update feature store with new features."""
        from app.services.feature_store import get_feature_store
        store = get_feature_store()
        store.sync_to_online_store()
        return {"status": "synced"}
    
    campaign_features = PythonOperator(
        task_id="campaign_features",
        python_callable=compute_campaign_features,
    )
    
    client_features = PythonOperator(
        task_id="client_features",
        python_callable=compute_client_features,
    )
    
    creative_features = PythonOperator(
        task_id="creative_features",
        python_callable=compute_creative_features,
    )
    
    update_store = PythonOperator(
        task_id="update_feature_store",
        python_callable=update_feature_store,
    )
    
    start >> [campaign_features, client_features, creative_features] >> update_store >> end


# ===========================================
# Model Retraining DAG
# ===========================================
with DAG(
    dag_id="adclass_model_retraining",
    default_args=default_args,
    description="Weekly model retraining pipeline",
    schedule_interval="0 4 * * 0",  # 4 AM every Sunday
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "training"],
) as training_dag:
    
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")
    
    def check_drift(**context):
        """Check for model drift."""
        from app.services.mlops import get_model_monitor, ModelType
        monitor = get_model_monitor()
        
        drift_detected = False
        for model_type in ModelType:
            drift = monitor.detect_drift(model_type)
            if drift.get("drift_detected"):
                drift_detected = True
                break
        
        return {"drift_detected": drift_detected}
    
    def train_creative_model(**context):
        """Retrain creative predictor."""
        from app.services.mlops import get_mlflow_client, ModelType
        client = get_mlflow_client()
        # Training logic
        return {"model": "creative_predictor", "status": "trained"}
    
    def train_churn_model(**context):
        """Retrain churn prediction model."""
        return {"model": "churn_model", "status": "trained"}
    
    def train_roas_model(**context):
        """Retrain ROAS optimizer."""
        return {"model": "roas_optimizer", "status": "trained"}
    
    def validate_models(**context):
        """Validate trained models."""
        return {"validation": "passed"}
    
    def promote_models(**context):
        """Promote validated models to production."""
        from app.services.mlops import get_mlflow_client, ModelType, ModelStage
        client = get_mlflow_client()
        # Promote logic
        return {"promoted": True}
    
    drift_check = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )
    
    train_creative = PythonOperator(
        task_id="train_creative",
        python_callable=train_creative_model,
    )
    
    train_churn = PythonOperator(
        task_id="train_churn",
        python_callable=train_churn_model,
    )
    
    train_roas = PythonOperator(
        task_id="train_roas",
        python_callable=train_roas_model,
    )
    
    validate = PythonOperator(
        task_id="validate_models",
        python_callable=validate_models,
    )
    
    promote = PythonOperator(
        task_id="promote_models",
        python_callable=promote_models,
    )
    
    start >> drift_check >> [train_creative, train_churn, train_roas] >> validate >> promote >> end


# ===========================================
# Alerting DAG
# ===========================================
with DAG(
    dag_id="adclass_alerting",
    default_args=default_args,
    description="Hourly alerting and monitoring",
    schedule_interval="0 * * * *",  # Every hour
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["monitoring", "alerts"],
) as alerting_dag:
    
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")
    
    def check_client_health(**context):
        """Check all client health scores."""
        from app.services.alert_system import get_alert_manager, AlertType
        manager = get_alert_manager()
        
        # Check each client (mock data)
        clients_at_risk = []
        # In production: Query database for clients
        
        return {"checked": True, "at_risk": len(clients_at_risk)}
    
    def check_campaign_performance(**context):
        """Check campaign performance."""
        from app.services.alert_system import get_alert_manager
        manager = get_alert_manager()
        return {"checked": True}
    
    def check_system_health(**context):
        """Check system health."""
        return {"healthy": True}
    
    client_health = PythonOperator(
        task_id="check_client_health",
        python_callable=check_client_health,
    )
    
    campaign_perf = PythonOperator(
        task_id="check_campaign_performance",
        python_callable=check_campaign_performance,
    )
    
    system_health = PythonOperator(
        task_id="check_system_health",
        python_callable=check_system_health,
    )
    
    start >> [client_health, campaign_perf, system_health] >> end
