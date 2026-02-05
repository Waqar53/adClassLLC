"""
ETL Pipeline Service

Comprehensive data extraction, transformation, and loading for ad platform data.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import hashlib


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataSource(str, Enum):
    META = "meta"
    GOOGLE = "google"
    TIKTOK = "tiktok"
    CRM = "crm"
    EMAIL = "email"
    ANALYTICS = "analytics"


@dataclass
class PipelineTask:
    """A single ETL task."""
    task_id: str
    name: str
    source: DataSource
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    records_processed: int = 0
    error_message: Optional[str] = None


@dataclass
class PipelineRun:
    """A complete pipeline execution run."""
    run_id: str
    pipeline_name: str
    status: PipelineStatus = PipelineStatus.PENDING
    tasks: List[PipelineTask] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_records: int = 0


class DataValidator:
    """
    Data quality validation for ETL pipelines.
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
    
    def add_rule(self, table: str, rule: Callable):
        """Add validation rule for a table."""
        if table not in self.validation_rules:
            self.validation_rules[table] = []
        self.validation_rules[table].append(rule)
    
    def validate(self, table: str, data: List[Dict]) -> Dict[str, Any]:
        """
        Validate data against rules.
        
        Returns validation results with any failures.
        """
        results = {
            "valid": True,
            "total_records": len(data),
            "valid_records": 0,
            "invalid_records": 0,
            "errors": []
        }
        
        rules = self.validation_rules.get(table, [])
        
        for i, record in enumerate(data):
            record_valid = True
            for rule in rules:
                try:
                    if not rule(record):
                        record_valid = False
                        results["errors"].append({
                            "record_index": i,
                            "rule": rule.__name__,
                            "record": record
                        })
                except Exception as e:
                    record_valid = False
                    results["errors"].append({
                        "record_index": i,
                        "error": str(e)
                    })
            
            if record_valid:
                results["valid_records"] += 1
            else:
                results["invalid_records"] += 1
        
        results["valid"] = results["invalid_records"] == 0
        return results


class DataTransformer:
    """
    Data transformation utilities.
    """
    
    @staticmethod
    def normalize_campaign_data(raw_data: Dict, source: DataSource) -> Dict:
        """Normalize campaign data from different sources to standard format."""
        if source == DataSource.META:
            return {
                "external_id": raw_data.get("campaign_id") or raw_data.get("id"),
                "name": raw_data.get("name", ""),
                "platform": "meta",
                "status": raw_data.get("status", "").lower(),
                "objective": raw_data.get("objective", ""),
                "daily_budget": float(raw_data.get("daily_budget", 0)) / 100,  # cents to dollars
                "lifetime_budget": float(raw_data.get("lifetime_budget", 0)) / 100 if raw_data.get("lifetime_budget") else None,
                "start_date": raw_data.get("start_time"),
                "end_date": raw_data.get("stop_time"),
            }
        elif source == DataSource.GOOGLE:
            return {
                "external_id": raw_data.get("campaign_id") or raw_data.get("id"),
                "name": raw_data.get("name", ""),
                "platform": "google",
                "status": raw_data.get("status", "").lower(),
                "objective": raw_data.get("advertising_channel_type", ""),
                "daily_budget": float(raw_data.get("budget_amount", 0)),
                "lifetime_budget": None,
                "start_date": raw_data.get("start_date"),
                "end_date": raw_data.get("end_date"),
            }
        elif source == DataSource.TIKTOK:
            return {
                "external_id": raw_data.get("campaign_id") or raw_data.get("id"),
                "name": raw_data.get("campaign_name", ""),
                "platform": "tiktok",
                "status": raw_data.get("status", "").lower(),
                "objective": raw_data.get("objective_type", ""),
                "daily_budget": float(raw_data.get("budget", 0)) if raw_data.get("budget_mode") == "BUDGET_MODE_DAY" else None,
                "lifetime_budget": float(raw_data.get("budget", 0)) if raw_data.get("budget_mode") == "BUDGET_MODE_TOTAL" else None,
                "start_date": None,
                "end_date": None,
            }
        return raw_data
    
    @staticmethod
    def normalize_metrics(raw_data: Dict, source: DataSource) -> Dict:
        """Normalize performance metrics from different sources."""
        if source == DataSource.META:
            return {
                "impressions": int(raw_data.get("impressions", 0)),
                "reach": int(raw_data.get("reach", 0)),
                "clicks": int(raw_data.get("clicks", 0)),
                "spend": float(raw_data.get("spend", 0)),
                "conversions": int(raw_data.get("actions", [{}])[0].get("value", 0)) if raw_data.get("actions") else 0,
                "conversion_value": float(raw_data.get("action_values", [{}])[0].get("value", 0)) if raw_data.get("action_values") else 0,
                "ctr": float(raw_data.get("ctr", 0)),
                "cpc": float(raw_data.get("cpc", 0)),
                "cpm": float(raw_data.get("cpm", 0)),
            }
        elif source == DataSource.GOOGLE:
            return {
                "impressions": int(raw_data.get("impressions", 0)),
                "reach": 0,  # Google doesn't provide reach
                "clicks": int(raw_data.get("clicks", 0)),
                "spend": float(raw_data.get("cost_micros", 0)) / 1_000_000,
                "conversions": float(raw_data.get("conversions", 0)),
                "conversion_value": float(raw_data.get("conversions_value", 0)),
                "ctr": float(raw_data.get("ctr", 0)),
                "cpc": float(raw_data.get("average_cpc", 0)) / 1_000_000,
                "cpm": float(raw_data.get("average_cpm", 0)) / 1_000_000,
            }
        elif source == DataSource.TIKTOK:
            return {
                "impressions": int(raw_data.get("impressions", 0)),
                "reach": int(raw_data.get("reach", 0)),
                "clicks": int(raw_data.get("clicks", 0)),
                "spend": float(raw_data.get("spend", 0)),
                "conversions": int(raw_data.get("conversions", 0)),
                "conversion_value": 0,  # Calculate separately
                "ctr": float(raw_data.get("ctr", 0)),
                "cpc": float(raw_data.get("cpc", 0)),
                "cpm": float(raw_data.get("cpm", 0)),
            }
        return raw_data
    
    @staticmethod
    def calculate_derived_metrics(metrics: Dict) -> Dict:
        """Calculate derived metrics from base metrics."""
        metrics["roas"] = metrics["conversion_value"] / max(metrics["spend"], 0.01)
        metrics["cvr"] = metrics["conversions"] / max(metrics["clicks"], 1) * 100
        metrics["cpa"] = metrics["spend"] / max(metrics["conversions"], 1)
        return metrics
    
    def transform(self, data: List[Dict], source: DataSource = None) -> List[Dict]:
        """
        Generic transform method for data normalization.
        
        Args:
            data: List of records to transform
            source: Optional data source for source-specific transformations
        
        Returns:
            Transformed data
        """
        result = []
        for record in data:
            if source and "spend" in record:
                # Normalize metrics
                normalized = self.normalize_metrics(record, source)
                normalized = self.calculate_derived_metrics(normalized)
                result.append(normalized)
            elif source:
                # Normalize campaign data
                result.append(self.normalize_campaign_data(record, source))
            else:
                result.append(record)
        return result
    
    @staticmethod
    def deduplicate(data: List[Dict], key_fields: List[str]) -> List[Dict]:
        """Remove duplicate records based on key fields."""
        seen = set()
        result = []
        
        for record in data:
            key = tuple(record.get(f) for f in key_fields)
            if key not in seen:
                seen.add(key)
                result.append(record)
        
        return result


class Deduplicator:
    """
    Data deduplication utility.
    """
    
    def __init__(self):
        self.seen_keys: set = set()
    
    def deduplicate(self, data: List[Dict], key_fields: List[str]) -> List[Dict]:
        """Remove duplicate records based on key fields."""
        self.seen_keys = set()
        result = []
        
        for record in data:
            # Create hashable key from fields
            key_parts = []
            for f in key_fields:
                val = record.get(f)
                if isinstance(val, (list, dict)):
                    key_parts.append(str(val))
                else:
                    key_parts.append(val)
            key = tuple(key_parts)
            
            if key not in self.seen_keys:
                self.seen_keys.add(key)
                result.append(record)
        
        return result
    
    def reset(self):
        """Reset seen keys."""
        self.seen_keys = set()


class ETLPipeline:
    """
    Main ETL pipeline orchestrator.
    """
    
    def __init__(self):
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.deduplicator = Deduplicator()
        self.runs: Dict[str, PipelineRun] = {}
        self._setup_validation_rules()
    
    def _setup_validation_rules(self):
        """Set up default validation rules."""
        # Campaign validation
        self.validator.add_rule("campaigns", lambda r: r.get("external_id") is not None)
        self.validator.add_rule("campaigns", lambda r: r.get("name") is not None)
        self.validator.add_rule("campaigns", lambda r: r.get("platform") in ["meta", "google", "tiktok"])
        
        # Metrics validation
        self.validator.add_rule("metrics", lambda r: r.get("impressions", -1) >= 0)
        self.validator.add_rule("metrics", lambda r: r.get("clicks", -1) >= 0)
        self.validator.add_rule("metrics", lambda r: r.get("spend", -1) >= 0)
    
    def create_run(self, pipeline_name: str, sources: List[DataSource]) -> PipelineRun:
        """Create a new pipeline run."""
        run_id = hashlib.md5(f"{pipeline_name}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        tasks = [
            PipelineTask(
                task_id=f"{run_id}_{source.value}",
                name=f"Extract {source.value}",
                source=source
            )
            for source in sources
        ]
        
        run = PipelineRun(
            run_id=run_id,
            pipeline_name=pipeline_name,
            tasks=tasks
        )
        
        self.runs[run_id] = run
        return run
    
    async def run_pipeline(
        self,
        run: PipelineRun,
        extractors: Dict[DataSource, Callable],
        loaders: Dict[str, Callable]
    ) -> PipelineRun:
        """
        Execute the full ETL pipeline.
        
        Args:
            run: Pipeline run configuration
            extractors: Functions to extract data from each source
            loaders: Functions to load data to destinations
        """
        run.status = PipelineStatus.RUNNING
        run.started_at = datetime.now()
        
        try:
            all_campaigns = []
            all_metrics = []
            
            # Extract phase
            for task in run.tasks:
                task.status = PipelineStatus.RUNNING
                task.started_at = datetime.now()
                
                try:
                    extractor = extractors.get(task.source)
                    if extractor:
                        raw_data = await extractor()
                        
                        # Transform
                        campaigns = [
                            self.transformer.normalize_campaign_data(c, task.source)
                            for c in raw_data.get("campaigns", [])
                        ]
                        metrics = [
                            self.transformer.calculate_derived_metrics(
                                self.transformer.normalize_metrics(m, task.source)
                            )
                            for m in raw_data.get("metrics", [])
                        ]
                        
                        all_campaigns.extend(campaigns)
                        all_metrics.extend(metrics)
                        
                        task.records_processed = len(campaigns) + len(metrics)
                        task.status = PipelineStatus.COMPLETED
                except Exception as e:
                    task.status = PipelineStatus.FAILED
                    task.error_message = str(e)
                
                task.completed_at = datetime.now()
            
            # Deduplicate
            all_campaigns = self.transformer.deduplicate(all_campaigns, ["external_id", "platform"])
            
            # Validate
            campaign_validation = self.validator.validate("campaigns", all_campaigns)
            metrics_validation = self.validator.validate("metrics", all_metrics)
            
            # Load phase
            if loaders.get("campaigns"):
                await loaders["campaigns"](all_campaigns)
            if loaders.get("metrics"):
                await loaders["metrics"](all_metrics)
            
            run.total_records = len(all_campaigns) + len(all_metrics)
            run.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            run.status = PipelineStatus.FAILED
        
        run.completed_at = datetime.now()
        return run
    
    def get_run_status(self, run_id: str) -> Optional[PipelineRun]:
        """Get status of a pipeline run."""
        return self.runs.get(run_id)


class IncrementalLoader:
    """
    Incremental data loading with change detection.
    """
    
    def __init__(self):
        self.last_sync_times: Dict[str, datetime] = {}
        self.checksums: Dict[str, str] = {}
    
    def get_last_sync_time(self, source: str) -> Optional[datetime]:
        """Get last successful sync time for a source."""
        return self.last_sync_times.get(source)
    
    def update_sync_time(self, source: str, sync_time: datetime):
        """Update last sync time after successful load."""
        self.last_sync_times[source] = sync_time
    
    def has_changed(self, record_id: str, data: Dict) -> bool:
        """Check if a record has changed since last sync."""
        new_checksum = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        old_checksum = self.checksums.get(record_id)
        
        if new_checksum != old_checksum:
            self.checksums[record_id] = new_checksum
            return True
        return False
    
    def get_changes_only(self, records: List[Dict], id_field: str = "id") -> List[Dict]:
        """Filter to only changed records."""
        return [r for r in records if self.has_changed(r.get(id_field, ""), r)]


# Singleton
_etl_pipeline: Optional[ETLPipeline] = None


def get_etl_pipeline() -> ETLPipeline:
    """Get or create ETL pipeline instance."""
    global _etl_pipeline
    if _etl_pipeline is None:
        _etl_pipeline = ETLPipeline()
    return _etl_pipeline
