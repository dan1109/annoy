#!/usr/bin/env python3
"""
OCI Data Science Job Monitoring and Cancellation Script

This script demonstrates the correct approach for monitoring and canceling
OCI Data Science Jobs when specific metric thresholds are reached.

The key principle: You must monitor the job externally and use the OCI SDK
to cancel the job, as you cannot cancel a job from within the job itself.
"""

import time
import logging
import json
import os
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import oci
from oci.data_science import DataScienceClient
from oci.data_science.models import Job, JobRun
from oci.logging import LoggingManagementClient
from oci.logging_search import LogSearchClient


@dataclass
class MetricThreshold:
    """Configuration for metric-based job cancellation"""
    metric_name: str
    threshold_value: float
    comparison: str  # 'greater_than', 'less_than', 'equals'
    consecutive_checks: int = 3  # Number of consecutive checks before canceling


@dataclass
class JobMonitorConfig:
    """Configuration for job monitoring"""
    compartment_id: str
    job_id: str
    job_run_id: Optional[str] = None
    check_interval: int = 30  # seconds
    max_monitoring_time: int = 3600  # seconds
    log_group_id: Optional[str] = None
    log_id: Optional[str] = None


class OCIJobMonitor:
    """
    External monitor for OCI Data Science Jobs that can cancel jobs
    based on metric thresholds.
    """
    
    def __init__(self, config: JobMonitorConfig, metric_thresholds: list[MetricThreshold]):
        self.config = config
        self.metric_thresholds = metric_thresholds
        
        # Initialize OCI clients
        self.oci_config = oci.config.from_file()
        self.ds_client = DataScienceClient(self.oci_config)
        self.log_client = LogSearchClient(self.oci_config)
        
        # Tracking variables
        self.consecutive_threshold_hits = {threshold.metric_name: 0 
                                         for threshold in metric_thresholds}
        self.start_time = datetime.now()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_job_run_status(self) -> Optional[str]:
        """Get the current status of the job run"""
        try:
            if self.config.job_run_id:
                response = self.ds_client.get_job_run(self.config.job_run_id)
                return response.data.lifecycle_state
            return None
        except Exception as e:
            self.logger.error(f"Error getting job run status: {e}")
            return None
    
    def get_latest_job_run(self) -> Optional[str]:
        """Get the latest job run ID for the job"""
        try:
            response = self.ds_client.list_job_runs(
                compartment_id=self.config.compartment_id,
                job_id=self.config.job_id,
                sort_order="DESC",
                sort_by="timeCreated",
                limit=1
            )
            
            if response.data:
                job_run_id = response.data[0].id
                self.config.job_run_id = job_run_id
                return job_run_id
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest job run: {e}")
            return None
    
    def extract_metrics_from_logs(self) -> Dict[str, float]:
        """
        Extract metrics from job logs using OCI Logging Search.
        This assumes metrics are logged in JSON format.
        """
        metrics = {}
        
        if not self.config.log_group_id or not self.config.log_id:
            self.logger.warning("Log configuration not provided, skipping log-based metrics")
            return metrics
        
        try:
            # Search for metric logs in the last check interval
            end_time = datetime.now()
            start_time = end_time - timedelta(seconds=self.config.check_interval + 10)
            
            search_query = f"""
            search "{self.config.log_group_id}/{self.config.log_id}" 
            | where data.message like "%METRIC%"
            | sort by datetime desc
            | limit 100
            """
            
            response = self.log_client.search_logs(
                search_logs_details=oci.logging_search.models.SearchLogsDetails(
                    time_start=start_time,
                    time_end=end_time,
                    search_query=search_query,
                    is_return_field_info=False
                )
            )
            
            # Parse metrics from log entries
            for log_entry in response.data.results:
                try:
                    message = log_entry.data.get('message', '')
                    if 'METRIC:' in message:
                        # Expected format: "METRIC: {'accuracy': 0.95, 'loss': 0.05}"
                        metric_str = message.split('METRIC:')[1].strip()
                        metric_data = json.loads(metric_str.replace("'", '"'))
                        metrics.update(metric_data)
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    self.logger.debug(f"Could not parse metric from log: {message}, error: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error extracting metrics from logs: {e}")
        
        return metrics
    
    def check_metric_thresholds(self, metrics: Dict[str, float]) -> bool:
        """
        Check if any metric thresholds have been exceeded.
        Returns True if a job should be canceled.
        """
        should_cancel = False
        
        for threshold in self.metric_thresholds:
            if threshold.metric_name not in metrics:
                continue
                
            metric_value = metrics[threshold.metric_name]
            threshold_exceeded = False
            
            if threshold.comparison == 'greater_than':
                threshold_exceeded = metric_value > threshold.threshold_value
            elif threshold.comparison == 'less_than':
                threshold_exceeded = metric_value < threshold.threshold_value
            elif threshold.comparison == 'equals':
                threshold_exceeded = abs(metric_value - threshold.threshold_value) < 1e-6
            
            if threshold_exceeded:
                self.consecutive_threshold_hits[threshold.metric_name] += 1
                self.logger.info(
                    f"Threshold exceeded for {threshold.metric_name}: "
                    f"{metric_value} {threshold.comparison} {threshold.threshold_value} "
                    f"(hit {self.consecutive_threshold_hits[threshold.metric_name]}/{threshold.consecutive_checks})"
                )
                
                if self.consecutive_threshold_hits[threshold.metric_name] >= threshold.consecutive_checks:
                    self.logger.warning(
                        f"Metric {threshold.metric_name} exceeded threshold {threshold.consecutive_checks} "
                        f"consecutive times. Marking job for cancellation."
                    )
                    should_cancel = True
            else:
                # Reset consecutive hits if threshold not exceeded
                self.consecutive_threshold_hits[threshold.metric_name] = 0
        
        return should_cancel
    
    def cancel_job_run(self) -> bool:
        """Cancel the current job run using OCI SDK"""
        if not self.config.job_run_id:
            self.logger.error("No job run ID available for cancellation")
            return False
        
        try:
            self.logger.info(f"Canceling job run: {self.config.job_run_id}")
            
            # Cancel the job run
            self.ds_client.cancel_job_run(self.config.job_run_id)
            
            self.logger.info(f"Successfully sent cancellation request for job run: {self.config.job_run_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error canceling job run: {e}")
            return False
    
    def monitor_job(self) -> Dict[str, Any]:
        """
        Main monitoring loop that checks job status and metrics,
        canceling the job if thresholds are exceeded.
        """
        self.logger.info(f"Starting job monitoring for job: {self.config.job_id}")
        
        # Get initial job run if not provided
        if not self.config.job_run_id:
            self.config.job_run_id = self.get_latest_job_run()
            if not self.config.job_run_id:
                return {"status": "error", "message": "No job run found to monitor"}
        
        monitoring_result = {
            "status": "completed",
            "job_run_id": self.config.job_run_id,
            "canceled": False,
            "final_metrics": {},
            "monitoring_duration": 0
        }
        
        while True:
            current_time = datetime.now()
            elapsed_time = (current_time - self.start_time).total_seconds()
            
            # Check maximum monitoring time
            if elapsed_time > self.config.max_monitoring_time:
                self.logger.warning("Maximum monitoring time exceeded")
                monitoring_result["status"] = "timeout"
                break
            
            # Check job run status
            job_status = self.get_job_run_status()
            self.logger.info(f"Job run status: {job_status}")
            
            if job_status in ['SUCCEEDED', 'FAILED', 'CANCELED']:
                self.logger.info(f"Job run completed with status: {job_status}")
                monitoring_result["status"] = job_status.lower()
                break
            
            # Extract and check metrics
            metrics = self.extract_metrics_from_logs()
            monitoring_result["final_metrics"] = metrics
            
            if metrics:
                self.logger.info(f"Current metrics: {metrics}")
                
                # Check if thresholds are exceeded
                if self.check_metric_thresholds(metrics):
                    self.logger.warning("Metric thresholds exceeded, canceling job")
                    
                    if self.cancel_job_run():
                        monitoring_result["canceled"] = True
                        monitoring_result["status"] = "canceled_by_monitor"
                        break
                    else:
                        self.logger.error("Failed to cancel job run")
            
            # Wait before next check
            time.sleep(self.config.check_interval)
        
        monitoring_result["monitoring_duration"] = elapsed_time
        return monitoring_result


def create_sample_monitoring_config() -> tuple[JobMonitorConfig, list[MetricThreshold]]:
    """Create sample configuration for demonstration"""
    
    config = JobMonitorConfig(
        compartment_id="ocid1.compartment.oc1..example",
        job_id="ocid1.datasciencejob.oc1..example",
        check_interval=30,
        max_monitoring_time=3600,
        log_group_id="ocid1.loggroup.oc1..example",
        log_id="ocid1.log.oc1..example"
    )
    
    thresholds = [
        MetricThreshold(
            metric_name="accuracy",
            threshold_value=0.95,
            comparison="greater_than",
            consecutive_checks=3
        ),
        MetricThreshold(
            metric_name="loss",
            threshold_value=0.01,
            comparison="less_than",
            consecutive_checks=2
        )
    ]
    
    return config, thresholds


def main():
    """Example usage of the job monitor"""
    
    # Create sample configuration
    config, thresholds = create_sample_monitoring_config()
    
    # Override with environment variables if available
    config.compartment_id = os.getenv('OCI_COMPARTMENT_ID', config.compartment_id)
    config.job_id = os.getenv('OCI_JOB_ID', config.job_id)
    config.job_run_id = os.getenv('OCI_JOB_RUN_ID', config.job_run_id)
    
    # Create and run monitor
    monitor = OCIJobMonitor(config, thresholds)
    result = monitor.monitor_job()
    
    print("\n" + "="*50)
    print("MONITORING RESULT:")
    print("="*50)
    print(json.dumps(result, indent=2, default=str))
    
    if result["canceled"]:
        print("\n✅ Job was successfully canceled based on metric thresholds!")
    else:
        print(f"\n📊 Job completed naturally with status: {result['status']}")


if __name__ == "__main__":
    main()