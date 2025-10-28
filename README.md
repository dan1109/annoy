# OCI Data Science Jobs - Monitoring and Cancellation

This repository demonstrates the **correct approach** for monitoring and canceling Oracle Cloud Infrastructure (OCI) Data Science Jobs based on metric thresholds.

## 🎯 Key Concept

**You cannot cancel a Data Science Job from within the job itself.** The correct approach is:

✅ **External monitoring process** uses OCI SDK to cancel jobs  
✅ Monitor watches job metrics via logs or external systems  
✅ Job code handles cancellation gracefully via signal handlers  
✅ Jobs naturally exit when code completes or `exit()` is called  

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Training Job   │    │ External Monitor │    │   OCI SDK      │
│                 │    │                  │    │                 │
│ 1. Log metrics  │───▶│ 2. Parse metrics │    │ 4. Cancel job   │
│ 3. Handle       │◀───│ 3. Check thresh. │───▶│    run          │
│    cancellation │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Files Overview

### Core Components

- **`oci_job_monitor.py`** - External monitoring script that watches job metrics and cancels jobs when thresholds are exceeded
- **`model_training_job.py`** - Example training job that logs metrics in a structured format and handles graceful shutdown
- **`job_management_utils.py`** - Utilities for creating, starting, and managing OCI Data Science Jobs

### Configuration

- **`requirements.txt`** - Python dependencies
- **`config_examples/`** - Sample configuration files

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OCI

Ensure you have OCI CLI configured or OCI config file at `~/.oci/config`:

```bash
oci setup config
```

### 3. Set Environment Variables

```bash
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your-compartment-id"
export OCI_PROJECT_ID="ocid1.datascienceproject.oc1..your-project-id"
export OCI_LOG_GROUP_ID="ocid1.loggroup.oc1..your-log-group-id"
export OCI_LOG_ID="ocid1.log.oc1..your-log-id"
```

### 4. Create and Monitor a Job

```python
from job_management_utils import OCIJobManager
from oci_job_monitor import OCIJobMonitor, JobMonitorConfig, MetricThreshold

# Create job manager
job_manager = OCIJobManager(compartment_id, project_id)

# Create and start job
job_id = job_manager.create_training_job("my-training-job", "model_training_job.py")
job_run_id = job_manager.create_job_run(job_id)

# Setup monitoring with thresholds
config = JobMonitorConfig(
    compartment_id=compartment_id,
    job_id=job_id,
    job_run_id=job_run_id,
    check_interval=30
)

thresholds = [
    MetricThreshold("accuracy", 0.95, "greater_than", consecutive_checks=3),
    MetricThreshold("loss", 0.01, "less_than", consecutive_checks=2)
]

# Start monitoring (this will cancel the job if thresholds are met)
monitor = OCIJobMonitor(config, thresholds)
result = monitor.monitor_job()
```

## 📊 Metric Logging Format

Training jobs should log metrics in this structured format for external monitoring:

```python
import json
import logging

def log_metric(metrics: dict):
    metric_str = json.dumps(metrics)
    logging.info(f"METRIC: {metric_str}")

# Example usage
log_metric({
    "epoch": 10,
    "accuracy": 0.94,
    "loss": 0.06,
    "timestamp": "2024-01-15T10:30:00Z"
})
```

## 🛡️ Graceful Shutdown Handling

Training jobs should handle cancellation signals properly:

```python
import signal
import sys

def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    # Save model checkpoint
    # Clean up resources
    sys.exit(1)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

## ⚙️ Configuration Examples

### Monitor Configuration

```python
from oci_job_monitor import JobMonitorConfig, MetricThreshold

config = JobMonitorConfig(
    compartment_id="ocid1.compartment.oc1..example",
    job_id="ocid1.datasciencejob.oc1..example",
    job_run_id="ocid1.datasciencejobrun.oc1..example",
    check_interval=30,  # seconds
    max_monitoring_time=3600,  # 1 hour
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
```

## 🔍 Monitoring Workflow

1. **Job Execution**: Training job runs and logs structured metrics
2. **External Monitoring**: Monitor script polls job logs for metrics
3. **Threshold Checking**: Compare metrics against configured thresholds
4. **Cancellation Decision**: Cancel job if thresholds exceeded consecutively
5. **Graceful Shutdown**: Job receives cancellation signal and shuts down cleanly

## ❌ Common Mistakes to Avoid

### ❌ Incorrect: Self-Cancellation
```python
# This WILL NOT WORK - jobs cannot cancel themselves
from oci.data_science import DataScienceClient

def train_model():
    if accuracy > 0.95:
        ds_client.cancel_job_run(current_job_run_id)  # ❌ Won't work
```

### ❌ Incorrect: Internal SDK Usage
```python
# This WILL NOT WORK - no access to job run context from within job
import oci

def check_and_cancel():
    job_run.cancel()  # ❌ No such method available internally
```

### ✅ Correct: External Monitoring
```python
# This WORKS - external process monitors and cancels
monitor = OCIJobMonitor(config, thresholds)
result = monitor.monitor_job()  # ✅ Proper external monitoring
```

## 🧪 Testing

Run the example scripts to test the monitoring system:

```bash
# Test job creation and management
python job_management_utils.py

# Test monitoring (requires running job)
python oci_job_monitor.py

# Test training job locally
python model_training_job.py
```

## 📚 Additional Resources

- [OCI Data Science Documentation](https://docs.oracle.com/en-us/iaas/data-science/using/overview.htm)
- [OCI Python SDK Documentation](https://docs.oracle.com/en-us/iaas/tools/python/latest/)
- [OCI Data Science Jobs Guide](https://docs.oracle.com/en-us/iaas/data-science/using/jobs.htm)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.