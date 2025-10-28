#!/usr/bin/env python3
"""
OCI Data Science Job Management Utilities

This module provides utilities for creating, starting, and managing
OCI Data Science Jobs with proper monitoring and cancellation capabilities.

Key Concepts Demonstrated:
1. Jobs cannot cancel themselves from within the job code
2. External monitoring is required to cancel jobs based on metrics
3. Proper job configuration for monitoring and logging
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import base64

import oci
from oci.data_science import DataScienceClient
from oci.data_science.models import (
    CreateJobDetails, CreateJobRunDetails, Job, JobRun,
    JobInfrastructureConfigurationDetails, JobConfigurationDetails,
    JobLogConfigurationDetails, JobShapeConfigDetails
)


class OCIJobManager:
    """
    Utility class for managing OCI Data Science Jobs with monitoring capabilities.
    """
    
    def __init__(self, compartment_id: str, project_id: str):
        self.compartment_id = compartment_id
        self.project_id = project_id
        
        # Initialize OCI client
        self.oci_config = oci.config.from_file()
        self.ds_client = DataScienceClient(self.oci_config)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_training_job(self, 
                          job_name: str,
                          training_script_path: str,
                          shape: str = "VM.Standard2.1",
                          block_storage_size_gb: int = 50,
                          log_group_id: Optional[str] = None,
                          log_id: Optional[str] = None,
                          environment_variables: Optional[Dict[str, str]] = None) -> str:
        """
        Create a Data Science Job for model training with proper logging configuration.
        
        Returns the job OCID.
        """
        
        # Prepare job infrastructure configuration
        infrastructure_config = JobInfrastructureConfigurationDetails(
            job_infrastructure_type="STANDALONE",
            shape_name=shape,
            block_storage_size_in_gbs=block_storage_size_gb,
            shape_config_details=JobShapeConfigDetails(
                ocpus=1.0,
                memory_in_gbs=15.0
            ) if "Flex" in shape else None
        )
        
        # Prepare environment variables
        env_vars = environment_variables or {}
        env_vars.update({
            "OCI_RESOURCE_PRINCIPAL_VERSION": "2.2",
            "CONDA_ENV_TYPE": "service",
            "CONDA_ENV_SLUG": "pytorch110_p38_cpu_v1"  # Example conda environment
        })
        
        # Prepare job configuration
        job_config = JobConfigurationDetails(
            job_type="DEFAULT",
            environment_variables=env_vars,
            command_line_arguments=[],
            maximum_runtime_in_minutes=360  # 6 hours max
        )
        
        # Setup logging configuration if provided
        log_config = None
        if log_group_id and log_id:
            log_config = JobLogConfigurationDetails(
                enable_logging=True,
                enable_auto_log_creation=False,
                log_group_id=log_group_id,
                log_id=log_id
            )
        
        # Create job details
        job_details = CreateJobDetails(
            compartment_id=self.compartment_id,
            project_id=self.project_id,
            display_name=job_name,
            description=f"Model training job: {job_name}",
            job_configuration_details=job_config,
            job_infrastructure_configuration_details=infrastructure_config,
            job_log_configuration_details=log_config
        )
        
        try:
            self.logger.info(f"Creating job: {job_name}")
            response = self.ds_client.create_job(job_details)
            job_id = response.data.id
            
            self.logger.info(f"Job created successfully: {job_id}")
            return job_id
            
        except Exception as e:
            self.logger.error(f"Error creating job: {e}")
            raise
    
    def create_job_run(self, 
                      job_id: str,
                      run_name: Optional[str] = None,
                      override_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create and start a job run.
        
        Returns the job run OCID.
        """
        
        display_name = run_name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Prepare job run configuration overrides
        job_config_override = {}
        if override_config:
            job_config_override = JobConfigurationDetails(
                job_type="DEFAULT",
                environment_variables=override_config.get("environment_variables", {}),
                command_line_arguments=override_config.get("command_line_arguments", []),
                maximum_runtime_in_minutes=override_config.get("maximum_runtime_minutes", 360)
            )
        
        job_run_details = CreateJobRunDetails(
            compartment_id=self.compartment_id,
            project_id=self.project_id,
            job_id=job_id,
            display_name=display_name,
            job_configuration_override_details=job_config_override if override_config else None
        )
        
        try:
            self.logger.info(f"Starting job run: {display_name}")
            response = self.ds_client.create_job_run(job_run_details)
            job_run_id = response.data.id
            
            self.logger.info(f"Job run started successfully: {job_run_id}")
            return job_run_id
            
        except Exception as e:
            self.logger.error(f"Error starting job run: {e}")
            raise
    
    def get_job_run_status(self, job_run_id: str) -> Dict[str, Any]:
        """Get detailed status of a job run"""
        try:
            response = self.ds_client.get_job_run(job_run_id)
            job_run = response.data
            
            return {
                "id": job_run.id,
                "display_name": job_run.display_name,
                "lifecycle_state": job_run.lifecycle_state,
                "lifecycle_details": job_run.lifecycle_details,
                "time_started": job_run.time_started,
                "time_finished": job_run.time_finished,
                "created_by": job_run.created_by
            }
            
        except Exception as e:
            self.logger.error(f"Error getting job run status: {e}")
            raise
    
    def cancel_job_run(self, job_run_id: str) -> bool:
        """
        Cancel a running job run.
        
        This is the key method that demonstrates the correct approach:
        External processes use the OCI SDK to cancel jobs.
        """
        try:
            self.logger.info(f"Canceling job run: {job_run_id}")
            
            # Check current status first
            status = self.get_job_run_status(job_run_id)
            current_state = status["lifecycle_state"]
            
            if current_state not in ["IN_PROGRESS", "ACCEPTED"]:
                self.logger.warning(f"Job run is in state {current_state}, cannot cancel")
                return False
            
            # Cancel the job run
            self.ds_client.cancel_job_run(job_run_id)
            
            self.logger.info(f"Cancellation request sent for job run: {job_run_id}")
            
            # Wait a moment and check if cancellation was successful
            time.sleep(5)
            updated_status = self.get_job_run_status(job_run_id)
            
            if updated_status["lifecycle_state"] in ["CANCELING", "CANCELED"]:
                self.logger.info("Job run cancellation confirmed")
                return True
            else:
                self.logger.warning("Job run cancellation status unclear")
                return False
                
        except Exception as e:
            self.logger.error(f"Error canceling job run: {e}")
            return False
    
    def list_job_runs(self, job_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent job runs"""
        try:
            response = self.ds_client.list_job_runs(
                compartment_id=self.compartment_id,
                job_id=job_id,
                sort_order="DESC",
                sort_by="timeCreated",
                limit=limit
            )
            
            job_runs = []
            for job_run in response.data:
                job_runs.append({
                    "id": job_run.id,
                    "display_name": job_run.display_name,
                    "job_id": job_run.job_id,
                    "lifecycle_state": job_run.lifecycle_state,
                    "time_created": job_run.time_created,
                    "created_by": job_run.created_by
                })
            
            return job_runs
            
        except Exception as e:
            self.logger.error(f"Error listing job runs: {e}")
            raise
    
    def wait_for_job_run_completion(self, 
                                  job_run_id: str, 
                                  check_interval: int = 30,
                                  max_wait_time: int = 3600) -> str:
        """
        Wait for a job run to complete (for testing purposes).
        
        In production, you would use the external monitor instead.
        """
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_wait_time:
                raise TimeoutError(f"Job run did not complete within {max_wait_time} seconds")
            
            status = self.get_job_run_status(job_run_id)
            state = status["lifecycle_state"]
            
            self.logger.info(f"Job run {job_run_id} status: {state}")
            
            if state in ["SUCCEEDED", "FAILED", "CANCELED"]:
                return state
            
            time.sleep(check_interval)


def create_sample_training_job_with_monitoring():
    """
    Example function showing how to create a complete training job
    with monitoring capabilities.
    """
    
    # Configuration (replace with actual values)
    config = {
        "compartment_id": os.getenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example"),
        "project_id": os.getenv("OCI_PROJECT_ID", "ocid1.datascienceproject.oc1..example"),
        "log_group_id": os.getenv("OCI_LOG_GROUP_ID", "ocid1.loggroup.oc1..example"),
        "log_id": os.getenv("OCI_LOG_ID", "ocid1.log.oc1..example")
    }
    
    # Create job manager
    job_manager = OCIJobManager(config["compartment_id"], config["project_id"])
    
    # Create the job
    job_id = job_manager.create_training_job(
        job_name="model-training-with-monitoring",
        training_script_path="model_training_job.py",
        shape="VM.Standard2.1",
        log_group_id=config["log_group_id"],
        log_id=config["log_id"],
        environment_variables={
            "TRAINING_DATA_PATH": "/opt/ml/input/data",
            "MODEL_OUTPUT_PATH": "/opt/ml/model"
        }
    )
    
    # Start a job run
    job_run_id = job_manager.create_job_run(
        job_id=job_id,
        run_name="training-run-001"
    )
    
    print(f"""
    ✅ Training job created and started successfully!
    
    Job ID: {job_id}
    Job Run ID: {job_run_id}
    
    Next steps:
    1. Use the external monitor (oci_job_monitor.py) to watch this job
    2. The monitor will cancel the job if metric thresholds are exceeded
    3. Check job status in OCI Console or via SDK
    
    Example monitor command:
    python oci_job_monitor.py --job-run-id {job_run_id}
    """)
    
    return job_id, job_run_id


def demonstrate_job_cancellation():
    """
    Demonstrate how external processes can cancel jobs based on conditions.
    
    This is the key concept: Jobs are canceled FROM OUTSIDE, not from within.
    """
    
    print("""
    🎯 OCI Data Science Jobs - Cancellation Approach
    ================================================
    
    CORRECT APPROACH:
    ✅ External monitoring process uses OCI SDK to cancel jobs
    ✅ Monitor watches job metrics via logs or external systems
    ✅ Job code handles cancellation gracefully via signal handlers
    ✅ Jobs naturally exit when code completes or exit() is called
    
    INCORRECT APPROACHES:
    ❌ Job code cannot cancel itself using OCI SDK
    ❌ Jobs cannot call their own cancel() function
    ❌ Internal cancellation based on metrics won't work
    
    ARCHITECTURE:
    
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │  Training Job   │    │ External Monitor │    │   OCI SDK      │
    │                 │    │                  │    │                 │
    │ 1. Log metrics  │───▶│ 2. Parse metrics │    │ 4. Cancel job   │
    │ 3. Handle       │◀───│ 3. Check thresh. │───▶│    run          │
    │    cancellation │    │                  │    │                 │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
    
    """)


if __name__ == "__main__":
    demonstrate_job_cancellation()
    
    # Uncomment to create a sample job (requires valid OCI configuration)
    # create_sample_training_job_with_monitoring()