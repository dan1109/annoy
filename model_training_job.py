#!/usr/bin/env python3
"""
Model Training Script for OCI Data Science Jobs

This script demonstrates how to structure model training code that:
1. Logs metrics in a format that can be monitored externally
2. Handles graceful shutdown when the job is canceled
3. Uses proper logging practices for OCI Data Science Jobs

Key Points:
- The job itself CANNOT cancel itself using OCI SDK
- External monitoring is required to cancel jobs based on metrics
- Jobs naturally exit when code execution completes or exit() is called
"""

import os
import sys
import time
import json
import signal
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


class ModelTrainingJob:
    """
    Example model training job that demonstrates proper metric logging
    and graceful shutdown handling for OCI Data Science Jobs.
    """
    
    def __init__(self):
        self.setup_logging()
        self.setup_signal_handlers()
        self.should_stop = False
        self.current_epoch = 0
        self.model = None
        
    def setup_logging(self):
        """Setup logging to output metrics in a format that can be monitored"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('/tmp/training.log') if os.path.exists('/tmp') else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True
        
    def log_metric(self, metrics: Dict[str, float]):
        """
        Log metrics in a structured format that can be parsed by external monitors.
        
        The format used here (METRIC: {...}) can be easily parsed by the
        external monitoring script using log search.
        """
        # Log in a parseable format for external monitoring
        metric_str = json.dumps(metrics)
        self.logger.info(f"METRIC: {metric_str}")
        
        # Also log human-readable format
        metric_display = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Training metrics - {metric_display}")
        
    def create_sample_data(self, n_samples: int = 10000) -> tuple:
        """Create sample classification data for training"""
        self.logger.info(f"Generating sample dataset with {n_samples} samples...")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.logger.info(f"Dataset split - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
        
    def train_model_iteratively(self, X_train, X_test, y_train, y_test, max_epochs: int = 50):
        """
        Train model iteratively to simulate long-running training with metric updates.
        
        This demonstrates how to structure training so that metrics are logged
        regularly and can be monitored by external processes.
        """
        self.logger.info("Starting iterative model training...")
        
        # Initialize with a small number of estimators
        base_estimators = 10
        
        for epoch in range(max_epochs):
            if self.should_stop:
                self.logger.info(f"Training stopped early at epoch {epoch} due to shutdown signal")
                break
                
            self.current_epoch = epoch
            
            # Gradually increase model complexity
            n_estimators = base_estimators + (epoch * 2)
            
            self.logger.info(f"Epoch {epoch + 1}/{max_epochs} - Training with {n_estimators} estimators")
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            
            start_time = time.time()
            self.model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            y_pred_proba_test = self.model.predict_proba(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_loss = log_loss(y_test, y_pred_proba_test)
            
            # Log metrics in structured format for external monitoring
            metrics = {
                "epoch": epoch + 1,
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
                "n_estimators": n_estimators,
                "training_time_seconds": float(training_time),
                "timestamp": datetime.now().isoformat()
            }
            
            self.log_metric(metrics)
            
            # Simulate some processing time
            time.sleep(2)
            
            # Check for early stopping conditions (this is internal logic, not OCI cancellation)
            if test_accuracy > 0.98:
                self.logger.info(f"Early stopping: High accuracy achieved ({test_accuracy:.4f})")
                break
                
            if test_loss < 0.05:
                self.logger.info(f"Early stopping: Low loss achieved ({test_loss:.4f})")
                break
        
        return self.model
    
    def save_model_artifacts(self, model, output_dir: str = "/tmp/model_artifacts"):
        """Save model artifacts and final metrics"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # In a real scenario, you would save the actual model
            # For this example, we'll just save metadata
            model_metadata = {
                "model_type": "RandomForestClassifier",
                "n_estimators": model.n_estimators if model else 0,
                "final_epoch": self.current_epoch,
                "training_completed": not self.should_stop,
                "timestamp": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(output_dir, "model_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
                
            self.logger.info(f"Model artifacts saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving model artifacts: {e}")
    
    def run_training(self):
        """
        Main training pipeline that demonstrates proper structure for
        OCI Data Science Jobs with external monitoring.
        """
        try:
            self.logger.info("="*50)
            self.logger.info("STARTING MODEL TRAINING JOB")
            self.logger.info("="*50)
            
            # Log job configuration
            job_config = {
                "job_type": "model_training",
                "framework": "scikit-learn",
                "start_time": datetime.now().isoformat(),
                "max_epochs": 50
            }
            self.log_metric(job_config)
            
            # Create training data
            X_train, X_test, y_train, y_test = self.create_sample_data()
            
            # Train model with iterative updates
            model = self.train_model_iteratively(X_train, X_test, y_train, y_test)
            
            # Save artifacts
            self.save_model_artifacts(model)
            
            # Log final status
            final_status = {
                "training_status": "completed" if not self.should_stop else "interrupted",
                "final_epoch": self.current_epoch,
                "end_time": datetime.now().isoformat()
            }
            self.log_metric(final_status)
            
            if self.should_stop:
                self.logger.info("Training was interrupted by external signal")
                sys.exit(1)  # Exit with error code to indicate interruption
            else:
                self.logger.info("Training completed successfully")
                sys.exit(0)  # Normal exit
                
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            error_metrics = {
                "training_status": "failed",
                "error_message": str(e),
                "end_time": datetime.now().isoformat()
            }
            self.log_metric(error_metrics)
            sys.exit(1)


def main():
    """
    Main entry point for the training job.
    
    This demonstrates the correct approach for OCI Data Science Jobs:
    1. The job runs its training code normally
    2. Metrics are logged in a structured format
    3. External monitoring can parse these metrics
    4. External monitoring uses OCI SDK to cancel the job if needed
    5. The job handles cancellation gracefully via signal handlers
    """
    
    # Create and run training job
    training_job = ModelTrainingJob()
    training_job.run_training()


if __name__ == "__main__":
    main()