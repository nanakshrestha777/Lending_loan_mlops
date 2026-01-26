"""
MLflow Model Deployment Script
This script:
1. Searches for the best model in MLflow
2. Registers it in the Model Registry
3. Transitions it to Production stage
4. Can be triggered manually or via Airflow DAG
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLflowModelDeployer:
    """Handle model registration and deployment to production"""
    
    def __init__(self, tracking_uri="http://127.0.0.1:5000"):
        """
        Initialize MLflow client
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        logger.info(f"✅ Connected to MLflow at {tracking_uri}")
    
    def find_best_model(
        self, 
        experiment_name="Lending_Club_XGBoost_Production",
        metric_name="f1_score",
        ascending=False
    ):
        """
        Find the best model run based on a metric
        
        Args:
            experiment_name: Name of the experiment
            metric_name: Metric to sort by (e.g., 'f1_score', 'accuracy')
            ascending: Sort order (False for higher is better)
            
        Returns:
            run_id and metric value of best model
        """
        logger.info(f"🔍 Searching for best model in experiment: {experiment_name}")
        
        try:
            # Search runs
            runs = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string="",
                run_view_type=ViewType.ACTIVE_ONLY,
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
            )
            
            if runs.empty:
                raise ValueError(f"No runs found in experiment '{experiment_name}'")
            
            best_run = runs.iloc[0]
            run_id = best_run.run_id
            metric_value = best_run[f"metrics.{metric_name}"]
            
            logger.info(f"✅ Found best model:")
            logger.info(f"   Run ID: {run_id}")
            logger.info(f"   {metric_name}: {metric_value:.4f}")
            
            return run_id, metric_value
            
        except Exception as e:
            logger.error(f"❌ Error finding best model: {e}")
            raise
    
    def register_model(
        self,
        run_id,
        model_name="lending_club_xgboost_model",
        artifact_path="model"
    ):
        """
        Register a model in MLflow Model Registry
        
        Args:
            run_id: MLflow run ID containing the model
            model_name: Name to register the model under
            artifact_path: Path to model artifact in the run
            
        Returns:
            ModelVersion object
        """
        logger.info(f"📝 Registering model: {model_name}")
        
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            logger.info(f"✅ Model registered:")
            logger.info(f"   Name: {model_name}")
            logger.info(f"   Version: {model_version.version}")
            logger.info(f"   Stage: {model_version.current_stage}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"❌ Error registering model: {e}")
            raise
    
    def transition_to_production(
        self,
        model_name="lending_club_xgboost_model",
        version=None,
        archive_existing=True
    ):
        """
        Transition a model version to Production stage
        
        Args:
            model_name: Registered model name
            version: Model version number (None = latest)
            archive_existing: Archive existing production models
            
        Returns:
            Updated ModelVersion
        """
        try:
            # Get latest version if not specified
            if version is None:
                latest_versions = self.client.get_latest_versions(
                    model_name, 
                    stages=["None", "Staging", "Production"]
                )
                if not latest_versions:
                    raise ValueError(f"No versions found for model '{model_name}'")
                version = latest_versions[0].version
            
            logger.info(f"🚀 Transitioning model to Production:")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Version: {version}")
            
            # Archive existing production models
            if archive_existing:
                prod_versions = self.client.get_latest_versions(
                    model_name, 
                    stages=["Production"]
                )
                for pv in prod_versions:
                    logger.info(f"📦 Archiving version {pv.version}")
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=pv.version,
                        stage="Archived"
                    )
            
            # Transition to production
            model_version = self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            logger.info(f"✅ Model transitioned to Production!")
            logger.info(f"   Version: {model_version.version}")
            logger.info(f"   Stage: {model_version.current_stage}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"❌ Error transitioning model: {e}")
            raise
    
    def deploy_best_model(
        self,
        experiment_name="Lending_Club_XGBoost_Production",
        model_name="lending_club_xgboost_model",
        metric_name="f1_score"
    ):
        """
        Complete deployment pipeline:
        1. Find best model
        2. Register it
        3. Transition to Production
        
        Args:
            experiment_name: MLflow experiment name
            model_name: Name for registered model
            metric_name: Metric to optimize
            
        Returns:
            Dictionary with deployment info
        """
        logger.info("=" * 60)
        logger.info("🚀 STARTING MODEL DEPLOYMENT PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Find best model
            run_id, metric_value = self.find_best_model(
                experiment_name=experiment_name,
                metric_name=metric_name
            )
            
            # Step 2: Register model
            model_version = self.register_model(
                run_id=run_id,
                model_name=model_name
            )
            
            # Step 3: Transition to production
            prod_version = self.transition_to_production(
                model_name=model_name,
                version=model_version.version
            )
            
            logger.info("=" * 60)
            logger.info("🎉 DEPLOYMENT SUCCESSFUL!")
            logger.info("=" * 60)
            
            deployment_info = {
                "model_name": model_name,
                "version": prod_version.version,
                "run_id": run_id,
                "metric": metric_name,
                "metric_value": metric_value,
                "stage": "Production",
                "model_uri": f"models:/{model_name}/Production"
            }
            
            logger.info(f"📊 Deployment Info:")
            for key, value in deployment_info.items():
                logger.info(f"   {key}: {value}")
            
            return deployment_info
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error("❌ DEPLOYMENT FAILED")
            logger.error(f"Error: {e}")
            logger.error("=" * 60)
            raise


def main():
    """Main execution function"""
    
    # Configuration
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "Lending_Club_XGBoost_Production")
    MODEL_NAME = os.getenv("MODEL_NAME", "lending_club_xgboost_model")
    METRIC_NAME = os.getenv("METRIC_NAME", "f1_score")
    
    # Initialize deployer
    deployer = MLflowModelDeployer(tracking_uri=MLFLOW_URI)
    
    # Deploy best model
    try:
        deployment_info = deployer.deploy_best_model(
            experiment_name=EXPERIMENT_NAME,
            model_name=MODEL_NAME,
            metric_name=METRIC_NAME
        )
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
