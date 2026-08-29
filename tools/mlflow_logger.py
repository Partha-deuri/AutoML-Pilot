import mlflow
from typing import Any

def log_experiment(run_name: str, task_type: str, metrics: dict[str, Any], best_model_name: str) -> None:
 
    mlflow.set_tracking_uri("sqlite:///outputs/mlruns.db")
    mlflow.set_experiment("AutoML-Pilot-Experiments")
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("best_model", best_model_name)
        
        for model_name, model_metrics in metrics.items():
            for metric_name, value in model_metrics.items():
                mlflow.log_metric(f"{model_name.replace(' ', '_')}_{metric_name}", value)
                
    print(f"Successfully logged experiment '{run_name}' to MLflow.")
