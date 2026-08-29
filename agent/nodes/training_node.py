from agent.state import PipelineState
from tools.modeling import train_and_evaluate
from tools.mlflow_logger import log_experiment
import uuid

def training_node(state: PipelineState) -> dict:
    """
    Trains models and logs them to MLflow.
    """
    df = state["cleaned_data"]
    target_col = state["data_profile"]["target_column"]
    task_type = state["data_profile"]["inferred_task_type"]
    
    metrics, models, X_test = train_and_evaluate(df, target_col, task_type)
    
    return {
        "model_metrics": metrics,
        "trained_models": models,
        "X_test": X_test
    }
