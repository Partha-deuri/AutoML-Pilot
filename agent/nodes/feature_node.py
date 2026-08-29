from agent.state import PipelineState
from tools.feature_engineering import engineer_features

def feature_node(state: PipelineState) -> dict:
    """
    Applies encoding and scaling to the cleaned dataset.
    """
    cleaned_df = state["cleaned_data"]
    target_col = state["data_profile"]["target_column"]
    
    engineered_df = engineer_features(cleaned_df, target_col)
    
    # Overwrite cleaned_data with the engineered version for the training node
    return {"cleaned_data": engineered_df}
