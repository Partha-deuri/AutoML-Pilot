from agent.state import PipelineState
from tools.profiling import profile_dataset
import os

def profiler_node(state: PipelineState) -> dict:
    """
    Reads the raw data and extracts the statistical profile.
    """
    raw_path = state["raw_data_path"]
    user_target = state.get("target_column", "").strip()
    
    import pandas as pd
    df = pd.read_csv(raw_path, nrows=1, sep=None, engine='python')
    
    lower_cols = [c.lower() for c in df.columns]
    target_col = None
    
    if user_target and user_target.lower() in lower_cols:
        target_col = df.columns[lower_cols.index(user_target.lower())]
    
    if not target_col:
        # Heuristic: check common names, else last column
        common_targets = ['target', 'label', 'class', 'survived', 'churn', 'price', 'quality']
        
        target_col = df.columns[-1] # Default fallback
        for common in common_targets:
            if common in lower_cols:
                target_col = df.columns[lower_cols.index(common)]
                break
    
    profile = profile_dataset(raw_path, target_col)
    
    return {"data_profile": profile}
