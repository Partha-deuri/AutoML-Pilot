import pandas as pd
from agent.state import PipelineState
from tools.cleaning import clean_data

def cleaning_node(state: PipelineState) -> dict:
    """
    Executes the JSON cleaning plan on the raw data.
    """
    raw_path = state["raw_data_path"]
    plan = state["cleaning_plan"]
    
    df = pd.read_csv(raw_path, sep=None, engine='python')
    cleaned_df = clean_data(df, plan)
    
    return {"cleaned_data": cleaned_df}
