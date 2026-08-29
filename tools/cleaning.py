import pandas as pd
from typing import Any

def clean_data(df: pd.DataFrame, cleaning_plan: dict[str, Any]) -> pd.DataFrame:

    cleaned_df = df.copy()
    actions = cleaning_plan.get("actions", [])
    
    for act in actions:
        action_type = act.get("action")
        col = act.get("column")
        
        if col not in cleaned_df.columns:
            continue
            
        if action_type == "drop_column":
            cleaned_df = cleaned_df.drop(columns=[col])
            
        elif action_type == "impute":
            strategy = act.get("strategy")
            if strategy == "median" and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif strategy == "mean" and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif strategy == "mode":
                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None
                if mode_val is not None:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
            elif strategy == "constant":
                constant_val = act.get("value", "Missing")
                cleaned_df[col] = cleaned_df[col].fillna(constant_val)

    return cleaned_df
