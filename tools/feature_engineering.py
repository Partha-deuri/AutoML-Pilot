import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def engineer_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:

    engineered_df = df.copy()
    
    # Process categorical columns
    cat_cols = engineered_df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col == target_column:
            continue
        
        num_unique = engineered_df[col].nunique()
        if num_unique <= 15:
            # One-Hot Encode
            engineered_df = pd.get_dummies(engineered_df, columns=[col], drop_first=True)
        else:
            # Label Encode (as a basic fallback for high cardinality)
            le = LabelEncoder()
            # Handle possible NaNs if cleaning missed them
            engineered_df[col] = engineered_df[col].astype(str)
            engineered_df[col] = le.fit_transform(engineered_df[col])
            
    # Process numeric columns
    num_cols = engineered_df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    
    cols_to_scale = [c for c in num_cols if c != target_column]
    if cols_to_scale:
        engineered_df[cols_to_scale] = scaler.fit_transform(engineered_df[cols_to_scale])
        
    # Ensure bools from get_dummies are ints
    bool_cols = engineered_df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        engineered_df[bool_cols] = engineered_df[bool_cols].astype(int)

    return engineered_df
