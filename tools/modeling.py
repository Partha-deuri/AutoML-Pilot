import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

def train_and_evaluate(df: pd.DataFrame, target_column: str, task_type: str) -> tuple[dict[str, Any], dict[str, Any], Any]:

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Label encoding for target if it's classification (XGBoost requires 0-indexed integer labels)
    if task_type == 'classification':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    metrics = {}
    
    if task_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            metrics[name] = {'accuracy': acc, 'f1_score': f1}
            
    elif task_type == 'regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            r2 = r2_score(y_test, preds)
            metrics[name] = {'rmse': rmse, 'r2_score': r2}
            
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    return metrics, models, X_test
