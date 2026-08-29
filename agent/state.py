from typing import TypedDict, Any, Optional

class PipelineState(TypedDict):
    """
    Represents the state of the machine learning pipeline as it moves through the LangGraph nodes.
    """
    # Raw Data Path provided by the user
    raw_data_path: str
    
    # Optional target column provided by the user (if unknown, empty string)
    target_column: str
    
    # Deterministic statistical summary extracted by the Profiler tool
    data_profile: dict[str, Any]
    
    # JSON-based plan for data cleaning, produced by the Planner LLM
    cleaning_plan: dict[str, Any]
    
    # Path to the cleaned dataset (or the DataFrame itself) after Cleaner node
    cleaned_data: Any
    
    # Information regarding scaling and categorical encoding
    feature_engineering_info: dict[str, Any]
    
    # Metrics logged during the training of various baseline models
    model_metrics: dict[str, Any]
    
    # Information on the top performing model
    best_model_info: dict[str, Any]
    
    # Dictionary of trained model objects
    trained_models: dict[str, Any]
    
    # Test dataset for SHAP evaluation
    X_test: Any
    
    # Path to the generated SHAP feature importance plot (if generated)
    shap_plot_path: str
    
    # Final synthesized Markdown report
    report: str
