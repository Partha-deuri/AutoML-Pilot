from langgraph.graph import StateGraph, END
from agent.state import PipelineState

# Import Nodes
from agent.nodes.profiler_node import profiler_node
from agent.nodes.planner_node import planner_node
from agent.nodes.cleaning_node import cleaning_node
from agent.nodes.feature_node import feature_node
from agent.nodes.training_node import training_node
from agent.nodes.evaluation_node import evaluation_node
from agent.nodes.report_node import report_node

def build_graph():
    """
    Constructs and compiles the LangGraph state machine for the AutoML-Pilot.
    """
    # Initialize the graph with the PipelineState schema
    workflow = StateGraph(PipelineState)
    
    # Add Nodes
    workflow.add_node("Profiler", profiler_node)
    workflow.add_node("Planner", planner_node)
    workflow.add_node("Cleaner", cleaning_node)
    workflow.add_node("Feature_Engineer", feature_node)
    workflow.add_node("Trainer", training_node)
    workflow.add_node("Evaluator", evaluation_node)
    workflow.add_node("Reporter", report_node)
    
    # Define Edges (Linear sequential flow)
    workflow.set_entry_point("Profiler")
    workflow.add_edge("Profiler", "Planner")
    workflow.add_edge("Planner", "Cleaner")
    workflow.add_edge("Cleaner", "Feature_Engineer")
    workflow.add_edge("Feature_Engineer", "Trainer")
    workflow.add_edge("Trainer", "Evaluator")
    workflow.add_edge("Evaluator", "Reporter")
    workflow.add_edge("Reporter", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app
