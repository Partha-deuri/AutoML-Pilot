import streamlit as st
import os
import tempfile
import warnings
import logging
from dotenv import load_dotenv
from tools.sql_chat import ask_dataset
from agent.graph import build_graph

# Suppress all warnings and unnecessary logs
warnings.filterwarnings("ignore")
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

# Load environment variables (API keys)
load_dotenv()

st.set_page_config(page_title="AutoML-Pilot", page_icon="🚀", layout="wide")

st.title("AutoML-Pilot")
st.subheader("Navigating your machine learning pipelines autonomously.")

st.markdown("""
Upload a CSV dataset, specify the target column, and let the agent autonomously clean the data, engineer features, train models, and generate a final model card!
""")

# Initialize session state for Chat
if "active_chat_dataset" not in st.session_state:
    st.session_state.active_chat_dataset = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dataset_path_for_chat" not in st.session_state:
    st.session_state.dataset_path_for_chat = None

if st.session_state.active_chat_dataset:
    st.button("🔙 Back to Report", on_click=lambda: st.session_state.update({"active_chat_dataset": None}))
    st.header(f"💬 Chat with Dataset: {st.session_state.active_chat_dataset}")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Input box
    if prompt := st.chat_input("Ask a question about the data (e.g., 'What is the average age?'):"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Executing SQL query..."):
                response = ask_dataset(st.session_state.dataset_path_for_chat, prompt)
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

else:
    tab1, tab2 = st.tabs(["Run Pipeline", "History"])

    with tab1:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        target_column = st.text_input("Target Column Name (Optional)", value="", help="Leave blank if you want the agent to guess.")
    
        if uploaded_file is not None:
            # Save uploaded file to datasets so chat and history can see it
            dataset_path = os.path.join("datasets", uploaded_file.name)
            os.makedirs("datasets", exist_ok=True)
            with open(dataset_path, "wb") as f:
                f.write(uploaded_file.getvalue())
                
            col1, col2 = st.columns(2)
            with col1:
                run_pipeline = st.button("Run Pipeline")
            with col2:
                dataset_name = uploaded_file.name.split('.')[0]
                if st.button("💬 Chat with this Dataset", key="chat_tab1"):
                    st.session_state.active_chat_dataset = dataset_name
                    st.session_state.dataset_path_for_chat = dataset_path
                    st.session_state.chat_history = []
                    st.rerun()
            
            st.markdown("---")
            if run_pipeline:
                with st.spinner("Agent is working..."):
                    try:
                        # We also create a tmp path just to be safe with the graph logic
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        
                        # Initialize Graph
                        app = build_graph()
                        
                        # Initial State
                        initial_state = {
                            "raw_data_path": tmp_path,
                            "target_column": target_column.strip()
                        }
                        
                        # Run the graph
                        final_state = app.invoke(initial_state)
                        
                        st.success("Pipeline execution complete!")
                        
                        # Save the report
                        report_content = final_state.get("report", "No report generated.")
                        report_path = os.path.join("outputs", "model_cards", f"{uploaded_file.name.split('.')[0]}_report.md")
                        os.makedirs(os.path.dirname(report_path), exist_ok=True)
                        with open(report_path, "w", encoding="utf-8") as f:
                            f.write(report_content)
                        
                        # Display Results
                        st.markdown(report_content)
                        
                        shap_path = final_state.get("shap_plot_path")
                        if shap_path and os.path.exists(shap_path):
                            st.subheader("Feature Importance (SHAP)")
                            st.image(shap_path)
                        
                        with st.expander("View Raw Cleaning Plan"):
                            st.json(final_state.get("cleaning_plan", {}))
                            
                        with st.expander("View Model Metrics"):
                            st.json(final_state.get("model_metrics", {}))
                            
                    except Exception as e:
                        st.error(f"An error occurred during execution: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
    
    with tab2:
        st.header("Past Model Cards")
        reports_dir = os.path.join("outputs", "model_cards")
        if os.path.exists(reports_dir):
            reports = sorted([f for f in os.listdir(reports_dir) if f.endswith("_report.md")])
            if reports:
                selected_report = st.selectbox("Select a model card", reports)
                if selected_report:
                    report_path = os.path.join(reports_dir, selected_report)
                    with open(report_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    
                    # Infer dataset name
                    dataset_name = selected_report.replace("_report.md", ".csv")
                    dataset_path = os.path.join("datasets", dataset_name)
                    
                    if os.path.exists(dataset_path):
                        st.info(f"Dataset Path: `{os.path.abspath(dataset_path)}`")
                        if st.checkbox("Show Dataset Sample", key=selected_report):
                            import pandas as pd
                            df = pd.read_csv(dataset_path, sep=None, engine='python', nrows=100)
                            st.dataframe(df)
                    else:
                        st.warning(f"Associated dataset ({dataset_name}) not found in the datasets folder.")
                        
                    # Display Report
                    st.markdown(content)
                    
                    # Try to load corresponding SHAP plot
                    shap_path = os.path.join("outputs", "shap_plots", f"{dataset_name.split('.')[0]}_shap.png")
                    if os.path.exists(shap_path):
                        st.subheader("Feature Importance (SHAP)")
                        st.image(shap_path)
                        
                    st.markdown("---")
                    if st.button("💬 Chat with this Dataset", key=f"chat_tab2_{selected_report}"):
                        st.session_state.active_chat_dataset = dataset_name.split('.')[0]
                        st.session_state.dataset_path_for_chat = dataset_path
                        st.session_state.chat_history = []
                        st.rerun()
            else:
                st.info("No past model cards found.")
        else:
            st.info("No past model cards found.")
