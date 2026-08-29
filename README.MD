# 🚀 ML Pipeline Agent: AutoML-Pilot

*Navigating your machine learning pipelines autonomously.*

**AutoML-Pilot** is an autonomous, LangGraph-based agent designed to ingest raw tabular datasets, reason about their statistical properties, and autonomously produce a cleaned dataset, trained models, and a comprehensive written model card.

Instead of relying on LLMs to write fragile, free-form Python scripts, this system enforces a strict **Separation of Concerns**: the LLM acts as the brain (planning, reasoning, and synthesis) while deterministic, pre-written Python tools act as the hands (executing pandas operations and scikit-learn/XGBoost training). This approach ensures reliability, safety, and true adaptability across completely unrelated datasets with zero per-dataset code changes.

---

## ✨ Key Capabilities

* **Deterministic Data Profiling:** Computes exact statistical summaries (missing values, cardinality, distributions) without relying on LLM hallucinations.
* **Autonomous Cleaning & Feature Engineering:** Generates strict JSON plans to handle imputation, outliers, and categorical encoding based on the data profile.
* **Safe Tool Execution:** Maps LLM-generated JSON plans to safe, predefined pandas transformations.
* **Automated Model Selection & Tracking:** Trains a registry of baseline models (Logistic Regression, Random Forest, XGBoost), selects the best performer based on task-appropriate metrics, and logs everything to MLflow.
* **Automated Documentation:** Synthesizes the agent's reasoning trace and model performance into a comprehensive, human-readable Markdown Model Card.

---

## 🧠 Pipeline Architecture (The State Machine)

The agent operates as a LangGraph state machine, passing a shared `PipelineState` object through the following nodes:

1. **Profiler (Code):** Scans the raw CSV and extracts statistical facts.
2. **Planner (LLM):** Reads the profile and outputs a JSON "Cleaning Plan".
3. **Cleaner (Code):** Parses the JSON plan and executes the data transformations.
4. **Feature Engineer (LLM + Code):** Decides on and executes scaling and encoding strategies.
5. **Trainer (Code):** Trains predefined models and logs metrics to MLflow.
6. **Evaluator (Code):** Compares MLflow metrics and selects the winning model.
7. **Reporter (LLM):** Ingests the agent's decision log and metrics to write the final Model Card.

---

## 📂 Project Structure

```text
ml-pipeline-agent/
├── agent/
│   ├── __init__.py
│   ├── graph.py              # LangGraph state machine definition
│   ├── state.py              # Shared state schema (TypedDict)
│   ├── nodes/                # Individual graph nodes
│   │   ├── profiler_node.py
│   │   ├── planner_node.py
│   │   ├── cleaning_node.py
│   │   ├── feature_node.py
│   │   ├── training_node.py
│   │   ├── evaluation_node.py
│   │   └── report_node.py
│   └── prompts/              # LLM system prompts
│       ├── planner_prompt.txt
│       └── report_prompt.txt
├── tools/
│   ├── profiling.py          # Deterministic pandas data profiling
│   ├── cleaning.py           # Imputation, outlier handling, dropping columns
│   ├── feature_engineering.py# Scaling, encoding
│   ├── modeling.py           # scikit-learn/XGBoost wrappers
│   └── mlflow_logger.py      # MLflow experiment tracking
├── datasets/                 # Generalization test suite
│   ├── titanic.csv           # (Classification, missing data)
│   ├── house_prices.csv      # (Regression, mixed types)
│   ├── customer_churn.csv    # (Classification, imbalanced)
│   └── wine_quality.csv      # (Multi-class)
├── outputs/
│   ├── model_cards/          # Generated markdown reports
│   └── mlruns/               # MLflow tracking store
├── evaluation/
│   └── run_benchmark.py      # Runs agent across all datasets to prove generalization
├── app.py                    # Streamlit interactive front end
└── requirements.txt

```

---

## 🚀 Installation & Setup

**1. Clone the repository and navigate to the project directory:**

```bash
git clone https://github.com/yourusername/ml-pipeline-agent.git
cd ml-pipeline-agent

```

**2. Create a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

```

**3. Configure Environment Variables:**
Create a `.env` file in the root directory and add your LLM API keys (e.g., Anthropic or OpenAI, depending on your LangChain configuration).

```env
ANTHROPIC_API_KEY=your_api_key_here
# OPENAI_API_KEY=your_api_key_here

```

---

## 🎮 Usage

### Option 1: Interactive Streamlit UI

Run the interactive front end to upload a CSV, watch the agent's reasoning stream in real-time, and view the final Model Card and MLflow metrics.

```bash
streamlit run app.py

```

### Option 2: Run the Generalization Benchmark

To prove the agent's adaptability, run the benchmark script. This executes the full pipeline across 4 structurally different datasets (regression, imbalanced classification, missing data, multi-class) and generates a comparison table of the agent's dataset-specific decisions.

```bash
python evaluation/run_benchmark.py

```

### Option 3: View MLflow Tracking

To view the experiment parameters, metrics, and logged agent artifacts (JSON plans and model cards), launch the MLflow UI:

```bash
mlflow ui --backend-store-uri outputs/mlruns

```

---

## 🧪 The Generalization Proof

The true test of this agent is its ability to adapt without hardcoded rules. The `evaluation/run_benchmark.py` script validates performance across:

* **Titanic:** Triggers missing data imputation logic.
* **House Prices:** Triggers regression metrics (RMSE) and high-cardinality categorical encoding.
* **Customer Churn:** Triggers class-imbalance awareness and prioritizes F1 scores over accuracy.
* **Wine Quality:** Triggers multi-class classification handling.

All handled by the exact same state machine, proving genuine zero-shot pipeline automation.