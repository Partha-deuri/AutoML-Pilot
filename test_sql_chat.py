import os
import pandas as pd
from sqlalchemy import create_engine
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv

load_dotenv()

def ask_dataset(csv_path: str, user_query: str) -> str:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    engine = create_engine("sqlite:///:memory:")
    df.to_sql("dataset", engine, index=False)
    db = SQLDatabase(engine=engine)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    
    # Try different agent types
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description", # Let's test this
        verbose=True,
        handle_parsing_errors=True
    )
    
    prompt = f"""
    You are an AI assistant interacting with a SQL database. The table name is "dataset".
    Always construct a syntactically correct SQLite query to answer the user's question.
    CRITICAL INSTRUCTION: You must strictly use the ReAct format (Action: ..., Action Input: ...). DO NOT use emojis (like 🍕) or HTML tags (like <I>) in your reasoning steps!
    CRITICAL INSTRUCTION: Generate ONLY ONE Action at a time. DO NOT hallucinate or generate the "Observation:" step yourself! Stop generating immediately after your "Action Input:" and wait for the system to provide the result.
    CRITICAL: When providing your Final Answer, you MUST include the exact SQL query you executed in a markdown code block (```sql), followed by the human-readable answer.
    
    User Question: {user_query}
    """
    
    try:
        response = agent_executor.invoke({"input": prompt})
        return response["output"]
    except Exception as e:
        return f"An error occurred while querying the database: {e}"

df = pd.DataFrame({"gender": ["female", "male", "female"], "age": [20, 22, 21]})
df.to_csv("dummy.csv", index=False)
print("Testing tool-calling...")
print(ask_dataset("dummy.csv", "How many females are there?"))
