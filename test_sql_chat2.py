import os
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_google_genai import ChatGoogleGenerativeAI
import textwrap
from dotenv import load_dotenv

load_dotenv()

def ask_dataset(csv_path: str, user_query: str) -> str:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    engine = create_engine("sqlite:///:memory:")
    df.to_sql("dataset", engine, index=False)
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    
    # 1. Get schema and sample
    schema_info = df.dtypes.to_string()
    sample_data = df.head(3).to_markdown()
    
    # 2. Prompt for SQL
    sql_prompt = textwrap.dedent(f"""
    You are an expert SQL assistant. We have a SQLite table named 'dataset' with the following schema:
    {schema_info}
    
    Sample data:
    {sample_data}
    
    Generate ONLY a raw, syntactically correct SQLite query to answer the user's question. DO NOT wrap the query in markdown formatting. DO NOT include any explanations.
    
    User Question: {user_query}
    """).strip()
    
    try:
        sql_response = llm.invoke(sql_prompt)
        query = sql_response.content.replace("```sql", "").replace("```", "").strip()
        
        # 3. Execute query
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
            
        # 4. Final Answer
        final_prompt = textwrap.dedent(f"""
        User Question: {user_query}
        SQL Query Executed: {query}
        SQL Result: {result}
        
        Provide a natural language answer to the user's question based on the SQL result.
        At the end of your answer, include the SQL Query that was executed in a markdown code block (```sql).
        """).strip()
        
        final_response = llm.invoke(final_prompt)
        return final_response.content
        
    except Exception as e:
        return f"An error occurred while querying the database: {e}"

df = pd.DataFrame({"gender": ["female", "male", "female"], "age": [20, 22, 21]})
df.to_csv("dummy.csv", index=False)
print("Testing direct chain...")
print(ask_dataset("dummy.csv", "How many females are there?"))
