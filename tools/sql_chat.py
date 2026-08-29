import pandas as pd
from sqlalchemy import create_engine, text
from langchain_google_genai import ChatGoogleGenerativeAI
import textwrap

def ask_dataset(csv_path: str, user_query: str) -> str:
  
    # 1. Load CSV to SQLite (auto-detect separator for files like wine_quality)
    df = pd.read_csv(csv_path, sep=None, engine='python')
    # Sanitize column names for SQL
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    
    engine = create_engine("sqlite:///:memory:")
    df.to_sql("dataset", engine, index=False)
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    
    # 2. Get schema and sample data for the prompt
    schema_info = df.dtypes.to_string()
    sample_data = df.head(3).to_csv(index=False)
    
    # 3. Prompt the LLM to generate ONLY the SQL query
    sql_prompt = textwrap.dedent(f"""
    You are an expert SQL assistant. We have a SQLite table named 'dataset' with the following schema:
    {schema_info}
    
    Sample data:
    {sample_data}
    
    Generate ONLY a raw, syntactically correct SQLite query to answer the user's question. 
    DO NOT wrap the query in markdown formatting. DO NOT include any explanations.
    
    User Question: {user_query}
    """).strip()
    
    try:
        sql_response = llm.invoke(sql_prompt)
        
        # Handle case where response content is a list of blocks
        content = sql_response.content
        if isinstance(content, list):
            query_text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        else:
            query_text = str(content)
            
        query = query_text.replace("```sql", "").replace("```", "").strip()
        
        # 4. Execute the generated query
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
            
        # 5. Prompt the LLM to summarize the result
        final_prompt = textwrap.dedent(f"""
        User Question: {user_query}
        SQL Query Executed: {query}
        SQL Result: {result}
        
        Provide a concise, natural language answer to the user's question based on the SQL result.
        At the end of your answer, you MUST include the exact SQL Query that was executed in a markdown code block (```sql).
        """).strip()
        
        final_response = llm.invoke(final_prompt)
        
        final_content = final_response.content
        if isinstance(final_content, list):
            return "".join(item.get("text", "") for item in final_content if isinstance(item, dict))
        return str(final_content)
        
    except Exception as e:
        return f"An error occurred while querying the database: {e}"
