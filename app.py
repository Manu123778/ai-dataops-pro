import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import urllib.parse
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import duckdb
import tempfile
import io
import requests
import zipfile
import numpy as np
# Attempt to import boto3 for AWS, handle nicely if missing
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# --- 1. CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="AI DataOps Pro 2.0", page_icon="⚡", layout="wide")

# Custom CSS for modern look
st.markdown("""
<style>
    .stButton>button { border-radius: 8px; font-weight: bold; }
    .reportview-container { background: #f0f2f6; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'db_engine' not in st.session_state: st.session_state.db_engine = None
if 'db_schema' not in st.session_state: st.session_state.db_schema = ""
if 'current_df' not in st.session_state: st.session_state.current_df = None
if 'original_df' not in st.session_state: st.session_state.original_df = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'cleaner_history' not in st.session_state: st.session_state.cleaner_history = []
if 'stats_history' not in st.session_state: st.session_state.stats_history = [] 
if 'source_type' not in st.session_state: st.session_state.source_type = None
if 'file_path' not in st.session_state: st.session_state.file_path = None
if 'cleaning_code' not in st.session_state: st.session_state.cleaning_code = ""
if 'last_query' not in st.session_state: st.session_state.last_query = ""
if 'last_error' not in st.session_state: st.session_state.last_error = None
if 'last_url' not in st.session_state: st.session_state.last_url = ""

# --- 3. UTILITY ---
def init_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        if 'user_api_key' in st.session_state and st.session_state.user_api_key:
            return Groq(api_key=st.session_state.user_api_key)
        return None
    return Groq(api_key=api_key)

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def get_df_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def check_safety(sql):
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "GRANT"]
    if any(x in sql.upper() for x in forbidden): 
        return False, "⚠️ Safety Block: Destructive SQL detected."
    return True, "Safe"

# --- 4. DATA ENGINE (Optimized with Caching) ---
def get_db_engine(server, db, driver):
    conn_str = f"DRIVER={{{driver}}};SERVER={server};DATABASE={db};Trusted_Connection=yes;"
    quoted = urllib.parse.quote_plus(conn_str)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}")

def get_schema_info(source_type, engine_or_path):
    try:
        if source_type == "database":
            if engine_or_path is None: return "Please connect to database first"
            insp = sqlalchemy.inspect(engine_or_path)
            schema_out = ""
            for t in insp.get_table_names():
                cols = [f"{c['name']} ({c['type']})" for c in insp.get_columns(t)]
                schema_out += f"Table: {t}\nColumns: {', '.join(cols)}\n\n"
            return schema_out if schema_out else "No tables found"
        else:
            if engine_or_path is None: return "Please upload a file first"
            conn = duckdb.connect(':memory:')
            q = f"SELECT * FROM '{engine_or_path}' LIMIT 1" if not engine_or_path.endswith('.xlsx') else "SELECT * FROM read_csv_auto('" + engine_or_path + "') LIMIT 1" 
            if engine_or_path.endswith('.csv'):
                 q = f"SELECT * FROM read_csv_auto('{engine_or_path}') LIMIT 1"
            elif engine_or_path.endswith('.parquet'):
                 q = f"SELECT * FROM read_parquet('{engine_or_path}') LIMIT 1"
            
            try:
                df = conn.execute(q).df()
                cols = list(df.columns)
                return f"Table: t (File Data)\nColumns: {', '.join(cols)}"
            except:
                return "Could not extract schema from file."
    except Exception as e: 
        return f"Schema Error: {e}"

# Cached query runner for speed
@st.cache_data(show_spinner=False)
def run_query_cached(query, source_type, engine_or_path_str):
    # Wrapper to allow caching (engine objects aren't hashable, so we pass string path if file)
    if source_type == "file":
        conn = duckdb.connect(':memory:')
        if engine_or_path_str.endswith('.csv'): 
            conn.execute(f"CREATE VIEW t AS SELECT * FROM read_csv_auto('{engine_or_path_str}')")
        elif engine_or_path_str.endswith('.xlsx'): 
            df = pd.read_excel(engine_or_path_str)
            conn.execute("CREATE VIEW t AS SELECT * FROM df")
        elif engine_or_path_str.endswith('.parquet'): 
            conn.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{engine_or_path_str}')")
        elif engine_or_path_str.endswith('.json'): 
            conn.execute(f"CREATE VIEW t AS SELECT * FROM read_json_auto('{engine_or_path_str}')")
        return conn.execute(query).df(), None
    return None, "DB Caching not implemented"

def run_query_unified(query, source_type, engine_or_path):
    try:
        if source_type == "database":
            if engine_or_path is None: return None, "Database not connected"
            with engine_or_path.connect() as conn: 
                return pd.read_sql(text(query), conn), None
        else:  # File or Web
            if engine_or_path is None: return None, "File not loaded"
            # Use cached version for files
            return run_query_cached(query, source_type, engine_or_path)
    except Exception as e:
        return None, str(e)

# --- 5. AI AGENTS ---

def agent_nl_to_sql(client, prompt, schema, source_type):
    if not client: return "-- No API Key."
    dialect = "T-SQL (SQL Server)" if source_type == "database" else "DuckDB SQL"
    sys = f"""You are an expert SQL writer specializing in {dialect}.
    SCHEMA: {schema}
    INSTRUCTIONS:
    1. Return ONLY the SQL query. No markdown.
    2. Use table 't' for files.
    """
    try:
        res = client.chat.completions.create(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", temperature=0
        )
        sql = res.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
        return sql
    except Exception as e: return f"-- Error: {e}"

def agent_fix_sql(client, bad_sql, error, schema, source_type):
    if not client: return "-- No API Key"
    sys = f"Fix this SQL for {source_type}. Error: {error}. Schema: {schema}"
    try:
        res = client.chat.completions.create(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": bad_sql}],
            model="llama-3.3-70b-versatile", temperature=0
        )
        return res.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
    except: return bad_sql

def agent_data_cleaner(client, df, intent="Scan and analyze this data for issues."):
    if not client: return "No API Key", ""
    info = get_df_info(df)
    head = df.head(5).to_string() 
    
    prompt = f"""
    You are a Data Cleaning Expert.
    TASK: {intent}
    
    DATA PROFILE:
    Sample: {head}
    Info: {info}
    
    CRITICAL RULES:
    1. The data is ALREADY loaded in a variable named `df`.
    2. DO NOT try to load files (like pd.read_csv). 
    3. Generate Python code that creates a 'clean_df' from 'df'.
    4. Provide the response as: [Explanation] ... [Code] ```python ... ```
    """
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", temperature=0.1
        )
        return res.choices[0].message.content
    except Exception as e: return f"Error: {e}"

def agent_dashboard_builder(client, df, intent):
    if not client: return "{}"
    columns = list(df.columns)
    # Include dtypes to make it smarter about selecting numeric vs categorical columns
    dtypes = df.dtypes.astype(str).to_dict()
    
    sys_msg = f"""Create a JSON spec for a Dashboard. 
    Columns: {columns}
    Data Types: {dtypes}
    User Intent: {intent}
    
    INSTRUCTIONS:
    1. Generate a JSON object with a "dashboard_title" and a list of "charts".
    2. Each chart must have: "type", "title", "x", "y" (optional), "color" (optional).
    3. Optional field: "filter" (Pandas query string, e.g. "Country == 'India'").
    4. Supported types: "bar", "line", "scatter", "pie", "histogram", "box".
    5. CRITICAL: Use EXACT column names from the provided list. Do not make up columns.
    6. If the user asks for a count or frequency, set "y" to null/None.
    
    JSON format: {{ "dashboard_title": "...", "charts": [ {{ "type": "...", "title": "...", "x": "...", "y": "...", "color": "...", "filter": "..." }} ] }}"""
    
    try:
        res = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": "Generate JSON"}],
            model="llama-3.3-70b-versatile", response_format={"type": "json_object"}, temperature=0.2
        )
        return res.choices[0].message.content
    except: return "{}"

def agent_stats_analyst(client, df, question):
    """New agent for statistical analysis"""
    if not client: return "No API Key", ""
    info = get_df_info(df)
    head = df.head(5).to_string()
    
    prompt = f"""
    You are a Senior Statistical Data Analyst.
    USER QUESTION: {question}
    
    DATA PROFILE:
    Sample: {head}
    Info: {info}
    
    INSTRUCTIONS:
    1. Write Python code to answer the question using pandas/numpy/scipy.
    2. The dataframe is available as `df`.
    3. IMPORTANT: If the user asks for a table, trend, or list (e.g., "year by year population"), create a DataFrame and assign it to `result`.
    4. If the user asks for a single value (e.g., "average age"), assign that number/string to `result`.
    5. Return format: [Explanation] ... [Code] ```python ... ```
    """
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", temperature=0.1
        )
        return res.choices[0].message.content
    except Exception as e: return f"Error: {e}"

# --- 6. UI MAIN ---
st.sidebar.title("⚡ AI DataOps Pro 2.0")

# API Key
groq_client = init_groq()
if not groq_client:
    st.sidebar.warning("⚠️ No API Key found.")
    key = st.sidebar.text_input("Enter Groq API Key:", type="password")
    if key:
        st.session_state.user_api_key = key
        st.rerun()

# SOURCE SELECTOR
st.sidebar.header("🔌 Connect Data")
mode = st.sidebar.radio("Source:", ["File Upload", "Web Scraping", "Cloud Storage (AWS S3)", "Database"])

# --- DATA CONTROLS ---
with st.sidebar.expander("⚙️ Data Controls", expanded=True):
    col_ref, col_reset = st.columns(2)
    with col_ref:
        if st.button("🔄 Refresh"):
            st.cache_data.clear() # Clear cache on refresh
            if st.session_state.source_type == "web" and st.session_state.last_url:
                try:
                    with st.spinner("Re-scraping..."):
                        headers = {"User-Agent": "Mozilla/5.0"}
                        response = requests.get(st.session_state.last_url, headers=headers)
                        response.raise_for_status()
                        dfs = pd.read_html(io.StringIO(response.text), header=0)
                        if dfs:
                            main_df = max(dfs, key=lambda x: x.size)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                                main_df.to_csv(tmp.name, index=False)
                                st.session_state.file_path = tmp.name
                            st.session_state.current_df = main_df
                            st.session_state.original_df = main_df.copy()
                            st.success("Refreshed!")
                except Exception as e: st.error(f"Error: {e}")
            elif st.session_state.db_engine:
                 try: 
                     with st.session_state.db_engine.connect() as c: c.execute(text("SELECT 1"))
                     st.success("DB Active")
                 except: st.error("DB Error")
    with col_reset:
        if st.button("❌ Remove"):
            st.session_state.current_df = None
            st.session_state.file_path = None
            st.session_state.db_engine = None
            st.session_state.source_type = None
            st.session_state.cleaner_history = []
            st.session_state.stats_history = []
            st.rerun()

# --- SOURCE LOGIC ---
if mode == "File Upload":
    st.session_state.source_type = "file"
    f = st.sidebar.file_uploader("Upload Data", type=['csv', 'xlsx', 'json', 'parquet', 'zip'])
    if f:
        path = save_uploaded_file(f)
        if path.endswith('.zip'):
            try:
                with zipfile.ZipFile(path, 'r') as z:
                    files = [n for n in z.namelist() if n.endswith(('.csv', '.xlsx', '.json', '.parquet'))]
                    if files:
                        extract_path = os.path.join(os.path.dirname(path), files[0])
                        with open(extract_path, 'wb') as target:
                            target.write(z.read(files[0]))
                        path = extract_path 
                        st.sidebar.success(f"📦 Unzipped: {files[0]}")
                    else:
                        st.sidebar.error("No data files found in ZIP")
            except Exception as e: st.sidebar.error(f"ZIP Error: {e}")

        st.session_state.file_path = path
        if st.session_state.current_df is None:
            # Use run_query_unified which now uses caching
            df, err = run_query_unified("SELECT * FROM t", "file", path)
            if df is not None:
                st.session_state.current_df = df
                st.session_state.original_df = df.copy()
                st.session_state.db_schema = get_schema_info("file", path)
                st.sidebar.success(f"Loaded {len(df)} rows")
            else:
                st.sidebar.error(err)

elif mode == "Web Scraping":
    st.sidebar.info("Extract data from website.")
    url = st.sidebar.text_input("Enter URL:", placeholder="https://...")
    if st.sidebar.button("🕸️ Extract"):
        with st.spinner("Scraping..."):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                dfs = pd.read_html(io.StringIO(response.text), header=0)
                if dfs:
                    main_df = max(dfs, key=lambda x: x.size)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        main_df.to_csv(tmp.name, index=False)
                        st.session_state.file_path = tmp.name
                    st.session_state.current_df = main_df
                    st.session_state.original_df = main_df.copy()
                    st.session_state.db_schema = get_schema_info("file", st.session_state.file_path)
                    st.session_state.source_type = "web"
                    st.session_state.last_url = url
                    st.sidebar.success(f"Loaded {len(main_df)} rows.")
                else: st.sidebar.warning("No tables found.")
            except Exception as e: st.sidebar.error(f"Error: {e}")

elif mode == "Cloud Storage (AWS S3)":
    st.session_state.source_type = "file" # Treated as file source once downloaded
    st.sidebar.info("Connect to AWS S3 Bucket.")
    if not HAS_BOTO3:
        st.sidebar.error("❌ 'boto3' library missing. pip install boto3")
    else:
        # Credential Inputs
        aws_id = st.sidebar.text_input("AWS Access Key ID", type="password")
        aws_secret = st.sidebar.text_input("AWS Secret Access Key", type="password")
        region = st.sidebar.text_input("Region (e.g. us-east-1)", "us-east-1")
        
        if aws_id and aws_secret:
            try:
                s3 = boto3.client('s3', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
                
                # List Buckets
                response = s3.list_buckets()
                buckets = [b['Name'] for b in response['Buckets']]
                selected_bucket = st.sidebar.selectbox("Select Bucket", buckets)
                
                if selected_bucket:
                    # List Objects (Limit 50 for performance)
                    objs = s3.list_objects_v2(Bucket=selected_bucket, MaxKeys=50)
                    if 'Contents' in objs:
                        files = [o['Key'] for o in objs['Contents'] if o['Key'].endswith(('.csv', '.json', '.parquet', '.xlsx'))]
                        selected_file = st.sidebar.selectbox("Select File", files)
                        
                        if st.sidebar.button("☁️ Load from S3"):
                            with st.spinner("Downloading..."):
                                obj = s3.get_object(Bucket=selected_bucket, Key=selected_file)
                                # Load directly into Pandas
                                if selected_file.endswith('.csv'):
                                    df = pd.read_csv(obj['Body'])
                                elif selected_file.endswith('.json'):
                                    df = pd.read_json(obj['Body'])
                                elif selected_file.endswith('.parquet'):
                                    df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
                                elif selected_file.endswith('.xlsx'):
                                    df = pd.read_excel(obj['Body'].read())
                                
                                # Save to session
                                st.session_state.current_df = df
                                st.session_state.original_df = df.copy()
                                # Save a temp local copy for consistency with SQL Engine
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                                    df.to_csv(tmp.name, index=False)
                                    st.session_state.file_path = tmp.name
                                st.session_state.db_schema = get_schema_info("file", st.session_state.file_path)
                                st.sidebar.success(f"Loaded {len(df)} rows from S3!")
                    else:
                        st.sidebar.warning("Bucket is empty or no supported files.")
            except Exception as e:
                st.sidebar.error(f"AWS Error: {e}")

elif mode == "Database":
    st.session_state.source_type = "database"
    srv = st.sidebar.text_input("Server", r"localhost\SQLEXPRESS")
    db = st.sidebar.text_input("Database", "Master")
    drv = st.sidebar.selectbox("Driver", ["ODBC Driver 17 for SQL Server", "SQL Server"])
    if st.sidebar.button("Connect DB"):
        try:
            eng = get_db_engine(srv, db, drv)
            with eng.connect() as c: c.execute(text("SELECT 1"))
            st.session_state.db_engine = eng
            st.session_state.db_schema = get_schema_info("database", eng)
            st.sidebar.success("Connected!")
        except Exception as e: st.sidebar.error(f"Error: {e}")

# NAVIGATION
page = st.sidebar.radio("Navigate:", ["🏠 Data Overview", "📈 Statistical Analysis", "🧹 AI Data Cleaner", "📊 Advanced Smart Dashboard", "🧠 AI Query Engine"])

# --- PAGES ---

if page == "🏠 Data Overview":
    st.title("Data Headquarters")
    if st.session_state.current_df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", st.session_state.current_df.shape[0])
        c2.metric("Columns", st.session_state.current_df.shape[1])
        try:
            dup_count = st.session_state.current_df.duplicated().sum()
        except TypeError:
            dup_count = st.session_state.current_df.astype(str).duplicated().sum()
        c3.metric("Duplicates", dup_count)
        st.dataframe(st.session_state.current_df.head(50), use_container_width=True)
    else:
        st.info("Please connect a data source.")

elif page == "📈 Statistical Analysis":
    st.title("📈 Statistical Analysis Hub")
    if st.session_state.current_df is not None:
        tab1, tab2 = st.tabs(["🔢 Descriptive Stats", "💬 AI Stat Analyst"])
        
        with tab1:
            st.subheader("Descriptive Statistics")
            # Select only numeric columns for stats
            numeric_df = st.session_state.current_df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe().T, use_container_width=True)
                
                st.subheader("Correlation Matrix")
                if len(numeric_df.columns) > 1:
                    corr = numeric_df.corr()
                    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for analysis.")
                
        with tab2:
            st.subheader("Ask the AI Statistician")
            for msg in st.session_state.stats_history:
                # Use st.write to automatically handle text vs dataframe rendering
                st.chat_message(msg['role']).write(msg['content'])
            
            if q := st.chat_input("E.g., 'What is the correlation between Sales and Profit?', 'Are there outliers in Age?'"):
                st.session_state.stats_history.append({"role": "user", "content": q})
                st.chat_message("user").write(q)
                
                with st.spinner("Calculating..."):
                    resp = agent_stats_analyst(groq_client, st.session_state.current_df, q)
                    if "```python" in resp:
                        parts = resp.split("```python")
                        exp = parts[0]
                        code = parts[1].split("```")[0]
                        
                        st.session_state.stats_history.append({"role": "assistant", "content": exp})
                        st.chat_message("assistant").write(exp)
                        
                        # Execute code
                        l_vars = {'df': st.session_state.current_df.copy(), 'pd': pd, 'np': np}
                        try:
                            # Capture stdout
                            import sys
                            from io import StringIO
                            old_stdout = sys.stdout
                            sys.stdout = mystdout = StringIO()
                            
                            exec(code, {}, l_vars)
                            
                            sys.stdout = old_stdout
                            output = mystdout.getvalue()
                            
                            # Optimized Result Handling for Tables
                            if 'result' in l_vars:
                                res_val = l_vars['result']
                                # Store the actual object (DataFrame/Value) in history
                                st.session_state.stats_history.append({"role": "assistant", "content": res_val})
                                st.chat_message("assistant").write(res_val)
                            elif output:
                                st.session_state.stats_history.append({"role": "assistant", "content": f"Output: {output}"})
                                st.chat_message("assistant").write(f"**Output:**\n{output}")
                                
                        except Exception as e:
                            st.error(f"Calculation Error: {e}")
                    else:
                        st.session_state.stats_history.append({"role": "assistant", "content": resp})
                        st.chat_message("assistant").write(resp)

elif page == "🧹 AI Data Cleaner":
    st.title("✨ AI Data Cleaning Agent")
    if st.session_state.current_df is not None:
        st.subheader("💬 Chat with Cleaner")
        for msg in st.session_state.cleaner_history:
            st.chat_message(msg['role']).write(msg['content'])
        if user_input := st.chat_input("Tell AI how to clean the data (e.g. 'Remove duplicates', 'Fill nulls in Age with 0')"):
            st.session_state.cleaner_history.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            with st.spinner("Thinking..."):
                analysis = agent_data_cleaner(groq_client, st.session_state.current_df, user_input)
                if "```python" in analysis:
                    parts = analysis.split("```python")
                    explanation = parts[0]
                    code = parts[1].split("```")[0]
                    st.session_state.cleaner_history.append({"role": "assistant", "content": explanation})
                    st.chat_message("assistant").write(explanation)
                    st.session_state.cleaning_explanation = explanation
                    st.session_state.cleaning_code = code
                else:
                    st.session_state.cleaner_history.append({"role": "assistant", "content": analysis})
                    st.chat_message("assistant").write(analysis)
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Automated Scan")
            if st.button("🔍 Auto-Detect Issues"):
                with st.spinner("Scanning..."):
                    analysis = agent_data_cleaner(groq_client, st.session_state.current_df)
                    if "```python" in analysis:
                        parts = analysis.split("```python")
                        st.session_state.cleaning_explanation = parts[0]
                        st.session_state.cleaning_code = parts[1].split("```")[0]
                    else:
                        st.session_state.cleaning_explanation = analysis
            if 'cleaning_explanation' in st.session_state:
                st.info("Analysis Result:")
                st.markdown(st.session_state.cleaning_explanation)
        with col2:
            st.subheader("Review & Execute Code")
            code = st.text_area("Python Code:", value=st.session_state.cleaning_code, height=250)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🚀 Run Cleaning Code", type="primary"):
                    l_vars = {'df': st.session_state.current_df.copy(), 'pd': pd, 'np': np}
                    try:
                        exec(code, {}, l_vars)
                        if 'clean_df' in l_vars:
                            st.session_state.current_df = l_vars['clean_df']
                            st.success("✅ Data Cleaned!")
                            st.rerun()
                        else: st.error("Code must produce 'clean_df'")
                    except Exception as e: st.error(f"Error: {e}")
            with c2:
                if st.button("↩️ Undo Changes"):
                    st.session_state.current_df = st.session_state.original_df.copy()
                    st.success("Reverted!")
                    st.rerun()

elif page == "📊 Advanced Smart Dashboard":
    st.title("🚀 AI Dashboard Architect")
    if st.session_state.current_df is not None:
        intent = st.text_input("Dashboard Goal:", placeholder="Sales by region...")
        if st.button("Generate"):
            with st.spinner("Building..."):
                resp = agent_dashboard_builder(groq_client, st.session_state.current_df, intent)
                try:
                    data = json.loads(resp)
                    charts = data.get("charts", [])
                    
                    st.subheader(data.get("dashboard_title", "AI Dashboard"))
                    
                    # PERFORMANCE OPTIMIZATION: REMOVED AGGRESSIVE DOWNSAMPLING
                    # We will use the full dataset and aggregate intelligently.
                    plot_df = st.session_state.current_df.copy()

                    for i in range(0, len(charts), 2):
                        c1, c2 = st.columns(2)
                        for col, idx in zip([c1, c2], [i, i+1]):
                            if idx < len(charts):
                                spec = charts[idx]
                                with col:
                                    st.subheader(spec.get("title"))
                                    try:
                                        t = spec.get('type', 'bar')
                                        x = spec.get('x')
                                        y = spec.get('y')
                                        c = spec.get('color')
                                        f = spec.get('filter') # Get the filter query from AI
                                        
                                        # 1. Apply AI Filter (e.g. Country == 'India')
                                        chart_data = plot_df
                                        if f:
                                            try:
                                                chart_data = chart_data.query(f)
                                            except Exception as e:
                                                st.warning(f"Could not apply filter '{f}': {e}")

                                        # 2. Validation
                                        if x and x not in chart_data.columns:
                                            st.error(f"Column '{x}' not found.")
                                            continue
                                            
                                        # 3. Smart Aggregation
                                        # If the resulting data is still huge, we aggregate to avoid messy charts
                                        if t in ['bar', 'line', 'pie'] and x:
                                            if y and y in chart_data.columns and pd.api.types.is_numeric_dtype(chart_data[y]):
                                                # Numeric Y: Sum/Mean
                                                chart_data = chart_data.groupby(x, as_index=False)[y].sum()
                                            elif not y:
                                                # No Y: Count frequency
                                                chart_data = chart_data[x].value_counts().reset_index()
                                                chart_data.columns = [x, 'Count']
                                                y = 'Count'
                                            
                                            # LIMIT: If we have > 20 categories, take top 20 to keep chart readable
                                            if len(chart_data) > 20:
                                                chart_data = chart_data.head(20)
                                        
                                        # 4. Safety Valve for Scatter/Raw plots
                                        # If it's a scatter plot and still has > 5000 points after filtering, THEN downsample
                                        if len(chart_data) > 5000:
                                            st.caption(f"⚠️ Sampling 5000 points from {len(chart_data)} results.")
                                            chart_data = chart_data.sample(5000)

                                        # 5. Plotting
                                        if t=='bar': 
                                            fig = px.bar(chart_data, x=x, y=y, color=c if c in chart_data.columns else None)
                                        elif t=='line': 
                                            fig = px.line(chart_data, x=x, y=y, color=c if c in chart_data.columns else None)
                                        elif t=='scatter': 
                                            fig = px.scatter(chart_data, x=x, y=y, color=c if c in chart_data.columns else None)
                                        elif t=='pie': 
                                            fig = px.pie(chart_data, names=x, values=y if y else None)
                                        elif t=='histogram':
                                            fig = px.histogram(chart_data, x=x, color=c if c in chart_data.columns else None)
                                        else: 
                                            fig = px.bar(chart_data, x=x, y=y)
                                            
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e: st.error(f"Error: {e}")
                except: st.error("Failed to generate dashboard.")

elif page == "🧠 AI Query Engine":
    st.title("🧠 AI SQL Engine")
    if st.session_state.source_type == "database" and not st.session_state.db_engine:
        st.warning("Please connect to a Database in the sidebar.")
        st.stop()
    if st.session_state.source_type == "file" and not st.session_state.file_path:
        st.warning("Please upload a File in the sidebar.")
        st.stop()
    with st.expander("View Schema"):
        st.code(st.session_state.db_schema)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Natural Language")
        q_text = st.text_area("Ask your data:", height=100, placeholder="Show top 10 rows...")
        if st.button("✨ Generate SQL", use_container_width=True):
            with st.spinner("Writing SQL..."):
                sql = agent_nl_to_sql(groq_client, q_text, st.session_state.db_schema, st.session_state.source_type)
                st.session_state.last_query = sql
    with col2:
        st.subheader("SQL Editor")
        sql_edit = st.text_area("Query:", value=st.session_state.last_query, height=100)
        c_run, c_fix = st.columns(2)
        with c_run:
            if st.button("▶️ Run", type="primary", use_container_width=True):
                safe, msg = check_safety(sql_edit)
                if not safe:
                    st.error(msg)
                else:
                    src = st.session_state.db_engine if st.session_state.source_type == "database" else st.session_state.file_path
                    df, err = run_query_unified(sql_edit, st.session_state.source_type, src)
                    if err:
                        st.error(f"Failed: {err}")
                        st.session_state.last_error = err
                    else:
                        st.success(f"Result: {len(df)} rows")
                        st.dataframe(df, use_container_width=True)
                        st.session_state.current_df = df 
        with c_fix:
            if st.session_state.last_error:
                if st.button("🔧 Auto-Fix", type="secondary", use_container_width=True):
                    fixed = agent_fix_sql(groq_client, sql_edit, st.session_state.last_error, st.session_state.db_schema, st.session_state.source_type)
                    st.session_state.last_query = fixed
                    st.rerun()