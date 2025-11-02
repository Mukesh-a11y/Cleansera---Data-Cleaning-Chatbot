import streamlit as st
import pandas as pd
import numpy as np
import random
import json
import time
import re
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai


GEMINI_API_KEY = "AIzaSyBmgl-SWUcQo14MBz9BCC3lE-ZgJWVCwGs" 
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


st.set_page_config(page_title="Cleansera – AI-Powered Data Assistant", layout="wide")
st.title("Cleansera – AI-Powered Data Assistant")

if "df_history" not in st.session_state:
    st.session_state.df_history = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_data_profile(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    profile = (
        f"Dataframe Info:\n{info_str}\n\n"
        f"Missing Values:\n{df.isnull().sum().to_string()}\n\n"
        f"Descriptive Statistics:\n{df.describe().to_string()}"
    )
    return profile


def get_initial_analysis(profile: str) -> str:
    """Generate an initial data summary from Gemini."""
    try:
        prompt = (
            "You are an expert data analyst. Summarize this dataframe "
            "and suggest initial cleaning steps. Keep it short and clear.\n\n"
            f"{profile}"
        )
        response = model.generate_content(prompt)
        return response.text if response and response.text else "No response generated."
    except Exception as e:
        return f"⚠️ Could not analyze the data. Error: {e}"


def execute_cleaning_code(code_to_execute: str, df_placeholder):
    """Safely execute Pandas code modifications."""
    if not st.session_state.df_history:
        return "No dataframe available."

    try:
        code_blocks = re.findall(r"```(?:python)?(.*?)```", code_to_execute, re.DOTALL)
        if code_blocks:
            code_to_execute = code_blocks[0].strip()

        current_df = st.session_state.df_history[-1].copy()
        local_vars = {"df": current_df}
        global_vars = {"pd": pd, "np": np, "random": random}

        exec(code_to_execute, global_vars, local_vars)
        updated_df = local_vars.get("df", None)

        if isinstance(updated_df, pd.DataFrame):
            st.session_state.df_history.append(updated_df)
            df_placeholder.dataframe(updated_df)
            return True
        else:
            return "No dataframe ('df') returned."
    except Exception as e:
        return f"Error executing code: {e}"


def fix_json_string(text: str) -> str:
    """Repair malformed JSON responses from Gemini."""
    try:
        text = text.strip().replace("```json", "").replace("```", "")
        text = re.sub(r"(\w+):", r'"\1":', text)  # unquoted keys
        text = re.sub(r"[\n\t]", "", text)
        match = re.search(r"\{.*\}", text)
        if match:
            text = match.group(0)
        return text
    except Exception:
        return text


def safe_parse_json(text: str) -> dict:
    """Attempt to parse Gemini output as JSON with multiple fallback layers."""
    text = fix_json_string(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try double repair pass
        text = fix_json_string(text)
        try:
            return json.loads(text)
        except Exception:
            # Final fallback: wrap non-JSON as message
            return {"type": "message", "content": text}

with st.sidebar:
    st.header("🧩 System Controls")

    if len(st.session_state.df_history) > 1:
        if st.button("↩ Undo Last Action"):
            st.session_state.df_history.pop()
            st.session_state.messages.pop() if st.session_state.messages else None
            st.rerun()

    if st.session_state.df_history:
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=False).encode("utf-8")

        csv = convert_df_to_csv(st.session_state.df_history[-1])
        st.download_button(
            label="⬇ Download Cleaned CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    
    # Display current columns
    if st.session_state.df_history:
        st.subheader(" Current Columns")
        cols = st.session_state.df_history[-1].columns.tolist()
        for col in cols:
            dtype = str(st.session_state.df_history[-1][col].dtype)
            st.text(f"• {col} ({dtype})")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_to_load = pd.read_csv(
                    uploaded_file, sep=None, engine="python", on_bad_lines="skip", encoding="utf-8-sig"
                )
            else:
                df_to_load = pd.read_excel(uploaded_file)

            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.df_history = [df_to_load]
            st.session_state.messages = []

            with st.spinner(" Analyzing your data..."):
                profile = get_data_profile(df_to_load)
                initial_analysis = get_initial_analysis(profile)
                st.session_state.messages.append({"role": "assistant", "content": initial_analysis})
                st.rerun()

        except Exception as e:
            st.error(f" Error reading the file: {e}")

placeholder = st.empty()

if st.session_state.df_history:
    placeholder.dataframe(st.session_state.df_history[-1])
else:
    st.info("📎 Please upload a file to start.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(" What do you want to do with the data?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(" Thinking..."):
            try:
                if not st.session_state.df_history:
                    st.error("Please upload a file first.")
                    st.stop()
                
                current_df = st.session_state.df_history[-1]
                columns_list = current_df.columns.tolist()
                dtypes_info = {col: str(dtype) for col, dtype in current_df.dtypes.items()}
                sample_data = current_df.head(3).to_string()

                system_prompt = f"""
You are an intelligent Python data assistant specializing in data manipulation and visualization.

CONTEXT:
- Working with pandas DataFrame called 'df'
- Current Columns: {columns_list}
- Column Data Types: {dtypes_info}
- Sample Data (first 3 rows):
{sample_data}

AVAILABLE LIBRARIES:
- pandas as pd
- numpy as np
- plotly.express as px
- plotly.graph_objects as go

SUPPORTED OPERATIONS:

1️⃣ CLARIFYING QUESTION:
{{"type": "question", "content": "Your clarifying question here"}}

2️⃣ DATA MODIFICATION (Cleaning, Adding, Removing Columns):
{{"type": "code", "content": "df.dropna(inplace=True)"}}
Examples:
- Remove column: df.drop('column_name', axis=1, inplace=True)
- Add column: df['new_col'] = df['col1'] + df['col2']
- Rename column: df.rename(columns={{'old': 'new'}}, inplace=True)
- Fill missing values: df['col'].fillna(value, inplace=True)
- Change data type: df['col'] = df['col'].astype('float')

3️⃣ DATA FILTERING/FINDING:
{{"type": "find", "content": "result = df[df['Price'] > 100]"}}

4️⃣ VISUALIZATION (Bar, Pie, Histogram, Line, Scatter, Box, etc.):
{{"type": "plot", "content": "fig = px.bar(df, x='category', y='value', title='My Chart')"}}

Available Plot Types:
- Bar Chart: px.bar(df, x='col1', y='col2')
- Pie Chart: px.pie(df, names='category', values='value')
- Histogram: px.histogram(df, x='col')
- Line Chart: px.line(df, x='date', y='value')
- Scatter Plot: px.scatter(df, x='col1', y='col2')
- Box Plot: px.box(df, x='category', y='value')
- Heatmap: px.imshow(df.corr())
- Area Chart: px.area(df, x='col1', y='col2')

5️⃣ STATISTICAL ANALYSIS:
{{"type": "analysis", "content": "result = df.groupby('category')['value'].mean()"}}

CRITICAL RULES:
- Output ONLY valid JSON (one of the 5 types above)
- DO NOT import any libraries
- Always use correct column names from the list provided
- For plots, always include appropriate x, y parameters and titles
- For modifications, always use inplace=True or reassign to df
- Ensure all Python syntax is correct
- If user asks for specific columns, use exact column names
- Handle potential errors (missing values, data types)

User Request: {prompt}

Analyze the request and return appropriate JSON response.
"""

                response = model.generate_content(system_prompt)
                if not response or not response.text:
                    st.error("No valid response from Gemini.")
                    st.stop()

                response_json = safe_parse_json(response.text)
                response_type = response_json.get("type", "message")
                response_content = response_json.get("content", "")

                # ===== HANDLE RESPONSE TYPES =====
                if response_type == "question":
                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

                elif response_type == "code":
                    st.code(response_content, language="python")
                    result = execute_cleaning_code(response_content, placeholder)
                    if result is True:
                        msg = "✅ Done! The dataframe has been updated."
                        st.markdown(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                    else:
                        st.error(result)
                        st.session_state.messages.append({"role": "assistant", "content": result})

                elif response_type == "find":
                    st.code(response_content, language="python")
                    local_vars = {"df": current_df, "pd": pd, "np": np}
                    exec(response_content, {}, local_vars)
                    result = local_vars.get("result")
                    if result is not None:
                        st.success("🔍 Here's what I found:")
                        st.write(result)
                        msg = f"Found results:\n```\n{result}\n```"
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                    else:
                        st.error("⚠️ No result found.")

                elif response_type == "plot":
                    st.code(response_content, language="python")
                    local_vars = {"df": current_df, "px": px, "go": go, "pd": pd, "np": np}
                    exec(response_content, {}, local_vars)
                    fig = local_vars.get("fig")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        msg = " Here's the chart you requested."
                        st.markdown(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                    else:
                        st.error("⚠️ Could not generate the plot.")

                elif response_type == "analysis":
                    st.code(response_content, language="python")
                    local_vars = {"df": current_df, "pd": pd, "np": np}
                    exec(response_content, {}, local_vars)
                    result = local_vars.get("result")
                    if result is not None:
                        st.success(" Analysis Results:")
                        st.write(result)
                        msg = f"Analysis:\n```\n{result}\n```"
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                    else:
                        st.error(" No result from analysis.")

                else:
                    st.warning("Unstructured response from Gemini.")
                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})