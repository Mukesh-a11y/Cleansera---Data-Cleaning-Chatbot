from flask import Flask, render_template, jsonify, request,send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, json, re, time, uuid, math
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
app = Flask(__name__)
CORS(app)

app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("cleaned", exist_ok=True)

GEMINI_API_KEY="YOUR API KEY HERE "

# Configure Gemini model
if HAS_GEMINI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    model = None


sessions = {}


def safe_to_records(df: pd.DataFrame, limit=None, page=0):
    """
    Convert DataFrame to JSON-serializable list-of-dicts.
    If limit is None => return all rows.
    Pagination: page is 0-indexed.
    """
    if limit is None:
        txt = df.to_json(orient="records", date_format="iso")
        return json.loads(txt)
    else:
        start = page * limit
        end = start + limit
        sub = df.iloc[start:end]
        txt = sub.to_json(orient="records", date_format="iso")
        return json.loads(txt)

def get_data_profile(df: pd.DataFrame):
    """Return columns, dtype mapping and missing counts."""
    return {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "rows": int(len(df))
    }

def analyze_with_ai(profile: dict):
    """Ask Gemini to summarize dataset. If Gemini not available, return a fallback message."""
    if not model:
        return "AI analysis is unavailable (Gemini not configured). The profile: " + json.dumps(profile, indent=2)
    prompt = f"""You are a professional data analyst. Summarize this dataset profile and suggest cleaning steps, quick visuals to run, and which columns may be numeric/datetime/categorical.
Profile:
{json.dumps(profile, indent=2)}
Provide a JSON output with keys: suggestions (list), recommended_plots (list of objects like {{type:'bar', x:'col', y:'col'}}), and summary (string).
Return strictly JSON only.
"""
    try:
        res = model.generate_content(prompt)
        # res.text may contain extra text — try to extract JSON
        return res.text
    except Exception as e:
        return f"⚠️ AI analysis failed: {e}"

def parse_json_response(text):
    """Try to salvage JSON-like responses from model (basic)."""
    if not text:
        return {"type": "message", "content": ""}
    txt = text.strip()
    # remove triple backticks blocks
    txt = txt.replace("```json", "").replace("```", "")
    # if it's valid JSON already, parse
    try:
        return json.loads(txt)
    except Exception:
        # attempt naive fix: convert unquoted keys to quoted (very simple)
        try:
            fixed = re.sub(r'(\b\w+\b)\s*:', r'"\1":', txt)
            return json.loads(fixed)
        except Exception:
            # fallback: return as message
            return {"type": "message", "content": txt}


@app.route('/')
def home():
    return render_template('start.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": ("gemini-2.5-flash" if model else "none")})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"})
    session_id = str(uuid.uuid4())
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            # supports xlsx, xls
            df = pd.read_excel(file_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed reading file: {e}"})

    # convert some common date-looking columns to datetime if possible (safe attempt)
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(10).tolist()
            # heuristics
            if any(re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', s) for s in sample):
                try:
                    df[col] = pd.to_datetime(df[col], errors="ignore", dayfirst=True)
                except Exception:
                    pass

    sessions[session_id] = [df.copy()]

    profile = get_data_profile(df)
    # return first page default (limit 100)
    preview = safe_to_records(df, limit=100, page=0)
    ai_analysis = analyze_with_ai(profile)

    return jsonify({
        "success": True,
        "session_id": session_id,
        "preview": preview,
        "profile": profile,
        "ai_analysis": ai_analysis,
        "total_rows": len(df)
    })


@app.route("/api/get_page", methods=["POST"])
def get_page():
    """
    Server-side pagination endpoint.
    POST body:
      session_id (required)
      page (required, 1-indexed)
      rows_per_page (required)
      search_query (optional, empty string means no search)
    """
    data = request.get_json() or {}
    session_id = data.get("session_id")
    page = data.get("page", 1)
    rows_per_page = data.get("rows_per_page", 50)
    search_query = data.get("search_query", "").strip()
    
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "Invalid session_id"}), 400
    
    df = sessions[session_id][-1]
    
    # Apply search filter if provided (frontend filtering backup)
    if search_query:
        # Simple case-insensitive search across all columns
        mask = df.astype(str).apply(lambda row: row.str.contains(search_query, case=False, na=False).any(), axis=1)
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    total_rows = len(filtered_df)
    total_pages = math.ceil(total_rows / rows_per_page) if total_rows > 0 else 1
    
    # Convert to 0-indexed for slicing
    page_idx = page - 1
    start = page_idx * rows_per_page
    end = start + rows_per_page
    
    page_data = filtered_df.iloc[start:end]
    preview = safe_to_records(page_data, limit=None)
    
    return jsonify({
        "success": True,
        "preview": preview,
        "page": page,
        "total_pages": total_pages,
        "total_rows": total_rows,
        "rows_per_page": rows_per_page
    })


@app.route("/api/search", methods=["POST"])
def search():
    """
    Server-side search endpoint for efficient searching through large datasets.
    POST body:
      session_id (required)
      query (required) - search term or column:value format
      page (optional, default 1)
      rows_per_page (optional, default 50)
    
    Search formats:
      - "keyword" - searches across all columns
      - "column:value" - searches specific column for value
      - "column1:value1 column2:value2" - multiple column filters
    """
    data = request.get_json() or {}
    session_id = data.get("session_id")
    query = data.get("query", "").strip()
    page = data.get("page", 1)
    rows_per_page = data.get("rows_per_page", 50)
    
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "Invalid session_id"}), 400
    
    if not query:
        return jsonify({"success": False, "error": "Empty search query"}), 400
    
    df = sessions[session_id][-1]
    
    try:
        # Parse query for column-specific searches
        column_searches = re.findall(r'(\w+):([^\s]+)', query)
        
        if column_searches:
            # Column-specific search
            mask = pd.Series([True] * len(df))
            
            for col_name, search_value in column_searches:
                if col_name in df.columns:
                    # Handle different data types
                    col_data = df[col_name]
                    
                    if pd.api.types.is_numeric_dtype(col_data):
                        # Try numeric comparison
                        try:
                            numeric_val = float(search_value)
                            mask &= (col_data == numeric_val)
                        except ValueError:
                            # If conversion fails, do string search
                            mask &= col_data.astype(str).str.contains(search_value, case=False, na=False)
                    else:
                        # String search (case-insensitive)
                        mask &= col_data.astype(str).str.contains(search_value, case=False, na=False, regex=False)
            
            filtered_df = df[mask]
        else:
            # Global search across all columns
            mask = df.astype(str).apply(
                lambda row: row.str.contains(query, case=False, na=False, regex=False).any(), 
                axis=1
            )
            filtered_df = df[mask]
        
        total_results = len(filtered_df)
        total_pages = math.ceil(total_results / rows_per_page) if total_results > 0 else 1
        
        # Paginate results
        page_idx = page - 1
        start = page_idx * rows_per_page
        end = start + rows_per_page
        
        results_page = filtered_df.iloc[start:end]
        results = safe_to_records(results_page, limit=None)
        
        return jsonify({
            "success": True,
            "results": results,
            "total_results": total_results,
            "total_pages": total_pages,
            "page": page,
            "rows_per_page": rows_per_page,
            "query": query
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Search failed: {str(e)}"
        }), 500


@app.route("/api/preview", methods=["GET"])
def preview():
    """
    GET parameters:
      session_id (required)
      limit (optional, integer or 'all')
      page (optional, 0-indexed integer)
    """
    session_id = request.args.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "Invalid session_id"}), 400
    df = sessions[session_id][-1]
    limit = request.args.get("limit", default="100")
    page = int(request.args.get("page", 0))
    if limit == "all":
        rows = safe_to_records(df, limit=None)
        return jsonify({"success": True, "preview": rows, "rows": int(len(rows))})
    else:
        try:
            limit_i = int(limit)
            rows = safe_to_records(df, limit=limit_i, page=page)
            total = len(df)
            return jsonify({"success": True, "preview": rows, "page": page, "limit": limit_i, "total_rows": total})
        except Exception as e:
            return jsonify({"success": False, "error": f"Bad limit param: {e}"}), 400

@app.route("/api/columns", methods=["GET"])
def columns():
    session_id = request.args.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "Invalid session_id"}), 400
    df = sessions[session_id][-1]
    return jsonify({"success": True, "profile": get_data_profile(df)})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    message = data.get("message", "")
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400

    df = sessions[session_id][-1].copy()
    cols = df.columns.tolist()
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    sample = df.head(5).to_string()

    # construct system prompt for AI
    system_prompt = f"""
You are an assistant that returns a single JSON dict describing the action to take.
You have the DataFrame 'df' with columns: {cols}
Data types: {dtypes}
Sample rows:
{sample}

If user asks to perform a transformation, return:
{{"type":"code", "content":"<python code using df, pd, np>", "explanation":"..."}}

If user asks to compute a numeric result, return:
{{"type":"analysis", "content":"<python code that sets result=...>", "explanation":"..."}}

If user asks to create a plot (bar/pie/line/scatter/histogram), return:
{{"type":"plot", "content":"fig = px.bar(df, x='col', y='col', title='...')", "explanation":"..."}}

If user asks a question you can't answer, return:
{{"type":"question", "content":"What column should I use for X?"}}

Respond in JSON only.
User: {message}
"""
    if not model:
        low = message.lower()
        if "sum" in low or "total" in low:
            num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if num_cols:
                col = num_cols[0]
                result = df[col].sum()
                return jsonify({"type": "analysis", "success": True, "content": f"result = df['{col}'].sum()", "result": str(result)})
            else:
                return jsonify({"type": "message", "content": "No numeric columns found to sum."})
        elif any(k in low for k in ["plot", "bar", "pie", "hist", "scatter", "visual"]):
            # recommend a simple bar using first categorical vs numeric
            cat_cols = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
            num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if cat_cols and num_cols:
                code = f"fig = px.bar(df.groupby('{cat_cols[0]}')[\"{num_cols[0]}\"].sum().reset_index(), x='{cat_cols[0]}', y='{num_cols[0]}', title='Total {num_cols[0]} by {cat_cols[0]}')"
                # execute to generate figure json
                try:
                    local = {"df": df, "px": px, "pd": pd, "np": np}
                    exec(code, {}, local)
                    fig = local.get("fig")
                    plot_json = fig.to_json()
                    return jsonify({"type": "plot", "success": True, "plot": plot_json})
                except Exception as e:
                    return jsonify({"type": "message", "content": f"Plot failed: {e}"})
            return jsonify({"type": "message", "content": "Not enough info to create plot."})
        else:
            return jsonify({"type": "message", "content": "AI model not configured. Use simple commands like 'sum sales' or 'plot sales by category'."})

    # If model exists, call it
    try:
        res = model.generate_content(system_prompt)
        parsed = parse_json_response(res.text)
    except Exception as e:
        return jsonify({"error": f"AI call failed: {e}"}), 500

    rtype = parsed.get("type", "message")
    content = parsed.get("content", "")
    explanation = parsed.get("explanation", "")

    # Execute code safely in a restricted namespace for allowed operations
    try:
        if rtype == "code":
            exec_globals = {"pd": pd, "np": np}
            exec_locals = {"df": df}
            exec(content, exec_globals, exec_locals)
            # after execution, get df (may be mutated)
            new_df = exec_locals.get("df", df)
            if not isinstance(new_df, pd.DataFrame):
                # maybe user used in-place operations
                new_df = df
            sessions[session_id].append(new_df.copy())
            return jsonify({
                "type": "code", 
                "success": True, 
                "content": content, 
                "explanation": explanation, 
                "preview": safe_to_records(new_df, limit=100, page=0), 
                "profile": get_data_profile(new_df)
            })

        elif rtype == "analysis":
            local_vars = {"df": df, "np": np, "pd": pd}
            exec(content, {}, local_vars)
            result = local_vars.get("result", None)
            return jsonify({
                "type": "analysis", 
                "success": True, 
                "content": content, 
                "result": result, 
                "explanation": explanation
            })

        elif rtype == "plot":
            local_vars = {"df": df, "px": px, "go": go, "pd": pd, "np": np}
            exec(content, {}, local_vars)
            fig = local_vars.get("fig")
            if fig:
                plot_json = fig.to_json()
                return jsonify({
                    "type": "plot", 
                    "success": True, 
                    "plot": plot_json, 
                    "explanation": explanation
                })
            else:
                return jsonify({"type": "message", "content": "No figure object produced."})

        elif rtype == "question":
            return jsonify({
                "type": "question", 
                "success": True, 
                "content": parsed.get("content", "")
            })

        else:
            return jsonify({"type": "message", "content": parsed.get("content", "")})
    except Exception as e:
        return jsonify({
            "type": rtype, 
            "success": False, 
            "error": str(e), 
            "content": content
        })


@app.route("/api/undo", methods=["POST"])
def undo():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "Invalid session"})
    if len(sessions[session_id]) <= 1:
        return jsonify({"success": False, "error": "No undo available"})
    sessions[session_id].pop()
    df = sessions[session_id][-1]
    return jsonify({
        "success": True, 
        "preview": safe_to_records(df, limit=100, page=0), 
        "profile": get_data_profile(df)
    })


@app.route("/api/download", methods=["POST"])
def download():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400
    df = sessions[session_id][-1]
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="cleaned_data.csv", mimetype="text/csv")

if __name__ == "__main__":
    print("🚀 Cleansera AI Backend running...")
    print("🌐 Open http://localhost:5000")
    print("📊 Features: Pagination, Server-side Search, AI-powered cleaning")
    app.run(host="0.0.0.0", port=5000, debug=True)