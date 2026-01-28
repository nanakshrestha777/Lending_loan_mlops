import logging
import json
import math
import os
import io
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
import redis
import mlflow.pyfunc
import pyarrow.parquet as pq
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# --- OFFICIAL IMPORTS FOR EVIDENTLY 0.4.15 ---
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LendingAPI")

app = FastAPI(title="Lending Club MLOps API")

# 2. Configuration
# Use environment variables to support both Docker and local development
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
LOCAL_ARTIFACT_ROOT = os.getenv("ARTIFACT_ROOT", "/home/nanak/mlops/mlflow_artifacts")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA", "/home/nanak/mlops/data/transformed.parquet")

# Redis Configuration - supports Docker (redis1) and local (127.0.0.1)
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "mysecurepassword")
r = redis.Redis(host=REDIS_HOST, port=6379, password=REDIS_PASSWORD, decode_responses=True)

# PostgreSQL Configuration - supports Docker (postgres_db) and local (localhost)
POSTGRES_CONFIG = {
    'host': os.getenv("POSTGRES_HOST", "127.0.0.1"), # Use IP
    'port': int(os.getenv("POSTGRES_PORT", "5432")),
    'database': os.getenv("POSTGRES_DB", "airflow"),
    'user': os.getenv("POSTGRES_USER", "airflow"),
    'password': os.getenv("POSTGRES_PASSWORD", "airflow"),
    'connect_timeout': 5,
    'gssencmode': 'disable' # Prevents GSSAPI handshake issues
}

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        return psycopg2.connect(**POSTGRES_CONFIG)
    except Exception as e:
        logger.error(f"Postgres Connection Detail Error: {e}")
        raise e

# Global Assets
model = None
scaling_params = None
reference_df = None
live_buffer = []
prediction_history = deque(maxlen=1000)  # Store last 1000 predictions
model_metadata = {}  # Store model info

@app.on_event("startup")
def load_assets():
    global model, scaling_params, reference_df, model_metadata
    
    # Create predictions table in PostgreSQL
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS loan_predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                grade VARCHAR(5),
                home_ownership VARCHAR(20),
                fico_score FLOAT,
                annual_inc FLOAT,
                int_rate FLOAT,
                dti FLOAT,
                prediction VARCHAR(10),
                confidence FLOAT,
                grade_rank FLOAT,
                is_rent INT,
                fico_scaled FLOAT,
                rate_scaled FLOAT,
                dti_scaled FLOAT,
                income_scaled FLOAT
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("✅ PostgreSQL predictions table ready")
    except Exception as e:
        logger.error(f"⚠️ PostgreSQL table creation error: {e}")
    
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        # Search Experiment ID 2 - Get MOST RECENT run (not just best F1)
        runs = mlflow.search_runs(experiment_ids=["2"], order_by=["start_time DESC"])
        
        if not runs.empty:
            # Use the most recent run
            best_run = runs.iloc[0]
            run_id = best_run.run_id
            run_folder = os.path.join(LOCAL_ARTIFACT_ROOT, "2", run_id, "artifacts")
            
            logger.info(f"📦 Loading most recent model from run: {run_id}")
            logger.info(f"   Start time: {best_run.get('start_time', 'N/A')}")
            
            # Store model metadata
            model_metadata = {
                'run_id': run_id,
                'f1_score': best_run.get('metrics.f1_score', 'N/A'),
                'accuracy': best_run.get('metrics.accuracy', 'N/A'),
                'precision': best_run.get('metrics.precision', 'N/A'),
                'recall': best_run.get('metrics.recall', 'N/A'),
                'start_time': best_run.get('start_time', 'N/A')
            }
            
            # Load Model
            model = mlflow.pyfunc.load_model(os.path.join(run_folder, "model"))
            # Load Scaling JSON
            with open(os.path.join(run_folder, "preprocessing_params.json"), "r") as f:
                scaling_params = json.load(f)
            
            # --- Load Reference Data (Baseline) ---
            logger.info(f"📂 Loading reference data from: {REFERENCE_DATA_PATH}")
            if os.path.exists(REFERENCE_DATA_PATH):
                try:
                    raw_ref_df = pd.read_parquet(REFERENCE_DATA_PATH)
                    logger.info(f"📊 Raw reference data loaded: {len(raw_ref_df)} rows")
                    
                    # Check if required columns exist
                    required_cols = ['grade', 'home_ownership', 'fico_score', 'int_rate', 'dti', 'annual_inc']
                    missing_cols = [col for col in required_cols if col not in raw_ref_df.columns]
                    if missing_cols:
                        logger.error(f"❌ Missing columns in reference data: {missing_cols}")
                        logger.error(f"   Available columns: {list(raw_ref_df.columns)}")
                        raise ValueError(f"Missing required columns: {missing_cols}")
                    
                    # Transform baseline to match the 6-feature model
                    ref_p = pd.DataFrame()
                    ref_p['grade_rank'] = raw_ref_df['grade'].map({"A": 1, "B": 2, "C": 3, "D": 4}).fillna(5)
                    ref_p['is_rent'] = (raw_ref_df['home_ownership'] == 'RENT').astype(int)
                    ref_p['fico_scaled'] = (raw_ref_df['fico_score'] - 300) / 550.0
                    ref_p['rate_scaled'] = raw_ref_df['int_rate'] / 35.0
                    ref_p['dti_scaled'] = raw_ref_df['dti'] / 100.0
                    
                    # Income scaling using the logic from your training
                    inc_log = (1 + raw_ref_df['annual_inc']).apply(math.log)
                    m_inc = scaling_params.get('min_inc_log') or scaling_params.get('min_inc') or 0
                    mx_inc = scaling_params.get('max_inc_log') or scaling_params.get('max_inc') or 15
                    
                    # Add epsilon to prevent division by zero in scaling
                    ref_p['income_scaled'] = (inc_log - m_inc) / (mx_inc - m_inc + 1e-9)
                    
                    # Final Clean: Drop any NaNs/Infs that might break statistical tests
                    reference_df = ref_p.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
                    
                    logger.info(f"✅ Baseline Ready: {len(reference_df)} rows")
                except Exception as ref_error:
                    logger.error(f"❌ Failed to load/transform reference data: {ref_error}")
                    logger.error(f"   File: {REFERENCE_DATA_PATH}")
                    reference_df = None
            else:
                logger.warning(f"⚠️ Reference data file not found: {REFERENCE_DATA_PATH}")
                reference_df = None
            
            logger.info(f"✅ API READY. Model Run: {run_id}")
        else:
            logger.warning("⚠️ No MLflow runs found in experiment 2")
            
    except Exception as e:
        logger.error(f"💥 Startup Error: {e}")
        
        # Provide helpful troubleshooting info
        if "binary format has been deprecated" in str(e):
            logger.error("=" * 60)
            logger.error("🔧 XGBoost Version Mismatch Detected!")
            logger.error("   Your model was saved with an older XGBoost version.")
            logger.error("   ")
            logger.error("   SOLUTION: Re-run your Airflow DAG to retrain the model:")
            logger.error("   docker exec airflow_webserver airflow dags trigger test__dag")
            logger.error("   ")
            logger.error("   Or check Airflow UI: http://localhost:8080")
            logger.error("=" * 60)
        
        # Don't let API crash completely - it can still handle health checks
        model = None
        scaling_params = None
        reference_df = None

class LoanRequest(BaseModel):
    grade: str
    home_ownership: str
    fico_score: float
    annual_inc: float
    int_rate: float
    dti: float

@app.post("/predict")
def predict(user_input: LoanRequest):
    if not model or not scaling_params:
        raise HTTPException(status_code=503, detail="Assets not loaded")
    try:
        # Pre-processing
        g_rank = {"A": 1, "B": 2, "C": 3, "D": 4}.get(user_input.grade.upper(), 5)
        rent_flag = 1 if user_input.home_ownership.upper() == "RENT" else 0
        inc_log = math.log(1 + user_input.annual_inc)
        m_inc = scaling_params.get('min_inc_log') or scaling_params.get('min_inc') or 0
        mx_inc = scaling_params.get('max_inc_log') or scaling_params.get('max_inc') or 15
        inc_scaled = (inc_log - m_inc) / (mx_inc - m_inc + 1e-9)

        features_dict = {
            "grade_rank": float(g_rank), 
            "is_rent": float(rent_flag),
            "fico_scaled": float((user_input.fico_score - 300) / 550.0),
            "rate_scaled": float(user_input.int_rate / 35.0),
            "dti_scaled": float(user_input.dti / 100.0),
            "income_scaled": float(inc_scaled)
        }
        
        # Save to buffer for Drift (Monitoring)
        live_buffer.append(features_dict)
        if len(live_buffer) > 500: live_buffer.pop(0)

        pred = int(model.predict(pd.DataFrame([features_dict]))[0])
        result = "REJECT" if pred == 1 else "APPROVE"
        
        # Store in prediction history
        prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'input': user_input.dict(),
            'prediction': result,
            'confidence': float(pred)
        })
        
        # Store in PostgreSQL database
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO loan_predictions 
                (grade, home_ownership, fico_score, annual_inc, int_rate, dti, 
                 prediction, confidence, grade_rank, is_rent, fico_scaled, 
                 rate_scaled, dti_scaled, income_scaled)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_input.grade, user_input.home_ownership, user_input.fico_score,
                user_input.annual_inc, user_input.int_rate, user_input.dti,
                result, float(pred), features_dict['grade_rank'], features_dict['is_rent'],
                features_dict['fico_scaled'], features_dict['rate_scaled'],
                features_dict['dti_scaled'], features_dict['income_scaled']
            ))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_error:
            logger.error(f"⚠️ PostgreSQL insert error: {db_error}")
        
        r.incr("api:predict_count")
        return {"prediction": result, "confidence": float(pred)}
    except Exception as e:
        logger.error(f"❌ Predict Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(requests: List[LoanRequest]):
    """Batch prediction endpoint - accepts multiple loan requests at once"""
    if not model or not scaling_params:
        raise HTTPException(status_code=503, detail="Assets not loaded")
    
    results = []
    for user_input in requests:
        try:
            # Pre-processing
            g_rank = {"A": 1, "B": 2, "C": 3, "D": 4}.get(user_input.grade.upper(), 5)
            rent_flag = 1 if user_input.home_ownership.upper() == "RENT" else 0
            inc_log = math.log(1 + user_input.annual_inc)
            m_inc = scaling_params.get('min_inc_log') or scaling_params.get('min_inc') or 0
            mx_inc = scaling_params.get('max_inc_log') or scaling_params.get('max_inc') or 15
            inc_scaled = (inc_log - m_inc) / (mx_inc - m_inc + 1e-9)

            features_dict = {
                "grade_rank": float(g_rank), 
                "is_rent": float(rent_flag),
                "fico_scaled": float((user_input.fico_score - 300) / 550.0),
                "rate_scaled": float(user_input.int_rate / 35.0),
                "dti_scaled": float(user_input.dti / 100.0),
                "income_scaled": float(inc_scaled)
            }
            
            live_buffer.append(features_dict)
            if len(live_buffer) > 500: live_buffer.pop(0)

            pred = int(model.predict(pd.DataFrame([features_dict]))[0])
            result = "REJECT" if pred == 1 else "APPROVE"
            
            prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'input': user_input.dict(),
                'prediction': result,
                'confidence': float(pred)
            })
            
            # Store in PostgreSQL
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO loan_predictions 
                    (grade, home_ownership, fico_score, annual_inc, int_rate, dti, 
                     prediction, confidence, grade_rank, is_rent, fico_scaled, 
                     rate_scaled, dti_scaled, income_scaled)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_input.grade, user_input.home_ownership, user_input.fico_score,
                    user_input.annual_inc, user_input.int_rate, user_input.dti,
                    result, float(pred), features_dict['grade_rank'], features_dict['is_rent'],
                    features_dict['fico_scaled'], features_dict['rate_scaled'],
                    features_dict['dti_scaled'], features_dict['income_scaled']
                ))
                conn.commit()
                cur.close()
                conn.close()
            except Exception as db_error:
                logger.error(f"⚠️ PostgreSQL insert error: {db_error}")
            
            r.incr("api:predict_count")
            results.append({"input": user_input.dict(), "prediction": result, "confidence": float(pred)})
        except Exception as e:
            logger.error(f"❌ Batch Predict Error for request: {e}")
            results.append({"input": user_input.dict(), "error": str(e)})
    
    return {"total": len(results), "results": results}

@app.get("/monitor/drift")
def get_drift():
    try:
        # Pull recent predictions from PostgreSQL (last 500 records)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT grade_rank, is_rent, fico_scaled, rate_scaled, dti_scaled, income_scaled
            FROM loan_predictions
            ORDER BY timestamp DESC
            LIMIT 500
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Check if we have enough predictions
        if len(rows) < 5:
            return HTMLResponse(content=f"""
                <h1>Need at least 5 predictions in database</h1>
                <p>Current predictions: {len(rows)}</p>
                <p>Please make some predictions first at <a href="/ui/predict">/ui/predict</a></p>
            """)
        
        # Convert to DataFrame
        current_df = pd.DataFrame(rows, columns=[
            'grade_rank', 'is_rent', 'fico_scaled', 'rate_scaled', 'dti_scaled', 'income_scaled'
        ]).astype(float)
        
        # Check if reference data is loaded
        if reference_df is None or len(reference_df) == 0:
            return HTMLResponse(content=f"""
                <h1>Reference Data Not Available</h1>
                <p>Reference baseline data is not loaded. Please check:</p>
                <ul>
                    <li>File exists: {REFERENCE_DATA_PATH}</li>
                    <li>Check API startup logs for errors</li>
                    <li>Ensure the DAG has run successfully to create transformed.parquet</li>
                </ul>
            """)
        
        # Sample Reference Data (baseline)
        reference_sample = reference_df.sample(n=min(5000, len(reference_df)), random_state=42)
        
        # Initialize Report
        report = Report(metrics=[DataDriftPreset()])
        
        # Run calculation
        report.run(reference_data=reference_sample, current_data=current_df)
        
        # Extract HTML
        buffer = io.StringIO()
        report.save_html(buffer)
        return HTMLResponse(content=buffer.getvalue())
            
    except Exception as e:
        logger.error(f"❌ Drift Error: {e}")
        return HTMLResponse(content=f"<h1>Report Error: {str(e)}</h1><p>Make sure you have made at least 5 predictions and PostgreSQL is running.</p>")

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Main Dashboard Homepage"""
    try:
        predict_count = r.get("api:predict_count") or 0
    except:
        predict_count = 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>s
    <head>
        <title>Lending Club Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                color: white;
                margin-bottom: 40px;
                padding: 20px;
            }}
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                transition: transform 0.3s;
            }}
            .stat-card:hover {{
                transform: translateY(-5px);
            }}
            .stat-card h3 {{
                color: #667eea;
                font-size: 0.9em;
                text-transform: uppercase;
                margin-bottom: 10px;
            }}
            .stat-card .value {{
                font-size: 2.5em;
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }}
            .stat-card .label {{
                color: #666;
                font-size: 0.9em;
            }}
            .nav-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
            }}
            .nav-card {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                text-decoration: none;
                color: inherit;
                transition: all 0.3s;
                border: 3px solid transparent;
            }}
            .nav-card:hover {{
                transform: translateY(-5px);
                border-color: #667eea;
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
            }}
            .nav-card h2 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.5em;
            }}
            .nav-card p {{
                color: #666;
                line-height: 1.6;
            }}
            .nav-card .icon {{
                font-size: 3em;
                margin-bottom: 15px;
            }}
            .status-indicator {{
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #4CAF50;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏦 Lending Club Dashboard</h1>
                <p><span class="status-indicator"></span>System Online & Ready</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Predictions</h3>
                    <div class="value">{predict_count}</div>
                    <div class="label">API calls made</div>
                </div>
                <div class="stat-card">
                    <h3>Model Status</h3>
                    <div class="value">{'✓' if model else '✗'}</div>
                    <div class="label">{'Loaded' if model else 'Not Loaded'}</div>
                </div>
                <div class="stat-card">
                    <h3>Recent Predictions</h3>
                    <div class="value">{len(prediction_history)}</div>
                    <div class="label">In history buffer</div>
                </div>
                <div class="stat-card">
                    <h3>Model F1 Score</h3>
                    <div class="value">{model_metadata.get('f1_score', 'N/A')}</div>
                    <div class="label">Best run metric</div>
                </div>
            </div>
            
            <div class="nav-grid">
                <a href="/ui/predict" class="nav-card">
                    <div class="icon">🎯</div>
                    <h2>Make Prediction</h2>
                    <p>Submit loan application data and get instant approval/rejection prediction</p>
                </a>
                
                <a href="/monitor/drift" class="nav-card">
                    <div class="icon">📈</div>
                    <h2>Drift Monitoring</h2>
                    <p>View data drift analysis and model performance monitoring</p>
                </a>
                
                <a href="/ui/history" class="nav-card">
                    <div class="icon">🕐</div>
                    <h2>Prediction History</h2>
                    <p>Browse recent predictions and download historical data</p>
                </a>
                
                <a href="/ui/metrics" class="nav-card">
                    <div class="icon">📉</div>
                    <h2>Model Metrics</h2>
                    <p>View detailed model performance metrics and metadata</p>
                </a>
                
                <a href="/docs" class="nav-card">
                    <div class="icon">📚</div>
                    <h2>API Documentation</h2>
                    <p>Interactive Swagger UI for testing API endpoints</p>
                </a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
def health():
    """Health check endpoint with database connection status"""
    health_status = {
        "status": "online",
        "model_loaded": model is not None,
        "redis_connected": False,
        "postgres_connected": False,
        "postgres_predictions_count": 0
    }
    
    # Test Redis
    try:
        r.ping()
        health_status["redis_connected"] = True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
    
    # Test PostgreSQL
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM loan_predictions")
        count = cur.fetchone()[0]
        health_status["postgres_connected"] = True
        health_status["postgres_predictions_count"] = count
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        health_status["postgres_error"] = str(e)
    
    return health_status

@app.get("/ui/predict", response_class=HTMLResponse)
def predict_form():
    """Interactive Prediction Form"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Loan Prediction Form</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
            .back-btn {
                display: inline-block;
                color: white;
                text-decoration: none;
                margin-bottom: 20px;
                padding: 10px 20px;
                background: rgba(255,255,255,0.2);
                border-radius: 5px;
                transition: all 0.3s;
            }
            .back-btn:hover {
                background: rgba(255,255,255,0.3);
            }
            .card {
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                color: #667eea;
                margin-bottom: 30px;
                text-align: center;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                color: #333;
                font-weight: 500;
            }
            input, select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1em;
                transition: border 0.3s;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            .btn {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1.1em;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.3s;
            }
            .btn:hover {
                transform: translateY(-2px);
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                font-size: 1.3em;
                font-weight: bold;
                display: none;
            }
            .result.approve {
                background: #4CAF50;
                color: white;
            }
            .result.reject {
                background: #f44336;
                color: white;
            }
            .loading {
                text-align: center;
                display: none;
                margin-top: 20px;
            }
            .hint {
                font-size: 0.85em;
                color: #666;
                margin-top: 3px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-btn">← Back to Dashboard</a>
            <div class="card">
                <h1>🎯 Loan Prediction Form</h1>
                <form id="predictionForm">
                    <div class="form-group">
                        <label>Grade</label>
                        <select name="grade" required>
                            <option value="">Select Grade</option>
                            <option value="A">A - Excellent</option>
                            <option value="B">B - Very Good</option>
                            <option value="C">C - Good</option>
                            <option value="D">D - Fair</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Home Ownership</label>
                        <select name="home_ownership" required>
                            <option value="">Select Status</option>
                            <option value="RENT">Rent</option>
                            <option value="OWN">Own</option>
                            <option value="MORTGAGE">Mortgage</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>FICO Score</label>
                        <input type="number" name="fico_score" min="300" max="850" required>
                        <div class="hint">Range: 300-850</div>
                    </div>
                    
                    <div class="form-group">
                        <label>Annual Income ($)</label>
                        <input type="number" name="annual_inc" min="0" step="1000" required>
                        <div class="hint">Enter full annual income</div>
                    </div>
                    
                    <div class="form-group">
                        <label>Interest Rate (%)</label>
                        <input type="number" name="int_rate" min="0" max="35" step="0.01" required>
                        <div class="hint">Range: 0-35%</div>
                    </div>
                    
                    <div class="form-group">
                        <label>Debt-to-Income Ratio (%)</label>
                        <input type="number" name="dti" min="0" max="100" step="0.01" required>
                        <div class="hint">Range: 0-100%</div>
                    </div>
                    
                    <button type="submit" class="btn">Get Prediction</button>
                </form>
                
                <div class="loading">⏳ Processing...</div>
                <div class="result" id="result"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = {
                    grade: formData.get('grade'),
                    home_ownership: formData.get('home_ownership'),
                    fico_score: parseFloat(formData.get('fico_score')),
                    annual_inc: parseFloat(formData.get('annual_inc')),
                    int_rate: parseFloat(formData.get('int_rate')),
                    dti: parseFloat(formData.get('dti'))
                };
                
                document.querySelector('.loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (result.prediction === 'APPROVE') {
                        resultDiv.className = 'result approve';
                        resultDiv.innerHTML = '✓ LOAN APPROVED';
                    } else {
                        resultDiv.className = 'result reject';
                        resultDiv.innerHTML = '✗ LOAN REJECTED';
                    }
                    
                    resultDiv.style.display = 'block';
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.querySelector('.loading').style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/ui/history", response_class=HTMLResponse)
def prediction_history_ui():
    """Prediction History Page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction History</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            .back-btn {
                display: inline-block;
                color: white;
                text-decoration: none;
                margin-bottom: 20px;
                padding: 10px 20px;
                background: rgba(255,255,255,0.2);
                border-radius: 5px;
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                color: #667eea;
                margin-bottom: 20px;
            }
            .controls {
                margin-bottom: 20px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
            .btn {
                padding: 10px 20px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            .btn:hover {
                background: #5568d3;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background: #667eea;
                color: white;
                font-weight: bold;
            }
            tr:hover {
                background: #f5f5f5;
            }
            .approve {
                color: #4CAF50;
                font-weight: bold;
            }
            .reject {
                color: #f44336;
                font-weight: bold;
            }
            .no-data {
                text-align: center;
                padding: 40px;
                color: #666;
                font-size: 1.2em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-btn">← Back to Dashboard</a>
            <div class="card">
                <h1>🕐 Prediction History</h1>
                <div class="controls">
                    <button class="btn" onclick="refreshHistory()">🔄 Refresh</button>
                    <button class="btn" onclick="exportCSV()">📥 Export CSV</button>
                    <button class="btn" onclick="clearHistory()">🗑️ Clear History</button>
                </div>
                <div id="historyTable"></div>
            </div>
        </div>
        
        <script>
            async function loadHistory() {
                try {
                    const response = await fetch('/api/history');
                    const data = await response.json();
                    
                    if (data.length === 0) {
                        document.getElementById('historyTable').innerHTML = 
                            '<div class="no-data">No predictions yet</div>';
                        return;
                    }
                    
                    let html = '<table><thead><tr>';
                    html += '<th>Timestamp</th><th>Grade</th><th>FICO</th><th>Income</th>';
                    html += '<th>Int Rate</th><th>DTI</th><th>Home</th><th>Result</th></tr></thead><tbody>';
                    
                    data.reverse().forEach(item => {
                        const input = item.input;
                        const predClass = item.prediction === 'APPROVE' ? 'approve' : 'reject';
                        html += `<tr>
                            <td>${new Date(item.timestamp).toLocaleString()}</td>
                            <td>${input.grade}</td>
                            <td>${input.fico_score}</td>
                            <td>$${input.annual_inc.toLocaleString()}</td>
                            <td>${input.int_rate}%</td>
                            <td>${input.dti}%</td>
                            <td>${input.home_ownership}</td>
                            <td class="${predClass}">${item.prediction}</td>
                        </tr>`;
                    });
                    
                    html += '</tbody></table>';
                    document.getElementById('historyTable').innerHTML = html;
                } catch (error) {
                    console.error('Error loading history:', error);
                }
            }
            
            function refreshHistory() {
                loadHistory();
            }
            
            async function exportCSV() {
                try {
                    const response = await fetch('/api/history/export');
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'prediction_history.csv';
                    a.click();
                } catch (error) {
                    alert('Error exporting CSV: ' + error.message);
                }
            }
            
            function clearHistory() {
                if (confirm('Are you sure you want to clear all prediction history?')) {
                    fetch('/api/history/clear', { method: 'POST' })
                        .then(() => loadHistory());
                }
            }
            
            loadHistory();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/ui/metrics", response_class=HTMLResponse)
def model_metrics_ui():
    """Model Metrics Page"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Metrics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
            }}
            .back-btn {{
                display: inline-block;
                color: white;
                text-decoration: none;
                margin-bottom: 20px;
                padding: 10px 20px;
                background: rgba(255,255,255,0.2);
                border-radius: 5px;
            }}
            .card {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-bottom: 20px;
            }}
            h1 {{
                color: #667eea;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.3em;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-box {{
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                text-align: center;
                border-left: 4px solid #667eea;
            }}
            .metric-box .label {{
                color: #666;
                font-size: 0.9em;
                margin-bottom: 10px;
            }}
            .metric-box .value {{
                font-size: 2em;
                font-weight: bold;
                color: #333;
            }}
            .info-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .info-table td {{
                padding: 12px;
                border-bottom: 1px solid #e0e0e0;
            }}
            .info-table td:first-child {{
                font-weight: bold;
                color: #667eea;
                width: 200px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-btn">← Back to Dashboard</a>
            <div class="card">
                <h1>📉 Model Performance Metrics</h1>
                
                <h2>Performance Scores</h2>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="label">F1 Score</div>
                        <div class="value">{model_metadata.get('f1_score', 'N/A')}</div>
                    </div>
                    <div class="metric-box">
                        <div class="label">Accuracy</div>
                        <div class="value">{model_metadata.get('accuracy', 'N/A')}</div>
                    </div>
                    <div class="metric-box">
                        <div class="label">Precision</div>
                        <div class="value">{model_metadata.get('precision', 'N/A')}</div>
                    </div>
                    <div class="metric-box">
                        <div class="label">Recall</div>
                        <div class="value">{model_metadata.get('recall', 'N/A')}</div>
                    </div>
                </div>
                
                <h2>Model Information</h2>
                <table class="info-table">
                    <tr>
                        <td>Run ID</td>
                        <td>{model_metadata.get('run_id', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Training Date</td>
                        <td>{model_metadata.get('start_time', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Model Status</td>
                        <td>{'✓ Loaded' if model else '✗ Not Loaded'}</td>
                    </tr>
                    <tr>
                        <td>Features</td>
                        <td>6 (grade_rank, is_rent, fico_scaled, rate_scaled, dti_scaled, income_scaled)</td>
                    </tr>
                    <tr>
                        <td>MLflow URI</td>
                        <td>{MLFLOW_URI}</td>
                    </tr>
                    <tr>
                        <td>Experiment ID</td>
                        <td>2</td>
                    </tr>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/history")
def get_history():
    """API endpoint to get prediction history from PostgreSQL"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, timestamp, grade, home_ownership, fico_score, 
                   annual_inc, int_rate, dti, prediction, confidence
            FROM loan_predictions 
            ORDER BY timestamp DESC 
            LIMIT 1000
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dicts and format timestamps
        results = []
        for row in rows:
            row_dict = dict(row)
            if row_dict.get('timestamp'):
                row_dict['timestamp'] = row_dict['timestamp'].isoformat()
            results.append(row_dict)
        
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"❌ Error fetching history: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/history/clear")
def clear_history():
    """Clear prediction history"""
    prediction_history.clear()
    return {"status": "cleared"}

@app.get("/api/history/export")
def export_history():
    """Export prediction history as CSV"""
    if not prediction_history:
        return StreamingResponse(
            iter(["No data to export"]),
            media_type="text/plain"
        )
    
    # Create CSV
    rows = []
    rows.append("timestamp,grade,home_ownership,fico_score,annual_inc,int_rate,dti,prediction")
    
    for item in prediction_history:
        inp = item['input']
        rows.append(f"{item['timestamp']},{inp['grade']},{inp['home_ownership']},{inp['fico_score']},{inp['annual_inc']},{inp['int_rate']},{inp['dti']},{item['prediction']}")
    
    csv_content = "\n".join(rows)
    
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"}
    )


