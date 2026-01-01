# main.py (under /src)
# ==========================================================
# Watt Tomorrow API
# Exposes:
# - /predict           → model prediction
# - /summarize         → HuggingFace summarizer
# - /predict/{id}      → read saved prediction by ID
# - /predict?page=...  → paginated predictions
# - /                  → serves /src/static/index.html UI
# ==========================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path
import json, joblib
import torch
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import transformers, os

# ---------------------------------------------------------
# 1) Setup FastAPI
# ---------------------------------------------------------
app = FastAPI(title="Watt Tomorrow API", version="1.0.0")

# ---------------------------------------------------------
# 2) Paths and model loading
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent  # src/
BM_DIR = PROJECT_ROOT / "best_model"
MODEL_PATH = BM_DIR / "model.joblib"
META_PATH = BM_DIR / "model_meta.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

# Load model and metadata
model = joblib.load(MODEL_PATH)
with open(META_PATH, 'r') as f:
    meta = json.load(f)

feature_names = meta["feature_names"]

# ---------------------------------------------------------
# 3) Prediction Input Schema
# ---------------------------------------------------------
class PricePredictionInput(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    dayofweek: int = Field(..., ge=0, le=6)
    day: int = Field(..., ge=1, le=31)
    dayofyear: int = Field(..., ge=1, le=366)
    weekofyear: int = Field(..., ge=1, le=53)
    month: int = Field(..., ge=1, le=12)
    year: int
    is_weekend: bool

    # Generation and forecast features
    generation_other: float
    generation_coal: float
    generation_oil: float
    generation_natural_gas: float
    generation_hydro: float
    generation_nuclear: float
    generation_solar: float
    generation_wind: float
    forecast_solar: float
    forecast_wind: float

    total_load_forecast: float
    total_load_actual: float
    price_day_ahead: float

    # Weather
    temp: float
    temp_min: float
    temp_max: float
    pressure: float
    wind_speed: float
    wind_deg: float
    rain_1h: float
    rain_3h: float
    snow_3h: float

    # Price lags
    price_actual: float
    price_actual_lag23: float
    price_actual_lag24: float
    price_actual_lag47: float
    price_actual_lag48: float
    price_actual_lag71: float
    price_actual_lag72: float
    price_actual_lag95: float
    price_actual_lag96: float
    price_actual_lag119: float
    price_actual_lag120: float
    price_actual_lag143: float
    price_actual_lag144: float

    # Rolling windows
    price_actual_rolling_mean_24: float
    price_actual_rolling_std_24: float
    total_load_actual_rolling_mean_24: float


# -----------------------------------------------------------------------------------------------------------------------
# Price Prediction Endpoint (/predict)
#
# Accepts:
#   - A full set of model input features defined in PricePredictionInput
#
# Behavior:
#   - Enforces strict input feature ordering (matching trained model)
#   - Converts the validated payload into a pandas DataFrame row
#   - Runs the machine-learning model to compute a price prediction
#
# Returns:
#   - JSON object: { "predicted_price": <rounded float> }
#
# Errors:
#   - 400 if any step in the prediction process fails
# -----------------------------------------------------------------------------------------------------------------------
@app.post("/predict")
def predict(payload: PricePredictionInput):

    try:
        # Strict feature ordering
        data = payload.model_dump()
        row = {f: data[f] for f in feature_names}

        df = pd.DataFrame([row], columns=feature_names)

        prediction = float(model.predict(df)[0])

        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
#=============================================================================================================================
#Summarizer section installing Total 3 model testet
# ---------------------------------------------------------
# 1) Summarization Model Microsoft phi-2
# ---------------------------------------------------------
DEVICE = "cpu"
'''
LLM_NAME = "microsoft/phi-3-mini-4k-instruct"
tokenizer_llm = AutoTokenizer.from_pretrained(LLM_NAME)
model_llm = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)
'''
# ---------------------------------------------------------
# 2) Summarization Model Qwen
# ---------------------------------------------------------
'''
LLM_NAME = "Qwen/Qwen2-1.5B-Instruct"

tokenizer_llm = AutoTokenizer.from_pretrained(LLM_NAME)
model_llm = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)
'''
# ---------------------------------------------------------
# 3) TinyLlama Summarization Model (1.1B)
# ---------------------------------------------------------

LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer_llm = AutoTokenizer.from_pretrained(
    LLM_NAME,
    trust_remote_code=True
)

model_llm = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# Avoid padding warnings
model_llm.config.pad_token_id = tokenizer_llm.eos_token_id


class SummarizationInput(BaseModel):
    predicted_price: float
    user_query: str
#-----------------------------------------------------------------------------------------------------------------------
# Summarization Enpoint
# Accepts:
#   - predicted_price : numeric forecast result
#   - user_query      : question related to the prediction
#
# Returns:
#   - LLM-generated explanation text based on the inputs
#
# ----------------------------------------------------------------------------------------------------------------------
# 1) Summarization Endpoint Post request for Microsoft phi-2
# ------------------------------------*******************************---------------------------------------------------

'''
@app.post("/summarize")
def summarize(payload: SummarizationInput):

    try:
        price = payload.predicted_price
        question = payload.user_query

        # A clean, strict, no-echo prompt
        prompt = (
            "You are an expert in electricity markets.\n"
            f"The predicted electricity price is {price} EUR/MWh.\n\n"
            "User question:\n"
            f"{question}\n\n"
        )

        inputs = tokenizer_llm(prompt, return_tensors="pt")
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.4
        )

        # Decode RAW text
        raw_text = tokenizer_llm.decode(outputs[0], skip_special_tokens=True)

        # Remove any prompt echo
        cleaned = raw_text.replace(prompt, "").strip()

        # Remove accidental prefix like "Answer:" or "Explanation:"
        for prefix in ["Answer:", "Explanation:", "Summary:", "Response:"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        print(cleaned)

        return cleaned

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )
'''
#=========================================================================================================================
# 2) Summarization Endpoint Post request for Qwen Model
#------------------------------------*******************************---------------------------------------------------
'''
@app.post("/summarize")
def summarize(payload: SummarizationInput):

    try:
        price = payload.predicted_price
        question = payload.user_query

        # ChatML format for a clean single-turn conversation
        prompt = (
            "<|system|>\n"
            "You are an expert in electricity markets. Provide clear, short explanations.\n"
            "<|user|>\n"
            f"The predicted electricity price is {price} EUR/MWh.\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "- Base your answer ONLY on this question.\n"
            "- Ignore any earlier questions or context.\n"
            "- Do NOT repeat the question.\n"
            "- Respond in 3–4 concise sentences.\n"
            "- Respond ONLY with the explanation.\n"
            "<|assistant|>\n"
        )

        inputs = tokenizer_llm(prompt, return_tensors="pt")

        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.25,
            top_p=0.9,
            eos_token_id=tokenizer_llm.eos_token_id
        )

        raw = tokenizer_llm.decode(outputs[0], skip_special_tokens=False)

        cleaned = (
            raw.replace(prompt, "")
               .replace("<|assistant|>", "")
               .replace("<|user|>", "")
               .replace("<|system|>", "")
               .replace("<|end|>", "")
               .strip()
        )
        print(cleaned)
        return cleaned

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
'''
# ===========================================================================================================================
# 3) Summarization Endpoint Post request for LIama model
# -----------------------****************************************------------------------------------------------------------


@app.post("/summarize")
def summarize(payload: SummarizationInput):

    try:
        price = payload.predicted_price
        question = payload.user_query

        # TinyLlama VICUNA chat format
        prompt = (
            "### System:\n"
            "You are an expert in electricity markets. Provide short, clear explanations.\n\n"

            "### User:\n"
            f"Predicted electricity price: {price} EUR/MWh.\n"
            f"Question: {question}\n\n"
            "Rules:\n"
            "- Do NOT repeat the question.\n"
            "- Base your answer ONLY on the predicted price.\n"
            "- Write 3–4 clear sentences.\n"
            "- Respond ONLY with the explanation.\n\n"

            "### Assistant:\n"
        )

        # Convert to tokens
        inputs = tokenizer_llm(prompt, return_tensors="pt")

        # Generate with TinyLlama (temperature works fine)
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.4,
            do_sample=True
        )

        # Decode
        raw = tokenizer_llm.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt
        cleaned = raw.replace(prompt, "").strip()

        # Remove accidental prefixes
        for prefix in ["Assistant:", "assistant:", "### Assistant:", "Answer:", "Explanation:"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        print(cleaned)
        return cleaned

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


#================================================================================================================================

# --------------------------------------------------------------------------------------------------------------------------
# GET Prediction Retrieval Endpoints
# These endpoints allow:
#   - Fetching a single prediction by ID
#   - Fetching paginated lists of predictions
#   - Selecting country-specific prediction files
# ***************************************************************************************************************************
#
# Get Prediction by ID Endpoint
# Accepts:
#   - id            : integer identifier for a saved prediction
#   - country_code  : ISO country code (default: "ES")
#
# Behavior:
#   - Loads the predictions CSV for the given country
#   - Searches for the row matching the provided ID
#
# Returns:
#   - A dictionary representing the prediction record
#
# Errors:
#   - 404 if CSV does not exist
#   - 404 if no matching ID is found
# -------------------------------------------------------------------------------------------------------------------------

def get_predictions_file_path(country_code: str) -> Path:
    """Return: project_root/data/NO/predictions/NO_predictions.csv"""
    return PROJECT_ROOT.parent / f"data/{country_code}/predictions/{country_code}_predictions.csv"


@app.get("/predict/{id}")
def get_predict_by_id(id: int, country_code: str = "ES"):

    file_path = get_predictions_file_path(country_code)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Predictions CSV not found")

    df = pd.read_csv(file_path)
    row = df[df["id"] == id]

    if row.empty:
        raise HTTPException(status_code=404, detail=f"No prediction with ID {id}")

    return row.iloc[0].to_dict()

# *********************************************************************************************************************
# Get All Predictions (Paginated)
# Accepts:
#   - country_code : ISO country code for dataset folder (default: "ES")
#   - page         : page number (starting from 1)
#   - page_size    : number of items per page
#
# Behavior:
#   - Loads the predictions CSV for the specified country
#   - Applies simple pagination using page + page_size
#
# Returns:
#   - page        : current page number
#   - total       : total number of records in the CSV
#   - page_size   : number of items returned per page
#   - predictions : list of prediction records for the given page
#
# Errors:
#   - 404 if the CSV file does not exist
# ----------------------------------------------------------------------------------------------------------------------
@app.get("/predict")
def get_all_predictions(country_code: str = "ES", page: int = 1, page_size: int = 10):

    file_path = get_predictions_file_path(country_code)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Predictions CSV not found")

    df = pd.read_csv(file_path)

    start = (page - 1) * page_size
    end = start + page_size

    return {
        "page": page,
        "total": len(df),
        "page_size": page_size,
        "predictions": df.iloc[start:end].to_dict(orient="records")
    }


# --------------------------------------------------------------------------------------------------------------------------
# Static Frontend Serving (index.html UI)
#
# Purpose:
#   - Serve the web-based UI directly from FastAPI
#   - Expose static assets (CSS, JS, images) under /static
#   - Return index.html when a user visits the root URL "/"
#
# Behavior:
#   - /static/* → serves files from src/static/
#   - /         → returns static/index.html
#
# Notes:
#   - Ensures the API and frontend run as a single application
#   - Provides a simple interface to interact with prediction/summarization APIs
# --------------------------------------------------------------------------------------------------------------------------
STATIC_DIR = PROJECT_ROOT / "static"

# Mount static folder under /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return "<h1>index.html not found in src/static/</h1>"
    return index_path.read_text()


# ---------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------