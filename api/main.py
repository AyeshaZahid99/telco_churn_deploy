"""
Telco Customer Churn — Prediction API

Serves the trained sklearn Pipeline (preprocessing + RandomForestClassifier)
behind a small FastAPI service so the Streamlit UI (or any other client)
never needs the .pkl files locally.

IMPORTANT — read before deploying:
The pipeline contains a custom transformer (SeniorCitizenTransformer) that
was pickled with `dill` under Python 3.10.5 (the Colab notebook's runtime).
`dill` embeds the function's raw bytecode. Python bytecode is NOT portable
across major/minor CPython versions — running this under Python 3.11/3.12
loads without error but silently produces corrupted/incorrect predictions
(confirmed while inspecting this pickle: it throws a nonsensical
`AttributeError: 'list' object has no attribute 'isinstance'` deep inside
sklearn's output-wrapping code when run on 3.12, and gives correct
predictions when run on 3.10). ALWAYS deploy this API on Python 3.10.x,
exactly as pinned in requirements.txt / Dockerfile. Do not "upgrade" the
Python version without re-testing predictions against known rows first.
"""

import warnings
from pathlib import Path
from typing import Dict

import dill
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pipeline.pkl"
FEATURE_DICT_PATH = BASE_DIR / "my_feature_dict.pkl"

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts whether a telecom customer will churn.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model + feature metadata once at startup -------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = dill.load(f)
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        f"Failed to load model pipeline from {MODEL_PATH}. "
        "Check you are running Python 3.10.x with the pinned requirements."
    ) from exc

feature_dict = joblib.load(FEATURE_DICT_PATH)


def _categorical_options() -> list[dict]:
    """De-duplicate the CATEGORICAL block in the feature dict (the source
    export has SENIORCITIZEN listed twice) into a clean list of
    {name, options} the UI can render directly."""
    cat = feature_dict["CATEGORICAL"]
    columns = list(cat["Column Name"].values())
    members = list(cat["Members"].values())
    seen: set[str] = set()
    result = []
    for col, mem in zip(columns, members):
        if col in seen:
            continue
        seen.add(col)
        result.append({"name": col, "options": mem})
    return result


CATEGORICAL_FEATURES = _categorical_options()
NUMERICAL_FEATURES = feature_dict["NUMERICAL"]["Column Name"]


# --- Request schema ----------------------------------------------------------
class CustomerInput(BaseModel):
    GENDER: str = Field(..., json_schema_extra={"example": "Female"})
    SENIORCITIZEN: int = Field(..., json_schema_extra={"example": 0})
    PARTNER: str = Field(..., json_schema_extra={"example": "Yes"})
    DEPENDENTS: str = Field(..., json_schema_extra={"example": "No"})
    TENURE: float = Field(..., json_schema_extra={"example": 12})
    PHONESERVICE: str = Field(..., json_schema_extra={"example": "Yes"})
    MULTIPLELINES: str = Field(..., json_schema_extra={"example": "No"})
    INTERNETSERVICE: str = Field(..., json_schema_extra={"example": "Fiber optic"})
    ONLINESECURITY: str = Field(..., json_schema_extra={"example": "No"})
    ONLINEBACKUP: str = Field(..., json_schema_extra={"example": "Yes"})
    DEVICEPROTECTION: str = Field(..., json_schema_extra={"example": "No"})
    TECHSUPPORT: str = Field(..., json_schema_extra={"example": "No"})
    STREAMINGTV: str = Field(..., json_schema_extra={"example": "Yes"})
    STREAMINGMOVIES: str = Field(..., json_schema_extra={"example": "No"})
    CONTRACT: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    PAPERLESSBILLING: str = Field(..., json_schema_extra={"example": "Yes"})
    PAYMENTMETHOD: str = Field(..., json_schema_extra={"example": "Electronic check"})
    MONTHLYCHARGES: float = Field(..., json_schema_extra={"example": 70.35})
    TOTALCHARGES: float = Field(..., json_schema_extra={"example": 845.5})


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float | None
    probabilities: Dict[str, float]


# --- Routes ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def get_features():
    """Feature metadata (categorical options + numerical field names) so a
    UI can build its form dynamically without ever touching the .pkl files."""
    return {"categorical": CATEGORICAL_FEATURES, "numerical": NUMERICAL_FEATURES}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerInput):
    try:
        row = pd.DataFrame([payload.model_dump()])
        prediction = model.predict(row)[0]
        proba = model.predict_proba(row)[0]
        classes = list(model.classes_)
        prob_map = {c: float(p) for c, p in zip(classes, proba)}
        return PredictionResponse(
            prediction=str(prediction),
            churn_probability=prob_map.get("Yes"),
            probabilities=prob_map,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
