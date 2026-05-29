"""
Ecomeal AI — FastAPI REST endpoints.
Provides programmatic access to inventory predictions and recommendations.
"""
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR
from src.data.preprocessor import DataPreprocessor
from src.models.wastage_predictor import WastagePredictor
from src.models.anomaly_detector import AnomalyDetector
from src.ai.recommendation_engine import RecommendationEngine

app = FastAPI(
    title="Ecomeal AI API",
    description="Food Waste Intelligence API — predictions, recommendations, and analytics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Optional[WastagePredictor] = None
_df: Optional[pd.DataFrame] = None


def get_predictor() -> WastagePredictor:
    global _predictor
    if _predictor is None:
        _predictor = WastagePredictor()
        try:
            _predictor.load()
        except Exception:
            raise HTTPException(500, "Model not trained. Run `python main.py` first.")
    return _predictor


def get_inventory() -> pd.DataFrame:
    global _df
    if _df is None:
        path = DATA_DIR / "inventory_predicted.csv"
        if not path.exists():
            raise HTTPException(404, "Processed data not found. Run `python main.py` first.")
        _df = pd.read_csv(path)
    return _df


class HealthResponse(BaseModel):
    status: str
    model_trained: bool
    inventory_records: int


class InventoryItem(BaseModel):
    name: str
    category: str
    quantity: float
    unit: str
    expiry_date: str
    daily_consumption: float
    price_per_unit: float
    storage_type: str = "Refrigerator"
    supplier: str = "Unknown"
    historical_wastage_rate: float = 0.15


@app.get("/", response_model=dict)
def root():
    return {"message": "Ecomeal AI API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        predictor = get_predictor()
        df = get_inventory()
        return HealthResponse(
            status="ok",
            model_trained=predictor._is_trained,
            inventory_records=len(df),
        )
    except HTTPException as e:
        return HealthResponse(status=f"degraded: {e.detail}", model_trained=False, inventory_records=0)


@app.get("/inventory/at-risk")
def get_at_risk_items(
    n: int = Query(20, ge=1, le=100),
    min_probability: float = Query(0.5, ge=0.0, le=1.0),
):
    """Get top N inventory items most at risk of being wasted."""
    predictor = get_predictor()
    df = get_inventory()
    top_risk = predictor.get_top_at_risk(df, n=n)
    if min_probability > 0:
        top_risk = top_risk[top_risk["waste_probability"] >= min_probability]
    return top_risk.to_dict(orient="records")


@app.get("/inventory/summary")
def get_inventory_summary():
    """Get high-level inventory waste intelligence summary."""
    df = get_inventory()
    return {
        "total_items": len(df),
        "predicted_to_waste": int(df["will_waste_predicted"].sum()) if "will_waste_predicted" in df.columns else 0,
        "critical_items": int((df["risk_label"] == "Critical").sum()) if "risk_label" in df.columns else 0,
        "total_waste_value_inr": round(float(df["waste_value_at_risk"].sum()), 2) if "waste_value_at_risk" in df.columns else 0,
        "by_category": df.groupby("category")["waste_value_at_risk"].sum().round(2).to_dict() if "category" in df.columns else {},
        "by_risk_level": df["risk_label"].value_counts().to_dict() if "risk_label" in df.columns else {},
    }


@app.post("/predict")
def predict_single_item(item: InventoryItem):
    """Predict waste risk for a single inventory item."""
    predictor = get_predictor()
    preprocessor = DataPreprocessor()

    item_df = pd.DataFrame([item.model_dump()])
    item_df["purchase_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    item_df["min_stock_level"] = item.daily_consumption * 2
    item_df["max_stock_level"] = item.daily_consumption * 30

    try:
        featured = preprocessor.engineer_features(preprocessor.clean(item_df))
        featured["will_waste"] = 0  # placeholder
        predicted = predictor.predict(featured)
        result = predicted.iloc[0]
        return {
            "name": item.name,
            "waste_probability": round(float(result.get("waste_probability", 0)), 4),
            "risk_label": result.get("risk_label", "Unknown"),
            "risk_score": float(result.get("risk_score", 0)),
            "days_to_expiry": int(result.get("days_to_expiry", 0)),
            "estimated_waste_qty": round(float(result.get("estimated_waste_qty", 0)), 3),
            "waste_value_at_risk": round(float(result.get("waste_value_at_risk", 0)), 2),
            "recommended_action": _get_action(result),
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.get("/recommendations/chef-specials")
def get_chef_specials(
    max_expiry_days: int = Query(7, ge=1, le=30),
    n_specials: int = Query(3, ge=1, le=5),
):
    """Generate AI-powered chef special recommendations for expiring ingredients."""
    df = get_inventory()
    rec_engine = RecommendationEngine()
    return rec_engine.generate_chef_specials(df, n_specials=n_specials, max_expiry_days=max_expiry_days)


@app.get("/anomalies")
def get_anomalies(severity: Optional[str] = Query(None, enum=["high", "medium", "low"])):
    """Detect inventory anomalies."""
    df = get_inventory()
    detector = AnomalyDetector(contamination=0.05)
    detector.fit_predict(df)
    return detector.get_anomaly_summary()


def _get_action(row) -> str:
    days = int(row.get("days_to_expiry", 99))
    if days <= 0:
        return "DISCARD: Item has expired"
    elif days <= 1:
        return "URGENT: Use today in specials or staff meals"
    elif days <= 3:
        return "PRIORITY: Feature in Chef Specials this week"
    elif days <= 7:
        return "PLAN: Schedule into this week's menu"
    return "MONITOR: Track consumption rate"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
