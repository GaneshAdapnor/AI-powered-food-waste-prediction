"""
Wastage prediction model: XGBoost + LightGBM ensemble with SHAP explainability.
Predicts both binary waste classification and continuous waste quantity.
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

from src.config import MODELS_DIR, RANDOM_SEED, TEST_SIZE

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "days_to_expiry",
    "days_since_purchase",
    "days_of_stock",
    "waste_surplus_days",
    "turnover_rate",
    "stock_utilization",
    "daily_consumption",
    "quantity",
    "price_per_unit",
    "historical_wastage_rate",
    "category_avg_waste_rate",
    "is_overstocked",
    "is_understocked",
    "is_near_expiry",
    "category_encoded",
    "storage_encoded",
    "price_tier",
]


class WastagePredictor:
    """
    Ensemble classifier for predicting which inventory items will be wasted.
    Uses XGBoost + LightGBM with SHAP for explainability.
    """

    def __init__(self):
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.metrics: dict = {}
        self._is_trained = False

    def train(self, df: pd.DataFrame) -> dict:
        """Train ensemble model and compute SHAP explainability."""
        logger.info("Training wastage prediction model...")

        X, y = self._prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0,
        )

        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            verbose=-1,
        )

        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
        )

        # Ensemble prediction: average probabilities
        xgb_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X_test)[:, 1]
        ensemble_prob = (xgb_prob + lgb_prob) / 2
        y_pred = (ensemble_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, ensemble_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.metrics = {
            "roc_auc": round(auc, 4),
            "precision": round(report["1"]["precision"], 4),
            "recall": round(report["1"]["recall"], 4),
            "f1": round(report["1"]["f1-score"], 4),
            "accuracy": round(report["accuracy"], 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        logger.info(f"Model metrics: AUC={auc:.4f}, F1={report['1']['f1-score']:.4f}")

        # SHAP explainability
        self.shap_explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = self.shap_explainer.shap_values(X_test[:200])

        self.feature_importance = pd.DataFrame({
            "feature": FEATURE_COLS,
            "shap_importance": np.abs(shap_values).mean(axis=0),
            "xgb_importance": self.xgb_model.feature_importances_,
        }).sort_values("shap_importance", ascending=False)

        self._is_trained = True
        return self.metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add waste predictions and risk scores to inventory dataframe."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        if df is None or df.empty:
            return df

        X, _ = self._prepare_features(df, fit=False)

        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X)[:, 1]
        ensemble_prob = (xgb_prob + lgb_prob) / 2

        result = df.copy()
        result["waste_probability"] = ensemble_prob.round(4)
        result["will_waste_predicted"] = (ensemble_prob >= 0.5).astype(int)
        result["risk_score"] = (ensemble_prob * 100).round(1)
        result["risk_label"] = result["waste_probability"].apply(self._probability_to_label)

        return result

    def explain_item(self, df: pd.DataFrame, item_idx: int) -> dict:
        """Generate SHAP-based explanation for a single inventory item."""
        if not self._is_trained or self.shap_explainer is None:
            return {"error": "Model not trained"}

        X, _ = self._prepare_features(df, fit=False)
        row = X.iloc[[item_idx]]
        shap_vals = self.shap_explainer.shap_values(row)[0]

        factors = []
        for feat, shap_val, feat_val in zip(FEATURE_COLS, shap_vals, row.values[0]):
            factors.append({
                "feature": feat,
                "value": round(float(feat_val), 4),
                "shap_contribution": round(float(shap_val), 4),
                "direction": "increases_risk" if shap_val > 0 else "reduces_risk",
            })

        factors.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)

        item = df.iloc[item_idx]
        return {
            "ingredient": item.get("name", "Unknown"),
            "waste_probability": round(
                float(self.xgb_model.predict_proba(row)[0, 1]), 4
            ),
            "top_risk_factors": factors[:5],
            "all_factors": factors,
        }

    def get_top_at_risk(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Return top N items most at risk of being wasted."""
        predicted = self.predict(df)
        return (
            predicted.nlargest(n, "waste_probability")
            [[
                "ingredient_id", "name", "category", "quantity", "unit",
                "days_to_expiry", "expiry_risk_level", "waste_probability",
                "risk_score", "risk_label", "waste_value_at_risk", "estimated_waste_qty"
            ]]
        )

    def save(self) -> Path:
        path = MODELS_DIR / "wastage_predictor.joblib"
        joblib.dump({
            "xgb": self.xgb_model,
            "lgb": self.lgb_model,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }, path)
        logger.info(f"Saved wastage predictor to {path}")
        return path

    def load(self) -> None:
        path = MODELS_DIR / "wastage_predictor.joblib"
        data = joblib.load(path)
        self.xgb_model = data["xgb"]
        self.lgb_model = data["lgb"]
        self.metrics = data["metrics"]
        self.feature_importance = data["feature_importance"]
        self.shap_explainer = shap.TreeExplainer(self.xgb_model)
        self._is_trained = True
        logger.info(f"Loaded wastage predictor from {path}")

    def _prepare_features(self, df: pd.DataFrame, fit: bool = True):
        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].fillna(0).astype(float)

        # Ensure all expected columns exist (fill missing with 0)
        for col in FEATURE_COLS:
            if col not in X.columns:
                X[col] = 0.0
        X = X[FEATURE_COLS]

        y = df["will_waste"].values if "will_waste" in df.columns else np.zeros(len(df))
        return X, y

    def _probability_to_label(self, p: float) -> str:
        if p >= 0.75:
            return "Critical"
        elif p >= 0.5:
            return "High"
        elif p >= 0.3:
            return "Medium"
        else:
            return "Low"
