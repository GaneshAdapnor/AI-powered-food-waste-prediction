"""
Anomaly detection for inventory patterns using Isolation Forest + statistical rules.
Catches unusual consumption spikes, suspicious pricing, and supply chain anomalies.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

ANOMALY_FEATURES = [
    "quantity",
    "daily_consumption",
    "days_to_expiry",
    "price_per_unit",
    "historical_wastage_rate",
    "turnover_rate",
    "stock_utilization",
]


class AnomalyDetector:
    """
    Hybrid anomaly detector: Isolation Forest + domain-specific statistical rules.
    Identifies unusual patterns in inventory data that may indicate problems.
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.anomalies: pd.DataFrame = pd.DataFrame()

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using ML + rule-based approaches."""
        logger.info("Running anomaly detection...")
        df = df.copy()

        df = self._ml_anomaly_detection(df)
        df = self._rule_based_anomalies(df)

        # Combine: flagged by either method
        df["is_anomaly"] = (
            (df.get("ml_anomaly", 0) == 1) | (df.get("rule_anomaly", 0) == 1)
        ).astype(int)

        self.anomalies = df[df["is_anomaly"] == 1].copy()
        n_anomalies = len(self.anomalies)
        logger.info(
            f"Detected {n_anomalies} anomalies "
            f"({n_anomalies / max(len(df), 1):.1%} of inventory)"
        )
        return df

    def get_anomaly_summary(self) -> list:
        """Return structured anomaly descriptions for reporting."""
        if self.anomalies.empty:
            return []
        results = []
        for _, row in self.anomalies.head(50).iterrows():
            reasons = []
            if row.get("ml_anomaly", 0):
                reasons.append(f"ML anomaly score: {row.get('anomaly_score', 0):.3f}")
            if row.get("anomaly_price_spike", 0):
                reasons.append("Price is 3+ standard deviations above category average")
            if row.get("anomaly_consumption_spike", 0):
                reasons.append("Consumption rate is unusually high for this category")
            if row.get("anomaly_zero_turnover", 0):
                reasons.append("Zero turnover — item may be obsolete or forgotten")
            if row.get("anomaly_expiry_mismatch", 0):
                reasons.append("Stock exceeds 5x normal consumption capacity before expiry")

            results.append({
                "ingredient_id": row.get("ingredient_id", ""),
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "anomaly_type": self._classify_anomaly_type(row),
                "reasons": reasons,
                "severity": self._get_severity(row),
                "quantity": row.get("quantity", 0),
                "unit": row.get("unit", ""),
                "waste_value_at_risk": row.get("waste_value_at_risk", 0),
            })
        return sorted(results, key=lambda x: x["severity"] == "high", reverse=True)

    def _ml_anomaly_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in ANOMALY_FEATURES if c in df.columns]
        X = df[available].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.fit_transform(X)

        raw_scores = self.model.fit(X_scaled).score_samples(X_scaled)
        predictions = self.model.predict(X_scaled)  # -1 = anomaly, 1 = normal

        # Normalize scores to [0, 1] where 1 = most anomalous
        min_score, max_score = raw_scores.min(), raw_scores.max()
        if max_score > min_score:
            normalized = 1 - (raw_scores - min_score) / (max_score - min_score)
        else:
            normalized = np.zeros(len(raw_scores))

        df["ml_anomaly"] = (predictions == -1).astype(int)
        df["anomaly_score"] = normalized.round(4)
        return df

    def _rule_based_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rule_anomaly"] = 0
        df["anomaly_price_spike"] = 0
        df["anomaly_consumption_spike"] = 0
        df["anomaly_zero_turnover"] = 0
        df["anomaly_expiry_mismatch"] = 0

        # Price spike: >3 std devs from category mean
        if "price_per_unit" in df.columns:
            price_stats = df.groupby("category")["price_per_unit"].agg(["mean", "std"])
            for cat, row in price_stats.iterrows():
                if row["std"] > 0:
                    mask = (
                        (df["category"] == cat) &
                        (df["price_per_unit"] > row["mean"] + 3 * row["std"])
                    )
                    df.loc[mask, "anomaly_price_spike"] = 1

        # Consumption spike: daily_consumption > 3 std devs in category
        if "daily_consumption" in df.columns:
            cons_stats = df.groupby("category")["daily_consumption"].agg(["mean", "std"])
            for cat, row in cons_stats.iterrows():
                if row["std"] > 0:
                    mask = (
                        (df["category"] == cat) &
                        (df["daily_consumption"] > row["mean"] + 3 * row["std"])
                    )
                    df.loc[mask, "anomaly_consumption_spike"] = 1

        # Zero/near-zero turnover for perishables
        if "turnover_rate" in df.columns and "category" in df.columns:
            perishables = ["Vegetables", "Fruits", "Dairy", "Meat/Protein"]
            mask = (
                df["category"].isin(perishables) &
                (df["turnover_rate"] < 0.001) &
                (df["quantity"] > 0)
            )
            df.loc[mask, "anomaly_zero_turnover"] = 1

        # Extreme stock-to-expiry mismatch
        if "days_of_stock" in df.columns and "days_to_expiry" in df.columns:
            mask = (
                (df["days_to_expiry"] > 0) &
                (df["days_of_stock"] > df["days_to_expiry"] * 5)
            )
            df.loc[mask, "anomaly_expiry_mismatch"] = 1

        anomaly_cols = [
            "anomaly_price_spike", "anomaly_consumption_spike",
            "anomaly_zero_turnover", "anomaly_expiry_mismatch",
        ]
        df["rule_anomaly"] = df[anomaly_cols].any(axis=1).astype(int)
        return df

    def _classify_anomaly_type(self, row: pd.Series) -> str:
        if row.get("anomaly_price_spike"):
            return "Price Anomaly"
        if row.get("anomaly_expiry_mismatch"):
            return "Overstock / Expiry Risk"
        if row.get("anomaly_zero_turnover"):
            return "Zero Turnover"
        if row.get("anomaly_consumption_spike"):
            return "Consumption Spike"
        return "Statistical Outlier"

    def _get_severity(self, row: pd.Series) -> str:
        score = row.get("anomaly_score", 0)
        if score > 0.8 or row.get("anomaly_expiry_mismatch"):
            return "high"
        elif score > 0.6:
            return "medium"
        return "low"
