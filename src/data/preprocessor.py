"""
Robust data preprocessing pipeline. Handles all real-world data quality issues
without crashing. Every transformation is logged and auditable.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import (
    CATEGORY_CONFIG, HIGH_RISK_DAYS, LOW_RISK_DAYS, MEDIUM_RISK_DAYS,
)

logger = logging.getLogger(__name__)

VALID_CATEGORIES = set(CATEGORY_CONFIG.keys())
VALID_STORAGE = {"Refrigerator", "Freezer", "Dry Storage", "Counter"}


class DataPreprocessor:
    """
    Cleans, validates, and engineers features from raw inventory data.
    Designed to never crash regardless of input quality.
    """

    def __init__(self):
        self.today = datetime.now().date()
        self.cleaning_report: dict = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full cleaning pipeline with audit trail."""
        if df is None or df.empty:
            logger.warning("Empty DataFrame passed to clean(). Returning empty.")
            return pd.DataFrame()
        logger.info(f"Starting cleaning pipeline on {len(df)} records...")
        original_count = len(df)
        df = df.copy()

        df = self._standardize_columns(df)
        df = self._remove_duplicates(df)
        df = self._fix_categories(df)
        df = self._parse_dates(df)
        df = self._fix_numeric_columns(df)
        df = self._fill_missing_values(df)
        df = self._validate_business_rules(df)

        self.cleaning_report["original_count"] = original_count
        self.cleaning_report["final_count"] = len(df)
        self.cleaning_report["records_cleaned"] = original_count - len(df)

        logger.info(
            f"Cleaning complete: {len(df)} records retained "
            f"({original_count - len(df)} removed)"
        )
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build ML-ready feature set from clean inventory data."""
        if df is None or df.empty:
            return df
        logger.info("Engineering features...")
        df = df.copy()

        # Core temporal features
        df["days_to_expiry"] = (
            pd.to_datetime(df["expiry_date"]).dt.date.apply(
                lambda x: (x - self.today).days if pd.notnull(x) else 0
            )
        )
        df["days_since_purchase"] = (
            pd.to_datetime(df["purchase_date"]).dt.date.apply(
                lambda x: (self.today - x).days if pd.notnull(x) else 0
            )
        )

        # Stock duration: how many days current stock will last at current consumption
        df["days_of_stock"] = np.where(
            df["daily_consumption"] > 0,
            df["quantity"] / df["daily_consumption"],
            df["days_to_expiry"] + 30,  # treat zero-consumption as long-lasting
        ).round(2)

        # Core waste risk signal: positive means we have MORE stock than we can use
        df["waste_surplus_days"] = (df["days_of_stock"] - df["days_to_expiry"]).clip(lower=0)

        df["estimated_waste_qty"] = (
            df["waste_surplus_days"] * df["daily_consumption"]
        ).clip(lower=0.0).round(3)

        df["waste_value_at_risk"] = (
            df["estimated_waste_qty"] * df["price_per_unit"]
        ).round(2)

        # Turnover and utilization
        df["turnover_rate"] = np.where(
            df["quantity"] > 0,
            df["daily_consumption"] / df["quantity"],
            0,
        ).round(4)

        df["stock_utilization"] = np.where(
            df["max_stock_level"] > 0,
            (df["quantity"] / df["max_stock_level"]).clip(0, 2),
            1.0,
        ).round(4)

        # Risk categorization
        df["expiry_risk_level"] = df["days_to_expiry"].apply(self._classify_expiry_risk)

        df["is_overstocked"] = (
            df["quantity"] > df["max_stock_level"]
        ).astype(int)

        df["is_understocked"] = (
            df["quantity"] < df["min_stock_level"]
        ).astype(int)

        df["is_near_expiry"] = (
            df["days_to_expiry"] <= HIGH_RISK_DAYS
        ).astype(int)

        # Waste probability (0-1 continuous signal before model)
        df["raw_waste_probability"] = self._compute_waste_probability(df)

        # Categorical encodings for ML
        df["category_encoded"] = pd.Categorical(df["category"]).codes
        df["storage_encoded"] = pd.Categorical(df["storage_type"]).codes

        # Category-level average waste rate
        cat_waste = df.groupby("category")["historical_wastage_rate"].transform("mean")
        df["category_avg_waste_rate"] = cat_waste.fillna(df["historical_wastage_rate"].mean())

        # Price tier (Low/Mid/High within category)
        def safe_price_tier(x):
            try:
                if len(x) < 3:
                    return pd.Series(1.0, index=x.index)
                return pd.qcut(x, q=3, labels=[0, 1, 2], duplicates="drop").astype(float)
            except Exception:
                return pd.Series(1.0, index=x.index)

        df["price_tier"] = df.groupby("category")["price_per_unit"].transform(
            safe_price_tier
        ).fillna(1)

        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def build_training_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create training labels based on domain logic + noise to simulate
        what actually gets wasted in practice.
        """
        df = df.copy()

        # Base probability from domain logic
        p_waste = df["raw_waste_probability"].values
        noise = np.random.normal(0, 0.08, len(df))
        p_noisy = np.clip(p_waste + noise, 0, 1)

        df["will_waste"] = (p_noisy > 0.5).astype(int)

        # Actual wastage amount (regression target) with noise
        waste_noise = np.random.lognormal(0, 0.3, len(df))
        df["actual_wastage_qty"] = np.clip(
            np.where(
                df["will_waste"] == 1,
                df["estimated_waste_qty"] * waste_noise,
                0,
            ),
            0, None
        ).round(3)

        df["actual_wastage_qty"] = np.where(
            df["will_waste"] == 0, 0, df["actual_wastage_qty"]
        )

        pos_rate = df["will_waste"].mean()
        logger.info(
            f"Training labels built. Wastage rate: {pos_rate:.1%} "
            f"({df['will_waste'].sum()} items predicted to waste)"
        )
        return df

    # --- Private helpers ---

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        if "ingredient_id" in df.columns:
            df = df.drop_duplicates(subset=["ingredient_id"])
        else:
            df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            logger.info(f"Removed {removed} duplicate records")
        self.cleaning_report["duplicates_removed"] = removed
        return df

    def _fix_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        if "category" not in df.columns:
            return df
        # Normalize case
        df["category"] = df["category"].str.strip().str.title()
        # Map near-matches
        category_map = {c.upper(): c for c in VALID_CATEGORIES}
        category_map.update({c: c for c in VALID_CATEGORIES})
        df["category"] = df["category"].apply(
            lambda x: category_map.get(x.upper() if isinstance(x, str) else x, x)
        )
        unknown = ~df["category"].isin(VALID_CATEGORIES)
        if unknown.sum():
            logger.warning(f"{unknown.sum()} records with unknown category → 'Grains'")
            df.loc[unknown, "category"] = "Grains"
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        for col in ["expiry_date", "purchase_date"]:
            if col not in df.columns:
                continue
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # Drop rows where expiry_date is completely unparseable
        bad_expiry = df["expiry_date"].isna()
        if bad_expiry.sum():
            logger.warning(f"Dropping {bad_expiry.sum()} rows with invalid expiry dates")
            df = df[~bad_expiry]

        # Fill missing purchase_date with (expiry - median shelf life)
        if "purchase_date" in df.columns:
            missing_purchase = df["purchase_date"].isna()
            if missing_purchase.sum():
                df.loc[missing_purchase, "purchase_date"] = (
                    df.loc[missing_purchase, "expiry_date"] - pd.Timedelta(days=14)
                )

        self.cleaning_report["invalid_dates_dropped"] = before - len(df)
        return df

    def _fix_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = ["quantity", "daily_consumption", "price_per_unit",
                        "historical_wastage_rate", "min_stock_level", "max_stock_level"]
        for col in numeric_cols:
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Negative values are data entry errors → take absolute value
            neg_mask = df[col] < 0
            if neg_mask.sum():
                logger.info(f"Fixed {neg_mask.sum()} negative values in '{col}'")
                df.loc[neg_mask, col] = df.loc[neg_mask, col].abs()
        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fill quantity with category median
        if "quantity" in df.columns and "category" in df.columns:
            df["quantity"] = df.groupby("category")["quantity"].transform(
                lambda x: x.fillna(x.median())
            )
            df["quantity"] = df["quantity"].fillna(df["quantity"].median())

        # Fill daily_consumption with category median
        if "daily_consumption" in df.columns and "category" in df.columns:
            df["daily_consumption"] = df.groupby("category")["daily_consumption"].transform(
                lambda x: x.fillna(x.median())
            )
            df["daily_consumption"] = df["daily_consumption"].fillna(1.0)
            # Replace zero consumption with small positive value
            df["daily_consumption"] = df["daily_consumption"].replace(0, 0.01)

        # Fill price
        if "price_per_unit" in df.columns:
            df["price_per_unit"] = df.groupby("category")["price_per_unit"].transform(
                lambda x: x.fillna(x.median())
            ).fillna(100.0)

        # Fill wastage rate
        if "historical_wastage_rate" in df.columns:
            df["historical_wastage_rate"] = df.groupby("category")["historical_wastage_rate"].transform(
                lambda x: x.fillna(x.mean())
            ).fillna(0.15)

        # Fill stock levels
        if "max_stock_level" in df.columns:
            df["max_stock_level"] = df["max_stock_level"].fillna(
                df["quantity"] * 2
            )
        if "min_stock_level" in df.columns:
            df["min_stock_level"] = df["min_stock_level"].fillna(
                df["daily_consumption"] * 3
            )

        return df

    def _validate_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        # Quantity must be non-negative
        df["quantity"] = df["quantity"].clip(lower=0)
        # Daily consumption must be positive
        df["daily_consumption"] = df["daily_consumption"].clip(lower=0.001)
        # Wastage rate must be in [0, 1]
        df["historical_wastage_rate"] = df["historical_wastage_rate"].clip(0, 1)
        # Price must be positive
        df["price_per_unit"] = df["price_per_unit"].clip(lower=0.01)
        return df

    def _classify_expiry_risk(self, days: int) -> str:
        if days <= 0:
            return "Expired"
        elif days <= HIGH_RISK_DAYS:
            return "High"
        elif days <= MEDIUM_RISK_DAYS:
            return "Medium"
        elif days <= LOW_RISK_DAYS:
            return "Low"
        else:
            return "Safe"

    def _compute_waste_probability(self, df: pd.DataFrame) -> pd.Series:
        """Heuristic waste probability before ML model."""
        # Primary signal: will we have leftover stock at expiry?
        stock_surplus_ratio = np.where(
            df["days_to_expiry"] > 0,
            (df["days_of_stock"] / df["days_to_expiry"]).clip(0, 5),
            5.0,
        )
        # Sigmoid-like transformation
        p_from_surplus = 1 / (1 + np.exp(-2 * (stock_surplus_ratio - 1)))

        # Blend with historical wastage rate
        p_historical = df["historical_wastage_rate"].values
        p_combined = 0.7 * p_from_surplus + 0.3 * p_historical

        # Boost for near-expiry items
        near_expiry_boost = np.where(df["days_to_expiry"] <= HIGH_RISK_DAYS, 0.15, 0)
        expired_boost = np.where(df["days_to_expiry"] <= 0, 0.5, 0)

        return pd.Series(
            np.clip(p_combined + near_expiry_boost + expired_boost, 0, 1),
            index=df.index,
        )
