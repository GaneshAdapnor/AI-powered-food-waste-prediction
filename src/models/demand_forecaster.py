"""
Demand forecasting using Exponential Smoothing (Holt-Winters).
Predicts future ingredient demand and detects overstock/shortage risks.
"""
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class DemandForecaster:
    """
    Per-ingredient time-series demand forecaster.
    Uses Holt-Winters Exponential Smoothing with weekly seasonality.
    Falls back to simple exponential smoothing for short series.
    """

    def __init__(self, forecast_horizon: int = 30):
        self.forecast_horizon = forecast_horizon
        self.forecasts: dict = {}
        self.alerts: list = []

    def fit_predict(
        self,
        history_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit forecaster for each ingredient and generate demand predictions."""
        logger.info("Running demand forecasting...")

        history_df = history_df.copy()
        history_df["date"] = pd.to_datetime(history_df["date"])
        history_df = history_df.sort_values("date")

        results = []
        ingredients = history_df["ingredient_id"].unique()
        logger.info(f"Forecasting demand for {len(ingredients)} ingredients")

        for ing_id in ingredients:
            ing_history = history_df[history_df["ingredient_id"] == ing_id].copy()
            ing_name = ing_history["name"].iloc[0]
            category = ing_history["category"].iloc[0]

            try:
                forecast_result = self._forecast_ingredient(ing_history, ing_id)
                forecast_result.update({"ingredient_id": ing_id, "name": ing_name, "category": category})
                results.append(forecast_result)
            except Exception as e:
                logger.warning(f"Forecast failed for {ing_name}: {e}. Using fallback.")
                results.append(self._fallback_forecast(ing_history, ing_id, ing_name, category))

        forecast_summary = pd.DataFrame(results)
        self._generate_alerts(forecast_summary, inventory_df)

        logger.info(f"Forecasting complete. {len(self.alerts)} alerts generated.")
        return forecast_summary

    def get_ingredient_forecast(self, ingredient_id: str) -> Optional[dict]:
        return self.forecasts.get(ingredient_id)

    def get_alerts(self, severity: Optional[str] = None) -> list:
        if severity:
            return [a for a in self.alerts if a["severity"] == severity]
        return self.alerts

    def _forecast_ingredient(self, history: pd.DataFrame, ing_id: str) -> dict:
        series = history.set_index("date")["consumption"].asfreq("D").ffill().fillna(0)

        if len(series) < 14:
            return self._simple_average_forecast(series, ing_id)

        trend, seasonal = "add", None
        seasonal_periods = None

        if len(series) >= 21:
            seasonal = "add"
            seasonal_periods = 7

        try:
            model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True, remove_bias=True)
            forecast = fitted.forecast(self.forecast_horizon)
            forecast = forecast.clip(lower=0)

            residuals = series - fitted.fittedvalues
            std_error = residuals.std()

            daily_forecasts = [
                {
                    "day": i + 1,
                    "predicted_consumption": max(0, round(v, 3)),
                    "lower_bound": max(0, round(v - 1.96 * std_error, 3)),
                    "upper_bound": max(0, round(v + 1.96 * std_error, 3)),
                }
                for i, v in enumerate(forecast.values)
            ]

            total_30d = float(forecast.sum())
            avg_daily = float(forecast.mean())
            trend_direction = self._detect_trend(series)

            result = {
                "avg_daily_forecast": round(avg_daily, 3),
                "total_30d_forecast": round(total_30d, 3),
                "trend_direction": trend_direction,
                "forecast_confidence": "high" if len(series) >= 30 else "medium",
                "daily_forecasts": daily_forecasts,
                "historical_avg": round(float(series.mean()), 3),
                "historical_std": round(float(series.std()), 3),
            }
            self.forecasts[ing_id] = result
            return result

        except Exception as e:
            logger.debug(f"HW model failed ({e}), using simple average")
            return self._simple_average_forecast(series, ing_id)

    def _simple_average_forecast(self, series: pd.Series, ing_id: str) -> dict:
        """Fallback: exponential moving average."""
        if len(series) == 0:
            avg = 0.1
        else:
            weights = np.exp(np.linspace(0, 1, len(series)))
            weights /= weights.sum()
            avg = float(np.average(series.values, weights=weights))

        avg = max(0, avg)
        std = float(series.std()) if len(series) > 1 else avg * 0.2

        daily_forecasts = [
            {
                "day": i + 1,
                "predicted_consumption": round(avg, 3),
                "lower_bound": max(0, round(avg - 1.96 * std, 3)),
                "upper_bound": round(avg + 1.96 * std, 3),
            }
            for i in range(self.forecast_horizon)
        ]

        result = {
            "avg_daily_forecast": round(avg, 3),
            "total_30d_forecast": round(avg * self.forecast_horizon, 3),
            "trend_direction": "stable",
            "forecast_confidence": "low",
            "daily_forecasts": daily_forecasts,
            "historical_avg": round(float(series.mean()) if len(series) else 0, 3),
            "historical_std": round(std, 3),
        }
        self.forecasts[ing_id] = result
        return result

    def _fallback_forecast(
        self, history: pd.DataFrame, ing_id: str, name: str, category: str
    ) -> dict:
        avg = history["consumption"].mean() if len(history) > 0 else 0.5
        return {
            "ingredient_id": ing_id,
            "name": name,
            "category": category,
            "avg_daily_forecast": round(float(avg), 3),
            "total_30d_forecast": round(float(avg) * self.forecast_horizon, 3),
            "trend_direction": "unknown",
            "forecast_confidence": "low",
            "daily_forecasts": [],
            "historical_avg": round(float(avg), 3),
            "historical_std": 0.0,
        }

    def _detect_trend(self, series: pd.Series) -> str:
        if len(series) < 7:
            return "stable"
        recent = series.iloc[-7:].mean()
        older = series.iloc[:-7].mean() if len(series) > 7 else series.mean()
        if older == 0:
            return "stable"
        change = (recent - older) / older
        if change > 0.10:
            return "increasing"
        elif change < -0.10:
            return "decreasing"
        else:
            return "stable"

    def _generate_alerts(
        self,
        forecast_summary: pd.DataFrame,
        inventory_df: pd.DataFrame,
    ) -> None:
        """Cross-reference forecasts with current inventory to find risks."""
        inv = inventory_df.copy()
        inv["expiry_date"] = pd.to_datetime(inv["expiry_date"])

        for _, row in forecast_summary.iterrows():
            ing_id = row["ingredient_id"]
            inv_rows = inv[inv["ingredient_id"] == ing_id]
            if inv_rows.empty:
                continue

            inv_row = inv_rows.iloc[0]
            current_qty = inv_row.get("quantity", 0)
            days_to_expiry = inv_row.get("days_to_expiry", 30)
            avg_daily = row["avg_daily_forecast"]
            total_30d = row["total_30d_forecast"]
            unit = inv_row.get("unit", "units")

            # Overstock: we have more than 30-day demand AND items may expire
            if current_qty > total_30d * 1.3 and days_to_expiry < 30:
                surplus = round(current_qty - total_30d, 2)
                self.alerts.append({
                    "ingredient_id": ing_id,
                    "name": row["name"],
                    "category": row["category"],
                    "type": "OVERSTOCK",
                    "severity": "high" if days_to_expiry < 7 else "medium",
                    "message": (
                        f"Overstocked by ~{surplus} {unit}. "
                        f"Current stock lasts {round(current_qty / max(avg_daily, 0.01), 1)} days "
                        f"but expires in {int(days_to_expiry)} days."
                    ),
                    "recommended_action": f"Use {surplus} {unit} within {int(days_to_expiry)} days or redistribute.",
                })

            # Shortage: demand trend is increasing and stock is low
            elif avg_daily > 0 and current_qty < avg_daily * 3:
                self.alerts.append({
                    "ingredient_id": ing_id,
                    "name": row["name"],
                    "category": row["category"],
                    "type": "SHORTAGE_RISK",
                    "severity": "medium",
                    "message": (
                        f"Low stock: only {current_qty} {unit} remaining "
                        f"({round(current_qty / max(avg_daily, 0.01), 1)} days supply). "
                        f"Trend: {row['trend_direction']}."
                    ),
                    "recommended_action": f"Reorder {round(total_30d - current_qty, 2)} {unit} soon.",
                })
