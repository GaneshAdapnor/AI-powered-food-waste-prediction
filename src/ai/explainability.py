"""
Natural language explainability layer.
Converts SHAP values and model outputs into human-readable insights.
"""
import logging
from typing import Optional

import pandas as pd

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)

FEATURE_DESCRIPTIONS = {
    "days_to_expiry": "days remaining until expiry",
    "days_of_stock": "days the current stock will last at current consumption",
    "waste_surplus_days": "days of unconsumed stock after expected expiry",
    "turnover_rate": "how fast inventory turns over (higher is better)",
    "stock_utilization": "current stock as a fraction of maximum stock level",
    "historical_wastage_rate": "historical wastage rate for this category",
    "is_overstocked": "whether current quantity exceeds maximum stock level",
    "is_near_expiry": "whether the item expires within 3 days",
    "daily_consumption": "average daily consumption rate",
    "quantity": "current quantity in stock",
    "category_avg_waste_rate": "average wastage rate for this food category",
}


class ExplainabilityEngine:
    """Converts ML model outputs into actionable, human-readable explanations."""

    def __init__(self):
        self._api_available = bool(ANTHROPIC_API_KEY)

    def explain_prediction(
        self,
        item_data: dict,
        shap_factors: Optional[list] = None,
    ) -> dict:
        """
        Generate a complete explanation for a single item's waste prediction.
        Includes natural language reason, risk drivers, and recommended actions.
        """
        explanation = {
            "ingredient": item_data.get("name", "Unknown"),
            "waste_probability": item_data.get("waste_probability", 0),
            "risk_label": item_data.get("risk_label", "Unknown"),
            "primary_reason": self._get_primary_reason(item_data),
            "risk_drivers": self._extract_risk_drivers(item_data, shap_factors),
            "mitigating_factors": self._extract_mitigating_factors(item_data, shap_factors),
            "recommended_action": self._get_recommended_action(item_data),
            "financial_impact": f"₹{item_data.get('waste_value_at_risk', 0):.2f} at risk",
        }
        return explanation

    def explain_category_risk(self, df: pd.DataFrame) -> dict:
        """Summarize waste risk by category with explanations."""
        if df.empty:
            return {}

        results = {}
        if "category" not in df.columns or "waste_probability" not in df.columns:
            return {}

        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            avg_prob = cat_df["waste_probability"].mean()
            high_risk = (cat_df["waste_probability"] > 0.5).sum()
            total_at_risk = cat_df["waste_value_at_risk"].sum() if "waste_value_at_risk" in cat_df.columns else 0

            results[category] = {
                "avg_waste_probability": round(float(avg_prob), 3),
                "items_at_high_risk": int(high_risk),
                "total_items": len(cat_df),
                "total_waste_value": round(float(total_at_risk), 2),
                "explanation": self._explain_category(category, avg_prob, high_risk, cat_df),
            }

        return results

    def generate_feature_narrative(self, shap_explanation: dict) -> str:
        """Convert SHAP explanation into readable narrative."""
        if not shap_explanation or "top_risk_factors" not in shap_explanation:
            return "Explanation unavailable."

        ingredient = shap_explanation.get("ingredient", "This item")
        prob = shap_explanation.get("waste_probability", 0)
        factors = shap_explanation["top_risk_factors"]

        increasing = [f for f in factors if f["direction"] == "increases_risk"]
        decreasing = [f for f in factors if f["direction"] == "reduces_risk"]

        narrative = f"{ingredient} has a {prob:.0%} waste probability.\n\n"

        if increasing:
            narrative += "WHY IT'S AT RISK:\n"
            for f in increasing[:3]:
                feat_desc = FEATURE_DESCRIPTIONS.get(f["feature"], f["feature"])
                narrative += (
                    f"  • {feat_desc.capitalize()}: {f['value']:.2f} "
                    f"(increases risk by {abs(f['shap_contribution']):.3f})\n"
                )

        if decreasing:
            narrative += "\nMITIGATING FACTORS:\n"
            for f in decreasing[:2]:
                feat_desc = FEATURE_DESCRIPTIONS.get(f["feature"], f["feature"])
                narrative += (
                    f"  • {feat_desc.capitalize()}: {f['value']:.2f} "
                    f"(reduces risk by {abs(f['shap_contribution']):.3f})\n"
                )

        return narrative

    def _get_primary_reason(self, item: dict) -> str:
        days_to_expiry = item.get("days_to_expiry", 99)
        days_of_stock = item.get("days_of_stock", 0)
        is_overstock = item.get("is_overstocked", 0)
        waste_prob = item.get("waste_probability", 0)

        if days_to_expiry <= 0:
            return "Item has already expired and must be discarded immediately."
        if days_to_expiry <= 1:
            return f"Item expires tomorrow — only {days_to_expiry} day remaining for consumption."
        if days_of_stock > days_to_expiry * 2:
            surplus = round(days_of_stock - days_to_expiry, 1)
            return (
                f"Current stock ({days_of_stock:.1f} days' worth) far exceeds time until expiry "
                f"({days_to_expiry} days). Surplus of ~{surplus} days will go to waste."
            )
        if is_overstock:
            return "Item is significantly overstocked relative to its maximum recommended level."
        if waste_prob > 0.7:
            return "High historical wastage rate combined with current stock level indicates elevated risk."
        return f"Moderate risk based on {days_to_expiry}-day expiry window and consumption patterns."

    def _extract_risk_drivers(self, item: dict, shap_factors: Optional[list]) -> list:
        drivers = []
        if shap_factors:
            for f in shap_factors:
                if f.get("direction") == "increases_risk" and abs(f.get("shap_contribution", 0)) > 0.01:
                    desc = FEATURE_DESCRIPTIONS.get(f["feature"], f["feature"])
                    drivers.append(f"{desc}: {f['value']:.2f}")
        else:
            # Rule-based fallback
            if item.get("days_to_expiry", 99) <= 3:
                drivers.append(f"Expires very soon ({item['days_to_expiry']} days)")
            if item.get("is_overstocked"):
                drivers.append("Quantity exceeds maximum stock level")
            if item.get("historical_wastage_rate", 0) > 0.2:
                drivers.append(f"High historical waste rate ({item['historical_wastage_rate']:.0%})")
        return drivers[:5]

    def _extract_mitigating_factors(self, item: dict, shap_factors: Optional[list]) -> list:
        factors = []
        if shap_factors:
            for f in shap_factors:
                if f.get("direction") == "reduces_risk" and abs(f.get("shap_contribution", 0)) > 0.01:
                    desc = FEATURE_DESCRIPTIONS.get(f["feature"], f["feature"])
                    factors.append(f"{desc}: {f['value']:.2f}")
        else:
            if item.get("turnover_rate", 0) > 0.5:
                factors.append("High turnover rate — item moves quickly")
            if item.get("days_to_expiry", 0) > 7:
                factors.append("Adequate time window for consumption")
        return factors[:3]

    def _get_recommended_action(self, item: dict) -> str:
        days = item.get("days_to_expiry", 99)
        name = item.get("name", "This item")
        qty = item.get("quantity", 0)
        unit = item.get("unit", "units")

        if days <= 0:
            return f"DISCARD: {name} has expired. Remove immediately."
        elif days <= 1:
            return f"URGENT: Use all {qty} {unit} of {name} today in kitchen specials or staff meals."
        elif days <= 3:
            return f"PRIORITY: Feature {name} in today's and tomorrow's Chef Specials. Prepare in bulk."
        elif days <= 7:
            return f"PLAN AHEAD: Schedule {name} into this week's menu. Consider batch preparation."
        else:
            return f"MONITOR: Track {name} consumption rate. Current stock should be consumed on schedule."

    def _explain_category(
        self,
        category: str,
        avg_prob: float,
        high_risk_count: int,
        cat_df: pd.DataFrame,
    ) -> str:
        cat_explanations = {
            "Vegetables": "Vegetables are highly perishable. Short shelf life + inconsistent consumption drives waste.",
            "Fruits": "Fruit waste is heavily influenced by ripening speed and service demand variability.",
            "Dairy": "Dairy items have strict expiry windows. Even slight overstocking leads to significant waste.",
            "Meat/Protein": "Highest per-unit cost category. Even small waste amounts have major financial impact.",
            "Grains": "Long shelf life keeps risk low, but bulk purchasing can create storage issues.",
            "Spices": "Low waste risk due to long shelf life, but expired spices reduce dish quality.",
            "Beverages": "Waste driven by demand forecasting errors and inconsistent customer preferences.",
            "Condiments": "Low base waste rate, but unused specialty condiments can accumulate.",
            "Frozen": "Freezer space limitations and power issues are the main risk factors.",
        }
        base = cat_explanations.get(category, "Category-specific waste patterns detected.")
        return f"{base} Currently {high_risk_count} item(s) at high risk with avg probability {avg_prob:.0%}."
