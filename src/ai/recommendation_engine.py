"""
AI-powered recommendation engine using Claude API.
Generates Chef Specials, waste reduction strategies, and inventory optimization advice.
Gracefully falls back to template-based recommendations if API is unavailable.
"""
import json
import logging
from typing import Optional

from google import genai
from google.genai import types
import pandas as pd

from src.config import GEMINI_API_KEY, GEMINI_MODEL, HIGH_RISK_DAYS, MEDIUM_RISK_DAYS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert chef and food waste reduction specialist at a busy modern restaurant.
You help kitchen teams make smart, creative decisions about using expiring ingredients to minimize waste.
You combine culinary creativity with operational pragmatism. Your recommendations are:
- Specific and actionable (not generic)
- Economically aware (mention cost savings)
- Culturally diverse (suggest dishes from multiple cuisines)
- Prioritized by urgency (most critical ingredients first)

Always respond in valid JSON format as specified by the user."""


class RecommendationEngine:
    """
    Generates AI-powered recommendations for reducing food waste.
    Uses Claude API with structured outputs; falls back to rule-based templates.
    """

    def __init__(self):
        self._client = None
        self._api_available = False
        self._setup_client()

    def generate_chef_specials(
        self,
        df: pd.DataFrame,
        n_specials: int = 3,
        max_expiry_days: int = MEDIUM_RISK_DAYS,
    ) -> dict:
        """
        Generate Chef Special dish suggestions for expiring ingredients.
        Returns structured recommendations with dishes, strategies, and action plan.
        """
        expiring = self._get_expiring_items(df, max_expiry_days)
        available_staples = self._get_available_staples(df, max_expiry_days)

        if expiring.empty:
            return {"message": "No items expiring soon. Inventory looks healthy!", "dishes": []}

        if self._api_available:
            return self._claude_chef_specials(expiring, available_staples, n_specials)
        else:
            return self._template_chef_specials(expiring, available_staples)

    def generate_inventory_report(self, df: pd.DataFrame, metrics: dict) -> str:
        """Generate a natural language inventory intelligence report."""
        if self._api_available:
            return self._claude_inventory_report(df, metrics)
        return self._template_inventory_report(df, metrics)

    def explain_waste_risk(self, item: dict) -> str:
        """Generate a plain-language explanation of why an item is at risk."""
        if self._api_available:
            return self._claude_explain_risk(item)
        return self._template_explain_risk(item)

    def _setup_client(self) -> None:
        if not GEMINI_API_KEY:
            logger.warning(
                "GEMINI_API_KEY not set. Using template-based recommendations. "
                "Set the key in .env for AI-powered insights."
            )
            return
        try:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._api_available = True
            logger.info(f"Gemini API client initialized ({GEMINI_MODEL})")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}. Using fallback.")

    def _claude_chef_specials(
        self,
        expiring: pd.DataFrame,
        staples: pd.DataFrame,
        n_specials: int,
    ) -> dict:
        expiring_list = self._format_ingredients_for_prompt(expiring)
        staples_list = self._format_ingredients_for_prompt(staples) if not staples.empty else "N/A"

        prompt = f"""Restaurant inventory alert. These ingredients are expiring soon and MUST be used:

EXPIRING INGREDIENTS (sorted by urgency):
{expiring_list}

AVAILABLE STAPLES (good stock):
{staples_list}

Please generate:
1. Exactly {n_specials} creative "Chef Special" dishes that primarily use the expiring ingredients
2. A waste reduction action plan with prioritized steps
3. Smart bulk usage strategies (batch cooking, prep-ahead, freezing)
4. Cost impact estimate (how much waste value can be saved)

Return ONLY valid JSON in this exact format:
{{
  "dishes": [
    {{
      "name": "Dish Name",
      "description": "2-3 sentence description",
      "cuisine_style": "e.g., Italian, Asian fusion",
      "primary_ingredients": ["ingredient1", "ingredient2"],
      "expiring_ingredients_used": ["ingredient1"],
      "estimated_portions": 15,
      "prep_time_minutes": 30,
      "urgency": "high/medium/low",
      "waste_saved_kg": 1.5,
      "chef_tips": "One practical tip"
    }}
  ],
  "action_plan": [
    {{
      "priority": 1,
      "action": "Specific action",
      "ingredient": "ingredient name",
      "deadline": "Today/Tomorrow/This week",
      "impact": "Expected impact"
    }}
  ],
  "bulk_strategies": ["strategy1", "strategy2"],
  "estimated_waste_savings_inr": 2500,
  "summary": "2-sentence executive summary for the kitchen manager"
}}"""

        try:
            full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
            response = self._client.models.generate_content(
                model=GEMINI_MODEL, contents=full_prompt
            )
            result = self._safe_json_parse(response.text)
            result["source"] = "gemini_ai"
            return result
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}. Using fallback.")
            return self._template_chef_specials(expiring, staples)

    def _claude_inventory_report(self, df: pd.DataFrame, metrics: dict) -> str:
        summary = {
            "total_items": len(df),
            "high_risk_items": int(df.get("will_waste_predicted", pd.Series()).sum() if "will_waste_predicted" in df.columns else 0),
            "total_waste_value": float(df.get("waste_value_at_risk", pd.Series(dtype=float)).sum()),
            "categories_at_risk": list(
                df[df.get("expiry_risk_level", "") == "High"]["category"].unique()
                if "expiry_risk_level" in df.columns else []
            )[:5],
            "model_metrics": metrics,
        }

        prompt = f"""Generate a concise, actionable inventory intelligence report for a restaurant manager.

Data Summary:
{json.dumps(summary, indent=2)}

Write a 3-paragraph report:
1. Current situation (what's critical right now)
2. Key risks and financial impact
3. Top 3 recommended actions for today

Be direct, specific, and business-focused. Use INR currency. Keep it under 200 words."""

        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL, contents=SYSTEM_PROMPT + "\n\n" + prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return self._template_inventory_report(df, metrics)

    def _claude_explain_risk(self, item: dict) -> str:
        prompt = f"""Explain in 2 sentences why this restaurant inventory item is at risk of being wasted.
Be specific about the numbers. Write from a practical kitchen manager perspective.

Item: {item.get('name')}
Category: {item.get('category')}
Current quantity: {item.get('quantity')} {item.get('unit')}
Expires in: {item.get('days_to_expiry')} days
Daily consumption: {item.get('daily_consumption')} {item.get('unit')}/day
Waste probability: {item.get('waste_probability', 0):.0%}
Estimated waste: {item.get('estimated_waste_qty', 0)} {item.get('unit')} worth ₹{item.get('waste_value_at_risk', 0):.0f}"""

        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
            return response.text
        except Exception as e:
            return self._template_explain_risk(item)

    def _template_chef_specials(self, expiring: pd.DataFrame, staples: pd.DataFrame) -> dict:
        """Rule-based fallback when API is unavailable."""
        dishes = []
        expiring_names = expiring["name"].tolist()

        dish_templates = [
            {
                "name": f"Chef's Quick Stir-Fry with {expiring_names[0]}",
                "description": f"A quick, flavorful stir-fry designed to use {expiring_names[0]} before it expires. "
                               "High heat cooking preserves nutrition while maximizing flavor.",
                "cuisine_style": "Asian Fusion",
                "primary_ingredients": expiring_names[:3],
                "expiring_ingredients_used": expiring_names[:2],
                "estimated_portions": 12,
                "prep_time_minutes": 20,
                "urgency": "high",
                "waste_saved_kg": round(expiring["quantity"].iloc[0] * 0.8, 2),
                "chef_tips": "Prep all ingredients before cooking — stir-fry waits for no one.",
            },
            {
                "name": f"Kitchen Sink Soup featuring {expiring_names[min(1, len(expiring_names)-1)]}",
                "description": "A hearty soup that combines multiple expiring ingredients into a rich, satisfying dish. "
                               "Great for batch cooking — freeze portions for later.",
                "cuisine_style": "Continental",
                "primary_ingredients": expiring_names[:4],
                "expiring_ingredients_used": expiring_names[:3],
                "estimated_portions": 20,
                "prep_time_minutes": 45,
                "urgency": "medium",
                "waste_saved_kg": sum(expiring["quantity"].head(3).tolist()),
                "chef_tips": "Make a large batch and freeze half. Soups only get better overnight.",
            },
            {
                "name": f"Today's Special Salad Bowl",
                "description": f"A vibrant composed salad using the freshest expiring ingredients. "
                               "Dressed to order to maintain crispness throughout service.",
                "cuisine_style": "Modern International",
                "primary_ingredients": expiring_names[:2],
                "expiring_ingredients_used": expiring_names[:2],
                "estimated_portions": 15,
                "prep_time_minutes": 15,
                "urgency": "high",
                "waste_saved_kg": round(expiring["quantity"].iloc[0] * 0.6, 2),
                "chef_tips": "Pre-portion into individual containers for quick service.",
            },
        ]

        action_plan = []
        for i, (_, row) in enumerate(expiring.head(5).iterrows()):
            action_plan.append({
                "priority": i + 1,
                "action": f"Use {row['name']} immediately in today's special",
                "ingredient": row["name"],
                "deadline": "Today" if row.get("days_to_expiry", 7) <= 1 else "Tomorrow",
                "impact": f"Saves ₹{row.get('waste_value_at_risk', 0):.0f} in potential waste",
            })

        total_savings = expiring["waste_value_at_risk"].sum() if "waste_value_at_risk" in expiring.columns else 0

        return {
            "dishes": dish_templates[:3],
            "action_plan": action_plan,
            "bulk_strategies": [
                "Blanch and freeze vegetables expiring in 1-2 days",
                "Prep large batches of sauces and soups using expiring produce",
                "Cross-utilize ingredients across multiple menu items today",
            ],
            "estimated_waste_savings_inr": round(float(total_savings), 2),
            "summary": (
                f"{len(expiring)} ingredients expiring within {MEDIUM_RISK_DAYS} days. "
                f"Potential waste value of ₹{total_savings:.0f} can be recovered with today's specials."
            ),
            "source": "template_fallback",
        }

    def _template_inventory_report(self, df: pd.DataFrame, metrics: dict) -> str:
        high_risk = df[df.get("expiry_risk_level", "") == "High"] if "expiry_risk_level" in df.columns else pd.DataFrame()
        total_waste = df["waste_value_at_risk"].sum() if "waste_value_at_risk" in df.columns else 0

        return (
            f"INVENTORY INTELLIGENCE REPORT\n\n"
            f"Current Situation: {len(df)} active inventory items tracked. "
            f"{len(high_risk)} items are in the critical expiry zone (≤{HIGH_RISK_DAYS} days). "
            f"Immediate action required to prevent ₹{total_waste:.0f} in food waste.\n\n"
            f"Key Risks: High-waste categories detected. Perishables with low turnover "
            f"represent the highest financial risk. Model confidence: AUC={metrics.get('roc_auc', 'N/A')}.\n\n"
            f"Top Actions: (1) Process all critical items today, (2) Run Chef Specials to clear expiring stock, "
            f"(3) Adjust purchase orders for overstocked items."
        )

    def _template_explain_risk(self, item: dict) -> str:
        days = item.get("days_to_expiry", 0)
        dc = item.get("daily_consumption", 0)
        qty = item.get("quantity", 0)
        days_of_stock = qty / max(dc, 0.001)
        surplus = max(0, days_of_stock - days)

        if surplus > 0:
            return (
                f"At current consumption of {dc:.2f} {item.get('unit', 'units')}/day, "
                f"this item will have ~{surplus:.1f} days of unconsumed stock when it expires in {days} days. "
                f"Approximately {item.get('estimated_waste_qty', 0):.2f} {item.get('unit', 'units')} "
                f"worth ₹{item.get('waste_value_at_risk', 0):.0f} will be wasted."
            )
        return f"This item is tracking well — expected to be consumed before its {days}-day expiry."

    @staticmethod
    def _get_expiring_items(df: pd.DataFrame, max_days: int) -> pd.DataFrame:
        if "days_to_expiry" not in df.columns:
            return pd.DataFrame()
        mask = (df["days_to_expiry"] >= 0) & (df["days_to_expiry"] <= max_days)
        return df[mask].sort_values("days_to_expiry").head(15)

    @staticmethod
    def _get_available_staples(df: pd.DataFrame, min_days: int) -> pd.DataFrame:
        if "days_to_expiry" not in df.columns:
            return pd.DataFrame()
        mask = df["days_to_expiry"] > min_days
        return df[mask].nlargest(10, "quantity")

    @staticmethod
    def _format_ingredients_for_prompt(df: pd.DataFrame) -> str:
        lines = []
        for _, row in df.iterrows():
            days = row.get("days_to_expiry", "?")
            qty = row.get("quantity", "?")
            unit = row.get("unit", "")
            val = row.get("waste_value_at_risk", 0)
            lines.append(
                f"  - {row['name']} ({row.get('category', '')}): "
                f"{qty} {unit} | expires in {days} days | waste value: ₹{val:.0f}"
            )
        return "\n".join(lines) if lines else "None"

    @staticmethod
    def _safe_json_parse(text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return {"raw_response": text, "dishes": [], "action_plan": []}
