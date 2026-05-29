import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_INVENTORY_RECORDS = 1200

# Risk thresholds (days to expiry)
HIGH_RISK_DAYS = 3
MEDIUM_RISK_DAYS = 7
LOW_RISK_DAYS = 14

def _get_gemini_key() -> str:
    # Try environment / .env first, then Streamlit secrets
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    return key

GEMINI_API_KEY = _get_gemini_key()
GEMINI_MODEL = "gemini-2.0-flash-lite"

CATEGORY_CONFIG = {
    "Vegetables": {
        "shelf_life": (2, 14), "waste_rate_base": 0.28,
        "storage": ["Refrigerator", "Dry Storage"],
        "unit": "kg", "price_range": (20, 120),
        "daily_consumption": (0.3, 3.0),
    },
    "Fruits": {
        "shelf_life": (4, 21), "waste_rate_base": 0.22,
        "storage": ["Refrigerator", "Counter"],
        "unit": "kg", "price_range": (30, 200),
        "daily_consumption": (0.2, 2.5),
    },
    "Dairy": {
        "shelf_life": (5, 30), "waste_rate_base": 0.18,
        "storage": ["Refrigerator"],
        "unit": "liters", "price_range": (40, 300),
        "daily_consumption": (0.5, 4.0),
    },
    "Meat/Protein": {
        "shelf_life": (2, 10), "waste_rate_base": 0.32,
        "storage": ["Refrigerator", "Freezer"],
        "unit": "kg", "price_range": (150, 600),
        "daily_consumption": (0.5, 5.0),
    },
    "Grains": {
        "shelf_life": (60, 365), "waste_rate_base": 0.04,
        "storage": ["Dry Storage"],
        "unit": "kg", "price_range": (30, 150),
        "daily_consumption": (0.5, 5.0),
    },
    "Spices": {
        "shelf_life": (90, 730), "waste_rate_base": 0.03,
        "storage": ["Dry Storage"],
        "unit": "grams", "price_range": (50, 800),
        "daily_consumption": (10, 100),
    },
    "Beverages": {
        "shelf_life": (7, 90), "waste_rate_base": 0.10,
        "storage": ["Refrigerator", "Dry Storage"],
        "unit": "liters", "price_range": (30, 200),
        "daily_consumption": (1.0, 10.0),
    },
    "Condiments": {
        "shelf_life": (30, 365), "waste_rate_base": 0.06,
        "storage": ["Dry Storage", "Refrigerator"],
        "unit": "liters", "price_range": (50, 400),
        "daily_consumption": (0.1, 1.0),
    },
    "Frozen": {
        "shelf_life": (30, 365), "waste_rate_base": 0.08,
        "storage": ["Freezer"],
        "unit": "kg", "price_range": (100, 500),
        "daily_consumption": (0.2, 3.0),
    },
}

SUPPLIERS = [
    "Fresh Farm Supplies", "Metro Wholesale", "City Fresh Mart",
    "Green Valley Farms", "Sunrise Distributors", "Quick Fresh Co",
    "Premium Foods Ltd", "Local Market Direct",
]

DAYS_IN_WEEK = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

WEEKDAY_DEMAND_FACTORS = {
    "Monday": 0.75, "Tuesday": 0.80, "Wednesday": 0.85,
    "Thursday": 0.90, "Friday": 1.20, "Saturday": 1.35, "Sunday": 1.10,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "ecomeal.log"),
        logging.StreamHandler(),
    ],
)
