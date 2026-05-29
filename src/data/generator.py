"""
Generates realistic restaurant inventory dataset with intentional data quality issues
to simulate real-world conditions.
"""
import logging
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import (
    CATEGORY_CONFIG, DATA_DIR, N_INVENTORY_RECORDS,
    RANDOM_SEED, SUPPLIERS, WEEKDAY_DEMAND_FACTORS,
)

logger = logging.getLogger(__name__)

INGREDIENTS_BY_CATEGORY = {
    "Vegetables": [
        "Onions", "Tomatoes", "Potatoes", "Bell Peppers", "Mushrooms",
        "Spinach", "Lettuce", "Carrots", "Broccoli", "Cauliflower",
        "Cabbage", "Cucumber", "Zucchini", "Eggplant", "Green Peas",
        "Corn", "Green Beans", "Celery", "Garlic", "Ginger",
        "Leeks", "Artichokes", "Asparagus", "Kale", "Bok Choy",
    ],
    "Fruits": [
        "Apples", "Bananas", "Lemons", "Limes", "Oranges",
        "Mangoes", "Grapes", "Strawberries", "Blueberries", "Pineapple",
        "Watermelon", "Avocados", "Peaches", "Pears", "Kiwi",
    ],
    "Dairy": [
        "Whole Milk", "Heavy Cream", "Unsalted Butter", "Mozzarella Cheese",
        "Cheddar Cheese", "Parmesan Cheese", "Greek Yogurt", "Eggs",
        "Cream Cheese", "Sour Cream", "Ricotta Cheese", "Feta Cheese",
    ],
    "Meat/Protein": [
        "Chicken Breast", "Ground Beef", "Pork Ribs", "Salmon Fillet",
        "Tuna Steak", "Tiger Shrimp", "Firm Tofu", "Paneer",
        "Lamb Chops", "Duck Breast", "Turkey Mince", "Sea Bass",
    ],
    "Grains": [
        "Basmati Rice", "Pasta (Penne)", "Sourdough Bread", "All-Purpose Flour",
        "Rolled Oats", "Quinoa", "Red Lentils", "Chickpeas",
        "Couscous", "Arborio Rice", "Whole Wheat Flour", "Semolina",
    ],
    "Spices": [
        "Black Pepper", "Cumin Seeds", "Coriander Powder", "Turmeric",
        "Paprika", "Dried Oregano", "Fresh Basil", "Fresh Thyme",
        "Rosemary", "Chili Flakes", "Cardamom", "Cinnamon Sticks",
        "Star Anise", "Bay Leaves", "Saffron",
    ],
    "Beverages": [
        "Fresh Orange Juice", "Apple Juice", "Sparkling Water",
        "Cold Brew Coffee", "Green Tea", "Mango Lassi Mix",
        "Coconut Water", "Lemonade Concentrate",
    ],
    "Condiments": [
        "Extra Virgin Olive Oil", "Soy Sauce", "Balsamic Vinegar",
        "Dijon Mustard", "Sriracha Hot Sauce", "Tahini",
        "Fish Sauce", "Worcestershire Sauce", "Coconut Milk",
    ],
    "Frozen": [
        "Frozen Peas", "Frozen Corn", "Frozen Shrimp", "Vanilla Ice Cream",
        "Frozen Mixed Berries", "Frozen Edamame", "Frozen Spinach",
    ],
}


class RestaurantDataGenerator:
    """Generates realistic restaurant inventory data with configurable noise."""

    def __init__(self, n_records: int = N_INVENTORY_RECORDS, seed: int = RANDOM_SEED):
        self.n_records = n_records
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.today = datetime.now().date()

    def generate_inventory(self, add_noise: bool = True) -> pd.DataFrame:
        """Generate the main inventory dataset."""
        logger.info(f"Generating {self.n_records} inventory records...")
        records = []

        ingredient_pool = self._build_ingredient_pool()

        for i in range(self.n_records):
            name, category = random.choice(ingredient_pool)
            cfg = CATEGORY_CONFIG[category]

            shelf_life = random.randint(*cfg["shelf_life"])
            purchase_days_ago = random.randint(0, max(1, shelf_life // 3))
            expiry_date = self.today + timedelta(
                days=shelf_life - purchase_days_ago + random.randint(-2, 3)
            )
            purchase_date = self.today - timedelta(days=purchase_days_ago)

            daily_consumption = round(
                random.uniform(*cfg["daily_consumption"]), 3
            )
            days_remaining = max(1, (expiry_date - self.today).days)

            # Realistic quantity: somewhere between 1-3x what we'd consume before expiry
            over_stock_factor = random.choice([0.7, 0.9, 1.0, 1.0, 1.2, 1.5, 2.0])
            quantity = round(
                daily_consumption * days_remaining * over_stock_factor
                + random.uniform(0, daily_consumption),
                2,
            )
            quantity = max(0.1, quantity)

            max_stock = round(daily_consumption * shelf_life * 1.2, 2)
            min_stock = round(daily_consumption * 2, 2)

            historical_wastage = round(
                max(0, cfg["waste_rate_base"] + np.random.normal(0, 0.05)), 4
            )

            record = {
                "ingredient_id": f"ING-{i+1:04d}",
                "name": name,
                "category": category,
                "quantity": quantity,
                "unit": cfg["unit"],
                "purchase_date": purchase_date.strftime("%Y-%m-%d"),
                "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                "daily_consumption": daily_consumption,
                "price_per_unit": round(random.uniform(*cfg["price_range"]), 2),
                "supplier": random.choice(SUPPLIERS),
                "storage_type": random.choice(cfg["storage"]),
                "historical_wastage_rate": historical_wastage,
                "min_stock_level": min_stock,
                "max_stock_level": max_stock,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            records.append(record)

        df = pd.DataFrame(records)

        if add_noise:
            df = self._inject_data_quality_issues(df)

        logger.info(f"Generated {len(df)} records across {df['category'].nunique()} categories")
        return df

    def generate_consumption_history(self, df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
        """Generate daily historical consumption data for demand forecasting."""
        logger.info(f"Generating {days}-day consumption history...")
        records = []
        base_date = self.today - timedelta(days=days)

        # Sample up to 80 ingredients for history (manageable size)
        sample_ingredients = df.drop_duplicates(subset="name").head(80)

        for _, row in sample_ingredients.iterrows():
            base_consumption = row["daily_consumption"]
            for day_offset in range(days):
                current_date = base_date + timedelta(days=day_offset)
                weekday = current_date.strftime("%A")

                # Weekly seasonality
                factor = WEEKDAY_DEMAND_FACTORS.get(weekday, 1.0)

                # Monthly trend (slight growth over time)
                trend = 1 + (day_offset / days) * 0.05

                # Random noise
                noise = np.random.normal(1.0, 0.12)

                # Occasional spikes (events, promotions)
                spike = 1.5 if random.random() < 0.03 else 1.0

                consumption = max(
                    0, base_consumption * factor * trend * noise * spike
                )
                records.append({
                    "ingredient_id": row["ingredient_id"],
                    "name": row["name"],
                    "category": row["category"],
                    "date": current_date.strftime("%Y-%m-%d"),
                    "consumption": round(consumption, 3),
                    "unit": row["unit"],
                })

        history_df = pd.DataFrame(records)
        logger.info(f"Generated {len(history_df)} consumption history records")
        return history_df

    def _build_ingredient_pool(self) -> list:
        pool = []
        for category, ingredients in INGREDIENTS_BY_CATEGORY.items():
            # Weight by number of records we want per category
            weight = max(1, len(ingredients))
            for ing in ingredients:
                pool.extend([(ing, category)] * weight)
        return pool

    def _inject_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intentionally inject realistic data quality problems."""
        df = df.copy()
        n = len(df)

        # ~3% missing quantities
        missing_qty_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
        df.loc[missing_qty_idx, "quantity"] = np.nan

        # ~2% missing daily consumption
        missing_dc_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
        df.loc[missing_dc_idx, "daily_consumption"] = np.nan

        # ~1.5% invalid expiry dates (clearly wrong)
        invalid_date_idx = np.random.choice(n, size=int(n * 0.015), replace=False)
        df.loc[invalid_date_idx, "expiry_date"] = "INVALID_DATE"

        # ~1% negative quantities (data entry errors)
        neg_qty_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
        df.loc[neg_qty_idx, "quantity"] = df.loc[neg_qty_idx, "quantity"].abs() * -1

        # ~2% duplicate records
        n_dupes = int(n * 0.02)
        dupe_idx = np.random.choice(n, size=n_dupes, replace=False)
        dupes = df.iloc[dupe_idx].copy()
        df = pd.concat([df, dupes], ignore_index=True)

        # ~1% inconsistent category casing
        mixed_case_idx = np.random.choice(len(df), size=int(len(df) * 0.01), replace=False)
        df.loc[mixed_case_idx, "category"] = df.loc[mixed_case_idx, "category"].str.upper()

        # ~0.5% zero consumption (inactive items)
        zero_dc_idx = np.random.choice(len(df), size=int(len(df) * 0.005), replace=False)
        df.loc[zero_dc_idx, "daily_consumption"] = 0

        logger.info(
            f"Injected data quality issues: {len(missing_qty_idx)} missing qty, "
            f"{len(invalid_date_idx)} invalid dates, {n_dupes} duplicates"
        )
        return df

    def save(self, inventory_df: pd.DataFrame, history_df: pd.DataFrame) -> tuple:
        inv_path = DATA_DIR / "inventory_raw.csv"
        hist_path = DATA_DIR / "consumption_history.csv"
        inventory_df.to_csv(inv_path, index=False)
        history_df.to_csv(hist_path, index=False)
        logger.info(f"Saved raw data to {DATA_DIR}")
        return inv_path, hist_path
