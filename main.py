"""
Ecomeal AI — Food Waste Intelligence Pipeline
Main entry point: generates data, trains models, produces insights.
"""
import logging
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, MODELS_DIR
from src.data.generator import RestaurantDataGenerator
from src.data.preprocessor import DataPreprocessor
from src.models.wastage_predictor import WastagePredictor
from src.models.demand_forecaster import DemandForecaster
from src.models.anomaly_detector import AnomalyDetector
from src.ai.recommendation_engine import RecommendationEngine
from src.ai.explainability import ExplainabilityEngine

logger = logging.getLogger(__name__)
console = Console()


def run_pipeline(
    regenerate: bool = True,
    n_records: int = 1200,
    skip_training: bool = False,
) -> dict:
    """Execute full Ecomeal AI pipeline."""

    console.print(Panel.fit(
        "[bold green]ECOMEAL AI — Food Waste Intelligence System[/bold green]\n"
        "Powered by XGBoost + LightGBM + Claude AI",
        border_style="green"
    ))

    # ─── STEP 1: Data Generation ───────────────────────────────────────────
    console.print("\n[bold cyan]Step 1/6: Data Generation[/bold cyan]")

    raw_inv_path = DATA_DIR / "inventory_raw.csv"
    hist_path = DATA_DIR / "consumption_history.csv"

    if regenerate or not raw_inv_path.exists():
        generator = RestaurantDataGenerator(n_records=n_records)
        raw_df = generator.generate_inventory(add_noise=True)
        history_df = generator.generate_consumption_history(raw_df)
        generator.save(raw_df, history_df)
        console.print(f"  ✓ Generated {len(raw_df)} inventory records + {len(history_df)} history rows")
    else:
        raw_df = pd.read_csv(raw_inv_path)
        history_df = pd.read_csv(hist_path)
        console.print(f"  ✓ Loaded existing data: {len(raw_df)} records")

    # ─── STEP 2: Data Cleaning & Feature Engineering ───────────────────────
    console.print("\n[bold cyan]Step 2/6: Data Preprocessing & Feature Engineering[/bold cyan]")

    preprocessor = DataPreprocessor()
    clean_df = preprocessor.clean(raw_df)
    featured_df = preprocessor.engineer_features(clean_df)
    labeled_df = preprocessor.build_training_labels(featured_df)

    console.print(f"  ✓ Cleaned: {len(labeled_df)} records ({preprocessor.cleaning_report})")
    console.print(f"  ✓ Features engineered: {labeled_df.shape[1]} columns")

    # Save processed data
    labeled_df.to_csv(DATA_DIR / "inventory_processed.csv", index=False)

    # ─── STEP 3: Wastage Prediction ────────────────────────────────────────
    console.print("\n[bold cyan]Step 3/6: Wastage Prediction Model[/bold cyan]")

    predictor = WastagePredictor()
    model_path = MODELS_DIR / "wastage_predictor.joblib"

    if skip_training and model_path.exists():
        predictor.load()
        console.print("  ✓ Loaded existing model")
    else:
        metrics = predictor.train(labeled_df)
        predictor.save()
        console.print(
            f"  ✓ Trained XGBoost + LightGBM ensemble\n"
            f"     AUC={metrics['roc_auc']} | F1={metrics['f1']} | "
            f"Precision={metrics['precision']} | Recall={metrics['recall']}"
        )

    predicted_df = predictor.predict(labeled_df)
    predicted_df.to_csv(DATA_DIR / "inventory_predicted.csv", index=False)

    top_at_risk = predictor.get_top_at_risk(labeled_df, n=15)

    # ─── STEP 4: Demand Forecasting ────────────────────────────────────────
    console.print("\n[bold cyan]Step 4/6: Demand Forecasting[/bold cyan]")

    forecaster = DemandForecaster(forecast_horizon=30)
    forecast_summary = forecaster.fit_predict(history_df, labeled_df)
    forecast_summary.to_csv(DATA_DIR / "demand_forecasts.csv", index=False)

    alerts = forecaster.get_alerts()
    overstock_alerts = forecaster.get_alerts("high")
    console.print(
        f"  ✓ Forecasted demand for {len(forecast_summary)} ingredients\n"
        f"     {len(alerts)} total alerts ({len(overstock_alerts)} high-severity)"
    )

    # ─── STEP 5: Anomaly Detection ─────────────────────────────────────────
    console.print("\n[bold cyan]Step 5/6: Anomaly Detection[/bold cyan]")

    detector = AnomalyDetector(contamination=0.05)
    anomaly_df = detector.fit_predict(predicted_df)
    anomaly_df.to_csv(DATA_DIR / "inventory_with_anomalies.csv", index=False)

    anomaly_summary = detector.get_anomaly_summary()
    console.print(
        f"  ✓ Detected {len(anomaly_summary)} anomalies in inventory\n"
        f"     High severity: {sum(1 for a in anomaly_summary if a['severity'] == 'high')}"
    )

    # ─── STEP 6: AI Recommendations ────────────────────────────────────────
    console.print("\n[bold cyan]Step 6/6: AI-Powered Recommendations[/bold cyan]")

    rec_engine = RecommendationEngine()
    chef_specials = rec_engine.generate_chef_specials(predicted_df, n_specials=3)
    inv_report = rec_engine.generate_inventory_report(predicted_df, predictor.metrics)

    api_mode = "Claude AI" if chef_specials.get("source") == "claude_ai" else "Template (set ANTHROPIC_API_KEY for AI)"
    console.print(f"  ✓ Generated {len(chef_specials.get('dishes', []))} Chef Specials ({api_mode})")
    console.print(
        f"  ✓ Estimated waste savings: ₹{chef_specials.get('estimated_waste_savings_inr', 0):.0f}"
    )

    # ─── RESULTS SUMMARY ───────────────────────────────────────────────────
    console.print("\n")
    _print_results_table(predicted_df, top_at_risk, alerts, anomaly_summary)
    _print_chef_specials(chef_specials)
    _print_inventory_report(inv_report)

    return {
        "inventory_df": anomaly_df,
        "predicted_df": predicted_df,
        "forecast_summary": forecast_summary,
        "anomaly_summary": anomaly_summary,
        "chef_specials": chef_specials,
        "inventory_report": inv_report,
        "model_metrics": predictor.metrics,
        "demand_alerts": alerts,
    }


def _print_results_table(df, top_at_risk, alerts, anomaly_summary):
    total_items = len(df)
    high_risk = (df.get("risk_label", pd.Series()) == "Critical").sum() if "risk_label" in df.columns else 0
    waste_value = df["waste_value_at_risk"].sum() if "waste_value_at_risk" in df.columns else 0
    will_waste = df["will_waste_predicted"].sum() if "will_waste_predicted" in df.columns else 0

    console.print(Panel.fit(
        f"[bold]INVENTORY INTELLIGENCE SUMMARY[/bold]\n\n"
        f"  Total Items:         {total_items:>6,}\n"
        f"  Predicted to Waste:  {int(will_waste):>6,} items\n"
        f"  Critical Risk:       {int(high_risk):>6,} items\n"
        f"  Total Waste Value:   ₹{waste_value:>9,.2f}\n"
        f"  Demand Alerts:       {len(alerts):>6,}\n"
        f"  Anomalies Detected:  {len(anomaly_summary):>6,}",
        title="[green]Pipeline Results[/green]",
        border_style="green",
    ))

    table = Table(title="Top 10 Items at Highest Risk", box=box.ROUNDED)
    table.add_column("Item", style="bold")
    table.add_column("Category")
    table.add_column("Qty")
    table.add_column("Expires In")
    table.add_column("Waste Prob", justify="right")
    table.add_column("Waste Value ₹", justify="right")
    table.add_column("Risk")

    risk_colors = {"Critical": "red", "High": "yellow", "Medium": "cyan", "Low": "green"}

    for _, row in top_at_risk.head(10).iterrows():
        color = risk_colors.get(row.get("risk_label", "Low"), "white")
        table.add_row(
            str(row.get("name", ""))[:20],
            str(row.get("category", "")),
            f"{row.get('quantity', 0):.1f} {row.get('unit', '')}",
            f"{int(row.get('days_to_expiry', 0))}d",
            f"{row.get('waste_probability', 0):.0%}",
            f"₹{row.get('waste_value_at_risk', 0):.0f}",
            f"[{color}]{row.get('risk_label', '')}[/{color}]",
        )
    console.print(table)


def _print_chef_specials(chef_specials: dict):
    dishes = chef_specials.get("dishes", [])
    if not dishes:
        return

    console.print(Panel.fit(
        f"[bold yellow]CHEF'S SPECIALS — TODAY[/bold yellow]\n" +
        "\n".join([
            f"\n  {i+1}. [bold]{d.get('name', 'Special')}[/bold] ({d.get('cuisine_style', '')})\n"
            f"     {d.get('description', '')[:100]}...\n"
            f"     Uses: {', '.join(d.get('expiring_ingredients_used', [])[:3])}\n"
            f"     Portions: {d.get('estimated_portions', '?')} | "
            f"Saves: {d.get('waste_saved_kg', 0):.1f} kg"
            for i, d in enumerate(dishes[:3])
        ]),
        title="[yellow]AI Recommendations[/yellow]",
        border_style="yellow",
    ))


def _print_inventory_report(report: str):
    console.print(Panel(
        report,
        title="[blue]Inventory Intelligence Report[/blue]",
        border_style="blue",
    ))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ecomeal AI Food Waste Intelligence")
    parser.add_argument("--no-regenerate", action="store_true", help="Use existing data")
    parser.add_argument("--skip-training", action="store_true", help="Use saved model")
    parser.add_argument("--n-records", type=int, default=1200, help="Number of inventory records")
    args = parser.parse_args()

    results = run_pipeline(
        regenerate=not args.no_regenerate,
        n_records=args.n_records,
        skip_training=args.skip_training,
    )

    console.print("\n[bold green]Pipeline complete![/bold green]")
    console.print("Run [bold]streamlit run dashboard/streamlit_app.py[/bold] to explore the dashboard.")
    console.print("Processed data saved to [bold]data/[/bold] directory.")
