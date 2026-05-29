"""
Ecomeal AI — Interactive Streamlit Dashboard
Multi-page interface for inventory intelligence, waste prediction, and recommendations.
"""
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, MODELS_DIR, HIGH_RISK_DAYS, MEDIUM_RISK_DAYS
from src.data.generator import RestaurantDataGenerator
from src.data.preprocessor import DataPreprocessor
from src.models.wastage_predictor import WastagePredictor
from src.models.demand_forecaster import DemandForecaster
from src.models.anomaly_detector import AnomalyDetector
from src.ai.recommendation_engine import RecommendationEngine
from src.ai.explainability import ExplainabilityEngine

st.set_page_config(
    page_title="Ecomeal AI — Food Waste Intelligence",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.risk-critical { color: #ff4b4b; font-weight: bold; }
.risk-high { color: #ffa500; font-weight: bold; }
.risk-medium { color: #ffd700; }
.risk-low { color: #00cc96; }
.section-header { border-left: 4px solid #00cc96; padding-left: 12px; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    """Load processed data or run pipeline if not available."""
    processed_path = DATA_DIR / "inventory_predicted.csv"
    history_path = DATA_DIR / "consumption_history.csv"
    forecast_path = DATA_DIR / "demand_forecasts.csv"

    if not processed_path.exists():
        with st.spinner("Running Ecomeal AI pipeline for first time..."):
            generator = RestaurantDataGenerator(n_records=1200)
            raw_df = generator.generate_inventory(add_noise=True)
            history_df = generator.generate_consumption_history(raw_df)
            generator.save(raw_df, history_df)

            preprocessor = DataPreprocessor()
            clean_df = preprocessor.clean(raw_df)
            featured_df = preprocessor.engineer_features(clean_df)
            labeled_df = preprocessor.build_training_labels(featured_df)

            predictor = WastagePredictor()
            predictor.train(labeled_df)
            predictor.save()
            predicted_df = predictor.predict(labeled_df)
            predicted_df.to_csv(processed_path, index=False)

            forecaster = DemandForecaster()
            forecast_summary = forecaster.fit_predict(history_df, labeled_df)
            forecast_summary.to_csv(forecast_path, index=False)

            detector = AnomalyDetector()
            anomaly_df = detector.fit_predict(predicted_df)
            anomaly_df.to_csv(DATA_DIR / "inventory_with_anomalies.csv", index=False)

    df = pd.read_csv(processed_path)
    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    forecast_df = pd.read_csv(forecast_path) if forecast_path.exists() else pd.DataFrame()

    return df, history_df, forecast_df


@st.cache_resource
def load_predictor():
    pred = WastagePredictor()
    try:
        pred.load()
    except Exception:
        pass
    return pred


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image("https://via.placeholder.com/200x60/00cc96/ffffff?text=ECOMEAL+AI", width=200)
    st.sidebar.markdown("## Navigation")

    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Inventory Analysis", "Wastage Predictions",
         "Demand Forecasting", "AI Recommendations", "Anomaly Detection", "Explainability"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    df, _, _ = load_data()

    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
    selected_cat = st.sidebar.selectbox("Category", categories)

    risk_levels = ["All", "Critical", "High", "Medium", "Low"]
    selected_risk = st.sidebar.selectbox("Risk Level", risk_levels)

    max_expiry = st.sidebar.slider("Max Days to Expiry", 1, 60, 30)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data")
    if st.sidebar.button("Regenerate Dataset"):
        st.cache_data.clear()
        st.rerun()

    return page, selected_cat, selected_risk, max_expiry


def filter_data(df, selected_cat, selected_risk, max_expiry):
    filtered = df.copy()
    if selected_cat != "All":
        filtered = filtered[filtered["category"] == selected_cat]
    if selected_risk != "All" and "risk_label" in filtered.columns:
        filtered = filtered[filtered["risk_label"] == selected_risk]
    if "days_to_expiry" in filtered.columns:
        filtered = filtered[filtered["days_to_expiry"] <= max_expiry]
    return filtered


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_overview(df):
    st.markdown('<h1 style="color:#00cc96;">🍃 Ecomeal AI — Food Waste Intelligence</h1>', unsafe_allow_html=True)
    st.markdown("Real-time inventory analysis powered by XGBoost + LightGBM + Claude AI")

    col1, col2, col3, col4, col5 = st.columns(5)

    total = len(df)
    will_waste = int(df["will_waste_predicted"].sum()) if "will_waste_predicted" in df.columns else 0
    critical = int((df["risk_label"] == "Critical").sum()) if "risk_label" in df.columns else 0
    waste_val = df["waste_value_at_risk"].sum() if "waste_value_at_risk" in df.columns else 0
    expired = int((df["days_to_expiry"] <= 0).sum()) if "days_to_expiry" in df.columns else 0

    with col1:
        st.metric("Total Items", f"{total:,}")
    with col2:
        st.metric("Predicted to Waste", f"{will_waste:,}", delta=f"{will_waste/max(total,1):.0%}", delta_color="inverse")
    with col3:
        st.metric("Critical Risk", f"{critical:,}", delta="Immediate action", delta_color="inverse")
    with col4:
        st.metric("Waste Value at Risk", f"₹{waste_val:,.0f}", delta_color="inverse")
    with col5:
        st.metric("Already Expired", f"{expired:,}", delta_color="inverse")

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Risk Distribution")
        if "risk_label" in df.columns:
            risk_counts = df["risk_label"].value_counts()
            color_map = {"Critical": "#ff4b4b", "High": "#ffa500", "Medium": "#ffd700", "Low": "#00cc96"}
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map=color_map,
                hole=0.45,
            )
            fig.update_layout(margin=dict(t=30, b=10), height=350, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Waste Value by Category")
        if "category" in df.columns and "waste_value_at_risk" in df.columns:
            cat_waste = df.groupby("category")["waste_value_at_risk"].sum().sort_values(ascending=True)
            fig = px.bar(
                x=cat_waste.values, y=cat_waste.index,
                orientation="h",
                color=cat_waste.values,
                color_continuous_scale="RdYlGn_r",
                labels={"x": "Waste Value (₹)", "y": "Category"},
            )
            fig.update_layout(margin=dict(t=30, b=10), height=350, paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Expiry Timeline — Next 14 Days")
    if "days_to_expiry" in df.columns:
        timeline = df[df["days_to_expiry"].between(0, 14)].copy()
        if not timeline.empty:
            fig = px.scatter(
                timeline,
                x="days_to_expiry",
                y="waste_probability",
                size="waste_value_at_risk",
                color="risk_label",
                hover_name="name",
                hover_data=["category", "quantity", "unit"],
                color_discrete_map={"Critical": "#ff4b4b", "High": "#ffa500", "Medium": "#ffd700", "Low": "#00cc96"},
                labels={"days_to_expiry": "Days to Expiry", "waste_probability": "Waste Probability"},
            )
            fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


def page_inventory_analysis(df):
    st.markdown('<h2 class="section-header">Inventory Analysis</h2>', unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("No data matches the current filters. Adjust the sidebar filters.")
        return

    display_cols = [
        "name", "category", "quantity", "unit", "days_to_expiry",
        "expiry_risk_level", "daily_consumption", "days_of_stock",
        "estimated_waste_qty", "waste_value_at_risk", "storage_type", "supplier",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].copy()

    if "waste_value_at_risk" in display_df.columns:
        display_df["waste_value_at_risk"] = display_df["waste_value_at_risk"].round(2)
    if "estimated_waste_qty" in display_df.columns:
        display_df["estimated_waste_qty"] = display_df["estimated_waste_qty"].round(3)

    st.dataframe(
        display_df.sort_values("days_to_expiry") if "days_to_expiry" in display_df.columns else display_df,
        use_container_width=True,
        height=450,
    )

    st.markdown("### Category Health Overview")
    if "category" in df.columns:
        cat_summary = df.groupby("category").agg(
            items=("name", "count"),
            avg_days_to_expiry=("days_to_expiry", "mean"),
            total_waste_value=("waste_value_at_risk", "sum"),
            avg_waste_prob=("waste_probability", "mean"),
        ).reset_index()
        cat_summary = cat_summary.round(2)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Avg Waste Probability", "Total Waste Value (₹)"))
        fig.add_trace(
            go.Bar(x=cat_summary["category"], y=cat_summary["avg_waste_prob"],
                   marker_color="#ffa500", name="Avg Waste Prob"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=cat_summary["category"], y=cat_summary["total_waste_value"],
                   marker_color="#ff4b4b", name="Waste Value"),
            row=1, col=2
        )
        fig.update_layout(height=350, showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def page_wastage_predictions(df):
    st.markdown('<h2 class="section-header">Wastage Predictions</h2>', unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("No data matches the current filters. Adjust the sidebar filters.")
        return

    predictor = load_predictor()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Waste Probability Distribution")
        if "waste_probability" in df.columns:
            fig = px.histogram(
                df, x="waste_probability", nbins=40,
                color="risk_label" if "risk_label" in df.columns else None,
                color_discrete_map={"Critical": "#ff4b4b", "High": "#ffa500", "Medium": "#ffd700", "Low": "#00cc96"},
                labels={"waste_probability": "Waste Probability"},
            )
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Model Performance")
        if predictor._is_trained and predictor.metrics:
            m = predictor.metrics
            st.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")
            st.metric("F1 Score", f"{m.get('f1', 0):.4f}")
            st.metric("Precision", f"{m.get('precision', 0):.4f}")
            st.metric("Recall", f"{m.get('recall', 0):.4f}")

    st.markdown("### Feature Importance (SHAP Values)")
    if predictor._is_trained and predictor.feature_importance is not None:
        fi = predictor.feature_importance.head(12)
        fig = px.bar(
            fi, x="shap_importance", y="feature", orientation="h",
            color="shap_importance", color_continuous_scale="Viridis",
            labels={"shap_importance": "Mean |SHAP Value|", "feature": "Feature"},
        )
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Items at Risk")
    if predictor._is_trained:
        top_risk = predictor.get_top_at_risk(df, n=20)
        if not top_risk.empty:
            risk_cols = ["name", "category", "quantity", "unit", "days_to_expiry",
                         "waste_probability", "risk_score", "risk_label", "waste_value_at_risk"]
            available = [c for c in risk_cols if c in top_risk.columns]
            st.dataframe(top_risk[available].round(3), use_container_width=True, height=400)


def page_demand_forecasting(df, forecast_df):
    st.markdown('<h2 class="section-header">Demand Forecasting</h2>', unsafe_allow_html=True)

    if forecast_df.empty:
        st.info("No forecast data available. Run the pipeline first.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Select Ingredient")
        ingredient_names = sorted(forecast_df["name"].unique().tolist())
        selected = st.selectbox("Ingredient", ingredient_names)

        ing_forecast = forecast_df[forecast_df["name"] == selected].iloc[0] if not forecast_df[forecast_df["name"] == selected].empty else None

        if ing_forecast is not None:
            st.metric("Avg Daily Forecast", f"{ing_forecast.get('avg_daily_forecast', 0):.2f}")
            st.metric("30-Day Total Forecast", f"{ing_forecast.get('total_30d_forecast', 0):.2f}")
            st.metric("Trend", ing_forecast.get("trend_direction", "stable").capitalize())
            st.metric("Confidence", ing_forecast.get("forecast_confidence", "medium").capitalize())

    with col2:
        st.markdown("### Forecast Chart")
        if ing_forecast is not None and "daily_forecasts" in ing_forecast:
            try:
                daily = ing_forecast["daily_forecasts"]
                if isinstance(daily, str):
                    try:
                        daily = json.loads(daily)
                    except (json.JSONDecodeError, ValueError):
                        import ast
                        daily = ast.literal_eval(daily)

                if daily:
                    forecast_chart_df = pd.DataFrame(daily)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=forecast_chart_df["day"],
                        y=forecast_chart_df["predicted_consumption"],
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#00cc96", width=2),
                    ))
                    if "upper_bound" in forecast_chart_df.columns:
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast_chart_df["day"], forecast_chart_df["day"][::-1]]),
                            y=pd.concat([forecast_chart_df["upper_bound"], forecast_chart_df["lower_bound"][::-1]]),
                            fill="toself", fillcolor="rgba(0,204,150,0.1)",
                            line=dict(color="rgba(255,255,255,0)"),
                            name="95% CI",
                        ))
                    fig.update_layout(
                        xaxis_title="Day", yaxis_title="Predicted Consumption",
                        height=350, paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render forecast chart: {e}")

    st.markdown("### Demand Summary — All Ingredients")
    summary_cols = ["name", "category", "avg_daily_forecast", "total_30d_forecast", "trend_direction", "forecast_confidence"]
    available = [c for c in summary_cols if c in forecast_df.columns]
    st.dataframe(forecast_df[available].round(3), use_container_width=True, height=400)


def page_ai_recommendations(df):
    st.markdown('<h2 class="section-header">AI-Powered Recommendations</h2>', unsafe_allow_html=True)

    rec_engine = RecommendationEngine()

    st.markdown("### Generate Chef Specials")

    col1, col2 = st.columns([1, 2])
    with col1:
        max_days = st.slider("Ingredients expiring within (days)", 1, 14, 7)
        n_specials = st.number_input("Number of chef specials", 1, 5, 3)
        generate_btn = st.button("Generate Recommendations", type="primary")

    with col2:
        if "days_to_expiry" in df.columns:
            expiring = df[df["days_to_expiry"].between(0, max_days)].sort_values("days_to_expiry")
            if not expiring.empty:
                st.markdown(f"**{len(expiring)} ingredients expiring within {max_days} days:**")
                cols_show = ["name", "category", "quantity", "unit", "days_to_expiry", "waste_value_at_risk"]
                available = [c for c in cols_show if c in expiring.columns]
                st.dataframe(expiring[available].head(10).round(2), use_container_width=True, height=200)

    if generate_btn or "chef_specials" not in st.session_state:
        with st.spinner("Generating AI recommendations..."):
            specials = rec_engine.generate_chef_specials(df, n_specials=n_specials, max_expiry_days=max_days)
            st.session_state["chef_specials"] = specials

    if "chef_specials" in st.session_state:
        specials = st.session_state["chef_specials"]

        source = specials.get("source", "template")
        if source == "claude_ai":
            st.success("Powered by Claude AI")
        else:
            st.info("Using template recommendations. Set ANTHROPIC_API_KEY for AI-powered suggestions.")

        summary = specials.get("summary", "")
        if summary:
            st.info(f"**Summary:** {summary}")

        savings = specials.get("estimated_waste_savings_inr", 0)
        st.metric("Estimated Waste Savings", f"₹{savings:,.0f}")

        dishes = specials.get("dishes", [])
        if dishes:
            st.markdown("### Today's Chef Specials")
            for i, dish in enumerate(dishes):
                with st.expander(f"🍽️ {dish.get('name', f'Special {i+1}')} — {dish.get('cuisine_style', '')}", expanded=i == 0):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"**Description:** {dish.get('description', '')}")
                        expiring_used = dish.get("expiring_ingredients_used", [])
                        if expiring_used:
                            st.markdown(f"**Expiring Ingredients Used:** {', '.join(expiring_used)}")
                        tip = dish.get("chef_tips", "")
                        if tip:
                            st.markdown(f"**Chef's Tip:** _{tip}_")
                    with col_b:
                        st.metric("Portions", dish.get("estimated_portions", "?"))
                        st.metric("Prep Time", f"{dish.get('prep_time_minutes', '?')} min")
                        st.metric("Waste Saved", f"{dish.get('waste_saved_kg', 0):.1f} kg")

        action_plan = specials.get("action_plan", [])
        if action_plan:
            st.markdown("### Action Plan")
            for action in action_plan:
                priority = action.get("priority", "")
                deadline = action.get("deadline", "")
                color = "🔴" if deadline == "Today" else "🟡" if deadline == "Tomorrow" else "🟢"
                st.markdown(
                    f"{color} **Priority {priority}** | {action.get('ingredient', '')} | "
                    f"_{action.get('action', '')}_ | Deadline: **{deadline}** | "
                    f"{action.get('impact', '')}"
                )

        strategies = specials.get("bulk_strategies", [])
        if strategies:
            st.markdown("### Bulk Usage Strategies")
            for s in strategies:
                st.markdown(f"- {s}")


def page_anomaly_detection(df):
    st.markdown('<h2 class="section-header">Anomaly Detection</h2>', unsafe_allow_html=True)

    detector = AnomalyDetector(contamination=0.05)

    with st.spinner("Running anomaly detection..."):
        anomaly_df = detector.fit_predict(df)
        anomaly_summary = detector.get_anomaly_summary()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Anomalies", len(anomaly_summary))
    with col2:
        high_sev = sum(1 for a in anomaly_summary if a.get("severity") == "high")
        st.metric("High Severity", high_sev)

    if anomaly_summary:
        st.markdown("### Detected Anomalies")
        anomaly_display = pd.DataFrame(anomaly_summary)
        if not anomaly_display.empty:
            cols = ["name", "category", "anomaly_type", "severity", "waste_value_at_risk"]
            available = [c for c in cols if c in anomaly_display.columns]

            styled_df = anomaly_display[available].copy()
            st.dataframe(styled_df, use_container_width=True, height=400)

        if "anomaly_type" in anomaly_display.columns:
            type_counts = anomaly_display["anomaly_type"].value_counts()
            fig = px.bar(
                x=type_counts.values, y=type_counts.index,
                orientation="h", color=type_counts.values,
                color_continuous_scale="RdYlGn_r",
                labels={"x": "Count", "y": "Anomaly Type"},
                title="Anomaly Types Distribution",
            )
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Anomaly Scatter Plot")
    if "anomaly_score" in anomaly_df.columns:
        fig = px.scatter(
            anomaly_df,
            x="days_to_expiry", y="anomaly_score",
            color="is_anomaly" if "is_anomaly" in anomaly_df.columns else None,
            size="waste_value_at_risk" if "waste_value_at_risk" in anomaly_df.columns else None,
            hover_name="name" if "name" in anomaly_df.columns else None,
            color_discrete_map={0: "#00cc96", 1: "#ff4b4b"},
            labels={"anomaly_score": "Anomaly Score", "days_to_expiry": "Days to Expiry"},
        )
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def page_explainability(df):
    st.markdown('<h2 class="section-header">Explainability & Insights</h2>', unsafe_allow_html=True)

    explainer = ExplainabilityEngine()
    predictor = load_predictor()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Item-Level Explanation")
        if "name" in df.columns and "waste_probability" in df.columns:
            high_risk_items = df.nlargest(50, "waste_probability")["name"].tolist()
            selected_item = st.selectbox("Select item to explain", high_risk_items)

            item_row = df[df["name"] == selected_item].iloc[0] if not df[df["name"] == selected_item].empty else None

            if item_row is not None:
                explanation = explainer.explain_prediction(item_row.to_dict())

                st.markdown(f"**Primary Reason:**")
                st.info(explanation["primary_reason"])

                st.markdown(f"**Recommended Action:**")
                st.warning(explanation["recommended_action"])

                st.markdown(f"**Financial Impact:** {explanation['financial_impact']}")

                if explanation.get("risk_drivers"):
                    st.markdown("**Risk Drivers:**")
                    for d in explanation["risk_drivers"]:
                        st.markdown(f"  🔴 {d}")

                if explanation.get("mitigating_factors"):
                    st.markdown("**Mitigating Factors:**")
                    for f in explanation["mitigating_factors"]:
                        st.markdown(f"  🟢 {f}")

    with col2:
        st.markdown("### Category Risk Analysis")
        if "category" in df.columns and "waste_probability" in df.columns:
            cat_explanations = explainer.explain_category_risk(df)
            for cat, info in cat_explanations.items():
                with st.expander(f"{cat} — {info['avg_waste_probability']:.0%} avg risk"):
                    st.markdown(info["explanation"])
                    st.metric("Items at High Risk", info["items_at_high_risk"])
                    st.metric("Total Waste Value", f"₹{info['total_waste_value']:,.0f}")

    st.markdown("### Feature Importance Heatmap")
    if predictor._is_trained and predictor.feature_importance is not None:
        fi = predictor.feature_importance.head(15)
        fig = go.Figure(go.Bar(
            x=fi["shap_importance"],
            y=fi["feature"],
            orientation="h",
            marker=dict(
                color=fi["shap_importance"],
                colorscale="RdYlGn_r",
            ),
        ))
        fig.update_layout(
            title="SHAP Feature Importance",
            xaxis_title="Mean |SHAP Value|",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    page, selected_cat, selected_risk, max_expiry = render_sidebar()

    try:
        df, history_df, forecast_df = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}. Please run `python main.py` first.")
        st.stop()

    filtered_df = filter_data(df, selected_cat, selected_risk, max_expiry)

    st.sidebar.markdown(f"**Showing:** {len(filtered_df):,} of {len(df):,} items")

    if page == "Overview":
        page_overview(filtered_df)
    elif page == "Inventory Analysis":
        page_inventory_analysis(filtered_df)
    elif page == "Wastage Predictions":
        page_wastage_predictions(filtered_df)
    elif page == "Demand Forecasting":
        page_demand_forecasting(filtered_df, forecast_df)
    elif page == "AI Recommendations":
        page_ai_recommendations(filtered_df)
    elif page == "Anomaly Detection":
        page_anomaly_detection(filtered_df)
    elif page == "Explainability":
        page_explainability(filtered_df)


if __name__ == "__main__":
    main()
