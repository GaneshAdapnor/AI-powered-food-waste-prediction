<div align="center">

# 🍃 Ecomeal AI
### Food Waste Intelligence System for Modern Restaurants

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ai-powered-food-waste-prediction.streamlit.app)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Gemini](https://img.shields.io/badge/Gemini%20AI-Recommendations-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **Predict food waste before it happens. Act before money walks out the kitchen door.**

<br/>

[🚀 Live Demo](https://ai-powered-food-waste-prediction.streamlit.app) • [📖 Docs](#-system-architecture) • [⚡ Quick Start](#-quick-start) • [🤖 AI Features](#-ai-integration)

<br/>

![Dashboard Preview](https://via.placeholder.com/900x400/0e1117/00cc96?text=Ecomeal+AI+%E2%80%94+Food+Waste+Intelligence+Dashboard)

</div>

---

## 🎯 The Problem

Restaurants lose **30–40% of purchased food** to waste every year. The root causes are invisible until it's too late:

- 📦 Overstocked perishables expire before they can be used
- 📉 Demand forecasting is done manually or not at all
- 🔔 No early warning system for items approaching expiry
- 💸 Kitchen staff have no financial context for waste decisions

**Ecomeal AI makes all of this visible — and actionable.**

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔮 **Waste Prediction** | XGBoost + LightGBM ensemble predicts which items will be wasted with **90% AUC** |
| 📊 **Demand Forecasting** | Holt-Winters time-series forecasting with weekly seasonality for 30-day demand |
| 🤖 **AI Chef Specials** | Gemini AI generates creative dish recommendations using expiring ingredients |
| 🔍 **Anomaly Detection** | Isolation Forest + domain rules catch unusual inventory patterns |
| 💡 **Explainability** | SHAP values translate model predictions into plain English for kitchen staff |
| 🛡️ **Robust Pipeline** | Handles missing values, invalid dates, duplicates — never crashes on bad data |
| 📱 **Interactive Dashboard** | 7-page Streamlit dashboard with real-time filters and Plotly charts |
| 🔌 **REST API** | FastAPI endpoints for integration with POS systems and inventory tools |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/GaneshAdapnor/AI-powered-food-waste-prediction.git
cd AI-powered-food-waste-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini API key (optional — works without it)
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Run the full pipeline
python3 main.py

# 5. Launch the dashboard
streamlit run dashboard/streamlit_app.py

# 6. Or start the REST API
python3 api/app.py  # → http://localhost:8000/docs
```

> 🌐 **Or just use the live app:** [ai-powered-food-waste-prediction.streamlit.app](https://ai-powered-food-waste-prediction.streamlit.app)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ECOMEAL AI PIPELINE                      │
├───────────┬──────────────┬────────────────┬────────────────────┤
│  DATA     │  ML MODELS   │  AI ENGINE     │  INTERFACES        │
│           │              │                │                    │
│ Generator │ Wastage      │ Gemini API     │ Streamlit          │
│ 1,200+    │ Predictor    │ Chef Specials  │ Dashboard          │
│ records   │ XGB + LGB    │                │ (7 pages)          │
│           │ AUC = 0.90   │ Template       │                    │
│ Preproc.  │              │ Fallback       │ FastAPI            │
│ Cleaning  │ Demand       │                │ REST API           │
│ Features  │ Forecaster   │ Explainability │                    │
│           │ Holt-Winters │ Engine         │ CLI                │
│           │              │ SHAP → NL      │ main.py            │
│           │ Anomaly      │                │                    │
│           │ Detector     │                │                    │
│           │ IsoForest    │                │                    │
└───────────┴──────────────┴────────────────┴────────────────────┘
```

```
ecomeal-ai/
├── 📂 src/
│   ├── config.py                  # Central configuration
│   ├── 📂 data/
│   │   ├── generator.py           # Realistic dataset simulation
│   │   └── preprocessor.py        # Robust cleaning + feature engineering
│   ├── 📂 models/
│   │   ├── wastage_predictor.py   # XGBoost + LightGBM + SHAP
│   │   ├── demand_forecaster.py   # Holt-Winters time-series
│   │   └── anomaly_detector.py    # Isolation Forest + rules
│   └── 📂 ai/
│       ├── recommendation_engine.py  # Gemini API integration
│       └── explainability.py         # SHAP → natural language
├── 📂 dashboard/
│   └── streamlit_app.py           # 7-page interactive dashboard
├── 📂 api/
│   └── app.py                     # FastAPI REST endpoints
├── 📂 tests/
│   └── test_pipeline.py           # 18 integration tests
└── main.py                        # Pipeline orchestrator
```

---

## 📊 Dashboard Pages

<table>
<tr>
<td width="50%">

**🏠 Overview**
- KPI cards: total items, waste value at risk, critical count
- Risk distribution pie chart
- Expiry timeline scatter plot

**📦 Inventory Analysis**
- Full filterable inventory table
- Category health comparison charts
- Waste value by category

**🔮 Wastage Predictions**
- Probability distribution histogram
- SHAP feature importance chart
- Top 20 items at highest risk

</td>
<td width="50%">

**📈 Demand Forecasting**
- Per-ingredient 30-day forecast
- Confidence interval visualization
- Overstock / shortage alerts table

**🤖 AI Recommendations**
- Gemini-powered Chef Specials
- Prioritized action plan
- Bulk usage strategies

**🔍 Anomaly Detection**
- Detected anomaly list with severity
- Anomaly type breakdown chart
- ML score scatter plot

**💡 Explainability**
- Per-item risk explanation
- Category-level risk analysis
- SHAP feature importance

</td>
</tr>
</table>

---

## 🤖 AI Integration

The recommendation engine uses **Google Gemini** to generate kitchen-ready insights:

```python
# Example output from Gemini AI
{
  "dishes": [
    {
      "name": "Mushroom & Spinach Risotto",
      "cuisine_style": "Italian",
      "expiring_ingredients_used": ["Mushrooms", "Spinach", "Heavy Cream"],
      "estimated_portions": 18,
      "waste_saved_kg": 3.2,
      "chef_tips": "Blanch spinach separately to preserve colour"
    }
  ],
  "action_plan": [...],
  "estimated_waste_savings_inr": 4800
}
```

> 💡 **Works without an API key** — falls back to intelligent template-based recommendations automatically.

---

## 🧠 ML Model Details

### Wastage Prediction

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.90 |
| **F1 Score** | 0.82 |
| **Precision** | 0.88 |
| **Recall** | 0.76 |

**Why XGBoost + LightGBM ensemble?**
- Both excel at tabular data with non-linear interactions
- Ensemble averaging reduces variance by ~3% AUC vs single model
- Native SHAP support for explainability
- `scale_pos_weight` handles class imbalance automatically

### Key Features (by SHAP importance)

```
waste_surplus_days      ████████████████████  (most important)
days_to_expiry          ████████████████
days_of_stock           ██████████████
turnover_rate           ████████████
historical_wastage_rate ██████████
stock_utilization       ████████
is_near_expiry          ██████
category_avg_waste_rate █████
```

### Demand Forecasting

Uses **Holt-Winters Exponential Smoothing** with:
- Additive trend component
- Weekly seasonality (7-day period)
- 95% confidence intervals
- Graceful fallback to EMA for short series

---

## 🛡️ Data Handling

The pipeline is **battle-hardened** against real-world data quality issues:

```
✅ Missing values      → Category median imputation
✅ Invalid dates       → Dropped with audit log
✅ Negative quantities → Absolute value correction
✅ Duplicates          → First occurrence retained
✅ Zero consumption    → Replaced with 0.01 (flagged as anomaly)
✅ Unknown categories  → Mapped to nearest known category
✅ Empty DataFrames    → Graceful return, never crashes
✅ API failures        → Template fallback, pipeline continues
```

---

## 🔌 REST API

```bash
# Start the API
python3 api/app.py  # → http://localhost:8000/docs
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System status + model info |
| `GET` | `/inventory/at-risk` | Top N items by waste probability |
| `GET` | `/inventory/summary` | Aggregated waste intelligence |
| `POST` | `/predict` | Predict waste risk for a single item |
| `GET` | `/recommendations/chef-specials` | Generate AI dish recommendations |
| `GET` | `/anomalies` | List detected inventory anomalies |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Spinach",
    "category": "Vegetables",
    "quantity": 5.0,
    "unit": "kg",
    "expiry_date": "2026-06-01",
    "daily_consumption": 0.8,
    "price_per_unit": 60.0,
    "storage_type": "Refrigerator"
  }'
```

---

## 🧪 Tests

```bash
python3 -m pytest tests/ -v
```

```
✅ TestDataGenerator::test_generates_correct_count
✅ TestDataGenerator::test_all_required_columns_present
✅ TestDataGenerator::test_noise_injection_creates_issues
✅ TestDataPreprocessor::test_clean_never_crashes
✅ TestDataPreprocessor::test_handles_completely_empty_dataframe
✅ TestDataPreprocessor::test_handles_all_invalid_dates
✅ TestWastagePredictor::test_trains_successfully
✅ TestWastagePredictor::test_predict_returns_probabilities
✅ TestAnomalyDetector::test_detects_anomalies
... 18 tests total — all passing ✅
```

---

## 📈 Scalability

| Concern | Current | Scale Path |
|---------|---------|-----------|
| **Data volume** | 1,200 records | TimescaleDB + batch processing |
| **Inference** | ~50ms per batch | Model serving with joblib singleton |
| **Forecasting** | 80 ingredients in ~500ms | `ThreadPoolExecutor` for parallelism |
| **Recommendations** | Per-request API call | Redis caching for repeated queries |
| **Dashboard** | 5-min Streamlit cache | CDN + edge caching |

---

## 🔮 Future Roadmap

- [ ] **POS Integration** — real daily sales data from restaurant POS systems
- [ ] **Computer Vision** — shelf camera integration for automated spoilage detection
- [ ] **RAG Recommendations** — vector database of 10,000+ recipes for semantic dish matching
- [ ] **Multi-location** — inter-kitchen transfer suggestions before re-ordering
- [ ] **Mobile App** — kitchen-facing alerts with one-tap action confirmation
- [ ] **Dynamic Pricing** — suggest discounts on at-risk menu items to drive demand
- [ ] **Automated Retraining** — weekly model updates as actual waste outcomes come in

---

## 🧑‍💻 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-orange?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-green?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini%20AI-4285F4?style=flat-square&logo=google&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-blue?style=flat-square)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

</div>

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with ❤️ for the Ecomeal AI/ML Internship Assignment**

*Reducing food waste, one prediction at a time.*

⭐ **Star this repo if you found it useful!**

</div>
