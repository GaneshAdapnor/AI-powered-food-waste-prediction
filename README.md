# Ecomeal AI — Food Waste Intelligence System

An end-to-end AI-powered system for predicting food waste, forecasting demand, detecting inventory anomalies, and generating intelligent Chef Special recommendations for restaurants.

---

## Quick Start

```bash
# 1. Clone and set up
cd ecomeal-ai
pip install -r requirements.txt

# 2. (Optional) Add your Anthropic API key for AI-powered recommendations
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=your_key_here

# 3. Run the full pipeline
python3 main.py

# 4. Launch the interactive dashboard
streamlit run dashboard/streamlit_app.py

# 5. Run tests
python3 -m pytest tests/ -v

# 6. Start REST API
python3 api/app.py  # → http://localhost:8000/docs
```

---

## System Architecture

```
ecomeal-ai/
├── src/
│   ├── config.py                 # Central configuration
│   ├── data/
│   │   ├── generator.py          # Realistic dataset simulation (1200+ records)
│   │   └── preprocessor.py       # Robust cleaning + feature engineering
│   ├── models/
│   │   ├── wastage_predictor.py  # XGBoost + LightGBM ensemble + SHAP
│   │   ├── demand_forecaster.py  # Holt-Winters exponential smoothing
│   │   └── anomaly_detector.py   # Isolation Forest + rule-based detection
│   └── ai/
│       ├── recommendation_engine.py  # Claude API + template fallback
│       └── explainability.py         # SHAP → natural language explanations
├── dashboard/
│   └── streamlit_app.py          # 7-page interactive dashboard
├── api/
│   └── app.py                    # FastAPI REST endpoints
├── tests/
│   └── test_pipeline.py          # 18 integration tests
└── main.py                       # Pipeline orchestrator with rich CLI output
```

---

## Dataset Approach

### Synthetic Data with Realistic Patterns

Since no public restaurant inventory dataset captures all required fields, I simulated **1,200+ inventory entries** across **9 food categories** and **70+ ingredient types** using domain-specific parameters.

**Key design decisions:**

- **Category-specific shelf lives**: Vegetables expire in 2-14 days; Spices in 90-730 days. Not uniform random.
- **Consumption patterns**: Weekend demand 35% higher than Monday (Friday=1.2x, Saturday=1.35x multiplier). Simulates real restaurant weekend rush.
- **Seasonal trends**: Slight 5% consumption growth over 90-day history window.
- **Occasional spikes**: 3% probability of 1.5x consumption spike per day (events, promotions).
- **Intentional data quality issues**: 3% missing quantities, 2% invalid dates, 1% negative values, 2% duplicates — to simulate real-world messiness and demonstrate robust handling.

**Inventory fields generated:**
`ingredient_id`, `name`, `category`, `quantity`, `unit`, `purchase_date`, `expiry_date`, `daily_consumption`, `price_per_unit`, `supplier`, `storage_type`, `historical_wastage_rate`, `min_stock_level`, `max_stock_level`

**Assumption**: Daily consumption follows a log-normal-ish distribution within category bounds. I did not simulate seasonality at the individual ingredient level (e.g., mango demand in summer) since that would require geo-specific data I don't have. Documented as a future improvement.

---

## Feature Engineering

The core insight: **waste risk = mismatch between stock duration and time until expiry**.

| Feature | Formula | Why It Matters |
|---|---|---|
| `days_to_expiry` | `expiry_date - today` | Primary signal |
| `days_of_stock` | `quantity / daily_consumption` | How long stock will last |
| `waste_surplus_days` | `max(0, days_of_stock - days_to_expiry)` | Direct waste estimate |
| `estimated_waste_qty` | `waste_surplus_days * daily_consumption` | Quantity that will expire |
| `waste_value_at_risk` | `estimated_waste_qty * price_per_unit` | Business-facing metric |
| `turnover_rate` | `daily_consumption / quantity` | Speed of inventory movement |
| `stock_utilization` | `quantity / max_stock_level` | Overstock indicator |
| `is_overstocked` | `quantity > max_stock_level` | Boolean overstock flag |
| `category_avg_waste_rate` | Group mean of `historical_wastage_rate` | Category-level prior |
| `raw_waste_probability` | Sigmoid of `(days_of_stock / days_to_expiry - 1)` blended with historical rate | Pre-model signal |

**Encoding**: Categories and storage types encoded as ordinal integers. Price tier computed per-category using quantile binning (Low=0, Mid=1, High=2) to capture relative pricing within a food category.

---

## Model Selection & Reasoning

### Wastage Prediction (Classification)

**Why XGBoost + LightGBM ensemble?**
- Both handle tabular data extremely well with small-to-medium datasets
- XGBoost: better accuracy on complex non-linear patterns
- LightGBM: faster training, handles categorical features better
- Ensemble via probability averaging reduces variance without overfitting
- Both support SHAP natively — critical for the explainability requirement
- Alternative considered: Random Forest (more stable but slower inference, less SHAP integration)
- Alternative considered: Logistic Regression (interpretable but can't capture non-linear interactions between `days_to_expiry` and `daily_consumption`)

**Training setup:**
- 80/20 train/test split, stratified by waste label
- `scale_pos_weight` adjusted for class imbalance (~45% waste rate in simulated data)
- No data leakage: labels generated from rule-based signal + noise before feature engineering runs

**Results:**
```
ROC-AUC: 0.90   (excellent discrimination)
F1:      0.82
Precision: 0.88  (few false alarms)
Recall:  0.76    (acceptable, use Chef Specials to catch near-misses)
```

The high precision matters for restaurant operations — chefs trust alerts they don't see many false positives on.

### Demand Forecasting

**Why Holt-Winters Exponential Smoothing?**
- Restaurant demand has strong weekly seasonality (weekend spikes)
- Holt-Winters handles both trend + seasonal components natively
- Lightweight, interpretable, no GPU needed
- Works well on 90-day history per ingredient
- Graceful fallback: if <14 days history, uses exponential moving average

Alternative considered: Prophet — more sophisticated but 5x slower, overkill for per-ingredient restaurant forecasting. ARIMA — doesn't handle seasonality as cleanly without explicit parameter tuning.

### Anomaly Detection

**Why Isolation Forest + Rule-based hybrid?**
- Isolation Forest: unsupervised, no labels needed, scales well, handles multivariate anomalies
- Rule-based layer: domain knowledge catches what statistics miss (e.g., zero turnover on perishables, price spikes within category)
- 5% contamination rate: reasonable for restaurant inventory where ~5% of items are genuinely anomalous

---

## AI Integration

### Claude API (Anthropic)

The recommendation engine uses Claude to generate:
1. **Chef Specials**: Creative dishes that use expiring ingredients as primary components
2. **Action Plans**: Prioritized kitchen team tasks sorted by urgency
3. **Bulk Strategies**: Batch cooking, freezing, and cross-utilization suggestions
4. **Inventory Report**: Natural language executive summary for kitchen managers

**Prompt engineering approach:**
- Structured JSON output spec in prompt → reliable parsing
- Ingredients formatted with urgency, quantity, and waste value → cost-aware recommendations
- System prompt establishes chef persona with waste reduction expertise
- Temperature: default (balanced creativity + reliability)

**Graceful fallback**: If `ANTHROPIC_API_KEY` is not set, the system switches to **template-based recommendations** that still use real inventory data (expiring ingredients, quantities, waste values). The system never crashes without the API key.

---

## Real-World Data Handling

The preprocessing pipeline handles every data quality issue without crashing:

| Issue | Handling |
|---|---|
| Missing `quantity` | Filled with category median |
| Missing `daily_consumption` | Filled with category median, zero → 0.01 |
| Invalid/unparseable dates | Rows dropped with warning logged |
| Negative quantities | Converted to absolute value (data entry error) |
| Duplicate `ingredient_id` | First occurrence retained |
| Inconsistent category casing | Normalized to Title Case, mapped to known categories |
| Zero consumption on perishables | Flagged as anomaly by rule-based detector |
| Null `price_per_unit` | Category median imputation |

Every cleaning step is logged with counts. The `cleaning_report` dict provides a full audit trail.

---

## Explainability

Three levels of explainability:

1. **SHAP values** (model-level): TreeExplainer on XGBoost gives per-feature SHAP contributions for each prediction. Feature importance chart shows `waste_surplus_days` and `days_to_expiry` dominate.

2. **Natural language per item** (instance-level): The `ExplainabilityEngine` converts SHAP values into plain English: "At current consumption of 0.8 kg/day, this item will have 4.2 days of unconsumed stock when it expires in 3 days. ~3.4 kg worth ₹680 will be wasted."

3. **Claude-powered explanations** (optional): With API key set, the explanation engine can query Claude for even richer, context-aware explanations.

---

## Scalability Considerations

- **Batch processing**: The pipeline processes 1,200+ records without any per-record loops in the model inference step — uses vectorized pandas/numpy operations throughout.
- **Incremental updates**: The preprocessor and predictor are stateless — new inventory rows can be processed independently and appended to the processed CSV.
- **Model serving**: XGBoost/LightGBM models are serialized with joblib. Loading is ~100ms. FastAPI endpoint wraps model in a singleton — no reload per request.
- **Caching**: Streamlit uses `@st.cache_data(ttl=300)` — data reloads every 5 minutes, not per user interaction.
- **Demand forecasting at scale**: Currently forecasts 80 ingredients in ~500ms. For 1000+ ingredients: parallelize with `concurrent.futures.ThreadPoolExecutor` (each ingredient is independent).
- **Future scale path**: Replace flat CSV with a time-series database (TimescaleDB or InfluxDB). Use Redis for recommendation caching.

---

## Tradeoffs Made

| Decision | Chosen | Tradeoff |
|---|---|---|
| Ensemble vs. single model | XGB + LGB ensemble | +2-3% AUC, +300ms training |
| SHAP vs simpler explainability | SHAP TreeExplainer | Richer explanations, depends on model structure |
| Streamlit vs Flask UI | Streamlit | 10x faster to build, less control over UX |
| Holt-Winters vs Prophet | Holt-Winters | Faster, less accurate for long-horizon forecasts |
| Synthetic data vs real data | Synthetic with domain constraints | Controllable, reproducible, no licensing issues |
| Claude API with fallback vs hard dependency | Fallback to templates | System always works, degrades gracefully |

---

## Future Improvements

1. **Real demand data integration**: Connect to POS system via API for actual daily sales by ingredient
2. **Supplier reliability modeling**: Track delivery delays, quality variations per supplier
3. **Multi-location inventory**: Suggest inter-kitchen transfers before ordering
4. **Computer vision**: Identify spoilage from shelf camera images
5. **RAG-powered recommendations**: Build a vector database of 10,000+ recipes; find optimal dishes given any set of expiring ingredients using semantic search
6. **Automated retraining**: Retrain weekly as new waste outcomes come in — closes the feedback loop
7. **Mobile app**: Kitchen-facing mobile interface for real-time alerts
8. **Pricing integration**: Suggest dynamic discounting for at-risk items (e.g., 20% off salads to clear expiring lettuce)

---

## API Endpoints

```
GET  /health                          System status
GET  /inventory/at-risk               Top N at-risk items
GET  /inventory/summary               Aggregated waste intelligence
POST /predict                         Predict waste risk for a single item
GET  /recommendations/chef-specials   Generate AI chef specials
GET  /anomalies                       List inventory anomalies
```

Full interactive docs at `http://localhost:8000/docs` (Swagger UI).

---

## Test Coverage

18 integration tests covering:
- Data generation (count, schema, noise injection)
- Preprocessing (cleaning, deduplication, invalid dates, empty DataFrames)
- Feature engineering (feature presence, valid probability ranges)
- Model training and prediction (AUC > 0.5, valid label set)
- Anomaly detection (detects anomalies, valid severity values)
- Recommendations (works without API, handles empty inventory)
