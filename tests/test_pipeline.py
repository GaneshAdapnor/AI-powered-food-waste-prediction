"""
Integration tests for the Ecomeal AI pipeline.
Tests data generation, preprocessing, ML models, and recommendations.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataGenerator:
    def test_generates_correct_count(self):
        from src.data.generator import RestaurantDataGenerator
        gen = RestaurantDataGenerator(n_records=100, seed=42)
        df = gen.generate_inventory(add_noise=False)
        assert len(df) == 100

    def test_all_required_columns_present(self):
        from src.data.generator import RestaurantDataGenerator
        required = ["ingredient_id", "name", "category", "quantity", "unit",
                    "expiry_date", "daily_consumption", "price_per_unit", "supplier"]
        gen = RestaurantDataGenerator(n_records=50, seed=42)
        df = gen.generate_inventory(add_noise=False)
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_noise_injection_creates_issues(self):
        from src.data.generator import RestaurantDataGenerator
        gen = RestaurantDataGenerator(n_records=500, seed=42)
        df = gen.generate_inventory(add_noise=True)
        # Should have some missing values after noise injection
        assert df.isnull().sum().sum() > 0

    def test_generates_consumption_history(self):
        from src.data.generator import RestaurantDataGenerator
        gen = RestaurantDataGenerator(n_records=50, seed=42)
        inv_df = gen.generate_inventory(add_noise=False)
        hist_df = gen.generate_consumption_history(inv_df, days=30)
        assert len(hist_df) > 0
        assert "consumption" in hist_df.columns
        assert "date" in hist_df.columns


class TestDataPreprocessor:
    @pytest.fixture
    def raw_df(self):
        from src.data.generator import RestaurantDataGenerator
        gen = RestaurantDataGenerator(n_records=200, seed=42)
        return gen.generate_inventory(add_noise=True)

    def test_clean_never_crashes(self, raw_df):
        from src.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        result = preprocessor.clean(raw_df)
        assert result is not None
        assert len(result) > 0

    def test_removes_duplicates(self, raw_df):
        from src.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        result = preprocessor.clean(raw_df)
        assert len(result) <= len(raw_df)

    def test_no_negative_quantities(self, raw_df):
        from src.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        result = preprocessor.clean(raw_df)
        if "quantity" in result.columns:
            assert (result["quantity"] >= 0).all()

    def test_feature_engineering_produces_expected_features(self, raw_df):
        from src.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        clean = preprocessor.clean(raw_df)
        featured = preprocessor.engineer_features(clean)
        expected_features = ["days_to_expiry", "days_of_stock", "waste_surplus_days",
                              "turnover_rate", "raw_waste_probability", "is_near_expiry"]
        for feat in expected_features:
            assert feat in featured.columns, f"Missing feature: {feat}"

    def test_waste_probability_in_valid_range(self, raw_df):
        from src.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        clean = preprocessor.clean(raw_df)
        featured = preprocessor.engineer_features(clean)
        if "raw_waste_probability" in featured.columns:
            assert (featured["raw_waste_probability"] >= 0).all()
            assert (featured["raw_waste_probability"] <= 1).all()

    def test_handles_completely_empty_dataframe(self):
        from src.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        empty_df = pd.DataFrame()
        result = preprocessor.clean(empty_df)
        assert result is not None  # Should not crash

    def test_handles_all_invalid_dates(self):
        from src.data.preprocessor import DataPreprocessor
        from src.data.generator import RestaurantDataGenerator
        gen = RestaurantDataGenerator(n_records=50, seed=42)
        df = gen.generate_inventory(add_noise=False)
        df["expiry_date"] = "INVALID"
        preprocessor = DataPreprocessor()
        result = preprocessor.clean(df)
        # Should handle gracefully, possibly returning empty or partially processed
        assert result is not None


class TestWastagePredictor:
    @pytest.fixture
    def labeled_df(self):
        from src.data.generator import RestaurantDataGenerator
        from src.data.preprocessor import DataPreprocessor
        gen = RestaurantDataGenerator(n_records=300, seed=42)
        raw = gen.generate_inventory(add_noise=True)
        prep = DataPreprocessor()
        clean = prep.clean(raw)
        featured = prep.engineer_features(clean)
        return prep.build_training_labels(featured)

    def test_trains_successfully(self, labeled_df):
        from src.models.wastage_predictor import WastagePredictor
        predictor = WastagePredictor()
        metrics = predictor.train(labeled_df)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] > 0.5  # Better than random

    def test_predict_returns_probabilities(self, labeled_df):
        from src.models.wastage_predictor import WastagePredictor
        predictor = WastagePredictor()
        predictor.train(labeled_df)
        result = predictor.predict(labeled_df)
        assert "waste_probability" in result.columns
        assert (result["waste_probability"] >= 0).all()
        assert (result["waste_probability"] <= 1).all()

    def test_risk_labels_are_valid(self, labeled_df):
        from src.models.wastage_predictor import WastagePredictor
        predictor = WastagePredictor()
        predictor.train(labeled_df)
        result = predictor.predict(labeled_df)
        valid_labels = {"Critical", "High", "Medium", "Low"}
        assert set(result["risk_label"].unique()).issubset(valid_labels)


class TestAnomalyDetector:
    def test_detects_anomalies(self):
        from src.data.generator import RestaurantDataGenerator
        from src.data.preprocessor import DataPreprocessor
        from src.models.anomaly_detector import AnomalyDetector

        gen = RestaurantDataGenerator(n_records=200, seed=42)
        raw = gen.generate_inventory(add_noise=False)
        prep = DataPreprocessor()
        clean = prep.clean(raw)
        featured = prep.engineer_features(clean)

        detector = AnomalyDetector(contamination=0.05)
        result = detector.fit_predict(featured)
        assert "is_anomaly" in result.columns
        # Should detect some anomalies
        assert result["is_anomaly"].sum() > 0

    def test_anomaly_summary_structure(self):
        from src.data.generator import RestaurantDataGenerator
        from src.data.preprocessor import DataPreprocessor
        from src.models.anomaly_detector import AnomalyDetector

        gen = RestaurantDataGenerator(n_records=200, seed=42)
        raw = gen.generate_inventory(add_noise=False)
        prep = DataPreprocessor()
        clean = prep.clean(raw)
        featured = prep.engineer_features(clean)

        detector = AnomalyDetector(contamination=0.05)
        detector.fit_predict(featured)
        summary = detector.get_anomaly_summary()

        assert isinstance(summary, list)
        for item in summary:
            assert "severity" in item
            assert item["severity"] in ["high", "medium", "low"]


class TestRecommendationEngine:
    def test_generates_recommendations_without_api(self):
        from src.data.generator import RestaurantDataGenerator
        from src.data.preprocessor import DataPreprocessor
        from src.ai.recommendation_engine import RecommendationEngine

        gen = RestaurantDataGenerator(n_records=100, seed=42)
        raw = gen.generate_inventory(add_noise=False)
        prep = DataPreprocessor()
        clean = prep.clean(raw)
        featured = prep.engineer_features(clean)
        featured = prep.build_training_labels(featured)

        # Force template mode by setting _api_available to False
        engine = RecommendationEngine()
        engine._api_available = False
        result = engine.generate_chef_specials(featured, n_specials=2)
        assert "dishes" in result
        assert isinstance(result["dishes"], list)

    def test_handles_empty_inventory(self):
        from src.ai.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        result = engine.generate_chef_specials(pd.DataFrame())
        assert "message" in result or "dishes" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
