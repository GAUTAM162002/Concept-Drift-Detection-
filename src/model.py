from river import compose, preprocessing, linear_model, metrics
from river.tree import HoeffdingTreeRegressor, HoeffdingAdaptiveTreeRegressor
import numpy as np
import pandas as pd


class OnlineModel:
    """Online learning model wrapper for electricity market prediction."""

    def __init__(self, model_type='linear', target='Investment_Feasibility', feature_columns=None, categorical_columns=None):
        """
        Initialize online learning model.

        Args:
            model_type: 'linear', 'tree', 'adaptive_tree', or 'ensemble'
            target: Target column to predict
            feature_columns: List of feature column names (auto-detected if None)
            categorical_columns: List of categorical column names (auto-detected if None)
        """
        self.model_type = model_type
        self.target = target
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns or []
        self.model = None
        self.metrics = {
            'mae': metrics.MAE(),
            'rmse': metrics.RMSE(),
            'r2': metrics.R2()
        }
        self._build_model()

    def set_feature_columns(self, feature_columns, categorical_columns=None):
        """Set feature columns after initialization (for custom datasets)."""
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns or []

    def _build_model(self):
        """Build the model pipeline."""
        if self.model_type == 'linear':
            base_model = linear_model.LinearRegression()
        elif self.model_type == 'tree':
            base_model = HoeffdingTreeRegressor(
                grace_period=50,
                max_depth=10,
                delta=1e-5
            )
        elif self.model_type == 'adaptive_tree':
            base_model = HoeffdingAdaptiveTreeRegressor(
                grace_period=50,
                max_depth=10,
                delta=1e-5
            )
        elif self.model_type == 'ensemble':
            # Use HoeffdingTree as base for ensemble
            base_model = HoeffdingAdaptiveTreeRegressor(
                grace_period=50,
                max_depth=10,
                delta=1e-5
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Build preprocessing pipeline
        self.model = compose.Pipeline(
            ('scale', preprocessing.StandardScaler()),
            ('model', base_model)
        )

    def _prepare_features(self, row):
        """
        Extract and prepare features from a data row.

        Args:
            row: pandas Series or dict

        Returns:
            dict: Feature dictionary for the model
        """
        features = {}

        # If feature_columns not set, auto-detect from row
        if self.feature_columns is None:
            self.feature_columns = [col for col in row.index if col != self.target]
            # Detect categorical columns (object dtype or few unique values)
            self.categorical_columns = []
            for col in self.feature_columns:
                val = row.get(col)
                if val is not None and not isinstance(val, (int, float, np.number)):
                    self.categorical_columns.append(col)

        # Process all feature columns
        for col in self.feature_columns:
            if col == self.target:
                continue
            if col not in row:
                continue

            val = row[col]
            if pd.isna(val):
                continue

            # Handle categorical columns
            if col in self.categorical_columns:
                # One-hot encode categorical values
                if isinstance(val, str):
                    safe_val = val.replace(' ', '_').replace('-', '_')
                    features[f"{col}_{safe_val}"] = 1.0
            else:
                # Numeric columns
                try:
                    features[col] = float(val)
                except (ValueError, TypeError):
                    # Skip non-numeric values in numeric columns
                    pass

        # Fallback: if no features detected, use all non-target columns as numeric
        if not features:
            for col in row.index:
                if col != self.target:
                    try:
                        features[col] = float(row[col])
                    except (ValueError, TypeError):
                        pass

        return features

    def _get_target(self, row):
        """Extract target value from row."""
        return float(row[self.target])

    def learn_one(self, row):
        """
        Learn from a single instance.

        Args:
            row: pandas Series or dict with features and target

        Returns:
            self
        """
        x = self._prepare_features(row)
        y = self._get_target(row)
        self.model.learn_one(x, y)
        return self

    def predict_one(self, row):
        """
        Predict for a single instance.

        Args:
            row: pandas Series or dict with features

        Returns:
            float: Predicted value
        """
        x = self._prepare_features(row)
        return self.model.predict_one(x)

    def learn_many(self, rows):
        """
        Learn from multiple instances.

        Args:
            rows: List of dicts or DataFrame
        """
        if isinstance(rows, pd.DataFrame):
            rows = rows.to_dict('records')

        for row in rows:
            self.learn_one(row)

    def update_metrics(self, y_true, y_pred):
        """
        Update evaluation metrics.

        Args:
            y_true: True value
            y_pred: Predicted value
        """
        for metric in self.metrics.values():
            metric.update(y_true, y_pred)

    def get_metrics(self):
        """Get current metric values."""
        return {
            name: metric.get()
            for name, metric in self.metrics.items()
        }

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            'mae': metrics.MAE(),
            'rmse': metrics.RMSE(),
            'r2': metrics.R2()
        }


class EnsembleModel:
    """Ensemble of online models with drift detection."""

    def __init__(self, models=None):
        """
        Initialize ensemble.

        Args:
            models: List of OnlineModel instances
        """
        if models is None:
            models = [
                OnlineModel('linear'),
                OnlineModel('adaptive_tree'),
            ]
        self.models = models
        self.weights = [1.0 / len(models)] * len(models)
        self.errors = [[] for _ in models]

    def predict_one(self, row):
        """
        Predict using weighted ensemble.

        Args:
            row: Feature row

        Returns:
            float: Weighted prediction
        """
        predictions = [m.predict_one(row) for m in self.models]

        # Filter out None predictions
        valid = [(p, w) for p, w in zip(predictions, self.weights) if p is not None]

        if not valid:
            return None

        total_weight = sum(w for _, w in valid)
        weighted_sum = sum(p * w for p, w in valid)

        return weighted_sum / total_weight if total_weight > 0 else None

    def learn_one(self, row):
        """Train all models on a row."""
        for model in self.models:
            model.learn_one(row)
        return self

    def update_weights(self, row, alpha=0.9):
        """
        Update ensemble weights based on recent performance.

        Args:
            row: Row with features and target
            alpha: Smoothing factor
        """
        y_true = self.models[0]._get_target(row)

        for i, model in enumerate(self.models):
            y_pred = model.predict_one(row)
            if y_pred is not None:
                error = abs(y_true - y_pred)
                self.errors[i].append(error)

                # Keep only recent errors
                if len(self.errors[i]) > 100:
                    self.errors[i].pop(0)

        # Recalculate weights based on inverse error
        if all(len(e) > 10 for e in self.errors):
            mean_errors = [np.mean(e) + 1e-10 for e in self.errors]
            inv_errors = [1.0 / e for e in mean_errors]
            total = sum(inv_errors)
            self.weights = [e / total for e in inv_errors]
