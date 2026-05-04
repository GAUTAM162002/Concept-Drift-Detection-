from river import compose, preprocessing, linear_model, metrics
from river.tree import HoeffdingTreeRegressor, HoeffdingAdaptiveTreeRegressor
from river.drift import ADWIN, PageHinkley
import numpy as np

class EnsembleModel:
    def __init__(self):
        # Initialize sub-models with full pipelines
        self.models = {
            'linear': ModelFactory.create_pipeline(linear_model.LinearRegression()),
            'tree': ModelFactory.create_pipeline(HoeffdingTreeRegressor(grace_period=50, max_depth=10)),
            'adaptive': ModelFactory.create_pipeline(HoeffdingAdaptiveTreeRegressor(grace_period=50, max_depth=10))
        }
        self.weights = {name: 1.0/3.0 for name in self.models}
        self.errors = {name: metrics.MAE() for name in self.models}

    def predict_one(self, x):
        preds = {name: m.predict_one(x) for name, m in self.models.items()}
        # Weighted average
        weighted_sum = sum(preds[name] * self.weights[name] for name in self.models if preds[name] is not None)
        total_weight = sum(self.weights[name] for name in self.models if preds[name] is not None)
        return weighted_sum / total_weight if total_weight > 0 else 0

    def learn_one(self, x, y):
        for name, m in self.models.items():
            # 1. Update weights based on current prediction error BEFORE learning
            y_pred = m.predict_one(x)
            if y_pred is not None:
                self.errors[name].update(y, y_pred)
                # Recalculate weights based on inverse error (smaller error -> higher weight)
                # We use 1/(MAE + epsilon)
                mae_vals = {n: e.get() + 1e-6 for n, e in self.errors.items()}
                inv_mae = {n: 1.0/v for n, v in mae_vals.items()}
                total_inv = sum(inv_mae.values())
                self.weights = {n: v/total_inv for n, v in inv_mae.items()}
            
            # 2. Learn
            m.learn_one(x, y)

class ModelFactory:
    @staticmethod
    def create_pipeline(base_model):
        return compose.Pipeline(
            ('preprocess', compose.TransformerUnion(
                ('numeric', compose.SelectType(int, float) | preprocessing.StandardScaler()),
                ('categorical', compose.SelectType(str) | preprocessing.OneHotEncoder())
            )),
            ('model', base_model)
        )

    @staticmethod
    def get_model(model_type: str):
        # Base model selection
        if model_type == 'linear':
            return ModelFactory.create_pipeline(linear_model.LinearRegression())
        elif model_type == 'tree':
            return ModelFactory.create_pipeline(HoeffdingTreeRegressor(grace_period=50, max_depth=10, delta=1e-5))
        elif model_type == 'adaptive_tree':
            return ModelFactory.create_pipeline(HoeffdingAdaptiveTreeRegressor(grace_period=50, max_depth=10, delta=1e-5))
        elif model_type == 'ensemble':
            return EnsembleModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_detector(detector_type: str, delta: float = 0.002, threshold: float = 50):
        if detector_type == 'adwin':
            return ADWIN(delta=delta)
        elif detector_type == 'page_hinkley':
            return PageHinkley(threshold=threshold, alpha=0.9999)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    @staticmethod
    def get_metrics():
        return {
            'MAE': metrics.MAE(),
            'RMSE': metrics.RMSE(),
            'R2': metrics.R2()
        }
