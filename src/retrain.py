import numpy as np
from collections import deque


class RetrainingStrategy:
    """Handles model retraining when drift is detected."""

    def __init__(self, model, strategy='window', window_size=1000, warmup_size=100):
        """
        Initialize retraining strategy.

        Args:
            model: OnlineModel instance
            strategy: 'window' (sliding), 'incremental', or 'reset'
            window_size: Size of training window after drift
            warmup_size: Minimum samples before using model
        """
        self.model = model
        self.strategy = strategy
        self.window_size = window_size
        self.warmup_size = warmup_size
        self.buffer = deque(maxlen=window_size)
        self.drift_count = 0
        self.samples_since_drift = 0
        self.is_warming_up = False
        self.retrain_history = []

    def on_drift_detected(self, current_sample_idx):
        """
        Handle drift detection event.

        Args:
            current_sample_idx: Index where drift was detected

        Returns:
            dict: Information about the retraining
        """
        self.drift_count += 1
        self.is_warming_up = True
        self.samples_since_drift = 0

        info = {
            'drift_point': current_sample_idx,
            'drift_count': self.drift_count,
            'strategy': self.strategy,
            'buffer_size': len(self.buffer)
        }

        if self.strategy == 'window':
            # Keep only recent samples in buffer
            self._apply_window_strategy()
        elif self.strategy == 'reset':
            # Clear buffer and rebuild model
            self._apply_reset_strategy()
        elif self.strategy == 'incremental':
            # Keep all data but increase learning rate (simulated)
            self._apply_incremental_strategy()

        self.retrain_history.append(info)
        return info

    def _apply_window_strategy(self):
        """Apply sliding window - model continues with buffer."""
        # The buffer already maintains window_size most recent samples
        # Model continues learning from here
        pass

    def _apply_reset_strategy(self):
        """Reset model and start fresh."""
        self.buffer.clear()
        self.model._build_model()
        self.model.reset_metrics()

    def _apply_incremental_strategy(self):
        """Apply incremental learning adjustment."""
        # For online models, this means continuing with current state
        # Could adjust internal parameters if supported
        pass

    def add_sample(self, row):
        """
        Add a sample to the buffer.

        Args:
            row: Data sample
        """
        self.buffer.append(row)

        if self.is_warming_up:
            self.samples_since_drift += 1
            if self.samples_since_drift >= self.warmup_size:
                self.is_warming_up = False

    def get_training_data(self):
        """Get current training data based on strategy."""
        if self.strategy == 'window':
            return list(self.buffer)
        elif self.strategy == 'reset':
            return list(self.buffer)
        else:
            return list(self.buffer)

    def should_predict(self):
        """Check if model should make predictions (past warmup)."""
        if self.strategy == 'reset':
            return self.samples_since_drift >= self.warmup_size
        return len(self.buffer) >= self.warmup_size

    def get_status(self):
        """Get current retraining status."""
        return {
            'drift_count': self.drift_count,
            'is_warming_up': self.is_warming_up,
            'samples_since_drift': self.samples_since_drift,
            'buffer_size': len(self.buffer),
            'strategy': self.strategy
        }


class AdaptiveEnsemble:
    """Ensemble of models with different retraining strategies."""

    def __init__(self, models, strategies=None):
        """
        Initialize adaptive ensemble.

        Args:
            models: List of OnlineModel instances
            strategies: List of retraining strategies (one per model)
        """
        if strategies is None:
            strategies = ['window', 'reset', 'incremental']

        self.models = models
        self.strategies = [
            RetrainingStrategy(model, strategy=strat)
            for model, strat in zip(models, strategies[:len(models)])
        ]
        self.performance_tracker = {i: [] for i in range(len(models))}
        self.best_model_idx = 0

    def on_drift(self, sample_idx):
        """Handle drift across all strategies."""
        results = []
        for i, strategy in enumerate(self.strategies):
            info = strategy.on_drift_detected(sample_idx)
            info['model_idx'] = i
            results.append(info)
        return results

    def update(self, row, predictions, true_value):
        """
        Update all strategies and track performance.

        Args:
            row: Data row
            predictions: List of predictions from each model
            true_value: Actual value
        """
        # Add sample to all buffers
        for strategy in self.strategies:
            strategy.add_sample(row)

        # Track errors
        for i, pred in enumerate(predictions):
            if pred is not None:
                error = abs(true_value - pred)
                self.performance_tracker[i].append(error)

                # Keep only recent performance
                if len(self.performance_tracker[i]) > 1000:
                    self.performance_tracker[i].pop(0)

        # Update best model
        if all(len(v) > 100 for v in self.performance_tracker.values()):
            mean_errors = {i: np.mean(v) for i, v in self.performance_tracker.items()}
            self.best_model_idx = min(mean_errors, key=mean_errors.get)

    def get_best_prediction(self, predictions):
        """Get prediction from best performing model."""
        if predictions[self.best_model_idx] is not None:
            return predictions[self.best_model_idx]

        # Fallback to first valid prediction
        for pred in predictions:
            if pred is not None:
                return pred
        return None

    def get_ensemble_prediction(self, predictions, method='mean'):
        """
        Get ensemble prediction.

        Args:
            predictions: List of predictions
            method: 'mean', 'median', or 'best'
        """
        valid = [p for p in predictions if p is not None]

        if not valid:
            return None

        if method == 'mean':
            return np.mean(valid)
        elif method == 'median':
            return np.median(valid)
        elif method == 'best':
            return self.get_best_prediction(predictions)

        return np.mean(valid)
