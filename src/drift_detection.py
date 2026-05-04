from river.drift import ADWIN, PageHinkley
import numpy as np


class DriftDetector:
    """Wrapper class for drift detection algorithms."""

    def __init__(self, detector_type='adwin', delta=0.002, threshold=50):
        """
        Initialize drift detector.

        Args:
            detector_type: 'adwin' or 'page_hinkley'
            delta: ADWIN sensitivity parameter
            threshold: Page-Hinkley detection threshold
        """
        self.detector_type = detector_type

        if detector_type == 'adwin':
            self.detector = ADWIN(delta=delta)
        elif detector_type == 'page_hinkley':
            self.detector = PageHinkley(threshold=threshold, alpha=0.9999)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

        self.drift_points = []
        self.sample_count = 0

    def update(self, y_true, y_pred):
        """
        Update detector with a new observation.

        Args:
            y_true: True label/value
            y_pred: Predicted label/value

        Returns:
            bool: True if drift was detected
        """
        # Use absolute error for regression
        error = abs(y_true - y_pred) if isinstance(y_true, (int, float, np.number)) else int(y_true != y_pred)

        self.detector.update(error)
        self.sample_count += 1

        if self.detector.drift_detected:
            self.drift_points.append(self.sample_count)
            return True
        return False

    def detect_drift(self, X=None, y=None, predictions=None):
        """
        Batch drift detection over a dataset.

        Args:
            X: Features
            y: True labels
            predictions: Model predictions

        Returns:
            list: Indices where drift was detected
        """
        if predictions is None or y is None:
            raise ValueError("predictions and y are required")

        drift_indices = []
        for i, (true, pred) in enumerate(zip(y, predictions)):
            if self.update(true, pred):
                drift_indices.append(i)

        return drift_indices

    def get_drift_points(self):
        """Get list of points where drift was detected."""
        return self.drift_points.copy()

    def reset(self):
        """Reset the detector to initial state."""
        if self.detector_type == 'adwin':
            self.detector = ADWIN(delta=self.detector.delta)
        elif self.detector_type == 'page_hinkley':
            self.detector = PageHinkley(threshold=self.detector.threshold)
        self.drift_points = []
        self.sample_count = 0


class MultiFeatureDriftDetector:
    """Monitor drift across multiple features simultaneously."""

    def __init__(self, feature_names, detector_type='adwin'):
        """
        Initialize multi-feature drift detection.

        Args:
            feature_names: List of feature names to monitor
            detector_type: 'adwin' or 'page_hinkley'
        """
        self.feature_names = feature_names
        self.detectors = {
            name: DriftDetector(detector_type=detector_type)
            for name in feature_names
        }
        self.drift_history = {name: [] for name in feature_names}

    def update_feature(self, feature_name, value):
        """
        Update a specific feature's drift detector.

        Args:
            feature_name: Name of the feature
            value: New feature value

        Returns:
            bool: True if drift detected in this feature
        """
        if feature_name not in self.detectors:
            return False

        # For feature drift, we track the feature value itself
        # using a simple mean-shift approach
        self.detectors[feature_name].detector.update(float(value))

        if self.detectors[feature_name].detector.drift_detected:
            self.drift_history[feature_name].append(
                self.detectors[feature_name].sample_count
            )
            return True
        return False

    def update(self, features_dict):
        """
        Update all feature detectors.

        Args:
            features_dict: Dictionary of feature names to values

        Returns:
            dict: Features where drift was detected
        """
        drifted = {}
        for name, value in features_dict.items():
            if name in self.detectors:
                if self.update_feature(name, value):
                    drifted[name] = self.detectors[name].sample_count
        return drifted

    def get_drift_summary(self):
        """Get summary of drift detections across all features."""
        return {
            name: {
                'count': len(points),
                'points': points
            }
            for name, points in self.drift_history.items()
        }
