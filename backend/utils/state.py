import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import deque

class AppState:
    def __init__(self):
        self.is_streaming: bool = False
        self.model_type: str = "linear"
        self.drift_handling_enabled: bool = True
        self.detector_type: str = "adwin"
        self.stream_speed: float = 0.01
        self.current_index: int = 0
        
        self.current_metrics: Dict[str, float] = {
            "MAE": 0.0,
            "RMSE": 0.0,
            "R2": 0.0
        }
        
        self.last_result: Dict[str, Any] = {
            "timestamp": None,
            "y_true": None,
            "y_pred": None,
            "error": None,
            "drift_detected": False,
            "model_type": "linear",
            "drift_handling_enabled": True,
            "MAE": 0.0,
            "RMSE": 0.0,
            "R2": 0.0,
            "top_features": []
        }
        
        # Buffer for last 2000 records
        self.history = deque(maxlen=2000)
        
        self.stream_task: Optional[asyncio.Task] = None
        self.data_path: str = "data/electricity_market_dataset.csv"
        self.dataset_type: str = "electricity"
        self.target_column: Optional[str] = None # Will be auto-detected
        
        # Lock for thread-safe (or rather task-safe) updates
        self.lock = asyncio.Lock()

    async def update_state(self, y_true, y_pred, error, drift_detected, metrics, top_features=None, drift_type=None):
        async with self.lock:
            result = {
                "timestamp": datetime.now().isoformat(),
                "y_true": y_true,
                "y_pred": y_pred,
                "error": error,
                "drift_detected": drift_detected,
                "drift_type": drift_type,
                "model_type": self.model_type,
                "drift_handling_enabled": self.drift_handling_enabled,
                "top_features": top_features or [],
                **metrics
            }
            self.last_result = result
            self.history.append(result)
            self.current_metrics = metrics

    async def reset(self, hard=True):
        async with self.lock:
            if hard:
                self.history.clear()
                self.current_index = 0
            self.current_metrics = {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0}
            self.last_result = {
                "timestamp": None,
                "y_true": None,
                "y_pred": None,
                "error": None,
                "drift_detected": False,
                "model_type": self.model_type,
                "drift_handling_enabled": self.drift_handling_enabled,
                "MAE": 0.0,
                "RMSE": 0.0,
                "R2": 0.0
            }

state = AppState()
