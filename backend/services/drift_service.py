import asyncio
import pandas as pd
import numpy as np
from backend.utils.state import state
from backend.models.model_factory import ModelFactory
import os

async def stream_data_task():
    """Background task to simulate real-time data streaming."""
    try:
        if not os.path.exists(state.data_path):
            print(f"Data file not found: {state.data_path}")
            state.is_streaming = False
            return

        # Load dataset
        df = pd.read_csv(state.data_path)
        
        # Auto-detect target column if not set (default to last column)
        if not state.target_column:
            state.target_column = df.columns[-1]
            print(f"Auto-detected target column: {state.target_column}")

        # Initialize model and metrics
        model = ModelFactory.get_model(state.model_type)
        detector = ModelFactory.get_detector(state.detector_type)
        eval_metrics = ModelFactory.get_metrics()
        
        # Explainability: Linear surrogate
        surrogate = ModelFactory.get_model('linear')
        
        current_model_type = state.model_type

        # Start from current_index
        while state.is_streaming:
            if state.current_index >= len(df):
                print("End of dataset reached.")
                state.current_index = 0
                state.is_streaming = False
                break

            row = df.iloc[state.current_index]
            
            # Check if model type was changed via API
            if state.model_type != current_model_type:
                model = ModelFactory.get_model(state.model_type)
                current_model_type = state.model_type
                print(f"Model switched to {current_model_type}")

            # Prepare data (handle both numeric and categorical)
            # We don't drop columns here, we just separate target
            x = row.drop(state.target_column).to_dict()
            
            # Ensure target is float
            try:
                y_true = float(row[state.target_column])
            except (ValueError, TypeError):
                # Skip rows where target is not a number (for regression)
                state.current_index += 1
                continue

            # 1. Predict
            y_pred = model.predict_one(x)
            
            # 2. Calculate error and update metrics
            error = None
            drift_detected = False
            
            if y_pred is not None:
                error = abs(y_true - y_pred)
                
                # Update metrics
                for m in eval_metrics.values():
                    m.update(y_true, y_pred)
                
                # 3. Drift Detection (Always detect)
            detector.update(error)
            drift_type = None
            if detector.drift_detected:
                drift_detected = True
                
                # --- Classification Logic ---
                recent_errors = [d['error'] for d in list(state.history)[-50:] if d['error'] is not None]
                if len(recent_errors) > 20:
                    short_window = np.mean(recent_errors[-5:])
                    long_window = np.mean(recent_errors)
                    
                    # 1. Abrupt: Sudden spike compared to short history
                    if short_window > long_window * 2.5:
                        drift_type = "abrupt"
                    # 2. Gradual: Sustained increase in error
                    elif short_window > long_window * 1.2:
                        drift_type = "gradual"
                    # 3. Recurring: Check if top features match past drift (Heuristic)
                    else:
                        drift_type = "recurring"
                else:
                    drift_type = "abrupt" # Default for first detections
                
                print(f"{drift_type.capitalize()} drift detected at index {state.current_index}!")
                
                # 4. Retraining Strategy
                if state.drift_handling_enabled:
                    print("Applying retraining strategy: Resetting model...")
                    model = ModelFactory.get_model(state.model_type)

            # 5. Learn
            model.learn_one(x, y_true)
            
            # Update surrogate if needed (if not already the main model)
            if state.model_type != 'linear':
                surrogate.learn_one(x, y_true)

            # 6. Extract Feature Importance (Top 5)
            # Use weights from the main model if linear, else from surrogate, or linear sub-model if ensemble
            if state.model_type == 'linear':
                target_estimator = model['model']
            elif state.model_type == 'ensemble':
                target_estimator = model.models['linear']['model']
            else:
                target_estimator = surrogate['model']
                
            weights = target_estimator.weights
            
            top_features = []
            if weights:
                # Get absolute weights and sort
                sorted_weights = sorted(
                    weights.items(), 
                    key=lambda item: abs(item[1]), 
                    reverse=True
                )[:5]
                
                # Normalize importance (0-1 range for UI)
                max_w = max(abs(w) for _, w in sorted_weights) if sorted_weights else 1
                top_features = [
                    {"name": name, "importance": round(abs(w) / max_w, 4)} 
                    for name, w in sorted_weights
                ]

            # 7. Update Global State
            current_metrics_vals = {name: m.get() for name, m in eval_metrics.items()}
            await state.update_state(
                y_true=y_true,
                y_pred=y_pred,
                error=error,
                drift_detected=drift_detected,
                metrics=current_metrics_vals,
                top_features=top_features,
                drift_type=drift_type
            )

            state.current_index += 1
            
            # Simulate delay using STREAM_SPEED
            await asyncio.sleep(state.stream_speed) 

        state.is_streaming = False
        print("Streaming stopped.")

    except Exception as e:
        print(f"Error in stream_data_task: {e}")
        import traceback
        traceback.print_exc()
        state.is_streaming = False
