import asyncio
import shutil
from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from backend.utils.state import state
from backend.services.drift_service import stream_data_task
import os
import pandas as pd

app = FastAPI(title="Concept Drift Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelUpdate(BaseModel):
    model_type: str

class DriftToggle(BaseModel):
    enabled: bool

class ConfigUpdate(BaseModel):
    stream_speed: float

class IndexUpdate(BaseModel):
    index: int

class DatasetUpdate(BaseModel):
    dataset_type: str # e.g., 'electricity', 'finance', 'weather'
    file_path: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Concept Drift Detection API is running"}

@app.post("/set-dataset")
async def set_dataset(update: DatasetUpdate):
    state.is_streaming = False # Stop current stream
    
    # Map dataset types to files if not explicitly provided
    dataset_map = {
        "electricity": "data/electricity_market_dataset.csv",
        "finance": "data/finance_dataset.csv",
        "weather": "data/weather_dataset.csv"
    }
    
    path = update.file_path or dataset_map.get(update.dataset_type)
    if not path or not os.path.exists(path):
        # Fallback to electricity if others don't exist for demo
        path = "data/electricity_market_dataset.csv"
    
    state.data_path = path
    state.dataset_type = update.dataset_type
    state.target_column = None # Force re-detection
    await state.reset(hard=True)
    
    return {"message": f"Dataset set to {update.dataset_type}", "path": path}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    state.is_streaming = False # Stop current stream
    
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    state.data_path = file_path
    state.dataset_type = "custom"
    state.target_column = None # Force re-detection of target column
    await state.reset(hard=True)
    
    return {"message": f"Dataset {file.filename} uploaded successfully", "path": file_path}

@app.get("/get-info")
async def get_info():
    try:
        df = pd.read_csv(state.data_path)
        return {
            "total_rows": len(df),
            "columns": list(df.columns),
            "current_index": state.current_index
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/set-index")
async def set_index(update: IndexUpdate):
    state.is_streaming = False # Pause on jump
    state.current_index = update.index
    await state.reset(hard=False) # Clear metrics but maybe keep some history? 
    # Actually, if we jump, we might want to clear history to avoid confusion
    async with state.lock:
        state.history.clear()
    return {"message": f"Index set to {update.index}", "current_index": state.current_index}

@app.get("/start-stream")
async def start_stream(background_tasks: BackgroundTasks):
    if state.is_streaming:
        return {"message": "Stream is already running"}
    
    state.is_streaming = True
    # If starting from scratch, reset metrics but maybe keep index if resuming
    # await state.reset(hard=False) 
    
    if state.stream_task is None or state.stream_task.done():
        state.stream_task = asyncio.create_task(stream_data_task())
    
    return {"message": "Streaming started"}

@app.get("/stop-stream")
async def stop_stream():
    if not state.is_streaming:
        return {"message": "Stream is not running"}
    
    state.is_streaming = False
    return {"message": "Streaming stopping..."}

@app.get("/restart-stream")
async def restart_stream():
    # Stop current stream if running
    state.is_streaming = False
    if state.stream_task:
        # Wait a bit for the task to catch the flag
        await asyncio.sleep(0.2)
    
    # Reset state and index
    await state.reset(hard=True)
    
    # Start again
    state.is_streaming = True
    state.stream_task = asyncio.create_task(stream_data_task())
    
    return {"message": "Stream restarted from beginning"}

@app.get("/get-state")
async def get_state():
    async with state.lock:
        result = dict(state.last_result)
        result["is_streaming"] = state.is_streaming
        return result

@app.get("/get-history")
async def get_history(limit: int = 200):
    async with state.lock:
        history_list = list(state.history)
        return history_list[-limit:] if limit else history_list

@app.post("/set-model")
async def set_model(update: ModelUpdate):
    valid_models = ["linear", "tree", "adaptive_tree", "ensemble"]
    if update.model_type not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Must be one of {valid_models}")
    
    state.model_type = update.model_type
    return {"message": f"Model type set to {update.model_type}", "is_streaming": state.is_streaming}

@app.post("/toggle-drift")
async def toggle_drift(toggle: DriftToggle):
    state.drift_handling_enabled = toggle.enabled
    return {"message": f"Drift handling {'enabled' if toggle.enabled else 'disabled'}"}

@app.post("/update-config")
async def update_config(config: ConfigUpdate):
    if config.stream_speed <= 0:
        raise HTTPException(status_code=400, detail="Stream speed must be positive")
    
    state.stream_speed = config.stream_speed
    return {"message": f"Stream speed set to {config.stream_speed}s"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
