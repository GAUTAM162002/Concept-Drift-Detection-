# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview



This is a concept drift detection project for electricity market data. It implements online learning and drift detection algorithms to monitor changes in electricity market patterns over time.

## Development Environment

This project uses a Python virtual environment:

```bash
# Activate the virtual environment
source venv/Scripts/activate  # Windows
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Key Dependencies

- **river** (0.23.0): Online machine learning library for drift detection
- **scikit-learn** (1.8.0): Traditional ML algorithms
- **pandas** (2.3.3): Data manipulation
- **numpy** (2.4.4): Numerical computing
- **jupyter** (1.1.1): Notebook environment
- **matplotlib** (3.10.8): Visualization
- **streamlit** (1.40.0): Web-based UI
- **plotly** (5.24.0): Interactive visualizations

## Project Structure

```
.
├── app.py                          # Streamlit web UI
├── run_app.bat                     # Windows launcher
├── run_app.sh                      # Linux/Mac launcher
├── UI_README.md                    # UI documentation
├── data/                           # Datasets
│   ├── electricity_market_dataset.csv   # Main dataset (61K+ rows, 27 features)
│   └── dataset.csv
├── notebooks/                      # Jupyter notebooks
│   ├── experiment.ipynb            # Initial exploration
│   └── evaluation.ipynb            # Full pipeline evaluation
├── outputs/                        # Generated results
│   ├── graphs/                     # Visualizations
│   └── results_*.json              # Pipeline results
├── src/                           # Source code
│   ├── data_stream.py              # DataStream class for streaming data
│   ├── drift_detection.py          # Drift detection algorithms (ADWIN, Page-Hinkley)
│   ├── model.py                    # Online learning models (Linear, Hoeffding Trees)
│   ├── retrain.py                  # Retraining strategies
│   └── main.py                     # Pipeline CLI
├── requirements.txt               # Python dependencies
└── venv/                          # Virtual environment
```

## Architecture

### Data Streaming

The `DataStream` class (`src/data_stream.py`) provides an iterator interface over CSV files:

```python
from data_stream import DataStream

stream = DataStream("data/electricity_market_dataset.csv")
while stream.has_more_data():
    row = stream.get_next_instance()
```

### Dataset Schema

The `electricity_market_dataset.csv` contains hourly electricity market data from 2018 with 27 columns:

- **Timestamp**: DateTime index
- **Pricing**: Historical_Electricity_Prices, Projected_Electricity_Prices, Electricity_Export_Prices, Electricity_Price_Forecast
- **Economic**: Inflation_Rates, GDP_Growth_Rate, Market_Elasticity
- **Energy Production**: Energy_Production_By_Solar, Energy_Production_By_Wind, Energy_Production_By_Coal
- **Costs**: Renewable_Investment_Costs, Fossil_Fuel_Costs, LCOE (Levelized Cost of Energy)
- **Market**: Energy_Market_Demand, Energy_Storage_Capacity, Renewable_Penetration_Rate
- **Environmental**: GHG_Emissions
- **Investment Metrics**: ROI, Net_Present_Value, Investment_Feasibility
- **Categorical**: Regulatory_Policies (Low/Medium/High), Energy_Access_Data (Urban/Rural), Project_Risk_Analysis (Low/High)

### Drift Detection (`src/drift_detection.py`)

```python
from drift_detection import DriftDetector

# Initialize ADWIN detector
detector = DriftDetector(detector_type='adwin', delta=0.002)

# Update with predictions
for y_true, y_pred in zip(actuals, predictions):
    if detector.update(y_true, y_pred):
        print(f"Drift detected at sample {detector.sample_count}")
```

Supported detectors:
- `adwin`: Adaptive Windowing algorithm
- `page_hinkley`: Page-Hinkley test

### Online Models (`src/model.py`)

```python
from model import OnlineModel

# Initialize model
model = OnlineModel(model_type='adaptive_tree', target='Investment_Feasibility')

# Online learning
for row in data_stream:
    prediction = model.predict_one(row)
    model.learn_one(row)  # Update model
```

Supported model types:
- `linear`: Linear regression with standard scaler
- `tree`: Hoeffding Tree Regressor
- `adaptive_tree`: Hoeffding Adaptive Tree Regressor (handles drift internally)

Features are automatically encoded (categoricals one-hot encoded).

### Retraining Strategies (`src/retrain.py`)

```python
from retrain import RetrainingStrategy

# Initialize with window strategy
retrainer = RetrainingStrategy(
    model,
    strategy='window',      # Options: 'window', 'reset', 'incremental'
    window_size=2000,
    warmup_size=1000
)

# On drift detection
retrainer.on_drift_detected(sample_index)
```

## Common Commands

### Launch Streamlit UI (Recommended)

```bash
# Windows
run_app.bat

# Linux/Mac
./run_app.sh

# Or manually
streamlit run app.py
```

The UI opens at `http://localhost:8501` with features:
- Upload custom datasets or use default
- Configure model and drift detector settings
- Real-time visualization during pipeline execution
- Compare multiple configurations
- Export results as JSON/CSV

### Run Pipeline (CLI)

```bash
# From project root with venv activated

# Basic run with default settings (adaptive_tree + adwin)
python src/main.py

# Specify model and detector
python src/main.py --model adaptive_tree --detector adwin

# Control sample size and warmup
python src/main.py --model linear --max-samples 10000 --warmup 1000

# Run model comparison across configurations
python src/main.py --compare --max-samples 5000
```

### CLI Options

```
--model {linear,tree,adaptive_tree,ensemble}   Model type (default: adaptive_tree)
--detector {adwin,page_hinkley}       Drift detector (default: adwin)
--warmup N                            Warmup samples before prediction (default: 1000)
--max-samples N                       Maximum samples to process (default: all)
--compare                             Run comparison of multiple configurations
--output-dir PATH                     Output directory (default: outputs)
```

### Start Jupyter Notebook

```bash
jupyter notebook notebooks/evaluation.ipynb
# or
jupyter lab
```

## Working with Components

### Import Pattern for Notebooks

```python
import sys
sys.path.append("../src")

from data_stream import DataStream
from drift_detection import DriftDetector, MultiFeatureDriftDetector
from model import OnlineModel, EnsembleModel
from retrain import RetrainingStrategy, AdaptiveEnsemble
```

### Example: Full Pipeline in Code

```python
from data_stream import DataStream
from drift_detection import DriftDetector
from model import OnlineModel
from retrain import RetrainingStrategy

stream = DataStream("data/electricity_market_dataset.csv")
model = OnlineModel(model_type='adaptive_tree')
detector = DriftDetector(detector_type='adwin')
retrainer = RetrainingStrategy(model, strategy='window', warmup_size=1000)

predictions = []
actuals = []

for i, row in enumerate(stream):
    if i < 1000:
        model.learn_one(row)
        continue
    
    pred = model.predict_one(row)
    actual = model._get_target(row)
    
    predictions.append(pred)
    actuals.append(actual)
    
    if detector.update(actual, pred):
        print(f"Drift at sample {i}")
        retrainer.on_drift_detected(i)
    
    model.learn_one(row)
```

## Notes

- The River library (`river`) is the primary tool for online learning and drift detection
- Data is time-series with hourly granularity
- The adaptive tree model internally handles drift, so fewer explicit drift detections are expected
- Results are saved as JSON files in `outputs/` with timestamps
- The evaluation notebook provides comprehensive visualization and testing utilities
