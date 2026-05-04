# Concept Drift Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io/)
[![River](https://img.shields.io/badge/River-0.23+-green.svg)](https://riverml.xyz/)

A professional web-based platform for detecting concept drift in streaming data using advanced online learning algorithms.

![Platform Screenshot](docs/screenshot.png)

## 🌟 Features

### Core Capabilities
- **🔬 Online Learning**: Train models incrementally on streaming data without storing entire datasets
- **🚨 Drift Detection**: Automatically detect concept drift using ADWIN and Page-Hinkley algorithms
- **📊 Real-time Visualization**: Interactive charts showing predictions, errors, and drift points
- **🔧 Model Comparison**: Compare multiple configurations side-by-side
- **📁 Custom Datasets**: Upload your own CSV files or use the default electricity market dataset

### Supported Algorithms
| Model | Description | Best For |
|-------|-------------|----------|
| **Adaptive Tree** | Hoeffding Adaptive Tree with internal drift handling | Non-stationary data |
| **Linear Regression** | Fast baseline with standard scaling | Quick experiments |
| **Hoeffding Tree** | Decision tree optimized for streaming | Feature interactions |
| **Ensemble** | Weighted combination of multiple models | Maximum accuracy |

### Drift Detectors
- **ADWIN** (Adaptive Windowing): Detects sudden drift with configurable sensitivity
- **Page-Hinkley Test**: Detects gradual drift patterns

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone or download the repository**
```bash
cd concept_drift_project
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Launch the Application

#### Option 1: Using Launcher Scripts (Recommended)
```bash
# Windows
run_app.bat

# Linux/Mac
./run_app.sh
```

#### Option 2: Manual Launch
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Dataset
- Go to the **"Run Pipeline"** tab
- Upload your CSV file **OR** click "Load Default Dataset"
- Select the **target column** (numeric column to predict)
- Preview your data to verify columns

### 2. Configure Settings
Use the sidebar to configure:

**Model Settings:**
- Select algorithm (Adaptive Tree recommended)
- Choose target column

**Drift Detection:**
- Select detector (ADWIN or Page-Hinkley)
- Adjust sensitivity parameters

**Training Settings:**
- Warmup samples (initial training period)
- Max samples (limit for faster experiments)
- Retraining strategy (window/reset/incremental)

### 3. Run Pipeline
- Click **"Start Pipeline"**
- Monitor real-time progress and metrics
- Watch live charts update during processing
- Drift events are highlighted automatically

### 4. View Results
Switch to the **"Results"** tab to see:
- Key metrics (MAE, RMSE, drift count)
- Prediction vs Actual charts
- Error distribution histograms
- Metrics history over time
- Drift analysis with intervals

### 5. Compare Models
Go to the **"Compare"** tab to:
- Select multiple model configurations
- Run comparison across all settings
- View performance side-by-side
- Identify best configuration for your data

### 6. Export Results
Download results in multiple formats:
- **JSON**: Complete results with configuration
- **CSV**: Predictions and errors for further analysis

## 🏗️ Project Structure

```
concept_drift_project/
├── app.py                      # Main Streamlit application
├── run_app.bat                 # Windows launcher
├── run_app.sh                  # Linux/Mac launcher
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── PROJECT_REPORT.md           # Detailed project report
├── UI_README.md                # UI-specific documentation
├── CLAUDE.md                   # Developer guide
│
├── src/                        # Source code
│   ├── data_stream.py          # Data streaming interface
│   ├── drift_detection.py      # Drift detection algorithms
│   ├── model.py                # Online learning models
│   ├── retrain.py              # Retraining strategies
│   └── main.py                 # CLI pipeline
│
├── data/                       # Datasets
│   └── electricity_market_dataset.csv
│
├── notebooks/                  # Jupyter notebooks
│   ├── experiment.ipynb
│   └── evaluation.ipynb
│
└── outputs/                    # Generated results
    ├── graphs/                 # Visualizations
    └── results_*.json          # Experiment results
```

## 🎯 Example Use Cases

### Electricity Market Analysis
```bash
# Use default dataset
python src/main.py --model adaptive_tree --detector adwin --warmup 1000
```

### Custom Dataset
```python
# In the UI:
# 1. Upload your CSV
# 2. Select target column
# 3. Configure model and detector
# 4. Run pipeline
```

### Batch Comparison
```bash
python src/main.py --compare --max-samples 10000
```

## 📊 Understanding Results

### Key Metrics
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Square Error): Penalizes large errors
- **Drift Count**: Number of detected concept drift events
- **Samples Processed**: Total data points analyzed

### Drift Detection
Drift points indicate when the data distribution changed significantly:
- **Orange vertical lines** on charts mark drift events
- **Higher errors** typically precede drift detection
- **Retraining** occurs automatically after drift

### Interpreting Charts
1. **Predictions vs Actual**: Shows how well the model tracks real values
2. **Error Chart**: Displays prediction errors over time
3. **Metrics History**: Tracks MAE/RMSE improvement
4. **Drift Intervals**: Distribution of time between drifts

## 🔧 Configuration Reference

### Model Parameters
```python
model_type: ['adaptive_tree', 'linear', 'tree', 'ensemble']
target_column: str  # Column to predict
```

### Drift Detector Parameters
```python
detector_type: ['adwin', 'page_hinkley']
delta: 0.0001 - 0.01  # ADWIN sensitivity
threshold: 10 - 200   # Page-Hinkley threshold
```

### Training Parameters
```python
warmup_size: 100 - 10000     # Initial training samples
max_samples: 0 - 100000      # 0 = process all
retrain_strategy: ['window', 'reset', 'incremental']
window_size: 500 - 10000     # Sliding window size
```

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Models
1. Extend `OnlineModel` class in `src/model.py`
2. Implement `learn_one()` and `predict_one()` methods
3. Update model selection in UI

### Adding New Detectors
1. Extend `DriftDetector` class in `src/drift_detection.py`
2. Implement `update()` method
3. Add detector configuration to UI

## 📚 Documentation

- **README.md** (this file): User guide and overview
- **PROJECT_REPORT.md**: Detailed technical report with results
- **UI_README.md**: UI-specific documentation
- **CLAUDE.md**: Developer guide and architecture

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **River ML**: For the excellent online learning library
- **Streamlit**: For the intuitive web framework
- **Plotly**: For interactive visualizations

## 📞 Support

For issues and questions:
1. Check the documentation files
2. Review example notebooks
3. Open an issue on GitHub

---

**Version**: 2.0  
**Last Updated**: April 21, 2026  
**Maintained by**: Concept Drift Detection Team
"# Concept-Drift-Detection-" 
