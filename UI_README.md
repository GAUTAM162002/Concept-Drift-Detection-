# Concept Drift Detection - Streamlit UI Documentation

Complete guide for using the Concept Drift Detection Platform web interface.

---

## 🚀 Quick Start

### Launch the Application

```bash
# Windows
run_app.bat

# Linux/Mac
./run_app.sh

# Manual
streamlit run app.py
```

Access at: **http://localhost:8501**

---

## 📱 Interface Overview

The application has four main tabs:

### 1. 🏠 Home
- Welcome message and feature overview
- Quick start guide with numbered steps
- Platform capabilities explanation

### 2. ▶️ Run Pipeline
- Upload custom datasets or use default
- Configure model and drift detection settings
- Execute pipeline with real-time monitoring

### 3. 📊 Results
- View comprehensive results dashboard
- Interactive visualizations
- Export options

### 4. 🔬 Compare
- Run multiple configurations
- Side-by-side performance comparison
- Identify best model settings

---

## 📁 Working with Datasets

### Upload Custom Dataset

1. Go to **"Run Pipeline"** tab
2. Click **"Browse files"** or drag & drop CSV
3. System automatically:
   - Detects numeric columns
   - Identifies categorical features
   - Shows data preview
4. Select **target column** from dropdown
5. Review detected feature count

### Use Default Dataset

1. Click **"📂 Load Default Dataset"** button
2. System loads electricity market dataset
3. Target column auto-set to "Investment_Feasibility"
4. 26 feature columns auto-detected

### Data Requirements

**Supported Formats:**
- CSV files (.csv)
- Numeric target column required
- Mix of numeric and categorical features supported

**Column Types:**
- **Numeric**: Automatically used as features or target
- **Categorical**: Automatically one-hot encoded
- **Date/Time**: Treated as categorical

---

## ⚙️ Configuration Guide

### Model Settings (Sidebar)

#### Algorithm Selection

| Option | Description | When to Use |
|--------|-------------|-------------|
| **Adaptive Tree** | Hoeffding Adaptive Tree with drift handling | Best overall, handles non-stationary data |
| **Linear** | Linear regression with scaling | Fast baseline, linear relationships |
| **Tree** | Hoeffding Tree regressor | Feature interactions, decision boundaries |
| **Ensemble** | Weighted combination of models | Maximum accuracy, slower training |

### Drift Detection Settings

#### Detector Type

**ADWIN (Adaptive Windowing)**
- Splits data into variable windows
- Detects when means differ significantly
- Recommended for most use cases

**Page-Hinkley Test**
- Monitors cumulative sum of differences
- Better for gradual drift patterns

#### Sensitivity Parameters

**ADWIN Delta:**
- Range: 0.0001 - 0.01
- Default: 0.002
- **Lower value** = More sensitive drift detection
- **Higher value** = Fewer false positives

**Page-Hinkley Threshold:**
- Range: 10 - 200
- Default: 50
- **Lower value** = More sensitive
- **Higher value** = More stable

### Training Settings

#### Warmup Samples
- Initial training before predictions begin
- Range: 100 - 10,000
- Default: 1,000
- **More warmup** = Better initial model
- **Less warmup** = Faster start

#### Max Samples
- Limit processing for quick experiments
- 0 = Process all data
- Useful for: Testing configurations, large datasets

#### Retraining Strategy

| Strategy | Behavior | Best For |
|----------|----------|----------|
| **Window** | Sliding window of recent samples | Balanced adaptation |
| **Reset** | Clear model, start fresh | Aggressive drift response |
| **Incremental** | Continue with adjustments | Conservative updates |

#### Window Size
- Size of training buffer after drift
- Range: 500 - 10,000
- Default: 2,000

---

## ▶️ Running the Pipeline

### Step-by-Step

1. **Configure Settings**
   - Select model and detector in sidebar
   - Adjust parameters as needed
   - Review configuration in expander

2. **Click "🚀 Start Pipeline"**
   - Progress bar shows completion percentage
   - Status badge shows current phase (Warmup/Processing/Drift)

3. **Monitor Progress**
   - **Metrics boxes** update every 500 samples:
     - MAE (Mean Absolute Error)
     - RMSE (Root Mean Square Error)
     - Drifts detected
     - Samples processed

4. **View Live Charts**
   - **Left chart**: Predictions vs Actual
     - Blue line: Actual values
     - Orange line: Predictions
     - Yellow dashed lines: Drift points
   - **Right chart**: Prediction Error
     - Shows error magnitude over time

5. **Completion**
   - Green "✅ Complete!" badge appears
   - Navigate to "📊 Results" tab

### Understanding Status Badges

| Badge | Meaning | Action |
|-------|---------|--------|
| 🔄 Warmup | Initial training phase | Wait for completion |
| ✅ Processing | Normal operation | Monitor metrics |
| 🚨 Drift at X | Drift detected | Model retraining |
| ✅ Complete | Pipeline finished | View results |

---

## 📊 Results Dashboard

### Key Metrics Section

Four metric boxes display:

1. **Total Samples**: Data points processed
2. **Valid Predictions**: Successful predictions made
3. **Drift Detections**: Number of drift events
4. **MAE**: Mean Absolute Error (lower is better)

### Visualization Tabs

#### 📈 Predictions Tab
- Time series overlay of actual vs predicted values
- Drift events marked with vertical lines
- Interactive zoom and pan
- Hover for exact values

#### 📉 Errors Tab
- **Error Over Time**: Line chart showing prediction errors
- **Error Distribution**: Histogram of error frequency
- Helps identify error patterns

#### 📊 History Tab
- MAE and RMSE trends over time
- Cumulative drift count
- Secondary Y-axis for drift count
- Identifies performance degradation

#### 🚨 Drifts Tab
- Total drift count
- Average interval between drifts
- Drift point log with timestamps
- Interval distribution histogram

### Export Options

#### Download JSON
- Complete results with configuration
- All predictions and metrics
- Drift points and timestamps

#### Download CSV
- Index, actual, predicted, error columns
- Easy import into Excel/other tools
- Time-stamped filename

---

## 🔬 Model Comparison

### Running Comparisons

1. Go to **"🔬 Compare"** tab
2. **Select configurations** to compare (hold Ctrl/Cmd for multiple)
3. Set **max samples** for test (default: 10,000)
4. Click **"🚀 Run Comparison"**
5. Wait for all configurations to complete

### Understanding Results

**Results Table:**
- Model type
- Detector type
- MAE and RMSE scores
- Number of drifts detected

**Visual Comparison:**
- Bar chart comparing MAE/RMSE across models
- Drift count comparison
- Easy identification of best performer

### Recommended Comparison Sets

**Accuracy Focus:**
- Adaptive Tree + ADWIN
- Ensemble + ADWIN
- Tree + ADWIN

**Speed Focus:**
- Linear + ADWIN
- Linear + Page-Hinkley
- Tree + Page-Hinkley

**Drift Sensitivity:**
- Adaptive Tree + ADWIN (delta=0.001)
- Adaptive Tree + ADWIN (delta=0.005)
- Tree + Page-Hinkley (threshold=30)

---

## 💡 Tips and Best Practices

### Configuration Tips

**For Best Accuracy:**
```
Model: Adaptive Tree
Detector: ADWIN
delta: 0.001
warmup: 2000
strategy: window
```

**For Fastest Processing:**
```
Model: Linear
Detector: Page-Hinkley
threshold: 100
max_samples: 10000
```

**For Maximum Drift Detection:**
```
Model: Adaptive Tree
Detector: ADWIN
delta: 0.0005
strategy: reset
```

### Troubleshooting

**Slow Performance:**
- Reduce max_samples
- Use Linear model instead of Ensemble
- Close other browser tabs

**No Drift Detected:**
- Decrease ADWIN delta (more sensitive)
- Use Reset strategy
- Check data has actual distribution shifts

**High Errors:**
- Increase warmup samples
- Try Adaptive Tree model
- Verify target column selection

**App Won't Start:**
- Check virtual environment activated
- Verify requirements installed: `pip install -r requirements.txt`
- Try manual launch: `streamlit run app.py`

### Performance Optimization

1. **Limit Samples**: Use max_samples for testing
2. **Linear Model**: Fastest option for exploration
3. **Lower Delta**: More sensitive but slower
4. **Close Charts**: Minimize browser load

---

## 🎨 Interface Features

### Design Elements

**Gradient Header:**
- Professional purple/indigo gradient
- Project title and subtitle

**Metric Cards:**
- Gradient backgrounds
- Large value display
- Clear labels

**Status Badges:**
- Color-coded (blue/running, green/success, orange/warning)
- Animated indicators
- Clear status messages

**Interactive Charts:**
- Plotly.js powered
- Zoom, pan, hover tooltips
- Responsive sizing

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| R | Rerun script (if "Always rerun" enabled) |
| Esc | Close modals/overlays |

### Responsive Design

The interface adapts to:
- Desktop browsers (full layout)
- Tablet screens (adjusted columns)
- Mobile browsers (stacked layout)

---

## 🔧 Advanced Features

### Custom CSS Styling

The app uses custom CSS for:
- Professional color scheme
- Smooth transitions
- Custom scrollbars
- Card-based layouts

Modify `st.markdown()` block in `app.py` to customize.

### Session State

The app maintains state for:
- Uploaded data path
- Selected target column
- Processing results
- Comparison results

Refresh page to reset state.

### Behind the Scenes

**Processing Flow:**
1. Data loaded via `DataStream` class
2. Features auto-detected and encoded
3. Model trained incrementally
4. Drift detector monitors errors
5. Retraining triggered on drift
6. Results cached in session state
7. Charts updated with Plotly

---

## 📞 Support

### Common Questions

**Q: Can I use my own dataset?**
A: Yes! Upload any CSV with numeric columns. Select target column and run.

**Q: Which model should I choose?**
A: Start with Adaptive Tree for best overall performance.

**Q: How do I know if drift was detected?**
A: Look for yellow vertical lines on charts or check "🚨 Drifts" tab.

**Q: Can I save my results?**
A: Yes, use Download JSON or CSV buttons in Results tab.

**Q: Is my data secure?**
A: Data is processed locally and not sent to any external servers.

### Getting Help

1. Check this documentation
2. Review example notebooks in `notebooks/`
3. Check PROJECT_REPORT.md for technical details
4. Review error messages in the app

---

## 📝 Changelog

### Version 2.0 (April 21, 2026)
- ✨ Complete UI redesign with professional theme
- 📊 Real-time chart updates during processing
- 📁 Custom dataset support with auto-detection
- 🔬 Model comparison feature
- 📤 JSON/CSV export options
- 🎨 Gradient design and improved typography

### Version 1.0 (April 18, 2026)
- Initial release with core functionality
- CLI pipeline execution
- Basic drift detection
- Jupyter notebook evaluation

---

**Last Updated:** April 21, 2026  
**UI Version:** 2.0  
**Documentation:** Complete
