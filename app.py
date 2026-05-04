"""
Concept Drift Detection Platform
Professional web interface for online learning and drift detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import tempfile
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_stream import DataStream
from drift_detection import DriftDetector
from model import OnlineModel
from retrain import RetrainingStrategy

st.set_page_config(
    page_title="Concept Drift Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(79, 70, 229, 0.3);
    }

    .app-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .app-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Section headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .metric-box {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }

    .metric-box .value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .metric-box .label {
        font-size: 0.85rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Status badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .badge-running {
        background: #dbeafe;
        color: #1d4ed8;
    }

    .badge-success {
        background: #d1fae5;
        color: #047857;
    }

    .badge-warning {
        background: #fef3c7;
        color: #b45309;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        height: 100%;
    }

    .feature-card h4 {
        color: #1e293b;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }

    .feature-card p {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }

    /* Step indicator */
    .step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .step-number {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        flex-shrink: 0;
    }

    .step-content h4 {
        margin: 0 0 0.25rem 0;
        color: #1e293b;
    }

    .step-content p {
        margin: 0;
        color: #64748b;
        font-size: 0.9rem;
    }

    /* Alert boxes */
    .alert {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }

    .alert-info {
        background: #eff6ff;
        border-color: #3b82f6;
        color: #1e40af;
    }

    .alert-success {
        background: #f0fdf4;
        border-color: #22c55e;
        color: #15803d;
    }

    .alert-warning {
        background: #fffbeb;
        border-color: #f59e0b;
        color: #b45309;
    }

    /* Chart container */
    .chart-wrapper {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin: 1rem 0;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border-radius: 8px !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
        border-radius: 10px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state."""
    defaults = {
        'results': None,
        'is_running': False,
        'comparison_results': [],
        'uploaded_data_path': None,
        'uploaded_df': None,
        'selected_target': None,
        'drift_log': [],
        'config': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Render app header."""
    st.markdown("""
    <div class="app-header">
        <h1>🔬 Concept Drift Detection</h1>
        <p>Advanced Online Learning & Drift Detection Platform</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render configuration sidebar."""
    with st.sidebar:
        st.markdown("## Configuration")

        # Model settings
        st.markdown("### Model")
        model_type = st.selectbox(
            "Algorithm",
            ['adaptive_tree', 'linear', 'tree', 'ensemble'],
            help="Choose the online learning algorithm"
        )

        model_desc = {
            'adaptive_tree': 'Hoeffding Adaptive Tree - handles drift internally',
            'linear': 'Linear Regression - fast baseline',
            'tree': 'Hoeffding Tree - efficient for streams',
            'ensemble': 'Ensemble - combines multiple models'
        }
        st.caption(model_desc[model_type])

        st.markdown("---")

        # Drift detection
        st.markdown("### Drift Detection")
        detector_type = st.selectbox(
            "Detector",
            ['adwin', 'page_hinkley'],
            help="ADWIN: Adaptive Windowing | Page-Hinkley: Statistical test"
        )

        if detector_type == 'adwin':
            delta = st.slider("Delta (sensitivity)", 0.0001, 0.01, 0.002, 0.0001, format="%.4f")
            threshold = 50
        else:
            delta = 0.002
            threshold = st.slider("Threshold", 10, 200, 50, 5)

        st.markdown("---")

        # Training settings
        st.markdown("### Training")
        col1, col2 = st.columns(2)
        with col1:
            warmup = st.number_input("Warmup", 100, 10000, 1000, 100, help="Initial training samples")
        with col2:
            max_samples = st.number_input("Max Samples", 0, 100000, 0, 1000, help="0 = all")

        strategy = st.selectbox("Strategy", ['window', 'reset', 'incremental'], help="Retraining approach")
        window = st.number_input("Window Size", 500, 10000, 2000, 500)

        # Last run stats
        if st.session_state.results:
            st.markdown("---")
            st.markdown("### Last Run")
            res = st.session_state.results['summary']
            st.metric("MAE", f"{res['final_mae']:.4f}" if res['final_mae'] else "N/A")
            st.metric("Drifts", res['drift_count'])

        return {
            'model_type': model_type,
            'detector_type': detector_type,
            'delta': delta,
            'threshold': threshold,
            'warmup_size': warmup,
            'max_samples': max_samples if max_samples > 0 else None,
            'retrain_strategy': strategy,
            'window_size': window
        }


def render_data_section():
    """Render data upload section."""
    st.markdown('<div class="section-title">Dataset Selection</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("Upload CSV file", type=['csv'])

    with col2:
        st.markdown("""
        <div class="alert alert-info">
            <strong>Tip:</strong> Upload a CSV with numeric columns.
            Select a target column for prediction.
        </div>
        """, unsafe_allow_html=True)

        default_path = "data/electricity_market_dataset.csv"
        if os.path.exists(default_path):
            if st.button("📂 Load Default Dataset", use_container_width=True):
                st.session_state.uploaded_data_path = default_path
                st.session_state.uploaded_df = None
                st.session_state.selected_target = "Investment_Feasibility"
                st.success("Default dataset loaded!")
                st.rerun()

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
            f.write(uploaded.getvalue())
            st.session_state.uploaded_data_path = f.name

        df = pd.read_csv(uploaded)
        st.session_state.uploaded_df = df

        st.markdown(f"""
        <div class="alert alert-success">
            <strong>Loaded:</strong> {len(df):,} rows × {len(df.columns)} columns
        </div>
        """, unsafe_allow_html=True)

        # Target selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            col1, col2 = st.columns([2, 1])
            with col1:
                target = st.selectbox(
                    "Target column",
                    numeric_cols,
                    index=numeric_cols.index('Investment_Feasibility') if 'Investment_Feasibility' in numeric_cols else 0
                )
                st.session_state.selected_target = target

            with col2:
                features = [c for c in df.columns if c != target]
                st.metric("Features", len(features))

            with st.expander("Preview data"):
                st.dataframe(df.head(10), use_container_width=True, hide_index=True)
                info = pd.DataFrame({
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Missing': df.isnull().sum()
                })
                st.dataframe(info, use_container_width=True)
        else:
            st.error("No numeric columns found!")

    return st.session_state.uploaded_data_path


def run_pipeline(config, data_path):
    """Execute pipeline with progress tracking."""
    st.session_state.is_running = True
    st.session_state.drift_log = []

    try:
        stream = DataStream(data_path)

        # Setup features
        feature_cols = None
        cat_cols = None
        target = st.session_state.get('selected_target', 'Investment_Feasibility')

        if st.session_state.uploaded_df is not None:
            df = st.session_state.uploaded_df
            feature_cols = [c for c in df.columns if c != target]
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            cat_cols = [c for c in cat_cols if c in feature_cols]

        model = OnlineModel(
            model_type=config['model_type'],
            target=target,
            feature_columns=feature_cols,
            categorical_columns=cat_cols
        )

        detector = DriftDetector(
            detector_type=config['detector_type'],
            delta=config['delta'],
            threshold=config.get('threshold', 50)
        )

        retrainer = RetrainingStrategy(
            model,
            strategy=config['retrain_strategy'],
            window_size=config['window_size'],
            warmup_size=config['warmup_size']
        )

        predictions, actuals, errors = [], [], []
        drift_points, metrics_history = [], []
        sample_count = 0
        total_rows = len(stream.data) if hasattr(stream, 'data') else 0

        # Progress UI
        progress_bar = st.progress(0)
        status = st.empty()

        # Metrics
        cols = st.columns(4)
        mae_box = cols[0].empty()
        rmse_box = cols[1].empty()
        drift_box = cols[2].empty()
        sample_box = cols[3].empty()

        # Charts
        c1, c2 = st.columns(2)
        pred_chart = c1.empty()
        err_chart = c2.empty()

        while stream.has_more_data():
            if config['max_samples'] and sample_count >= config['max_samples']:
                break

            row = stream.get_next_instance()
            if row is None:
                break

            if target not in row or pd.isna(row[target]):
                continue

            sample_count += 1
            y_true = model._get_target(row)

            # Progress
            if total_rows > 0:
                progress = min(sample_count / min(total_rows, config['max_samples'] or total_rows), 1.0)
                progress_bar.progress(progress)

            # Warmup
            if sample_count <= config['warmup_size']:
                model.learn_one(row)
                retrainer.add_sample(row)
                actuals.append(y_true)
                predictions.append(None)
                status.markdown(f'<span class="badge badge-running">🔄 Warmup: {sample_count}/{config["warmup_size"]}</span>', unsafe_allow_html=True)
                continue

            # Predict
            y_pred = model.predict_one(row)
            predictions.append(y_pred)
            actuals.append(y_true)

            if y_pred is not None:
                error = abs(y_true - y_pred)
                errors.append(error)

                if detector.update(y_true, y_pred):
                    drift_point = sample_count
                    drift_points.append(drift_point)
                    st.session_state.drift_log.append({
                        'sample': drift_point,
                        'error': error,
                        'time': datetime.now().strftime('%H:%M:%S')
                    })
                    retrainer.on_drift_detected(drift_point)
                    status.markdown(f'<span class="badge badge-warning">🚨 Drift at {drift_point}</span>', unsafe_allow_html=True)

                model.learn_one(row)
                retrainer.add_sample(row)

                # Update metrics every 500 samples
                if sample_count % 500 == 0:
                    recent = errors[-500:]
                    mae = np.mean(recent)
                    rmse = np.sqrt(np.mean([e**2 for e in recent]))

                    mae_box.metric("MAE", f"{mae:.4f}")
                    rmse_box.metric("RMSE", f"{rmse:.4f}")
                    drift_box.metric("Drifts", len(drift_points))
                    sample_box.metric("Samples", f"{sample_count:,}")

                    metrics_history.append({
                        'sample': sample_count,
                        'mae': mae,
                        'rmse': rmse,
                        'drift_count': len(drift_points)
                    })

                    status.markdown(f'<span class="badge badge-running">✅ Processing: {sample_count:,} | MAE: {mae:.4f}</span>', unsafe_allow_html=True)

                    # Update charts
                    update_charts(predictions, actuals, errors, drift_points, pred_chart, err_chart)

        # Final results
        valid = [(p, a) for p, a in zip(predictions, actuals) if p is not None]

        if valid:
            preds, trues = zip(*valid)
            final_mae = np.mean([abs(p - t) for p, t in zip(preds, trues)])
            final_rmse = np.sqrt(np.mean([(p - t)**2 for p, t in zip(preds, trues)]))
        else:
            final_mae = final_rmse = None

        st.session_state.results = {
            'config': config,
            'summary': {
                'total_samples': sample_count,
                'valid_predictions': len(valid) if valid else 0,
                'drift_count': len(drift_points),
                'drift_points': drift_points,
                'final_mae': final_mae,
                'final_rmse': final_rmse
            },
            'predictions': predictions,
            'actuals': actuals,
            'errors': errors,
            'metrics_history': metrics_history
        }

        st.session_state.is_running = False
        status.markdown('<span class="badge badge-success">✅ Complete!</span>', unsafe_allow_html=True)

    except Exception as e:
        st.session_state.is_running = False
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def update_charts(predictions, actuals, errors, drift_points, pred_chart, err_chart):
    """Update live charts."""
    valid_idx = [i for i, p in enumerate(predictions) if p is not None]
    valid_preds = [predictions[i] for i in valid_idx]
    valid_actuals = [actuals[i] for i in valid_idx]

    if len(valid_preds) < 10:
        return

    n = min(500, len(valid_preds))
    show_preds = valid_preds[-n:]
    show_actuals = valid_actuals[-n:]
    show_errors = errors[-n:]

    # Prediction chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=show_actuals, mode='lines', name='Actual', line=dict(color='#64748b', width=1.5)))
    fig1.add_trace(go.Scatter(y=show_preds, mode='lines', name='Predicted', line=dict(color='#4f46e5', width=2)))

    window_start = valid_idx[-n] if n < len(valid_idx) else 0
    for d in drift_points:
        if window_start <= d < window_start + n:
            fig1.add_vline(x=d - window_start, line_dash="dash", line_color="#f59e0b", opacity=0.6)

    fig1.update_layout(
        title="Predictions vs Actual",
        height=320,
        margin=dict(l=40, r=20, t=40, b=30),
        template='plotly_white',
        legend=dict(orientation="h", y=1.1)
    )
    pred_chart.plotly_chart(fig1, use_container_width=True, key=f"pred_{time.time()}")

    # Error chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=show_errors, mode='lines', fill='tozeroy', line=dict(color='#f97316'), fillcolor='rgba(249,115,22,0.2)'))

    fig2.update_layout(
        title="Prediction Error",
        height=320,
        margin=dict(l=40, r=20, t=40, b=30),
        template='plotly_white',
        showlegend=False
    )
    err_chart.plotly_chart(fig2, use_container_width=True, key=f"err_{time.time()}")


def render_results():
    """Render results dashboard."""
    if not st.session_state.results:
        return

    results = st.session_state.results
    summary = results['summary']

    st.markdown('<div class="section-title">Results Dashboard</div>', unsafe_allow_html=True)

    # Metrics
    st.markdown("### Key Metrics")
    cols = st.columns(4)

    metrics = [
        ("Total Samples", f"{summary['total_samples']:,}"),
        ("Valid Predictions", f"{summary['valid_predictions']:,}"),
        ("Drift Detections", str(summary['drift_count'])),
        ("MAE", f"{summary['final_mae']:.4f}" if summary['final_mae'] else "N/A")
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    if summary['final_mae']:
        st.metric("RMSE", f"{summary['final_rmse']:.4f}")

    # Tabs
    st.markdown("---")
    st.markdown("### Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "📉 Errors", "📊 History", "🚨 Drifts"])

    preds = results.get('predictions', [])
    actuals = results.get('actuals', [])
    errors = results.get('errors', [])
    drift_points = summary['drift_points']

    valid_idx = [i for i, p in enumerate(preds) if p is not None]
    valid_preds = [preds[i] for i in valid_idx]
    valid_actuals = [actuals[i] for i in valid_idx]

    with tab1:
        if valid_preds:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=valid_actuals, mode='lines', name='Actual', line=dict(color='#64748b')))
            fig.add_trace(go.Scatter(y=valid_preds, mode='lines', name='Predicted', line=dict(color='#4f46e5')))

            for d in drift_points[:50]:
                if d < len(valid_actuals):
                    fig.add_vline(x=valid_idx.index(d) if d in valid_idx else d, line_dash="dash", line_color="#f59e0b")

            fig.update_layout(height=450, template='plotly_white', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if errors:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=errors, mode='lines', fill='tozeroy', line=dict(color='#f97316'), fillcolor='rgba(249,115,22,0.15)'))
            fig.update_layout(height=450, template='plotly_white', title="Error Over Time")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.histogram(x=errors, nbins=50, template='plotly_white', color_discrete_sequence=['#4f46e5'])
            fig2.update_layout(height=350, title="Error Distribution", xaxis_title="Absolute Error")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        history = results.get('metrics_history', [])
        if history:
            df = pd.DataFrame(history)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df['sample'], y=df['mae'], mode='lines', name='MAE', line=dict(color='#4f46e5')), secondary_y=False)
            fig.add_trace(go.Scatter(x=df['sample'], y=df['rmse'], mode='lines', name='RMSE', line=dict(color='#7c3aed')), secondary_y=False)
            fig.add_trace(go.Scatter(x=df['sample'], y=df['drift_count'], mode='lines', name='Drifts', line=dict(color='#f59e0b', dash='dot')), secondary_y=True)

            fig.update_layout(height=450, template='plotly_white', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View data"):
                st.dataframe(df, use_container_width=True, hide_index=True)

    with tab4:
        if drift_points:
            st.metric("Total Drifts", len(drift_points))

            if len(drift_points) > 1:
                intervals = [drift_points[i+1] - drift_points[i] for i in range(len(drift_points)-1)]
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Interval", f"{np.mean(intervals):.0f}")
                c2.metric("Min", min(intervals))
                c3.metric("Max", max(intervals))

                fig = px.histogram(x=intervals, nbins=20, template='plotly_white', color_discrete_sequence=['#7c3aed'])
                fig.update_layout(height=350, xaxis_title="Samples Between Drifts")
                st.plotly_chart(fig, use_container_width=True)

            if st.session_state.drift_log:
                st.dataframe(pd.DataFrame(st.session_state.drift_log), use_container_width=True, hide_index=True)
        else:
            st.info("No drift detected with current configuration.")

    # Export
    st.markdown("---")
    st.markdown("### Export Results")

    col1, col2 = st.columns(2)
    with col1:
        json_data = json.dumps(results, indent=2, default=str)
        st.download_button("📥 Download JSON", json_data, f"results_{datetime.now():%Y%m%d_%H%M%S}.json", "application/json", use_container_width=True)

    with col2:
        if valid_preds:
            csv_data = pd.DataFrame({
                'index': valid_idx,
                'actual': valid_actuals,
                'predicted': valid_preds,
                'error': [abs(p-a) for p,a in zip(valid_preds, valid_actuals)]
            }).to_csv(index=False)
            st.download_button("📥 Download CSV", csv_data, f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", use_container_width=True)


def render_comparison():
    """Render model comparison."""
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    presets = [
        {'model_type': 'adaptive_tree', 'detector_type': 'adwin', 'name': '🌳 Adaptive Tree + ADWIN'},
        {'model_type': 'adaptive_tree', 'detector_type': 'page_hinkley', 'name': '🌳 Adaptive Tree + Page-Hinkley'},
        {'model_type': 'linear', 'detector_type': 'adwin', 'name': '📈 Linear + ADWIN'},
        {'model_type': 'linear', 'detector_type': 'page_hinkley', 'name': '📈 Linear + Page-Hinkley'},
        {'model_type': 'ensemble', 'detector_type': 'adwin', 'name': '🔀 Ensemble + ADWIN'},
    ]

    selected = st.multiselect("Select configurations", range(len(presets)), format_func=lambda i: presets[i]['name'], default=[0, 2, 4])
    max_samples = st.slider("Max samples", 1000, 50000, 10000, 1000)

    if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
        if not st.session_state.uploaded_data_path:
            st.error("Upload a dataset first!")
            return

        configs = []
        for i in selected:
            cfg = presets[i].copy()
            cfg.update({'warmup_size': 1000, 'max_samples': max_samples, 'retrain_strategy': 'window', 'window_size': 2000, 'delta': 0.002})
            configs.append(cfg)

        progress = st.empty()
        progress.progress(0)

        results = []
        target = st.session_state.get('selected_target', 'Investment_Feasibility')

        for idx, cfg in enumerate(configs):
            try:
                stream = DataStream(st.session_state.uploaded_data_path)
                model = OnlineModel(cfg['model_type'], target=target)
                detector = DriftDetector(cfg['detector_type'], delta=0.002)

                predictions, actuals, errors, drifts = [], [], [], []
                sample_count = 0

                while stream.has_more_data():
                    if cfg['max_samples'] and sample_count >= cfg['max_samples']:
                        break

                    row = stream.get_next_instance()
                    if row is None:
                        break

                    if target not in row or pd.isna(row[target]):
                        continue

                    sample_count += 1
                    y_true = float(row[target])

                    if sample_count <= cfg['warmup_size']:
                        model.learn_one(row)
                        continue

                    y_pred = model.predict_one(row)
                    predictions.append(y_pred)
                    actuals.append(y_true)

                    if y_pred is not None:
                        error = abs(y_true - y_pred)
                        errors.append(error)
                        if detector.update(y_true, y_pred):
                            drifts.append(sample_count)
                        model.learn_one(row)

                valid = [(p,a) for p,a in zip(predictions, actuals) if p is not None]
                if valid:
                    p, a = zip(*valid)
                    mae = np.mean([abs(x-y) for x,y in zip(p,a)])
                    rmse = np.sqrt(np.mean([(x-y)**2 for x,y in zip(p,a)]))
                else:
                    mae = rmse = None

                results.append({'config': cfg, 'summary': {'mae': mae, 'rmse': rmse, 'drifts': len(drifts), 'samples': sample_count}})

            except Exception as e:
                st.error(f"Error with {cfg['name']}: {e}")

            progress.progress((idx + 1) / len(configs))

        progress.empty()
        st.session_state.comparison_results = results

    if st.session_state.comparison_results:
        st.markdown("### Comparison Results")

        data = []
        for r in st.session_state.comparison_results:
            data.append({
                'Model': r['config']['model_type'],
                'Detector': r['config']['detector_type'],
                'MAE': f"{r['summary']['mae']:.4f}" if r['summary']['mae'] else 'N/A',
                'RMSE': f"{r['summary']['rmse']:.4f}" if r['summary']['rmse'] else 'N/A',
                'Drifts': r['summary']['drifts']
            })

        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


def render_home():
    """Render home page."""
    st.markdown('<div class="section-title">Welcome</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="alert alert-info">
        <strong>Concept Drift Detection Platform</strong> - Detect changes in data patterns
        using advanced online learning algorithms.
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("### Features")

    cols = st.columns(3)
    features = [
        ("🔄", "Online Learning", "Train models incrementally on streaming data without storing entire datasets"),
        ("🚨", "Drift Detection", "Automatically detect concept drift using ADWIN and Page-Hinkley algorithms"),
        ("📊", "Visualization", "Interactive charts and metrics for model performance analysis")
    ]

    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <h4>{icon} {title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick start
    st.markdown("### Quick Start")

    steps = [
        ("1", "Upload Data", "Go to 'Run Pipeline' and upload your CSV or use the default dataset"),
        ("2", "Configure", "Select model, drift detector, and parameters in the sidebar"),
        ("3", "Run", "Click 'Start Pipeline' to begin processing"),
        ("4", "Analyze", "View results, compare models, and export findings")
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step">
            <div class="step-number">{num}</div>
            <div class="step-content">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main entry point."""
    render_header()
    init_session_state()

    config = render_sidebar()

    tabs = st.tabs(["🏠 Home", "▶️ Run Pipeline", "📊 Results", "🔬 Compare"])

    with tabs[0]:
        render_home()

    with tabs[1]:
        st.markdown('<div class="section-title">Run Pipeline</div>', unsafe_allow_html=True)

        data_path = render_data_section()

        if data_path is None:
            st.warning("Please upload a dataset or load the default dataset to continue.")
        else:
            with st.expander("Current Configuration"):
                st.json(config)

            col1, col2 = st.columns([1, 4])

            with col1:
                if st.button("🚀 Start Pipeline", type="primary", disabled=st.session_state.is_running, use_container_width=True):
                    run_pipeline(config, data_path)

            with col2:
                if st.session_state.is_running:
                    st.markdown('<span class="badge badge-running">⏳ Running...</span>', unsafe_allow_html=True)

            if st.session_state.results and not st.session_state.is_running:
                st.markdown('<span class="badge badge-success">✅ Complete!</span>', unsafe_allow_html=True)

    with tabs[2]:
        if st.session_state.results is None:
            st.info("Run a pipeline first to see results.")
        else:
            render_results()

    with tabs[3]:
        render_comparison()


if __name__ == "__main__":
    main()
