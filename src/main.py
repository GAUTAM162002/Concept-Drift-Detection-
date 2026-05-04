import sys
import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_stream import DataStream
from drift_detection import DriftDetector
from model import OnlineModel
from retrain import RetrainingStrategy


def run_concept_drift_pipeline(
    data_path,
    model_type='adaptive_tree',
    detector_type='adwin',
    warmup_size=1000,
    max_samples=None,
    output_dir='outputs'
):
    """
    Run the complete concept drift detection pipeline.

    Args:
        data_path: Path to the CSV data file
        model_type: Type of model ('linear', 'tree', 'adaptive_tree', 'ensemble')
        detector_type: Drift detector type ('adwin', 'page_hinkley')
        warmup_size: Number of samples before drift detection starts
        max_samples: Maximum samples to process (None for all)
        output_dir: Directory to save results

    Returns:
        dict: Results summary
    """
    print(f"=" * 60)
    print(f"Concept Drift Detection Pipeline")
    print(f"Data: {data_path}")
    print(f"Model: {model_type}, Drift Detector: {detector_type}")
    print(f"=" * 60)

    # Initialize components
    stream = DataStream(data_path)
    model = OnlineModel(model_type=model_type)
    drift_detector = DriftDetector(detector_type=detector_type)
    retrainer = RetrainingStrategy(model, strategy='window', window_size=2000, warmup_size=warmup_size)

    # Tracking variables
    predictions = []
    actuals = []
    prediction_errors = []
    drift_points = []
    metrics_history = []

    sample_count = 0
    samples_since_last_drift = 0

    print(f"\nProcessing samples...")

    while stream.has_more_data():
        if max_samples and sample_count >= max_samples:
            break

        row = stream.get_next_instance()
        if row is None:
            break

        # Skip rows with missing target
        if model.target not in row or pd.isna(row[model.target]):
            continue

        sample_count += 1
        samples_since_last_drift += 1

        # Get actual value
        y_true = model._get_target(row)

        # Warmup phase: just learn, don't predict
        if sample_count <= warmup_size:
            model.learn_one(row)
            retrainer.add_sample(row)
            actuals.append(y_true)
            predictions.append(None)
            continue

        # Prediction phase
        y_pred = model.predict_one(row)
        predictions.append(y_pred)
        actuals.append(y_true)

        if y_pred is not None:
            error = abs(y_true - y_pred)
            prediction_errors.append(error)

            # Update drift detector
            drift_detected = drift_detector.update(y_true, y_pred)

            if drift_detected:
                drift_point = sample_count
                drift_points.append(drift_point)
                print(f"  [DRIFT] Detected at sample {drift_point} (error: {error:.4f})")

                # Trigger retraining
                retrain_info = retrainer.on_drift_detected(drift_point)
                samples_since_last_drift = 0

            # Update model
            model.learn_one(row)
            retrainer.add_sample(row)

            # Periodic metrics
            if sample_count % 1000 == 0:
                recent_errors = prediction_errors[-1000:]
                mae = np.mean(recent_errors)
                rmse = np.sqrt(np.mean([e**2 for e in recent_errors]))
                print(f"  [Progress] Sample {sample_count}: MAE={mae:.4f}, RMSE={rmse:.4f}")

                metrics_history.append({
                    'sample': sample_count,
                    'mae': mae,
                    'rmse': rmse,
                    'drift_count': len(drift_points)
                })

    print(f"\n{'=' * 60}")
    print(f"Pipeline Complete")
    print(f"{'=' * 60}")

    # Calculate final metrics
    valid_predictions = [(p, a) for p, a in zip(predictions, actuals) if p is not None]

    if valid_predictions:
        preds, trues = zip(*valid_predictions)
        final_mae = np.mean([abs(p - t) for p, t in zip(preds, trues)])
        final_rmse = np.sqrt(np.mean([(p - t)**2 for p, t in zip(preds, trues)]))

        print(f"\nFinal Metrics:")
        print(f"  Samples processed: {sample_count}")
        print(f"  Valid predictions: {len(valid_predictions)}")
        print(f"  Drift detections: {len(drift_points)}")
        print(f"  Final MAE: {final_mae:.4f}")
        print(f"  Final RMSE: {final_rmse:.4f}")
    else:
        print("  No valid predictions made")
        final_mae = None
        final_rmse = None

    # Prepare results
    results = {
        'config': {
            'data_path': data_path,
            'model_type': model_type,
            'detector_type': detector_type,
            'warmup_size': warmup_size,
            'max_samples': max_samples
        },
        'summary': {
            'total_samples': sample_count,
            'valid_predictions': len(valid_predictions) if valid_predictions else 0,
            'drift_count': len(drift_points),
            'drift_points': drift_points,
            'final_mae': final_mae,
            'final_rmse': final_rmse
        },
        'metrics_history': metrics_history,
        'retrain_history': retrainer.retrain_history if hasattr(retrainer, 'retrain_history') else []
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'results_{model_type}_{detector_type}_{timestamp}.json')

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results


def compare_models(data_path, max_samples=10000, output_dir='outputs'):
    """
    Compare different model and drift detector combinations.

    Args:
        data_path: Path to data file
        max_samples: Maximum samples to process
        output_dir: Output directory
    """
    print("\n" + "=" * 60)
    print("Model Comparison Experiment")
    print("=" * 60)

    configs = [
        ('linear', 'adwin'),
        ('linear', 'page_hinkley'),
        ('adaptive_tree', 'adwin'),
        ('adaptive_tree', 'page_hinkley'),
        ('ensemble', 'adwin'),
    ]

    all_results = []

    for model_type, detector_type in configs:
        try:
            results = run_concept_drift_pipeline(
                data_path=data_path,
                model_type=model_type,
                detector_type=detector_type,
                max_samples=max_samples,
                output_dir=output_dir
            )
            all_results.append({
                'model': model_type,
                'detector': detector_type,
                'results': results
            })
        except Exception as e:
            print(f"Error with {model_type}/{detector_type}: {e}")

    # Save comparison
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_file = os.path.join(output_dir, f'comparison_{timestamp}.json')

    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("Comparison Summary")
    print("=" * 60)
    for r in all_results:
        summary = r['results']['summary']
        print(f"{r['model']}/{r['detector']}: MAE={summary.get('final_mae', 'N/A'):.4f}, "
              f"Drifts={summary['drift_count']}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concept Drift Detection Pipeline')
    parser.add_argument('--data', type=str, default='data/electricity_market_dataset.csv',
                        help='Path to data file')
    parser.add_argument('--model', type=str, default='adaptive_tree',
                        choices=['linear', 'tree', 'adaptive_tree', 'ensemble'],
                        help='Model type')
    parser.add_argument('--detector', type=str, default='adwin',
                        choices=['adwin', 'page_hinkley'],
                        help='Drift detector type')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Warmup samples')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to process')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison of multiple configurations')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')

    args = parser.parse_args()

    if args.compare:
        compare_models(
            data_path=args.data,
            max_samples=args.max_samples or 10000,
            output_dir=args.output_dir
        )
    else:
        run_concept_drift_pipeline(
            data_path=args.data,
            model_type=args.model,
            detector_type=args.detector,
            warmup_size=args.warmup,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
