
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from ppg_hr_rr_model import (
    HRRRPredictor, 
    PPGDataset,
    load_data,
    filter_signals,
    normalize_signals,
    evaluate_model
)
from plot import trend_plot, bland_altman_plot


def load_model(model_path, device='cpu', use_separate_heads=True, use_dsp_features=True, sampling_rate=100):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pth file)
        device: Device to load model on ('cpu' or 'cuda')
        use_separate_heads: Whether model uses separate heads (should match training config)
        use_dsp_features: Whether model uses DSP features (should match training config)
        sampling_rate: Sampling rate in Hz (should match training config)
    
    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading model from: {model_path}")
    model = HRRRPredictor(
        input_length=1000, 
        use_separate_heads=use_separate_heads,
        use_dsp_features=use_dsp_features,
        sampling_rate=sampling_rate
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    return model


def load_training_config(results_dir):
    """
    Load training configuration from results directory.
    
    Args:
        results_dir: Path to results directory containing training_config.json
    
    Returns:
        Dictionary with training configuration
    """
    config_path = os.path.join(results_dir, 'training_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded training configuration from: {config_path}")
        return config
    else:
        print(f"Warning: Training config not found at {config_path}. Using defaults.")
        return None


def evaluate_trained_model(model_path, data_paths=None, results_dir=None, 
                          batch_size=32, device='cpu', save_results=True):
    """
    Comprehensive evaluation of a trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        data_paths: List of data paths (if None, will try to load from config)
        results_dir: Results directory (if None, will use model's parent directory)
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        save_results: Whether to save evaluation results and plots
    
    Returns:
        Dictionary with evaluation results
    """
    # Determine results directory
    if results_dir is None:
        results_dir = os.path.dirname(model_path)
    
    # Load training configuration if available
    config = load_training_config(results_dir)
    
    # Get configuration parameters
    if config:
        use_separate_heads = config.get('use_separate_heads', True)
        use_dsp_features = config.get('use_dsp_features', True)
        apply_filter = config.get('apply_filter', True)
        sampling_rate = config.get('sampling_rate', 100)
        low_cutoff = config.get('low_cutoff', 0.5)
        high_cutoff = config.get('high_cutoff', 8.0)
        filter_order = config.get('filter_order', 4)
        test_size = config.get('test_size', 0.2)
        val_size = config.get('val_size', 0.2)
        
        if data_paths is None:
            data_paths = config.get('data_paths', None)
    else:
        # Default values
        use_separate_heads = True
        use_dsp_features = True
        apply_filter = True
        sampling_rate = 100
        low_cutoff = 0.5
        high_cutoff = 8.0
        filter_order = 4
        test_size = 0.2
        val_size = 0.2
    
    # Default to four datasets if not provided
    if data_paths is None:
        default_data_dir = "/home/ghosn/Project/csee8300_3/data"
        data_paths = [
            os.path.join(default_data_dir, "dataset_constant_ibi_constant_wa.npy"),
            os.path.join(default_data_dir, "dataset_constant_ibi_dynamic_wa.npy"),
            os.path.join(default_data_dir, "dataset_dynamic_ibi_constant_wa.npy"),
            os.path.join(default_data_dir, "dataset_dynamic_ibi_dynamic_wa.npy")
        ]
        print(f"Using default four datasets for evaluation:")
        for path in data_paths:
            print(f"  - {path}")
    
    # Load model
    model = load_model(
        model_path, 
        device=device, 
        use_separate_heads=use_separate_heads,
        use_dsp_features=use_dsp_features,
        sampling_rate=sampling_rate
    )
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    signals, hr_labels, rr_labels = load_data(data_paths)
    print(f"Data shape: {signals.shape}")
    print(f"HR range: [{hr_labels.min():.2f}, {hr_labels.max():.2f}] bpm")
    print(f"RR range: [{rr_labels.min():.2f}, {rr_labels.max():.2f}] bpm")
    
    # Apply filter
    if apply_filter:
        print(f"Filtering signals (bandpass: {low_cutoff}-{high_cutoff} Hz)...")
        signals = filter_signals(
            signals,
            sampling_rate=sampling_rate,
            lowcut=low_cutoff,
            highcut=high_cutoff,
            order=filter_order
        )
        print("Filtering completed.")
    
    # Normalize signals
    print("Normalizing signals...")
    signals = normalize_signals(signals)
    
    # Split data (same split as training if using same random seed)
    print(f"\nSplitting data (test_size={test_size}, val_size={val_size})...")
    X_temp, X_test, y_hr_temp, y_hr_test, y_rr_temp, y_rr_test = train_test_split(
        signals, hr_labels, rr_labels, test_size=test_size, random_state=42
    )
    X_train, X_val, y_hr_train, y_hr_val, y_rr_train, y_rr_val = train_test_split(
        X_temp, y_hr_temp, y_rr_temp, test_size=val_size/(1-test_size), random_state=42
    )
    
    print(f"Test samples: {len(X_test)}")
    
    # Create test dataset and dataloader
    test_dataset = PPGDataset(X_test, y_hr_test, y_rr_test, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, test_loader, device=device)
    
    # Save results if requested
    if save_results:
        eval_dir = os.path.join(results_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save evaluation results
        results_to_save = {
            'hr_mae': float(results['hr_mae']),
            'hr_rmse': float(results['hr_rmse']),
            'hr_std': float(results['hr_std']),
            'hr_mare': float(results['hr_mare']),
            'rr_mae': float(results['rr_mae']),
            'rr_rmse': float(results['rr_rmse']),
            'rr_std': float(results['rr_std']),
            'rr_mare': float(results['rr_mare']),
            'num_test_samples': len(X_test)
        }
        
        results_path = os.path.join(eval_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nEvaluation results saved to: {results_path}")
        
        # Plot predictions using methods from plot.py
        # Format data as [ID, pred, label] for HR
        hr_data = np.column_stack([
            np.zeros(len(results['hr_preds'])),  # ID column (set to 0)
            results['hr_preds'],
            results['hr_trues']
        ])
        
        # Format data as [ID, pred, label] for RR
        rr_data = np.column_stack([
            np.zeros(len(results['rr_preds'])),  # ID column (set to 0)
            results['rr_preds'],
            results['rr_trues']
        ])
        
        # Create Fig directory in evaluation directory
        fig_dir = os.path.join(eval_dir, 'Fig')
        
        # Generate trend plots
        print("\nGenerating trend plots...")
        trend_plot(hr_data, 'HR', output_dir=fig_dir)
        trend_plot(rr_data, 'RR', output_dir=fig_dir)
        print(f"Trend plots saved to: {fig_dir}")
        
        # Generate Bland-Altman plots
        print("Generating Bland-Altman plots...")
        bland_altman_plot(hr_data, 'HR', output_dir=fig_dir)
        bland_altman_plot(rr_data, 'RR', output_dir=fig_dir)
        print(f"Bland-Altman plots saved to: {fig_dir}")
        

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained HR/RR prediction model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--data_paths', type=str, nargs='+', default=None,
                        help='Paths to data files (if not provided, will use config)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Results directory (default: parent of model_path)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run evaluation on')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save evaluation results')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Run evaluation
    results = evaluate_trained_model(
        model_path=args.model_path,
        data_paths=args.data_paths,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        device=args.device,
        save_results=not args.no_save
    )
    
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"HR - MAE: {results['hr_mae']:.2f} bpm, RMSE: {results['hr_rmse']:.2f} bpm, MARE: {results['hr_mare']:.4f}")
    print(f"RR - MAE: {results['rr_mae']:.2f} bpm, RMSE: {results['rr_rmse']:.2f} bpm, MARE: {results['rr_mare']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

