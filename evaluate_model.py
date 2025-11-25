
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
    evaluate_model,
    plot_predictions
)


def load_model(model_path, device='cpu', use_separate_heads=True):
    print(f"Loading model from: {model_path}")
    model = HRRRPredictor(input_length=1000, use_separate_heads=use_separate_heads).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    return model


def load_training_config(results_dir):
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
    if results_dir is None:
        results_dir = os.path.dirname(model_path)
    
    config = load_training_config(results_dir)
    
    if config:
        use_separate_heads = config.get('use_separate_heads', True)
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
        use_separate_heads = True
        apply_filter = True
        sampling_rate = 100
        low_cutoff = 0.5
        high_cutoff = 8.0
        filter_order = 4
        test_size = 0.2
        val_size = 0.2
    
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
    
    model = load_model(model_path, device=device, use_separate_heads=use_separate_heads)
    
    print("\nLoading and preprocessing data...")
    signals, hr_labels, rr_labels = load_data(data_paths)
    print(f"Data shape: {signals.shape}")
    print(f"HR range: [{hr_labels.min():.2f}, {hr_labels.max():.2f}] bpm")
    print(f"RR range: [{rr_labels.min():.2f}, {rr_labels.max():.2f}] bpm")

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
    
    print("Normalizing signals...")
    signals = normalize_signals(signals)

    print(f"\nSplitting data (test_size={test_size}, val_size={val_size})...")
    X_temp, X_test, y_hr_temp, y_hr_test, y_rr_temp, y_rr_test = train_test_split(
        signals, hr_labels, rr_labels, test_size=test_size, random_state=42
    )
    X_train, X_val, y_hr_train, y_hr_val, y_rr_train, y_rr_val = train_test_split(
        X_temp, y_hr_temp, y_rr_temp, test_size=val_size/(1-test_size), random_state=42
    )
    
    print(f"Test samples: {len(X_test)}")

    test_dataset = PPGDataset(X_test, y_hr_test, y_rr_test, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nEvaluating model on test set...")
    results = evaluate_model(model, test_loader, device=device)

    if save_results:
        eval_dir = os.path.join(results_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)

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
        
        plot_predictions(results, results_dir=eval_dir)
        

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

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
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

