import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import random
import os
from datetime import datetime
import json
import torch.fft

class DSPFeatureExtractor(nn.Module):
    """
    DSP Feature Extraction Block for PPG signals.
    Extracts time-domain, frequency-domain, and statistical features.
    """
    def __init__(self, input_length=1000, sampling_rate=100):
        super(DSPFeatureExtractor, self).__init__()
        self.input_length = input_length
        self.sampling_rate = sampling_rate
        
    def forward(self, x):
        """
        Extract DSP features from PPG signal.
        
        Args:
            x: Input signal tensor of shape (batch_size, input_length)
        
        Returns:
            features: Extracted features tensor of shape (batch_size, num_features)
        """
        batch_size = x.shape[0]
        features_list = []
        
        # Time-domain features
        # 1. Statistical moments
        mean = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1)
        std = torch.std(x, dim=1, keepdim=True)  # (batch_size, 1)
        variance = torch.var(x, dim=1, keepdim=True)  # (batch_size, 1)
        
        # 2. Peak-to-peak amplitude
        peak_to_peak = torch.max(x, dim=1)[0] - torch.min(x, dim=1)[0]  # (batch_size,)
        peak_to_peak = peak_to_peak.unsqueeze(1)  # (batch_size, 1)
        
        # 3. Zero-crossing rate
        # Count sign changes
        diff_sign = torch.diff(torch.sign(x), dim=1)  # (batch_size, input_length-1)
        zero_crossings = torch.sum(torch.abs(diff_sign) > 0, dim=1, keepdim=True).float() / (self.input_length - 1)  # (batch_size, 1)
        
        # 4. Signal energy
        energy = torch.sum(x ** 2, dim=1, keepdim=True) / self.input_length  # (batch_size, 1)
        
        # 5. Root Mean Square (RMS)
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True))  # (batch_size, 1)
        
        # Waveform features
        # 6. First derivative (slope features)
        first_deriv = torch.diff(x, dim=1)  # (batch_size, input_length-1)
        mean_slope = torch.mean(first_deriv, dim=1, keepdim=True)  # (batch_size, 1)
        std_slope = torch.std(first_deriv, dim=1, keepdim=True)  # (batch_size, 1)
        
        # 7. Second derivative (curvature features)
        second_deriv = torch.diff(first_deriv, dim=1)  # (batch_size, input_length-2)
        mean_curvature = torch.mean(second_deriv, dim=1, keepdim=True)  # (batch_size, 1)
        std_curvature = torch.std(second_deriv, dim=1, keepdim=True)  # (batch_size, 1)
        
        # Statistical features
        # 8. Skewness (third moment)
        centered = x - mean  # (batch_size, input_length)
        skewness = torch.mean((centered ** 3), dim=1, keepdim=True) / (std ** 3 + 1e-8)  # (batch_size, 1)
        
        # 9. Kurtosis (fourth moment)
        kurtosis = torch.mean((centered ** 4), dim=1, keepdim=True) / (std ** 4 + 1e-8)  # (batch_size, 1)
        
        # Frequency-domain features
        # 10. FFT-based features
        # Compute FFT
        fft_vals = torch.fft.rfft(x, dim=1)  # (batch_size, input_length//2+1)
        magnitude = torch.abs(fft_vals)  # (batch_size, input_length//2+1)
        power_spectrum = magnitude ** 2  # (batch_size, input_length//2+1)
        
        # Frequency bins (move to same device as input)
        freqs = torch.fft.rfftfreq(self.input_length, 1.0/self.sampling_rate).to(x.device)  # (input_length//2+1,)
        
        # 11. Dominant frequency (frequency with maximum power)
        max_power_idx = torch.argmax(power_spectrum, dim=1)  # (batch_size,)
        dominant_freq = freqs[max_power_idx]  # (batch_size,)
        dominant_freq = dominant_freq.unsqueeze(1)  # (batch_size, 1)
        
        # 12. Spectral power in HR band (0.8-3.0 Hz, ~48-180 bpm)
        hr_mask = (freqs >= 0.8) & (freqs <= 3.0)  # (input_length//2+1,)
        hr_power = torch.sum(power_spectrum[:, hr_mask], dim=1, keepdim=True)  # (batch_size, 1)
        total_power = torch.sum(power_spectrum, dim=1, keepdim=True)  # (batch_size, 1)
        hr_power_ratio = hr_power / (total_power + 1e-8)  # (batch_size, 1)
        
        # 13. Spectral power in RR band (0.1-0.5 Hz, ~6-30 bpm)
        rr_mask = (freqs >= 0.1) & (freqs <= 0.5)  # (input_length//2+1,)
        rr_power = torch.sum(power_spectrum[:, rr_mask], dim=1, keepdim=True)  # (batch_size, 1)
        rr_power_ratio = rr_power / (total_power + 1e-8)  # (batch_size, 1)
        
        # 14. Spectral centroid (weighted average frequency)
        freqs_expanded = freqs.unsqueeze(0).expand(batch_size, -1)  # (batch_size, input_length//2+1)
        spectral_centroid = torch.sum(freqs_expanded * power_spectrum, dim=1, keepdim=True) / (total_power + 1e-8)  # (batch_size, 1)
        
        # 15. Spectral bandwidth (spread of frequencies)
        freq_diff = freqs_expanded - spectral_centroid  # (batch_size, input_length//2+1)
        spectral_bandwidth = torch.sqrt(torch.sum((freq_diff ** 2) * power_spectrum, dim=1, keepdim=True) / (total_power + 1e-8))  # (batch_size, 1)
        
        # Combine all features
        features_list = [
            mean, std, variance,
            peak_to_peak, zero_crossings, energy, rms,
            mean_slope, std_slope, mean_curvature, std_curvature,
            skewness, kurtosis,
            dominant_freq, hr_power_ratio, rr_power_ratio,
            spectral_centroid, spectral_bandwidth
        ]
        
        features = torch.cat(features_list, dim=1)  # (batch_size, num_features)
        
        return features


class PPGDataset(Dataset):
    """Dataset class for PPG signal data with optional augmentation"""
    def __init__(self, signals, hr_labels, rr_labels, augment=False):
        self.signals = torch.FloatTensor(signals)
        self.hr_labels = torch.FloatTensor(hr_labels)
        self.rr_labels = torch.FloatTensor(rr_labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx].clone()
        
        # Data augmentation during training
        if self.augment:
            # Add small random noise
            noise = torch.randn_like(signal) * 0.01
            signal = signal + noise
            
            # Random scaling (0.95 to 1.05)
            scale = torch.empty(1).uniform_(0.95, 1.05)
            signal = signal * scale
        
        return {
            'signal': signal,
            'hr': self.hr_labels[idx],
            'rr': self.rr_labels[idx]
        }


class HRRRPredictor(nn.Module):
    """
    Improved CNN-based model for predicting HR and RR from PPG signals.
    
    Architecture:
    - DSP Feature Extraction Block (time-domain, frequency-domain features)
    - 1D Convolutional layers with residual connections
    - Separate heads for HR and RR prediction
    - Global Average Pooling to reduce dimensionality
    - Fully connected layers for regression
    """
    def __init__(self, input_length=1000, use_separate_heads=True, use_dsp_features=True, sampling_rate=100):
        super(HRRRPredictor, self).__init__()
        self.use_separate_heads = use_separate_heads
        self.use_dsp_features = use_dsp_features
        
        # DSP Feature Extraction Block
        if use_dsp_features:
            self.dsp_extractor = DSPFeatureExtractor(input_length=input_length, sampling_rate=sampling_rate)
            # DSP features dimension (18 features)
            dsp_feature_dim = 18
        else:
            dsp_feature_dim = 0
        
        # First conv block: extract low-level features
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Second conv block: extract mid-level features
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Third conv block: extract high-level features
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Additional conv block for deeper features
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Shared feature extractor (CNN features + DSP features)
        self.shared_fc = nn.Sequential(
            nn.Linear(256 + dsp_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        if use_separate_heads:
            # Separate heads for HR and RR
            self.hr_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
            self.rr_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
        else:
            # Shared head
            self.fc = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 2)  # Output: HR and RR
            )
    
    def forward(self, x):
        # x shape: (batch_size, 1000)
        
        # Extract DSP features from raw input signal
        if self.use_dsp_features:
            dsp_features = self.dsp_extractor(x)  # (batch_size, 18)
        
        # Add channel dimension: (batch_size, 1, 1000)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.conv1(x)  # (batch_size, 32, 500)
        x = self.conv2(x)  # (batch_size, 64, 250)
        x = self.conv3(x)  # (batch_size, 128, 125)
        x = self.conv4(x)  # (batch_size, 256, 62)
        
        # Global pooling: (batch_size, 256, 1)
        x = self.global_pool(x)
        
        # Flatten: (batch_size, 256)
        cnn_features = x.squeeze(-1)
        
        # Concatenate CNN features with DSP features
        if self.use_dsp_features:
            combined_features = torch.cat([cnn_features, dsp_features], dim=1)  # (batch_size, 256+18)
        else:
            combined_features = cnn_features  # (batch_size, 256)
        
        # Shared feature extraction
        x = self.shared_fc(combined_features)  # (batch_size, 64)
        
        if self.use_separate_heads:
            # Separate predictions
            hr = self.hr_head(x)  # (batch_size, 1)
            rr = self.rr_head(x)  # (batch_size, 1)
            return torch.cat([hr, rr], dim=1)  # (batch_size, 2)
        else:
            # Shared head
            return self.fc(x)  # (batch_size, 2)


def load_data(data_paths):
    """
    Load and preprocess data from one or multiple numpy files.
    
    Args:
        data_paths: Single path (str) or list of paths (list) to numpy files
    
    Returns:
        signals: Combined signals from all datasets
        hr_labels: Combined HR labels from all datasets
        rr_labels: Combined RR labels from all datasets
    """
    # Convert single path to list for uniform processing
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    
    all_signals = []
    all_hr_labels = []
    all_rr_labels = []
    
    for data_path in data_paths:
        print(f"Loading data from: {data_path}")
        data = np.load(data_path)
        
        # Extract signals (first 1000 columns) and labels (HR at -5, RR at -4)
        signals = data[:, :1000]  # Shape: (n_samples, 1000)
        hr_labels = data[:, -5]    # Heart Rate
        rr_labels = data[:, -4]    # Respiratory Rate
        
        # Remove samples where rr_labels == 0
        # keep_idx = rr_labels != 0
        # signals = signals[keep_idx]
        # hr_labels = hr_labels[keep_idx]
        # rr_labels = rr_labels[keep_idx]
        
        all_signals.append(signals)
        all_hr_labels.append(hr_labels)
        all_rr_labels.append(rr_labels)
        
        print(f"  Loaded {len(signals)} samples (after removing rr=0)")
    
    # Concatenate all datasets
    combined_signals = np.concatenate(all_signals, axis=0)
    combined_hr_labels = np.concatenate(all_hr_labels, axis=0)
    combined_rr_labels = np.concatenate(all_rr_labels, axis=0)
    
    print(f"Total combined samples: {len(combined_signals)}")
    
    return combined_signals, combined_hr_labels, combined_rr_labels


def filter_ppg_signal(signal, sampling_rate=100, lowcut=0.5, highcut=8.0, order=4):
    """
    Apply bandpass filter to PPG signal to remove noise and stabilize the signal.
    
    Args:
        signal: 1D array of PPG signal values
        sampling_rate: Sampling rate in Hz (default: 100 Hz)
        lowcut: Low cutoff frequency in Hz (default: 0.5 Hz)
        highcut: High cutoff frequency in Hz (default: 8.0 Hz)
        order: Filter order (default: 4)
    
    Returns:
        Filtered signal
    """
    nyquist = sampling_rate / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth bandpass filter
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered_signal = scipy_signal.filtfilt(b, a, signal)
    
    return filtered_signal


def filter_signals(signals, sampling_rate=100, lowcut=0.5, highcut=8.0, order=4):
    """
    Apply bandpass filter to all PPG signals.
    
    Args:
        signals: 2D array of PPG signals (n_samples, signal_length)
        sampling_rate: Sampling rate in Hz (default: 100 Hz)
        lowcut: Low cutoff frequency in Hz (default: 0.5 Hz)
        highcut: High cutoff frequency in Hz (default: 8.0 Hz)
        order: Filter order (default: 4)
    
    Returns:
        Filtered signals
    """
    signals_filtered = np.zeros_like(signals)
    for i in range(len(signals)):
        signals_filtered[i] = filter_ppg_signal(
            signals[i], sampling_rate=sampling_rate, 
            lowcut=lowcut, highcut=highcut, order=order
        )
    return signals_filtered


def normalize_signals(signals):
    """Normalize each signal to [0, 1] range"""
    signals_normalized = np.zeros_like(signals)
    for i in range(len(signals)):
        signal = signals[i]
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val - min_val > 0:
            signals_normalized[i] = (signal - min_val) / (max_val - min_val)
        else:
            signals_normalized[i] = signal
    return signals_normalized


def train_model(model, train_loader, val_loader, num_epochs=300, learning_rate=0.001, 
                device='cuda', hr_weight=1.0, rr_weight=2.0, use_huber_loss=True, 
                results_dir='./results'):
    """
    Train the model with improved loss function
    
    Args:
        hr_weight: Weight for HR loss (default: 1.0)
        rr_weight: Weight for RR loss (default: 2.0, higher because RR is harder to predict)
        use_huber_loss: Use Huber loss instead of MSE for robustness to outliers
    """
    if use_huber_loss:
        criterion = nn.HuberLoss(delta=1.0)  # More robust to outliers
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            signals = batch['signal'].to(device)
            hr_true = batch['hr'].to(device)
            rr_true = batch['rr'].to(device)
            
            # Forward pass
            outputs = model(signals)
            hr_pred = outputs[:, 0]
            rr_pred = outputs[:, 1]
            
            # Compute weighted loss (give more weight to RR since it's harder to predict)
            hr_loss = criterion(hr_pred, hr_true)
            rr_loss = criterion(rr_pred, rr_true)
            loss = hr_weight * hr_loss + rr_weight * rr_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                signals = batch['signal'].to(device)
                hr_true = batch['hr'].to(device)
                rr_true = batch['rr'].to(device)
                
                outputs = model(signals)
                hr_pred = outputs[:, 0]
                rr_pred = outputs[:, 1]
                
                # Use same weighted loss as training
                hr_loss = criterion(hr_pred, hr_true)
                rr_loss = criterion(rr_pred, rr_true)
                loss = hr_weight * hr_loss + rr_weight * rr_loss
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model and compute metrics"""
    model.eval()
    hr_preds = []
    hr_trues = []
    rr_preds = []
    rr_trues = []
    
    with torch.no_grad():
        for batch in test_loader:
            signals = batch['signal'].to(device)
            hr_true = batch['hr'].to(device)
            rr_true = batch['rr'].to(device)
            
            outputs = model(signals)
            hr_pred = outputs[:, 0]
            rr_pred = outputs[:, 1]
            
            hr_preds.extend(hr_pred.cpu().numpy())
            hr_trues.extend(hr_true.cpu().numpy())
            rr_preds.extend(rr_pred.cpu().numpy())
            rr_trues.extend(rr_true.cpu().numpy())
    
    hr_preds = np.array(hr_preds)
    hr_trues = np.array(hr_trues)
    rr_preds = np.array(rr_preds)
    rr_trues = np.array(rr_trues)
    
    # Compute errors
    hr_errors = hr_preds - hr_trues
    rr_errors = rr_preds - rr_trues
    
    # Compute metrics using sklearn
    hr_mae = mean_absolute_error(hr_trues, hr_preds)
    hr_rmse = np.sqrt(mean_squared_error(hr_trues, hr_preds))
    hr_std = np.std(hr_errors)
    # MARE: Add small epsilon to denominator to avoid division by zero
    # skip the samples with 0 in hr_trues
    # mask = hr_trues != 0
    mask = hr_trues > 0
    hr_mare = np.mean(np.abs(hr_errors[mask] / (hr_trues[mask])))
    
    rr_mae = mean_absolute_error(rr_trues, rr_preds)
    rr_rmse = np.sqrt(mean_squared_error(rr_trues, rr_preds))
    rr_std = np.std(rr_errors)
    # MARE: Add small epsilon to denominator to avoid division by zero
    # mask = rr_trues != 0
    mask = rr_trues > 0
    rr_mare = np.mean(np.abs(rr_errors[mask] / (rr_trues[mask])))
    
    print("\n=== Evaluation Results ===")
    print(f"Heart Rate (HR):")
    print(f"  MAE:  {hr_mae:.2f} bpm")
    print(f"  RMSE: {hr_rmse:.2f} bpm")
    print(f"  STD:  {hr_std:.2f} bpm")
    print(f"  MARE: {hr_mare:.4f}")
    print(f"\nRespiratory Rate (RR):")
    print(f"  MAE:  {rr_mae:.2f} bpm")
    print(f"  RMSE: {rr_rmse:.2f} bpm")
    print(f"  STD:  {rr_std:.2f} bpm")
    print(f"  MARE: {rr_mare:.4f}")
    
    return {
        'hr_preds': hr_preds,
        'hr_trues': hr_trues,
        'rr_preds': rr_preds,
        'rr_trues': rr_trues,
        'hr_mae': hr_mae,
        'hr_rmse': hr_rmse,
        'hr_std': hr_std,
        'hr_mare': hr_mare,
        'rr_mae': rr_mae,
        'rr_rmse': rr_rmse,
        'rr_std': rr_std,
        'rr_mare': rr_mare
    }


def plot_training_curves(train_losses, val_losses, results_dir='./results'):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to '{save_path}'")
    
    # Save loss values to file
    losses_data = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    losses_path = os.path.join(results_dir, 'losses.json')
    with open(losses_path, 'w') as f:
        json.dump(losses_data, f, indent=2)
    print(f"Loss values saved to '{losses_path}'")


def plot_predictions(results, results_dir='./results'):
    """Plot predicted vs true values"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # HR predictions
    axes[0].scatter(results['hr_trues'], results['hr_preds'], alpha=0.5)
    axes[0].plot([results['hr_trues'].min(), results['hr_trues'].max()],
                  [results['hr_trues'].min(), results['hr_trues'].max()], 'r--', lw=2)
    axes[0].set_xlabel('True HR (bpm)')
    axes[0].set_ylabel('Predicted HR (bpm)')
    axes[0].set_title(f'HR: MAE={results["hr_mae"]:.2f}, RMSE={results["hr_rmse"]:.2f}, STD={results["hr_std"]:.2f}')
    axes[0].grid(True)
    
    # RR predictions
    axes[1].scatter(results['rr_trues'], results['rr_preds'], alpha=0.5)
    axes[1].plot([results['rr_trues'].min(), results['rr_trues'].max()],
                  [results['rr_trues'].min(), results['rr_trues'].max()], 'r--', lw=2)
    axes[1].set_xlabel('True RR (bpm)')
    axes[1].set_ylabel('Predicted RR (bpm)')
    axes[1].set_title(f'RR: MAE={results["rr_mae"]:.2f}, RMSE={results["rr_rmse"]:.2f}, STD={results["rr_std"]:.2f}')
    axes[1].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'predictions.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction plots saved to '{save_path}'")
    
    # Save evaluation results to file
    results_to_save = {
        'hr_mae': float(results['hr_mae']),
        'hr_rmse': float(results['hr_rmse']),
        'hr_std': float(results['hr_std']),
        'hr_mare': float(results['hr_mare']),
        'rr_mae': float(results['rr_mae']),
        'rr_rmse': float(results['rr_rmse']),
        'rr_std': float(results['rr_std']),
        'rr_mare': float(results['rr_mare'])
    }
    results_path = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Evaluation results saved to '{results_path}'")


if __name__ == "__main__":
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('./results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Configuration - combine multiple datasets
    DATA_PATHS = [
        "/home/ghosn/Project/csee8300_3/data/dataset_constant_ibi_constant_wa.npy",
        "/home/ghosn/Project/csee8300_3/data/dataset_constant_ibi_dynamic_wa.npy",
        "/home/ghosn/Project/csee8300_3/data/dataset_dynamic_ibi_constant_wa.npy",
        "/home/ghosn/Project/csee8300_3/data/dataset_dynamic_ibi_dynamic_wa.npy"
    ]
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2

    # set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Filtering configuration
    APPLY_FILTER = True  # Set to False to disable filtering
    SAMPLING_RATE = 100  # Hz
    LOW_CUTOFF = 0.5     # Hz (removes DC and very low frequency drift)
    HIGH_CUTOFF = 8.0    # Hz (removes high frequency noise)
    FILTER_ORDER = 4     # Butterworth filter order

    # Training configuration
    HR_WEIGHT = 1.0   # Weight for HR loss
    RR_WEIGHT = 1.6   # Higher weight for RR (it's harder to predict)
    USE_HUBER_LOSS = True  # Use Huber loss for robustness
    

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading data...")
    signals, hr_labels, rr_labels = load_data(DATA_PATHS)
    print(f"Data shape: {signals.shape}")
    print(f"HR range: [{hr_labels.min():.2f}, {hr_labels.max():.2f}]")
    print(f"RR range: [{rr_labels.min():.2f}, {rr_labels.max():.2f}]")
    
    # Apply filter to stabilize PPG signals
    if APPLY_FILTER:
        print(f"Filtering signals (bandpass: {LOW_CUTOFF}-{HIGH_CUTOFF} Hz)...")
        signals = filter_signals(
            signals, 
            sampling_rate=SAMPLING_RATE,
            lowcut=LOW_CUTOFF,
            highcut=HIGH_CUTOFF,
            order=FILTER_ORDER
        )
        print("Filtering completed.")
    
    # Normalize signals
    print("Normalizing signals...")
    signals = normalize_signals(signals)
    
    # Split data: train -> val/test -> val/test
    X_temp, X_test, y_hr_temp, y_hr_test, y_rr_temp, y_rr_test = train_test_split(
        signals, hr_labels, rr_labels, test_size=TEST_SIZE, random_state=42
    )
    X_train, X_val, y_hr_train, y_hr_val, y_rr_train, y_rr_val = train_test_split(
        X_temp, y_hr_temp, y_rr_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets and dataloaders (enable augmentation for training)
    train_dataset = PPGDataset(X_train, y_hr_train, y_rr_train, augment=True)
    val_dataset = PPGDataset(X_val, y_hr_val, y_rr_val, augment=False)
    test_dataset = PPGDataset(X_test, y_hr_test, y_rr_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create improved model with separate heads and DSP features
    USE_SEPARATE_HEADS = True  # Use separate heads for HR and RR
    USE_DSP_FEATURES = True  # Use DSP feature extraction block
    model = HRRRPredictor(
        input_length=1000, 
        use_separate_heads=USE_SEPARATE_HEADS,
        use_dsp_features=USE_DSP_FEATURES,
        sampling_rate=SAMPLING_RATE
    ).to(device)
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save training configuration
    config = {
        'timestamp': timestamp,
        'data_paths': DATA_PATHS,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'test_size': TEST_SIZE,
        'val_size': VAL_SIZE,
        'apply_filter': APPLY_FILTER,
        'sampling_rate': SAMPLING_RATE,
        'low_cutoff': LOW_CUTOFF,
        'high_cutoff': HIGH_CUTOFF,
        'filter_order': FILTER_ORDER,
        'hr_weight': HR_WEIGHT,
        'rr_weight': RR_WEIGHT,
        'use_huber_loss': USE_HUBER_LOSS,
        'use_separate_heads': USE_SEPARATE_HEADS,
        'use_dsp_features': USE_DSP_FEATURES,
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'device': str(device)
    }
    config_path = os.path.join(results_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training configuration saved to '{config_path}'")
    
    # Train model
    print("\nStarting training...")
    print(f"Training with: HR weight={HR_WEIGHT}, RR weight={RR_WEIGHT}, Huber loss={USE_HUBER_LOSS}")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE,
        device=device,
        hr_weight=HR_WEIGHT,
        rr_weight=RR_WEIGHT,
        use_huber_loss=USE_HUBER_LOSS,
        results_dir=results_dir
    )
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, device=device)
    
    # Plot results
    plot_training_curves(train_losses, val_losses, results_dir=results_dir)
    plot_predictions(results, results_dir=results_dir)
    
    print(f"\nTraining completed! All results saved to: {results_dir}")

