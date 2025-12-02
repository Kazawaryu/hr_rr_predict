# Heart Rate and Respiratory Rate Prediction Project Report

## Executive Summary

This project presents a deep learning-based system for simultaneous prediction of Heart Rate (HR) and Respiratory Rate (RR) from Photoplethysmography (PPG) signals. The system employs a hybrid architecture combining 1D Convolutional Neural Networks (CNN) with Digital Signal Processing (DSP) feature extraction to achieve high-accuracy predictions. The model processes 10-second PPG signal segments sampled at 100 Hz and outputs both HR and RR values in beats per minute (bpm).

**Key Achievements:**

- HR Prediction: MAE of 0.63 bpm, RMSE of 1.36 bpm, MARE of 2.00%

- RR Prediction: MAE of 0.15 bpm, RMSE of 0.32 bpm, MARE of 1.99%

- Dataset: Use all four datasets, 60% for training, 20% for validation, 20% for testing

- Model size: 182466 trainable parameters

- Evaluation on 19,200 test samples

Please note that the newest code is published on my Github repository. More new feature can be found [here](https://github.com/Kazawaryu/hr_rr_predict).

---

## 1. Introduction

### 1.1 Background

The task is to design an algorithm to calculate the inter-beat-interval (IBI) and waveform amplitude (WA) from the simulated cardiorespiratory sensor data. For the real-time live demo purpose, please send the sensor data, prediction results and labels to InfluxDB through MQTT and visualize them in the Grafana.

### 1.2 Objectives

The primary objectives of this project are:

1. **Develop a deep learning model** capable of simultaneously predicting HR and RR from PPG signals

2. **Integrate DSP feature extraction** to combine handcrafted features with learned CNN features

3. **Achieve high prediction accuracy** suitable for clinical and wearable device applications

4. **Provide comprehensive evaluation** using multiple metrics and visualization techniques

5. **Visualization in Grafana** appling Grafana to design a online data visualization platform

### 1.3 Dataset

The project utilizes four datasets representing different physiological conditions based on inter-beat-interval (IBI) and waveform amplitude (WA) variations:

- `dataset_constant_ibi_constant_wa.npy`: Constant inter-beat-interval (IBI), constant waveform amplitude (WA)

- `dataset_constant_ibi_dynamic_wa.npy`: Constant inter-beat-interval (IBI), dynamic waveform amplitude (WA)

- `dataset_dynamic_ibi_constant_wa.npy`: Dynamic inter-beat-interval (IBI), constant waveform amplitude (WA)

- `dataset_dynamic_ibi_dynamic_wa.npy`: Dynamic inter-beat-interval (IBI), dynamic waveform amplitude (WA)

Each dataset contains:

- **Input**: 1000-dimensional cardiorespiratory sensor data (10 seconds at 100 Hz sampling rate)

- **Output**: HR and RR labels in bpm (only use HR and RR in this project)

- **Total samples**: 96,000 samples across all datasets

### 1.4 MQTT Publisher

This script is provided by the template, only add the interface to online predict the HR and RR.

---

## 2. Methodology

### 2.1 Model Architecture

The model employs a hybrid architecture that combines:

1. **DSP Feature Extraction Block**: Extracts 18 handcrafted features from raw cardiorespiratory sensor data

2. **1D Convolutional Neural Network**: Learns hierarchical temporal features

3. **Separate Prediction Heads**: Task-specific heads for HR and RR prediction

The structure of this model is shown as follow:

```shell
Model architecture:
HRRRPredictor(
  (dsp_extractor): DSPFeatureExtractor()
  (conv1): Sequential(
    (0): Conv1d(1, 32, kernel_size=(7,), stride=(1,), padding=(3,))
    (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv1d(32, 64, kernel_size=(5,), stride=(1,), padding=(2,))
    (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv3): Sequential(
    (0): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv4): Sequential(
    (0): Conv1d(128, 256, kernel_size=(3,), stride=(1,), padding=(1,))
    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (global_pool): AdaptiveAvgPool1d(output_size=1)
  (shared_fc): Sequential(
    (0): Linear(in_features=274, out_features=128, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=128, out_features=64, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.2, inplace=False)
  )
  (hr_head): Sequential(
    (0): Linear(in_features=64, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=32, out_features=1, bias=True)
  )
  (rr_head): Sequential(
    (0): Linear(in_features=64, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=32, out_features=1, bias=True)
  )
)
```

#### 2.1.1 DSP Feature Extractor

The DSP feature extraction block extracts features directly from raw input signals (not filtered data), including:

**Time-Domain Features (7 features):**

- Statistical moments: mean, standard deviation, variance

- Signal characteristics: peak-to-peak amplitude, zero-crossing rate

- Energy metrics: signal energy, root mean square (RMS)

**Waveform Features (4 features):**

- First derivative statistics: mean slope, standard deviation of slope

- Second derivative statistics: mean curvature, standard deviation of curvature

**Statistical Features (2 features):**

- Skewness (third moment)

- Kurtosis (fourth moment)

**Frequency-Domain Features (5 features):**

- Dominant frequency (frequency with maximum power)

- Spectral power ratio in HR band (0.8-3.0 Hz, ~48-180 bpm)

- Spectral power ratio in RR band (0.1-0.5 Hz, ~6-30 bpm)

- Spectral centroid (weighted average frequency)

- Spectral bandwidth (spread of frequencies)

**Total DSP Features: 18**

#### 2.1.2 Convolutional Neural Network

The CNN architecture consists of four sequential 1D convolutional blocks:

1. **Conv Block 1** (Low-level features):
   
   - 1D Convolution: 1 → 32 channels, kernel size 7
   - Batch Normalization + ReLU + Max Pooling (kernel size 2)
   - Output: (batch_size, 32, 500)

2. **Conv Block 2** (Mid-level features):
   
   - 1D Convolution: 32 → 64 channels, kernel size 5
   - Batch Normalization + ReLU + Max Pooling (kernel size 2)
   - Output: (batch_size, 64, 250)

3. **Conv Block 3** (High-level features):
   
   - 1D Convolution: 64 → 128 channels, kernel size 3
   - Batch Normalization + ReLU + Max Pooling (kernel size 2)
   - Output: (batch_size, 128, 125)

4. **Conv Block 4** (Deep features):
   
   - 1D Convolution: 128 → 256 channels, kernel size 3
   - Batch Normalization + ReLU + Max Pooling (kernel size 2)
   - Output: (batch_size, 256, 62)

**Global Average Pooling**: Reduces spatial dimensions to (batch_size, 256, 1) → (batch_size, 256)

#### 2.1.3 Feature Fusion and Prediction

- **Feature Concatenation**: CNN features (256-dim) + DSP features (18-dim) = 274-dim combined features

- **Shared Feature Extractor**:
  
  - Linear: 274 → 128 (ReLU, Dropout 0.3)
  - Linear: 128 → 64 (ReLU, Dropout 0.2)

- **Separate Prediction Heads**:
  
  - **HR Head**: 64 → 32 → 1 (ReLU, Dropout 0.1)
  - **RR Head**: 64 → 32 → 1 (ReLU, Dropout 0.1)

**Total Parameters**: ~182,466 trainable parameters

### 2.2 Data Preprocessing

#### 2.2.1 Signal Filtering

A 4th-order Butterworth bandpass filter is applied to remove noise and stabilize the cardiorespiratory sensor signal:

- **Low Cutoff Frequency**: 0.5 Hz (removes DC component and low-frequency drift)

- **High Cutoff Frequency**: 8.0 Hz (removes high-frequency noise)

- **Sampling Rate**: 100 Hz

- **Filter Type**: Butterworth bandpass (bidirectional filtering using `filtfilt`)

**Rationale**: 

- HR typically ranges from 0.8-3 Hz (48-180 bpm)

- RR typically ranges from 0.1-0.5 Hz (6-30 bpm)

- The 0.5-8 Hz bandpass preserves both cardiac and respiratory components while removing artifacts

#### 2.2.2 Normalization

Each signal is normalized to [0, 1] range using min-max normalization:

```
signal_normalized = (signal - min(signal)) / (max(signal) - min(signal))
```

This ensures consistent scale across different signal amplitudes and improves training stability.

#### 2.2.3 Data Augmentation

During training, the following augmentations are applied to improve generalization:

- **Random Noise Injection**: Gaussian noise with standard deviation 0.01

- **Random Scaling**: Uniform scaling factor between 0.95 and 1.05

These augmentations are applied only during training, not during validation or testing.

#### 2.2.4 Data Splitting

- **Training Set**: 60% of data

- **Validation Set**: 20% of data  

- **Test Set**: 20% of data

### 2.3 Training Procedure

#### 2.3.1 Loss Function

The model uses a **weighted Huber loss** combining HR and RR prediction errors:

```
Loss = HR_weight × HuberLoss(HR_pred, HR_true) + RR_weight × HuberLoss(RR_pred, RR_true)
```

**Parameters**:

- HR_weight: 1.0

- RR_weight: 1.6 (higher weight due to RR being more challenging to predict)

- Huber loss delta: 1.0

**Rationale**: 

- Huber loss is more robust to outliers than MSE

- Higher weight for RR compensates for its typically higher prediction error

- Separate loss terms allow independent optimization of HR and RR predictions

#### 2.3.2 Optimizer

- **Algorithm**: Adam optimizer

- **Learning Rate**: 0.001

- **Weight Decay**: 1e-5 (L2 regularization)

#### 2.3.3 Learning Rate Scheduling

- **Scheduler**: ReduceLROnPlateau

- **Mode**: Minimize validation loss

- **Factor**: 0.5 (halve learning rate)

- **Patience**: 7 epochs

#### 2.3.4 Training Configuration

- **Batch Size**: 64

- **Number of Epochs**: 100

- **Early Stopping**: Model checkpoint saved based on best validation loss

- **Device**: CUDA (if available), otherwise CPU

- **Random Seeds**: Set to 42 for reproducibility (PyTorch, NumPy, Python random)

#### 2.3.5 Training Progress

The training process shows stable convergence with decreasing loss values over epochs. The following figure illustrates the training and validation loss curves:

![Training Curves](results/20251201_191821/training_curves.png)

*Figure 1: Training and validation loss curves over 100 epochs. The model demonstrates stable convergence with both training and validation losses decreasing consistently, indicating good generalization without overfitting.*

---

## 3. Results

### 3.1 Model Performance

The model was evaluated on a test set of **19,200 samples**. The following results were achieved:

| Metric   | HR       | RR       |
| -------- | -------- | -------- |
| **MAE**  | 0.63 bpm | 0.15 bpm |
| **RMSE** | 1.36 bpm | 0.32 bpm |
| **STD**  | 1.35 bpm | 0.32 bpm |
| **MARE** | 2.00%    | 1.99%    |

### 3.2 Evaluation Metrics

The model is evaluated using four comprehensive metrics:

1. **Mean Absolute Error (MAE)**:
   
   ```
   MAE = mean(|y_pred - y_true|)
   ```
   
   Measures average prediction error magnitude.

2. **Root Mean Squared Error (RMSE)**:
   
   ```
   RMSE = sqrt(mean((y_pred - y_true)²))
   ```
   
   Penalizes larger errors more heavily.

3. **Standard Deviation (STD)**:
   
   ```
   STD = std(y_pred - y_true)
   ```
   
   Measures the spread of prediction errors.

4. **Mean Absolute Relative Error (MARE)**:
   
   ```
   MARE = mean(|y_pred - y_true| / y_true)
   ```
   
   Provides relative error assessment, useful for comparing performance across different value ranges.

### 3.3 Visualization

The evaluation includes comprehensive visualizations:

#### 3.3.1 Trend Plots

Trend plots show predictions vs. labels sorted by label values, with density-based coloring. These plots provide visual assessment of prediction accuracy across the entire value range.

**Heart Rate (HR) Trend Plot:**

![HR Trend Plot](results/20251201_191821/evaluation/Fig/Trend_Plot_HR.png)

*Figure 2: Trend plot for Heart Rate predictions. Predictions (blue) closely follow true labels (red), demonstrating excellent model performance across the entire HR range.*

**Respiratory Rate (RR) Trend Plot:**

![RR Trend Plot](results/20251201_191821/evaluation/Fig/Trend_Plot_RR.png)

*Figure 3: Trend plot for Respiratory Rate predictions. The model shows consistent accuracy across the entire RR range with high correlation between predictions and true values.*

#### 3.3.2 Bland-Altman Plots

Bland-Altman plots visualize agreement between predictions and labels, showing mean difference and ±1.96 SD limits of agreement. These plots are essential for assessing clinical acceptability of the predictions.

**Heart Rate (HR) Bland-Altman Plot:**

![HR Bland-Altman Plot](results/20251201_191821/evaluation/Fig/Bland_Altman_Plot_HR.png)

*Figure 4: Bland-Altman plot for Heart Rate predictions. The mean difference and limits of agreement demonstrate excellent agreement between predictions and true values.*

**Respiratory Rate (RR) Bland-Altman Plot:**

![RR Bland-Altman Plot](results/20251201_191821/evaluation/Fig/Bland_Altman_Plot_RR.png)

*Figure 5: Bland-Altman plot for Respiratory Rate predictions. The tight limits of agreement indicate high precision in RR estimation.*

These visualizations are saved in the `evaluation/Fig/` directory within each results folder and provide comprehensive assessment of model performance.

### 3.4 Model Comparison

The hybrid architecture (CNN + DSP features) demonstrates significant advantages:

- **DSP Features**: Provide domain knowledge about signal characteristics

- **CNN Features**: Learn complex temporal patterns from data

- **Separate Heads**: Allow task-specific optimization for HR and RR

- **Robust Training**: Weighted Huber loss handles outliers effectively

### 3.5 Grafana Visualization Platform

In this project, we design a Grafana board to visualize data from InfluxDB, which is received from the MQTT broker. The compoment is given from the template, and the the data pipeline is shown with detail in the Homework 1.

![grafana](Fig/image.png)

*Figure 6: Grafana data visualization platform*

---

## 4. Implementation Details

### 4.1 Framework, Sofware, and Hardware

#### Framework

- **Deep Learning Framework**: PyTorch

- **Signal Processing**: SciPy

- **Data Handling**: NumPy

- **Visualization**: Matplotlib

- **Machine Learning Utilities**: scikit-learn

#### Software Environment

- **OS**: Ubuntu 22.04

- **Python**: 3.10

- **CUDA**: 12.8

- **Pytorch**: 2.9.1

#### Hardware Environment

- **CPU**: AMD Ryzen7 - 9600x

- **GPU**: Nvidia 5070 Ti

- **MEM**: Adie c28 DDR5 32G x 2

- **Board**: MSI B650i



### 4.2 Code Structure

```
data/
├── ppg_hr_rr_model.py          # Main model definition and training script
├── evaluate_model.py           # Model evaluation script
├── plot.py                     # Visualization functions (trend plots, Bland-Altman)
├── inference_example.py        # Example inference script
├── mqtt_publisher_model.py     # MQTT integration for real-time prediction
└── results/                    # Training results directory
    └── YYYYMMDD_HHMMSS/        # Timestamped result folders
        ├── best_model.pth      # Trained model weights
        ├── training_config.json
        ├── training_curves.png
        ├── losses.json
        └── evaluation/
            ├── evaluation_results.json
            └── Fig/
                ├── Trend_Plot_HR.png
                ├── Trend_Plot_RR.png
                ├── Bland_Altman_Plot_HR.png
                └── Bland_Altman_Plot_RR.png
```

### 4.3 Model Usage

#### Training

```bash
python ppg_hr_rr_model.py
```

#### Evaluation

```bash
python evaluate_model.py --model_path results/YYYYMMDD_HHMMSS/best_model.pth
```

#### Inference

```python
from ppg_hr_rr_model import HRRRPredictor, load_model
import torch

# Load model
model = load_model('path/to/best_model.pth', device='cpu')

# Prepare signal (1000 points, normalized)
signal = torch.FloatTensor(your_signal).unsqueeze(0)

# Predict
model.eval()
with torch.no_grad():
    output = model(signal)
    hr_pred = output[0, 0].item()
    rr_pred = output[0, 1].item()
```

---

## 5. Key Features and Innovations

### 5.1 Hybrid Architecture

The combination of DSP feature extraction and CNN learning provides:

- **Domain Knowledge**: Handcrafted features capture known signal characteristics

- **Learned Patterns**: CNN discovers complex temporal relationships

- **Complementary Information**: Both feature types contribute to final predictions

### 5.2 DSP Feature Extraction

The DSP feature extractor is:

- **Fully Differentiable**: Uses PyTorch operations, enabling end-to-end training

- **Device-Aware**: Automatically handles CPU/CUDA device placement

- **Efficient**: Computes 18 features in a single forward pass

- **Raw Signal Processing**: Extracts features from unfiltered input signals

### 5.3 Separate Prediction Heads

Task-specific heads allow:

- **Independent Optimization**: HR and RR can be optimized separately

- **Specialized Representations**: Each head learns features relevant to its task

- **Flexible Weighting**: Different loss weights for HR and RR

### 5.4 Robust Training Strategy

- **Weighted Loss**: Higher weight for RR compensates for prediction difficulty

- **Huber Loss**: Robust to outliers in training data

- **Data Augmentation**: Improves generalization to unseen signal variations

- **Learning Rate Scheduling**: Adaptive learning rate based on validation performance

---

## 6. Limitations

1. **Signal Quality Dependency**: Performance may degrade with poor signal quality or motion artifacts
2. **Improve the limbo value of RR**: The model can still improve the performance for the extremely low and high RR values.
3. **Fixed Segment Length**: Model is trained on 10-second segments; performance may vary for different lengths
4. **Subject Variability**: Model performance may vary across different individuals or physiological conditions
5. **Dataset Specificity**: Model is trained on specific datasets; generalization to other datasets may require fine-tuning

## 7. Conclusions

This project successfully developed a deep learning system for simultaneous HR and RR prediction from PPG signals. The hybrid architecture combining DSP feature extraction with CNN learning achieved excellent performance:

- **HR Prediction**: MAE of 0.63 bpm, demonstrating high accuracy suitable for clinical applications

- **RR Prediction**: MAE of 0.15 bpm, showing exceptional performance for respiratory rate estimation

- **Model Efficiency**: ~182K parameters, suitable for deployment on edge devices

- **Comprehensive Evaluation**: Multiple metrics and visualizations provide thorough performance assessment

The integration of DSP features with CNN learning represents a novel approach that leverages both domain knowledge and data-driven learning. The separate prediction heads and weighted loss function enable effective multi-task learning for HR and RR prediction.

The system is ready for deployment in wearable devices, clinical monitoring systems, and research applications requiring accurate HR and RR estimation from PPG signals.

---

## 8. References

[1] Allen, J. (2007). Photoplethysmography and its application in clinical physiological measurement. Physiological measurement, 28(3), R1.

[2] Charlton, P. H., Birrenkott, D. A., Bonnici, T., Pimentel, M. A., Johnson, A. E., Alastruey, J., ... & Clifton, D. A. (2017). Breathing rate estimation from the electrocardiogram and photoplethysmogram: A review. IEEE reviews in biomedical engineering, 11, 2-20.

[3] Elgendi, M. (2012). On the analysis of fingertip photoplethysmogram signals. Current cardiology reviews, 8(1), 14-25.
[4] Schäfer, A., & Kratky, K. W. (2008). Estimation of breathing rate from respiratory sinus arrhythmia: comparison of various methods. Annals of Biomedical Engineering, 36(3), 476-485.

[5] Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). Deep PPG: Large-scale heart rate estimation with convolutional neural networks. Sensors, 19(14), 3079.

[6] Slapničar, G., Mlakar, N., & Luštrek, M. (2019). Blood pressure estimation from photoplethysmogram using a spectro-temporal deep neural network. Sensors, 19(15), 3420.

[7] Rodriguez-Labra, J. I., Kosik, C., Maddipatla, D., Narakathu, B. B., & Atashbar, M. Z. (2021). Development of a PPG sensor array as a wearable device for monitoring cardiovascular metrics. IEEE Sensors Journal, 21(23), 26320-26327.

---

## Appendix: Model Configuration

### Training Configuration (Example)

```json
{
  "timestamp": "20251201_191821",
  "batch_size": 64,
  "num_epochs": 100,
  "learning_rate": 0.001,
  "test_size": 0.2,
  "val_size": 0.2,
  "apply_filter": true,
  "sampling_rate": 100,
  "low_cutoff": 0.5,
  "high_cutoff": 8.0,
  "filter_order": 4,
  "hr_weight": 1.0,
  "rr_weight": 1.6,
  "use_huber_loss": true,
  "use_separate_heads": true,
  "use_dsp_features": true,
  "total_params": 182466,
  "trainable_params": 182466,
  "device": "cuda"
}
```

### Model Architecture Summary

- **Input**: 1000-dimensional cardiorespiratory sensor data

- **DSP Features**: 18 features

- **CNN Features**: 256 features (after global pooling)

- **Combined Features**: 274 features

- **Shared FC**: 274 → 128 → 64

- **HR Head**: 64 → 32 → 1

- **RR Head**: 64 → 32 → 1

- **Output**: [HR, RR] in bpm

---
