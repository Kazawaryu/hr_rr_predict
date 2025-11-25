import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.signal import hilbert

data = np.load("/home/ghosn/Project/csee8300_3/data/dataset_constant_ibi_constant_wa.npy")

signal = data[0, :1000]
fs = 100


b1, a1 = butter(2, [0.8/(fs/2), 3.0/(fs/2)], btype='band')
heart_sig = filtfilt(b1, a1, signal)
peaks, _ = find_peaks(heart_sig, distance=fs*0.4)
heart_rate = len(peaks) * (60 / 10)
print(f"Heart Rate ≈ {heart_rate:.1f} bpm")

analytic = hilbert(signal)
envelope = np.abs(analytic)
b2, a2 = butter(2, [0.1/(fs/2), 0.5/(fs/2)], btype='band')
resp_sig = filtfilt(b2, a2, envelope)
rpeaks, _ = find_peaks(resp_sig, distance=fs*2)
resp_rate = len(rpeaks) * (60 / 10)
print(f"Respiration Rate ≈ {resp_rate:.1f} rpm")


plt.figure(figsize=(10,4))
plt.plot(signal, label="Raw Signal")
plt.plot(peaks, heart_sig[peaks], "r.", label="Heart Peaks")
plt.legend()
plt.show()