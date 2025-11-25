import paho.mqtt.client as mqtt
import time
import json
import numpy as np
import torch
from ppg_hr_rr_model import HRRRPredictor, filter_ppg_signal


BROKER = '127.0.0.1'
PORT = 1883
TOPIC_Signal = 'signal'
TOPIC_HR_Label = 'hr_label'
TOPIC_RR_Label = 'rr_label'
TOPIC_HR_Pred = 'hr_pred'
TOPIC_RR_Pred = 'rr_pred'

def publish_data(client, data):
    for i in range(data.shape[0]):
        hr_label_dict = { "hr_label": float(data[i, -5])}
        rr_label_dict = { "rr_label": float(data[i, -4])}
        hr, rr = predict_hr_rr(model, data[i, :1000])
        hr_pred_dict = {"hr_pred": float(hr)}
        rr_pred_dict = {"rr_pred": float(rr)}

        client.publish(TOPIC_HR_Label, json.dumps(hr_label_dict))
        client.publish(TOPIC_RR_Label, json.dumps(rr_label_dict))
        client.publish(TOPIC_HR_Pred, json.dumps(hr_pred_dict))
        client.publish(TOPIC_RR_Pred, json.dumps(rr_pred_dict))
        
        print('hr_label: ', hr_label_dict["hr_label"], 'rr_label: ', rr_label_dict["rr_label"])
        print('hr_pred: ', hr_pred_dict["hr_pred"], 'rr_pred: ', rr_pred_dict["rr_pred"])
        print('*'*100)
        time.sleep(0.5)


def predict_hr_rr(model, signal, apply_filter=True, device='cpu'):
    signal = np.array(signal)

    if apply_filter:
        signal = filter_ppg_signal(
            signal, 
            sampling_rate=100,
            lowcut=0.5,
            highcut=8.0,
            order=4
        )
    
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val > 0:
        signal_normalized = (signal - min_val) / (max_val - min_val)
    else:
        signal_normalized = signal
    signal_tensor = torch.FloatTensor(signal_normalized).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(signal_tensor)  # Shape: (1, 2)
        hr = output[0, 0].item()  # Extract HR from first column
        rr = output[0, 1].item()  # Extract RR from second column
    
    return hr, rr

if __name__ == "__main__":
    # r7PO-Dav9ST782CdjhCorctTpNwVJvMlQRRwSTpgu66SmvQu3ox50WzUGDQmv4DqMCpmxai2MJkzOiYH28wbsA==
    data = np.load("/home/ghosn/Project/csee8300_3/data/dataset_constant_ibi_constant_wa.npy")

    device = torch.device('cpu')
    model = HRRRPredictor(input_length=1000, use_separate_heads=True).to(device)
    model.load_state_dict(torch.load('./results/20251122_130937/best_model.pth', map_location=device))
    model.eval()
    
    print(f"Model loaded on device: {device}")
    print("Starting MQTT publisher...")
    
    client = mqtt.Client()
    client.connect(BROKER, PORT, 60)
    publish_data(client, data)