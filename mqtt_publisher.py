import paho.mqtt.client as mqtt
import time
import json
import numpy as np

BROKER = '127.0.0.1'
PORT = 1883
TOPIC_Signal = 'signal'
TOPIC_HR_Label = 'hr_label'
TOPIC_RR_Label = 'rr_label'

def publish_data(client, data):
    for i in range(data.shape[0]):
        signal_dict = {"timestamp": float(data[i, -6]), "value": data[i, :1000].tolist()}
        hr_label_dict = {"timestamp": float(data[i, -6]), "value": float(data[i, -5])}
        rr_label_dict = {"timestamp": float(data[i, -6]), "value": float(data[i, -4])}
        client.publish(TOPIC_Signal, json.dumps(signal_dict))
        client.publish(TOPIC_HR_Label, json.dumps(hr_label_dict))
        client.publish(TOPIC_RR_Label, json.dumps(rr_label_dict))
        print(f"Published data for timestamp: {data[i, -6]}")
        time.sleep(1)


if __name__ == "__main__":
    # r7PO-Dav9ST782CdjhCorctTpNwVJvMlQRRwSTpgu66SmvQu3ox50WzUGDQmv4DqMCpmxai2MJkzOiYH28wbsA==
    data = np.load("/home/ghosn/Project/csee8300_3/data/dataset_constant_ibi_constant_wa.npy")
    client = mqtt.Client()
    client.connect(BROKER, PORT, 60)
    publish_data(client, data)