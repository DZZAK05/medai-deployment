# simulateur.py

import time
import json
import random
import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT   = 1883
TOPIC  = "medai/capteurs"

client = mqtt.Client()
client.connect(BROKER, PORT)

try:
    while True:
        payload = json.dumps({
            "timestamp": time.time(),
            "temperature": round(random.normalvariate(37, 0.5), 2),
            "fc": random.randint(60, 100),
            "pa": random.randint(100, 130)
        })
        client.publish(TOPIC, payload)
        print(f"Publié → {payload}")
        time.sleep(1)
except KeyboardInterrupt:
    client.disconnect()
