# subscriber_debug.py

import json
import paho.mqtt.client as mqtt

print("DEBUG — subscriber_debug.py lancé")

BROKER = "localhost"
PORT   = 1883
TOPIC  = "medai/capteurs"

def on_connect(client, userdata, flags, rc):
    print("📶 Connecté au broker MQTT, code de retour =", rc)
    client.subscribe(TOPIC)
    print("✅ Abonné au topic", TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        data = msg.payload.decode()
    print("→ Reçu :", data)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT)
client.loop_forever()
