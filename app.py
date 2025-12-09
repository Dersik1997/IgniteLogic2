import streamlit as st
import paho.mqtt.client as mqtt
import json
import threading

# -----------------------------
# Variabel global
# -----------------------------
latest_data = {
    "light": None,
    "suhu": None,
    "lembap": None,
    "status": None
}

# -----------------------------
# MQTT CALLBACK
# -----------------------------
def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe("Iot/IgniteLogic/sensor")

def on_message(client, userdata, msg):
    global latest_data
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)

        latest_data["light"] = data.get("light")
        latest_data["suhu"] = data.get("suhu")
        latest_data["lembap"] = data.get("lembap")
        latest_data["status"] = data.get("status")

        print("DATA MASUK:", latest_data)

    except Exception as e:
        print("Error:", e)

# -----------------------------
# MQTT CLIENT
# -----------------------------
def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    # GANTI DENGAN BROKER KAMU
    client.connect("broker.emqx.io", 1883, 60)

    client.loop_forever()

# -----------------------------
# JALANKAN MQTT DI BACKGROUND
# -----------------------------
threading.Thread(target=mqtt_thread, daemon=True).start()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Dashboard Sensor IoT - IgniteLogic")

sensor_box = st.empty()

while True:
    sensor_box.markdown(f"""
    ### 🔎 Data Sensor Terbaru  
    **Cahaya (light):** `{latest_data['light']}`  
    **Suhu:** `{latest_data['suhu']} °C`  
    **Kelembapan:** `{latest_data['lembap']} %`  
    **Status:** `{latest_data['status']}`  
    """)

    st.sleep(1)
