# app.py - IoT ML Realtime Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import queue
import threading
from datetime import datetime, timezone, timedelta
import plotly.graph_objs as go
import paho.mqtt.client as mqtt
import joblib

# --------------------------------
# CONFIGURATIONS
# --------------------------------
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
TOPIC_SENSOR = "Iot/IgniteLogic/sensor"
TOPIC_OUTPUT = "Iot/IgniteLogic/output"
MODEL_PATH = "model.pkl"

TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

GLOBAL_MQ = queue.Queue()

st.set_page_config(page_title="IgniteLogic ML Dashboard", layout="wide")
st.title("🔥 IgniteLogic IoT + Machine Learning Dashboard")

# --------------------------------
# SESSION STATE
# --------------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

if "ml_model" not in st.session_state:
    st.session_state.ml_model = joblib.load(MODEL_PATH)


# --------------------------------
# MQTT CALLBACKS
# --------------------------------
def _on_connect(client, userdata, flags, rc):
    client.subscribe(TOPIC_SENSOR)
    GLOBAL_MQ.put({"_type": "status", "connected": (rc == 0)})

def _on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        GLOBAL_MQ.put({"_type": "sensor", "data": payload})
    except:
        pass


# --------------------------------
# MQTT THREAD
# --------------------------------
def start_mqtt_thread_once():
    def worker():
        client = mqtt.Client()
        client.on_connect = _on_connect
        client.on_message = _on_message

        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, 60)
                client.loop_forever()
            except Exception as e:
                GLOBAL_MQ.put({"_type": "error", "msg": str(e)})
                time.sleep(3)

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        st.session_state.mqtt_thread_started = True

start_mqtt_thread_once()


# --------------------------------
# ML PREDICT HELPER
# --------------------------------
def predict_status(light, temp, hum):
    model = st.session_state.ml_model
    X = [[light, temp, hum]]

    try:
        label = model.predict(X)[0]
    except:
        label = "ERR"

    if hasattr(model, "predict_proba"):
        try:
            conf = float(np.max(model.predict_proba(X)))
        except:
            conf = None
    else:
        conf = None

    return label, conf


# --------------------------------
# PROCESS QUEUE
# --------------------------------
def process_queue():
    q = GLOBAL_MQ

    while not q.empty():
        item = q.get()
        t = now_str()

        if item["_type"] == "sensor":
            d = item["data"]

            light = float(d.get("light", 0))
            temp = float(d.get("temperature", 0))
            hum = float(d.get("humidity", 0))

            # ML Prediction
            pred, conf = predict_status(light, temp, hum)

            # Publish output to ESP32
            out_msg = "AMAN" if pred == "Aman" else "TIDAK_AMAN"
            try:
                pub = mqtt.Client()
                pub.connect(MQTT_BROKER, MQTT_PORT, 60)
                pub.publish(TOPIC_OUTPUT, out_msg)
                pub.disconnect()
            except:
                pass

            row = {
                "ts": t,
                "light": light,
                "temp": temp,
                "hum": hum,
                "pred": pred,
                "conf": conf,
            }

            st.session_state.last = row
            st.session_state.logs.append(row)

            # Limit logs to 2000
            if len(st.session_state.logs) > 2000:
                st.session_state.logs = st.session_state.logs[-2000:]

process_queue()


# --------------------------------
# UI SECTION
# --------------------------------
left, right = st.columns([1, 2])

# LEFT PANEL
with left:
    st.header("📡 Last Sensor Reading")

    if st.session_state.last:
        last = st.session_state.last

        st.write(f"⏱ Time: {last['ts']}")
        st.write(f"💡 Light: {last['light']}")
        st.write(f"🌡 Temperature: {last['temp']} °C")
        st.write(f"💧 Humidity: {last['hum']} %")

        st.markdown("---")
        st.subheader("ML Prediction")

        if last["pred"] == "Aman":
            st.success(f"🟢 Status: {last['pred']}")
        else:
            st.error(f"🔴 Status: {last['pred']}")

        st.write(f"Confidence: {last['conf']}")
    else:
        st.info("Menunggu data dari ESP32...")

# RIGHT PANEL
with right:
    st.header("📊 Live Chart")

    df = pd.DataFrame(st.session_state.logs[-200:])

    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], mode="lines", name="Temp"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], mode="lines", name="Humidity"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["light"], mode="lines", name="Light"))

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Logs")
    if not df.empty:
        st.dataframe(df[::-1])
