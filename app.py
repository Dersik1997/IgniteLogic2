# app.py - IgniteLogic IoT ML Dashboard (MQTT Stabil)

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


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
TOPIC_SENSOR = "Iot/IgniteLogic/sensor"
TOPIC_OUTPUT = "Iot/IgniteLogic/output"
MODEL_PATH = "model.pkl"

TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")


# -------------------------------------------------
# GLOBAL QUEUE (untuk MQTT thread)
# -------------------------------------------------
GLOBAL_MQ = queue.Queue()


# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(page_title="IgniteLogic ML IoT Dashboard", layout="wide")
st.title("🔥 IgniteLogic IoT + Machine Learning Dashboard (Stable Version)")


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None

# Load ML model
if "ml_model" not in st.session_state:
    try:
        st.session_state.ml_model = joblib.load(MODEL_PATH)
        st.success(f"Model loaded: {MODEL_PATH}")
    except:
        st.session_state.ml_model = None
        st.error("Model gagal dimuat! Pastikan model.pkl ada di root folder.")


# -------------------------------------------------
# MQTT CALLBACKS
# -------------------------------------------------
def _on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)
        GLOBAL_MQ.put({"_type": "status", "msg": "Connected"})
    else:
        GLOBAL_MQ.put({"_type": "status", "msg": "Failed"})


def _on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        GLOBAL_MQ.put({"_type": "sensor", "data": data})
    except:
        GLOBAL_MQ.put({"_type": "raw", "payload": msg.payload.decode()})


# -------------------------------------------------
# MQTT THREAD — SUPER STABIL
# -------------------------------------------------
def start_mqtt_thread_once():

    def worker():
        client = mqtt.Client()
        client.on_connect = _on_connect
        client.on_message = _on_message

        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, 60)
                st.session_state.mqtt_client = client
                client.loop_forever()  # ← Kunci stabilitas Streamlit Cloud
            except Exception as e:
                GLOBAL_MQ.put({"_type": "error", "msg": str(e)})
                time.sleep(3)

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        st.session_state.mqtt_thread_started = True


start_mqtt_thread_once()


# -------------------------------------------------
# ML PREDICT
# -------------------------------------------------
def predict_status(light, temp, hum):
    model = st.session_state.ml_model
    if model is None:
        return "ERR", None

    X = [[light, temp, hum]]

    try:
        label = model.predict(X)[0]
    except:
        label = "ERR"

    try:
        conf = float(np.max(model.predict_proba(X)))
    except:
        conf = None

    return label, conf


# -------------------------------------------------
# PROCESS INCOMING MQTT
# -------------------------------------------------
def process_queue():
    while not GLOBAL_MQ.empty():
        item = GLOBAL_MQ.get()
        t = now_str()

        if item["_type"] == "sensor":
            d = item["data"]

            light = float(d.get("light", 0))
            temp = float(d.get("temperature", 0))
            hum = float(d.get("humidity", 0))

            pred, conf = predict_status(light, temp, hum)

            # publish hasil ke ESP32
            try:
                client = st.session_state.mqtt_client
                if client is not None:
                    msg = "AMAN" if pred == "Aman" else "TIDAK_AMAN"
                    client.publish(TOPIC_OUTPUT, msg)
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

            # limit log
            if len(st.session_state.logs) > 2000:
                st.session_state.logs = st.session_state.logs[-2000:]


# jalankan sebelum render UI
process_queue()


# -------------------------------------------------
# UI LAYOUT
# -------------------------------------------------
left, right = st.columns([1, 2])

# LEFT SIDE
with left:
    st.header("📡 Sensor Terakhir")

    if st.session_state.last:
        last = st.session_state.last

        st.write(f"⏱ Waktu: {last['ts']}")
        st.write(f"💡 Light: {last['light']}")
        st.write(f"🌡 Temp: {last['temp']} °C")
        st.write(f"💧 Hum: {last['hum']} %")

        st.markdown("---")

        st.subheader("🤖 ML Prediction")

        if last["pred"] == "Aman":
            st.success("🟢 Aman")
        else:
            st.error("🔴 Tidak Aman")

        st.write(f"Confidence: {last['conf']}")
    else:
        st.info("Menunggu data dari ESP32...")

# RIGHT SIDE
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
