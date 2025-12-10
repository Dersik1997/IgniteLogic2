import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import queue
import threading
from datetime import datetime, timezone, timedelta
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# Optional: lightweight auto-refresh helper
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ---------------------------
# Config (edit if needed)
# ---------------------------
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
TOPIC_SENSOR = "Iot/IgniteLogic/sensor"
TOPIC_OUTPUT = "Iot/IgniteLogic/output" 
MODEL_PATH = "model.pkl" 

# timezone GMT+7 helper
TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# module-level queue used by MQTT thread
# ---------------------------
GLOBAL_MQ = queue.Queue()

# ---------------------------
# Streamlit page setup
# ---------------------------

st.set_page_config(page_title="IoT Realtime Dashboard (Aman/Waspada/Tidak Aman)", layout="wide")
st.title("💡 Dashboard Monitoring Lingkungan Realtime (ESP32) - Tiga Label")

# ---------------------------
# session_state init
# ---------------------------
if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = GLOBAL_MQ

if "logs" not in st.session_state:
    st.session_state.logs = [] # list of dict rows

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

if "ml_model" in st.session_state:
    del st.session_state["ml_model"]
    
# ---------------------------
# MQTT callbacks (use GLOBAL_MQ, NOT st.session_state inside callbacks)
# ---------------------------
def _on_connect(client, userdata, flags, rc):
    try:
        client.subscribe(TOPIC_SENSOR)
        client.subscribe(TOPIC_OUTPUT) 
    except Exception:
        pass
    GLOBAL_MQ.put({"_type": "status", "connected": (rc == 0), "ts": time.time()})

def _on_message(client, userdata, msg):
    payload = msg.payload.decode(errors="ignore")
    
    # Pesan dari TOPIC_OUTPUT (misal: label Aman/Waspada/Tidak Aman)
    if msg.topic == TOPIC_OUTPUT:
        GLOBAL_MQ.put({"_type": "output", "payload": payload, "ts": time.time()})
        return

    try:
        data = json.loads(payload)
    except Exception:
        GLOBAL_MQ.put({"_type": "raw", "payload": payload, "ts": time.time()})
        return

    # push structured sensor message
    GLOBAL_MQ.put({"_type": "sensor", "data": data, "ts": time.time(), "topic": msg.topic})

# ---------------------------
# Start MQTT thread (worker)
# ---------------------------
def start_mqtt_thread_once():
    def worker():
        client = mqtt.Client()
        client.on_connect = _on_connect
        client.on_message = _on_message
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                GLOBAL_MQ.put({"_type": "error", "msg": f"MQTT worker error: {e}", "ts": time.time()})
                time.sleep(5)  # backoff then retry

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True, name="mqtt_worker")
        t.start()
        st.session_state.mqtt_thread_started = True
        time.sleep(0.05)

# start thread
start_mqtt_thread_once()

# ---------------------------
# Drain queue (process incoming msgs)
# ---------------------------
def process_queue():
    updated = False
    q = st.session_state.msg_queue
    while not q.empty():
        item = q.get()
        ttype = item.get("_type")
        
        if ttype == "status":
            st.session_state.last_status = item.get("connected", False)
            updated = True
        
        elif ttype == "error":
            st.error(item.get("msg"))
            updated = True
        
        elif ttype == "output":
            pass # diabaikan karena status diambil dari data 'sensor'

        elif ttype == "sensor":
            d = item.get("data", {})
            
            # --- PENTING: Mengubah kunci JSON dari "status" ke "label" ---
            try:
                suhu = float(d.get("suhu"))
            except Exception:
                suhu = None
            try:
                lembap = float(d.get("lembap"))
            except Exception:
                lembap = None
            try:
                light = int(d.get("light"))
            except Exception:
                light = None
            
            # >>> MENGAMBIL KUNCI "label" DARI KODE ESP32 YANG BARU <<<
            status_esp = d.get("label", "N/A") # Mengambil "label" dari Arduino

            row = {
                "ts": datetime.fromtimestamp(item.get("ts", time.time()), TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "suhu": suhu,
                "lembap": lembap,
                "light": light,
                "status_esp": status_esp # Status dari logika di ESP32 (Aman, Waspada, Tidak Aman)
            }
            
            # --- Nonaktifkan Logika ML/Anomaly (Menggunakan status dari ESP32) ---
            row["pred"] = status_esp # Gunakan status ESP32 sebagai 'prediksi'
            # Kita bisa set conf ke 1.0 untuk menunjukkan kepastian, atau None
            row["conf"] = 1.0 
            row["anomaly"] = ("Tidak Aman" in status_esp) # Menandai Tidak Aman sebagai anomali
            # -----------------------------------------------------------------

            st.session_state.last = row
            st.session_state.logs.append(row)
            
            # keep bounded
            if len(st.session_state.logs) > 5000:
                st.session_state.logs = st.session_state.logs[-5000:]
            updated = True
            
    return updated

# run once here to pick up immediately available messages
_ = process_queue()

# ---------------------------
# UI layout
# ---------------------------
# optionally auto refresh UI; requires streamlit-autorefresh in requirements
if HAS_AUTOREFRESH:
    # Refresh setiap 2 detik, sinkron dengan delay di Arduino
    st_autorefresh(interval=2000, limit=None, key="autorefresh") 

left, right = st.columns([1, 2])

# --- Helper function for status color ---
def get_status_color(status):
    if "Aman" in status:
        return "green"
    elif "Waspada" in status:
        return "orange" # Kuning diwakili dengan orange di HTML/CSS dasar
    else:
        return "red"
# ----------------------------------------


with left:
    st.header("Connection Status")
    st.write("Broker:", f"**{MQTT_BROKER}:{MQTT_PORT}**")
    connected = getattr(st.session_state, "last_status", None)
    st.metric("MQTT Connected", "Yes" if connected else "No")
    st.write("Topic Sensor:", TOPIC_SENSOR)
    st.write("Topic Output:", TOPIC_OUTPUT)
    st.markdown("---")

    st.header("Last Reading")
    if st.session_state.last:
        last = st.session_state.last
        st.write(f"Time: **{last.get('ts')}**")
        st.write(f"Light: **{last.get('light')}** (0-4095)")
        st.write(f"Suhu: **{last.get('suhu')} °C**")
        st.write(f"Lembap: **{last.get('lembap')} %**")
        st.markdown("### Status ESP32")
        
        # LOGIKA WARNA BARU DENGAN 3 KATEGORI
        status_text = last.get('status_esp', 'N/A')
        status_color = get_status_color(status_text)
        
        st.markdown(f"**<p style='font-size: 24px; color: {status_color};'>● {status_text}</p>**", unsafe_allow_html=True)
        
    else:
        st.info("Waiting for data...")

    st.markdown("---")
    st.header("Manual Output Control")
    # Kontrol disesuaikan untuk mengirimkan label yang jelas ke TOPIC_OUTPUT
    col1, col2, col3 = st.columns(3)
    
    if col1.button("Send Aman"):
        try:
            pubc = mqtt.Client()
            pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
            pubc.publish(TOPIC_OUTPUT, "Aman")
            pubc.disconnect()
            st.success("Published Aman")
        except Exception as e:
            st.error(f"Publish failed: {e}")
            
    if col2.button("Send Waspada"):
        try:
            pubc = mqtt.Client()
            pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
            pubc.publish(TOPIC_OUTPUT, "Waspada (Cahaya Masuk)")
            pubc.disconnect()
            st.warning("Published Waspada")
        except Exception as e:
            st.error(f"Publish failed: {e}")
            
    if col3.button("Send Tidak Aman"):
        try:
            pubc = mqtt.Client()
            pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
            pubc.publish(TOPIC_OUTPUT, "Tidak Aman (Suhu/Lembap Tinggi)")
            pubc.disconnect()
            st.error("Published Tidak Aman")
        except Exception as e:
            st.error(f"Publish failed: {e}")

    st.markdown("---")
    st.header("Download Logs")
    if st.button("Download CSV"):
        if st.session_state.logs:
            df_dl = pd.DataFrame(st.session_state.logs)
            csv = df_dl.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV file", data=csv, file_name=f"iot_logs_{int(time.time())}.csv")
        else:
            st.info("No logs to download")

with right:
    st.header("Live Chart (last 200 points)")
    df_plot = pd.DataFrame(st.session_state.logs[-200:])
    
    if (not df_plot.empty) and {"suhu", "lembap", "light"}.issubset(df_plot.columns):
        fig = go.Figure()
        
        # Suhu
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["suhu"], mode="lines+markers", name="Suhu (°C)"))
        
        # Kelembaban
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["lembap"], mode="lines+markers", name="Lembap (%)", yaxis="y2"))
        
        # Cahaya (Light)
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["light"], mode="lines", name="Light (0-4095)", yaxis="y3", opacity=0.3))


        fig.update_layout(
            yaxis=dict(title="Suhu (°C)", side="left"),
            yaxis2=dict(title="Lembap (%)", overlaying="y", side="right", showgrid=False),
            yaxis3=dict(title="Light", overlaying="y", side="right", showgrid=False, range=[0, 4100], anchor="free", position=0.98),
            height=520,
            hovermode="x unified"
        )
        
        # color markers by status ESP32 (3-way logic)
        colors = []
        for _, r in df_plot.iterrows():
            stat = r.get("status_esp", "")
            colors.append(get_status_color(stat)) # Menggunakan fungsi helper
            
        # Terapkan pewarnaan pada Suhu dan Lembap
        fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="lines+markers", name="Suhu (°C)"))
        fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="lines+markers", name="Lembap (%)"))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Make sure ESP32 publishes to correct topic.")

    st.markdown("### Recent Logs")
    if st.session_state.logs:
        # Tampilkan kolom yang relevan
        df_display = pd.DataFrame(st.session_state.logs)[["ts", "light", "suhu", "lembap", "status_esp"]].rename(columns={
            "status_esp": "Status ESP32"
        })
        st.dataframe(df_display[::-1].head(100), use_container_width=True)
    else:
        st.write("—")

# after UI render, drain queue (so next rerun shows fresh data)
process_queue()
