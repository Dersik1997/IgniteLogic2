import streamlit as st
import pandas as pd
import numpy as np
import joblib # Digunakan untuk memuat model scikit-learn
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
MODEL_PATH = "model.pkl" # Pastikan file model scikit-learn ada di folder yang sama

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

st.set_page_config(page_title="IoT Realtime Dashboard (ML Server)", layout="wide")
st.title("💡 Dashboard Monitoring Lingkungan Realtime (Prediksi scikit-learn)")

# ---------------------------
# session_state init
# ---------------------------
if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = GLOBAL_MQ

if "logs" not in st.session_state:
    # Menambahkan kolom 'prediksi_server' ke logs
    st.session_state.logs = [] 

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False
    
# Inisialisasi/Muat Model scikit-learn
if "ml_model" not in st.session_state:
    try:
        # Coba muat model yang sudah dilatih (pkl file)
        st.session_state.ml_model = joblib.load(MODEL_PATH)
        st.info(f"Model scikit-learn ({MODEL_PATH}) berhasil dimuat.")
    except FileNotFoundError:
        st.session_state.ml_model = None
        st.error(f"File model ML tidak ditemukan di: {MODEL_PATH}. Prediksi server dinonaktifkan.")
    except Exception as e:
        st.session_state.ml_model = None
        st.error(f"Error memuat model ML: {e}")

# ---------------------------
# MQTT callbacks (use GLOBAL_MQ, NOT st.session_state inside callbacks)
# (Tidak diubah dari kode Anda sebelumnya)
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
# Start MQTT thread (worker) (Tidak diubah)
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
                time.sleep(5) 

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True, name="mqtt_worker")
        t.start()
        st.session_state.mqtt_thread_started = True
        time.sleep(0.05)

# start thread
start_mqtt_thread_once()

# --- Helper function for status color ---
def get_status_color(status):
    if "Aman" in status:
        return "green"
    elif "Waspada" in status or "Cek" in status:
        return "orange"
    else:
        return "red"
# ----------------------------------------


# ---------------------------
# Drain queue (process incoming msgs) - LOGIKA PREDIKSI BARU DI SINI
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
            pass

        elif ttype == "sensor":
            d = item.get("data", {})
            
            try:
                suhu = float(d.get("suhu"))
            except Exception:
                suhu = None
            try:
                lembap = float(d.get("lembap"))
            except Exception:
                lembap = None
            try:
                # Menggunakan light (yang sudah dibalik) jika model ML dilatih dengan nilai terbalik
                # Jika model ML dilatih dengan rawLight, gunakan nilai mentah.
                # Kita akan asumsikan model ML dilatih dengan 3 fitur: suhu, lembap, light (terbalik)
                light = int(d.get("light")) 
            except Exception:
                light = None
            
            # Mendapatkan label dari ESP32
            status_esp = d.get("label", "N/A") 
            
            row = {
                "ts": datetime.fromtimestamp(item.get("ts", time.time()), TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "suhu": suhu,
                "lembap": lembap,
                "light": light,
                "status_esp": status_esp, 
                "prediksi_server": "N/A" # Default
            }
            
            # =========================================================
            # START: LOGIKA PREDIKSI SCKIT-LEARN (SERVER)
            # =========================================================
            if st.session_state.ml_model and suhu is not None and lembap is not None and light is not None:
                try:
                    # Model scikit-learn memerlukan input fitur dalam bentuk array 2D
                    # PENTING: Urutan fitur (suhu, lembap, light) HARUS sama dengan urutan saat model dilatih
                    fitur_input = np.array([[suhu, lembap, light]])
                    
                    # Lakukan prediksi (Klasifikasi atau Regresi)
                    prediksi_server = st.session_state.ml_model.predict(fitur_input)[0]
                    
                    # Tambahkan hasil prediksi server ke baris data
                    row["prediksi_server"] = str(prediksi_server)
                    
                except Exception as e:
                    row["prediksi_server"] = "Prediksi Error"
                    # st.error(f"Error prediksi scikit-learn: {e}") # Nonaktifkan agar UI tidak terlalu ramai
            # =========================================================
            # END: LOGIKA PREDIKSI SCKIT-LEARN
            # =========================================================

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
if HAS_AUTOREFRESH:
    st_autorefresh(interval=2000, limit=None, key="autorefresh") 

left, right = st.columns([1, 2])

with left:
    st.header("Connection Status")
    # ... (Bagian Connection Status tidak diubah)
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
        st.markdown("---")

        st.markdown("### Status ESP32 (Edge)")
        status_text = last.get('status_esp', 'N/A')
        status_color = get_status_color(status_text)
        st.markdown(f"**<p style='font-size: 24px; color: {status_color};'>● {status_text}</p>**", unsafe_allow_html=True)
        
        st.markdown("### Prediksi Server (scikit-learn)")
        pred_text = last.get('prediksi_server', 'N/A')
        pred_color = get_status_color(pred_text)
        st.markdown(f"**<p style='font-size: 24px; color: {pred_color};'>● {pred_text}</p>**", unsafe_allow_html=True)

    else:
        st.info("Waiting for data...")

    st.markdown("---")
    # ... (Bagian Manual Output Control dan Download Logs tidak diubah, namun disingkat di sini)
    
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
    # ... (Bagian Live Chart tidak diubah, namun disingkat di sini)
    df_plot = pd.DataFrame(st.session_state.logs[-200:])
    
    if (not df_plot.empty) and {"suhu", "lembap", "light"}.issubset(df_plot.columns):
        # ... (Logika Plotly Chart sama)
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
        
        # color markers by status ESP32 (menggunakan Status Edge)
        colors = []
        for _, r in df_plot.iterrows():
            stat = r.get("status_esp", "")
            colors.append(get_status_color(stat))
            
        fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="lines+markers", name="Suhu (°C)"))
        fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="lines+markers", name="Lembap (%)"))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Make sure ESP32 publishes to correct topic.")


    st.markdown("### Recent Logs")
    if st.session_state.logs:
        # Menampilkan kolom prediksi_server
        df_display = pd.DataFrame(st.session_state.logs)[["ts", "light", "suhu", "lembap", "status_esp", "prediksi_server"]].rename(columns={
            "status_esp": "Status ESP32 (Edge)",
            "prediksi_server": "Prediksi Server (ML)"
        })
        st.dataframe(df_display[::-1].head(100), use_container_width=True)
    else:
        st.write("—")

# after UI render, drain queue (so next rerun shows fresh data)
process_queue()
