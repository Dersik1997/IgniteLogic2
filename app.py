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
TOPIC_OUTPUT = "Iot/IgniteLogic/output" # Digunakan untuk mengirim perintah LED balik ke ESP32
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

st.set_page_config(page_title="IoT Realtime Dashboard (scikit-learn)", layout="wide")
st.title("💡 Dashboard Monitoring Lingkungan Realtime (Prediksi scikit-learn)")
st.caption("ESP32 mengirim data mentah. Server (Streamlit) membuat prediksi ML dan mengirim perintah LED balik.")

# ---------------------------
# session_state init
# ---------------------------
if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = GLOBAL_MQ

if "logs" not in st.session_state:
    st.session_state.logs = [] 

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False
    
# Inisialisasi/Muat Model scikit-learn
if "ml_model" not in st.session_state:
    try:
        st.session_state.ml_model = joblib.load(MODEL_PATH)
        st.info(f"Model scikit-learn ({MODEL_PATH}) berhasil dimuat.")
    except FileNotFoundError:
        st.session_state.ml_model = None
        st.error(f"File model ML tidak ditemukan di: {MODEL_PATH}. Prediksi server dinonaktifkan.")
    except Exception as e:
        st.session_state.ml_model = None
        st.error(f"Error memuat model ML: {e}")

# ---------------------------
# MQTT callbacks
# ---------------------------
def _on_connect(client, userdata, flags, rc):
    try:
        client.subscribe(TOPIC_SENSOR)
    except Exception:
        pass
    GLOBAL_MQ.put({"_type": "status", "connected": (rc == 0), "ts": time.time()})

def _on_message(client, userdata, msg):
    payload = msg.payload.decode(errors="ignore")
    
    try:
        data = json.loads(payload)
    except Exception:
        GLOBAL_MQ.put({"_type": "raw", "payload": payload, "ts": time.time()})
        return

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
    elif "Waspada" in status or "Kuning" in status:
        return "orange"
    elif "Tidak Aman" in status or "Merah" in status:
        return "red"
    else:
        return "gray"
# ----------------------------------------

# ---------------------------
# Drain queue (process incoming msgs) - LOGIKA PREDIKSI & KONTROL DI SINI
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
        
        elif ttype == "sensor":
            d = item.get("data", {})
            
            # Ambil data dari ESP32
            suhu = float(d.get("suhu", np.nan))
            lembap = float(d.get("lembap", np.nan))
            light = int(d.get("light", np.nan)) # Nilai dibalik (4095=Terang)
            rawLight = int(d.get("rawLight", np.nan)) # Nilai mentah (0=Terang)
            
            # Label dari ESP32 hanya sebagai penanda
            status_esp = d.get("label", "N/A") 
            
            row = {
                "ts": datetime.fromtimestamp(item.get("ts", time.time()), TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "suhu": suhu,
                "lembap": lembap,
                "light": light,
                "rawLight": rawLight,
                "status_esp": status_esp, 
                "prediksi_server": "Menunggu Prediksi",
                "perintah_terkirim": "N/A"
            }
            
            # =========================================================
            # START: LOGIKA PREDIKSI SCKIT-LEARN (SERVER)
            # =========================================================
            prediksi_server = "N/A"
            if st.session_state.ml_model and not np.isnan([suhu, lembap, light]).any():
                try:
                    # Input Model: [[suhu, lembap, light (dibalik)]]. PENTING: Gunakan np.float64
                    fitur_input = np.array([[np.float64(suhu), np.float64(lembap), np.float64(light)]]) 
                    
                    # Prediksi
                    prediksi_server = st.session_state.ml_model.predict(fitur_input)[0]
                    
                    # -----------------------------------------------------------------
                    # KIRIM PERINTAH KONTROL BALIK KE ESP32 MELALUI MQTT
                    # -----------------------------------------------------------------
                    perintah_led = ""
                    if "Aman" in prediksi_server:
                        perintah_led = "LED_HIJAU"
                    elif "Waspada" in prediksi_server:
                        perintah_led = "LED_KUNING"
                    else: # Tidak Aman atau label lainnya
                        perintah_led = "LED_MERAH"
                        
                    try:
                        # Klien publish harus dibuat di Streamlit thread
                        pubc = mqtt.Client()
                        pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
                        pubc.publish(TOPIC_OUTPUT, perintah_led) 
                        pubc.disconnect()
                        row["perintah_terkirim"] = perintah_led
                    except Exception as e:
                        row["perintah_terkirim"] = f"ERROR PUBLISH: {e}" 

                except Exception as e:
                    prediksi_server = f"Prediksi Error: {e}"
            
            row["prediksi_server"] = str(prediksi_server)
            # =========================================================
            # END: LOGIKA PREDIKSI SCKIT-LEARN (SERVER)
            # =========================================================

            st.session_state.last = row
            st.session_state.logs.append(row)
            
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
    st.write("Broker:", f"**{MQTT_BROKER}:{MQTT_PORT}**")
    connected = getattr(st.session_state, "last_status", None)
    st.metric("MQTT Connected", "Yes" if connected else "No")
    st.write("Topic Sensor (Input):", TOPIC_SENSOR)
    st.write("Topic Output (Control):", TOPIC_OUTPUT)
    st.markdown("---")

    st.header("Last Reading")
    if st.session_state.last:
        last = st.session_state.last
        st.write(f"Time: **{last.get('ts')}**")
        st.write(f"Suhu: **{last.get('suhu')} °C**")
        st.write(f"Lembap: **{last.get('lembap')} %**")
        st.write(f"Light (Dibalik): **{last.get('light')}**")
        st.write(f"RAW Light: **{last.get('rawLight')}**")
        st.markdown("---")

        st.markdown("### Prediksi Server (scikit-learn)")
        pred_text = last.get('prediksi_server', 'N/A')
        pred_color = get_status_color(pred_text)
        st.markdown(f"**<p style='font-size: 24px; color: {pred_color};'>● {pred_text}</p>**", unsafe_allow_html=True)
        
        st.caption(f"Perintah Terakhir ke ESP32: **{last.get('perintah_terkirim', 'N/A')}**")

    else:
        st.info("Waiting for data...")

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
        
        # Suhu dan Kelembaban (Primary & Secondary Axis)
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["suhu"], mode="lines+markers", name="Suhu (°C)"))
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
        
        # Pewarnaan marker chart berdasarkan Prediksi Server
        colors = []
        for _, r in df_plot.iterrows():
            stat = r.get("prediksi_server", "")
            colors.append(get_status_color(stat))
            
        fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="lines+markers", name="Suhu (°C)"))
        fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="lines+markers", name="Lembap (%)"))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Make sure ESP32 publishes to correct topic.")


    st.markdown("### Recent Logs")
    if st.session_state.logs:
        # Menampilkan kolom prediksi_server
        df_display = pd.DataFrame(st.session_state.logs)[["ts", "suhu", "lembap", "light", "prediksi_server", "perintah_terkirim"]].rename(columns={
            "light": "Light (Dibalik)",
            "prediksi_server": "Prediksi Server (ML)",
            "perintah_terkirim": "Perintah Ke ESP32"
        })
        st.dataframe(df_display[::-1].head(100), use_container_width=True)
    else:
        st.write("—")

process_queue()
