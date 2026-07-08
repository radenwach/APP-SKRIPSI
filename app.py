import streamlit as st
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import time
import pandas as pd
import plotly.express as px
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="APLIKASI SKRIPSI", page_icon="🔍", layout="wide")

MODEL_PATH = 'model_dioptimasi.tflite'

# --- LOAD MODEL TFLITE ---
@st.cache_resource
def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan di {os.getcwd()}! Pastikan file ada di repositori.")
        return None
    
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Inisialisasi model
interpreter = load_tflite_model(MODEL_PATH)

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# --- FUNGSI PREDIKSI TFLITE ---
def predict(img, interpreter):
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { border-radius: 5px; height: 3em; background-color: #007BFF; color: white; font-weight: bold; }
    .result-text { font-size: 24px; font-weight: bold; text-align: center; }
    [data-testid="stFileUploadDropzone"] { border: 2px dashed #007BFF; background-color: #f0f8ff; border-radius: 15px; padding: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'menu' not in st.session_state: st.session_state.menu = 'Beranda'

with st.sidebar:
    st.title("Navigasi Sistem")
    if st.button("BERANDA", use_container_width=True): st.session_state.menu = "Beranda"
    if st.button("PREDIKSI", use_container_width=True): st.session_state.menu = "Prediksi"
    if st.button("STATISTIK", use_container_width=True): st.session_state.menu = "Statistik"
    if st.button("TENTANG", use_container_width=True): st.session_state.menu = "Tentang"

# --- LOGIKA APLIKASI ---
menu = st.session_state.menu

if menu == "Beranda":
    st.title("Sistem Deteksi AI-Generated Portrait")
    st.write("Aplikasi untuk mendeteksi potret asli vs deepfake menggunakan model EfficientNetB0.")
    if not interpreter: st.warning("Model belum dimuat dengan benar.")

elif menu == "Prediksi":
    st.header("Deteksi Citra")
    if not interpreter:
        st.error("Sistem tidak dapat melakukan prediksi karena model tidak ditemukan.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Unggah Potret (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file: st.image(Image.open(uploaded_file), width=230)
        with col2:
            if uploaded_file and st.button("Mulai Prediksi", use_container_width=True):
                with st.spinner('Menganalisis...'):
                    start = time.time()
                    img = Image.open(uploaded_file)
                    result = predict(img, interpreter)
                    waktu = time.time() - start
                    
                    prob_real = float(result[0][0])
                    # Sesuaikan threshold jika hasil prediksi Anda tidak masuk akal
                    label = "ASLI (REAL)" if prob_real > 0.5 else "AI (FAKE)"
                    color = "#155724" if label == "ASLI (REAL)" else "#FF0000"
                    bg = "#d4edda" if label == "ASLI (REAL)" else "#f8d7da"
                    
                    st.markdown(f"<div style='background-color:{bg}; padding:15px; border-radius:8px;'><div class='result-text' style='color:{color};'>PREDIKSI: {label}</div></div>", unsafe_allow_html=True)
                    st.info(f"Waktu Inferensi: {waktu:.2f} detik")

elif menu == "Statistik":
    st.header("Dasbor Statistik")
    st.write("Fitur statistik dalam pengembangan.")

elif menu == "Tentang":
    st.header("Tentang Penelitian")
    st.write("**Peneliti:** R. Muhammad Wachyu Fajar Sidik")
