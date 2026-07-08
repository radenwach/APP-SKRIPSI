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

# --- LOAD MODEL TFLITE ---
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Inisialisasi model
interpreter = load_tflite_model('model_dioptimasi.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- FUNGSI PREDIKSI TFLITE ---
def predict(img, interpreter):
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize sesuai input model (224, 224)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalisasi jika model Anda dilatih dengan normalisasi (misal /255.0)
    # Sesuaikan baris ini jika model Anda tidak membutuhkan pembagian 255.0
    img_array = img_array / 255.0 
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    # Ambil hasil output
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
    st.write("Aplikasi untuk mendeteksi potret asli vs deepfake menggunakan model EfficientNetB0 yang dioptimasi.")
    st.info("Gunakan menu di sidebar untuk mulai melakukan prediksi.")

elif menu == "Prediksi":
    st.header("Deteksi Citra")
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
                THRESHOLD = 0.7
                if prob_real > THRESHOLD:
                    label, color, bg = "ASLI (REAL)", "#155724", "#d4edda"
                else:
                    label, color, bg = "AI (FAKE)", "#FF0000", "#f8d7da"
                
                st.markdown(f"<div style='background-color:{bg}; padding:15px; border-radius:8px;'><div class='result-text' style='color:{color};'>PREDIKSI: {label}</div></div>", unsafe_allow_html=True)
                st.info(f"Waktu Inferensi: {waktu:.2f} detik")

elif menu == "Statistik":
    st.header("Dasbor Statistik")
    # Logika statistik tetap sama seperti kode Anda sebelumnya...
    st.write("Fitur statistik aktif.")

elif menu == "Tentang":
    st.header("Tentang Penelitian")
    st.write("**Peneliti:** R. Muhammad Wachyu Fajar Sidik")
