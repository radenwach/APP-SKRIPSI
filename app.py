import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import pandas as pd
import plotly.express as px
import os
from tensorflow.keras.preprocessing import image as keras_image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="APLIKASI SKRIPSI",
    page_icon="🔍",
    layout="wide"
)

# --- STATE MANAGEMENT UNTUK MENU NAVIGATION ---
if 'menu' not in st.session_state:
    st.session_state.menu = 'Beranda'

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Mengatur jarak (padding) blok utama agar konten lebih ke atas */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main {
        background-color: #f5f7f9;
    }
    /* Memastikan desain tombol kotak biru tetap konsisten */
    .stButton>button {
        border-radius: 5px;
        height: 3em;
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin: 0;
    }
    
    /* --- GAYA BARU UNTUK BOX UPLOAD --- */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #007BFF; /* Garis putus-putus warna biru */
        background-color: #f0f8ff;  /* Latar belakang biru sangat muda */
        border-radius: 15px;        /* Sudut membulat */
        padding: 20px;
        transition: all 0.3s ease;  /* Efek transisi halus */
    }
    
    /* Efek saat mouse diarahkan ke kotak upload */
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: #e1effe;
        border-color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    # Pastikan file model berada di folder yang sama dengan app.py
    model = tf.keras.models.load_model('Dataset Skripsi Split Camera Smartphone & Profesional V2.keras')
    return model

model = load_my_model()

# --- FUNGSI PREDIKSI ---
def predict(img, model):
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img_resized = img.resize((224, 224), Image.NEAREST)
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    return prediction

# --- SIDEBAR ---
with st.sidebar:
    st.title("Navigasi Sistem")
    
    # Menambahkan parameter use_container_width=True agar tombol memanjang penuh
    if st.button("BERANDA", use_container_width=True):
        st.session_state.menu = "Beranda"
    if st.button("PREDIKSI", use_container_width=True):
        st.session_state.menu = "Prediksi"
    if st.button("STATISTIK", use_container_width=True):
        st.session_state.menu = "Statistik"
    if st.button("TENTANG", use_container_width=True):
        st.session_state.menu = "Tentang"

# Ambil menu yang sedang aktif dari session state
menu = st.session_state.menu

# --- BERANDA ---
if menu == "Beranda":
    # Menambahkan jarak ke bawah khusus untuk halaman Beranda
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    
    # Membuat proporsi 1.5 untuk teks (kiri) dan 1 untuk gambar (kanan)
    col_kiri, col_kanan = st.columns([1.5, 1])
    
    with col_kiri:
        st.title("Sistem Deteksi AI-Generated Portrait")
        st.write("""
        Aplikasi ini digunakan untuk mendeteksi apakah suatu citra merupakan potret manusia asli atau hasil sintesis generasi AI (*deepfake*). 
        Menggunakan model *Convolutional Neural Network* jenis **EfficientNetB0**, sistem menganalisis anomali piksel, tekstur, hingga konteks pencahayaan gambar secara menyeluruh.
        
        Gunakan menu di sidebar untuk mulai melakukan prediksi.
        """)
        st.info("💡 Sistem ini dirancang untuk membaca konteks pencahayaan, tekstur, dan latar belakang potret secara utuh.")

    with col_kanan:
        st.image("ilustrasi.png", use_container_width=True)

# --- PREDIKSI ---
elif menu == "Prediksi":
    st.header("Deteksi Citra")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Unggah Potret Manusia (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_uploaded = Image.open(uploaded_file)
            st.image(image_uploaded, caption="Citra yang Diunggah", width=230)
            
    with col2:
        # --- TAMBAHAN JARAK AGAR HASIL PREDIKSI AGAK KE BAWAH ---
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        
        st.subheader("Hasil Prediksi")
        if uploaded_file is not None:
            if st.button("Mulai Prediksi", use_container_width=True):
                with st.spinner('Menganalisis fitur citra...'):
                    start_time = time.time()
                    result = predict(image_uploaded, model)
                    end_time = time.time()
                    
                    waktu_inferensi = end_time - start_time
                    prob_real = float(result[0][0])
                    prob_fake = 1 - prob_real
                    
                    # --- IMPLEMENTASI THRESHOLD & KOTAK HASIL KUSTOM ---
                    THRESHOLD = 0.7
                    if prob_real > THRESHOLD:
                        hasil_label = "ASLI (REAL)"
                        prob_final = prob_real
                        warna_teks = '#155724' # Hijau tua untuk teks
                        bg_color = '#d4edda'   # Hijau muda untuk latar
                    else:
                        hasil_label = "AI (FAKE)"
                        prob_final = prob_fake
                        warna_teks = '#FF0000' # Merah cerah menyala untuk teks
                        bg_color = '#f8d7da'   # Merah muda untuk latar
                    
                    # Menampilkan kotak hasil dengan font besar dari HTML
                    st.markdown(f"""
                        <div style='background-color: {bg_color}; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid {warna_teks}40;'>
                            <div class='result-text' style='color: {warna_teks};'>
                                PREDIKSI: {hasil_label}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Box info bawaan untuk metrik tambahan
                    st.info(f"Waktu Pemrosesan: {waktu_inferensi:.2f} detik")
                    
                    # --- SIMPAN PERMANEN KE CSV ---
                    file_path = 'riwayat_statistik.csv'
                    df_baru = pd.DataFrame([{
                        "Nama Berkas": uploaded_file.name,
                        "Hasil": hasil_label,
                        "Akurasi (%)": prob_final * 100,
                        "Waktu (s)": waktu_inferensi
                    }])
                    
                    if not os.path.exists(file_path):
                        df_baru.to_csv(file_path, index=False)
                    else:
                        df_baru.to_csv(file_path, mode='a', header=False, index=False)
                        
        else:
            st.info("Silakan unggah potret terlebih dahulu untuk memulai prediksi.")

# --- STATISTIK ---
elif menu == "Statistik":
    st.header("Dasbor Statistik")
    st.write("Panel pemantauan analitik penggunaan sistem secara keseluruhan.")
    
    file_path = 'riwayat_statistik.csv'
    
    if os.path.exists(file_path):
        df_stat = pd.read_csv(file_path)
        
        if len(df_stat) > 0:
            total_data = len(df_stat)
            rata_waktu = df_stat["Waktu (s)"].mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Total Citra Diproses", value=f"{total_data} Citra")
            col2.metric(label="Akurasi Model Yang Digunakan", value="87.0%") 
            col3.metric(label="Rata-rata Waktu Inferensi", value=f"{rata_waktu:.2f} s")
            
            st.markdown("---")
            
            st.subheader("Distribusi Klasifikasi")
            distribusi = df_stat["Hasil"].value_counts().reset_index()
            distribusi.columns = ['Klasifikasi', 'Jumlah']
            
            fig = px.pie(distribusi, values='Jumlah', names='Klasifikasi', hole=0.4, 
                         color_discrete_map={'AI':'#ff9999', 'ASLI':'#66b3ff'})
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Tampilkan Detail Riwayat Prediksi"):
                df_history = df_stat.copy()
                df_display = df_history.drop(columns=["Akurasi (%)", "Tingkat Keyakinan"], errors='ignore')
                df_display.index = df_display.index + 1 
                st.dataframe(df_display, use_container_width=True)
                
                if st.button("Hapus Semua Riwayat", use_container_width=True):
                    os.remove(file_path)
                    st.rerun()
        else:
            st.info("File riwayat ada, tetapi masih kosong.")
    else:
        st.info("Belum ada data statistik. Silakan lakukan deteksi pada potret di menu 'Prediksi' terlebih dahulu.")

# --- TENTANG ---
elif menu == "Tentang":
    st.header("Tentang Penelitian")
    st.write("**Peneliti:** R. Muhammad Wachyu Fajar Sidik")
    st.write("**Instansi:** Universitas PGRI Kediri (UNP Kediri)")
    st.write("**Topik:** Deteksi AI-Generated Portrait menggunakan EfficientNetB0 dengan Optimizer AdamW.")
    st.write("Sistem ini dibangun sebagai perwujudan purwarupa dari penelitian akademis untuk meningkatkan keamanan digital dalam mendeteksi ancaman potret sintesis (*deepfake*).")