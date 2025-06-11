import streamlit as st
import numpy as np
import joblib

# Memuat model yang sudah dilatih untuk penyakit paru-paru
model = joblib.load('river.pkl')

st.title("Prediksi Penyakit Paru-Paru")

# Form input data pasien
with st.form("Form_paru"):
    st.header("Masukkan data pasien:")

    # Contoh input berdasarkan kolom umum dari dataset paru-paru
    # Silakan sesuaikan dengan nama fitur dari dataset kamu
    age = st.number_input('Usia (tahun)', min_value=1, max_value=120)
    gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
    smoking = st.selectbox('Perokok?', ['Ya', 'Tidak'])
    coughing = st.selectbox('Batuk Kronis?', ['Ya', 'Tidak'])
    fatigue = st.selectbox('Kelelahan?', ['Ya', 'Tidak'])
    wheezing = st.selectbox('Mengi?', ['Ya', 'Tidak'])

    # Tombol submit
    submit = st.form_submit_button("Prediksi")

# Saat tombol ditekan
if submit:
    # Konversi input ke bentuk numerik (sesuaikan sesuai model training kamu)
    gender_val = 1 if gender == 'Laki-laki' else 0
    smoking_val = 1 if smoking == 'Ya' else 0
    coughing_val = 1 if coughing == 'Ya' else 0
    fatigue_val = 1 if fatigue == 'Ya' else 0
    wheezing_val = 1 if wheezing == 'Ya' else 0

    # Gabungkan fitur
    features = np.array([[age, gender_val, smoking_val, coughing_val, fatigue_val, wheezing_val]])

    # Lakukan prediksi
    prediction = model.predict(features)[0]

    # Tampilkan hasil
    st.header("Hasil Prediksi")
    if prediction == 1:
        st.error("Hasil: Terindikasi Penyakit Paru-Paru")
    else:
        st.success("Hasil: Tidak Terindikasi Penyakit Paru-Paru")
