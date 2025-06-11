import streamlit as st
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load('river.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("Prediksi Kategori Pasien Hepatitis")

st.markdown("*Gunakan **titik (.)** sebagai pemisah desimal, bukan koma (,).*")

# Form input
with st.form("form_hepatitis"):
    st.header("Masukkan data pasien:")

    age = st.number_input("Usia", min_value=1, max_value=120)
    sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    alb = st.number_input("Albumin (ALB)", format="%.2f")
    alp = st.number_input("Alkaline Phosphatase (ALP)", format="%.2f")
    alt = st.number_input("Alanine Aminotransferase (ALT)", format="%.2f")
    ast = st.number_input("Aspartate Aminotransferase (AST)", format="%.2f")
    bil = st.number_input("Bilirubin (BIL)", format="%.2f")
    che = st.number_input("Cholinesterase (CHE)", format="%.2f")
    chol = st.number_input("Kolesterol (CHOL)", format="%.2f")
    crea = st.number_input("Creatinine (CREA)", format="%.2f")
    ggt = st.number_input("GGT", format="%.2f")
    prot = st.number_input("Protein Total (PROT)", format="%.2f")

    submitted = st.form_submit_button("Prediksi")

# Proses prediksi
if submitted:
    sex_val = 1 if sex == "Laki-laki" else 0
    data = np.array([[age, sex_val, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]])

    prediction = model.predict(data)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    st.subheader("Hasil Prediksi:")
    st.success(f"Kategori Pasien: {label}")
