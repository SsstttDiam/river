import streamlit as st
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load('river.pkl')

st.title("Prediksi Kategori Pasien Hepatitis")

# Form input
with st.form("form_hepatitis"):
    st.header("Masukkan data pasien:")

    age = st.number_input("Usia", min_value=1, max_value=120)
    sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    alb = st.number_input("Albumin (ALB)")
    alp = st.number_input("Alkaline Phosphatase (ALP)")
    alt = st.number_input("Alanine Aminotransferase (ALT)")
    ast = st.number_input("Aspartate Aminotransferase (AST)")
    bil = st.number_input("Bilirubin (BIL)")
    che = st.number_input("Cholinesterase (CHE)")
    chol = st.number_input("Kolesterol (CHOL)")
    crea = st.number_input("Creatinine (CREA)")
    ggt = st.number_input("GGT")
    prot = st.number_input("Protein Total (PROT)")

    submitted = st.form_submit_button("Prediksi")

# Proses prediksi
if submitted:
    sex_val = 1 if sex == "Laki-laki" else 0
    data = np.array([[age, sex_val, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]])

    prediction = model.predict(data)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    st.subheader("Hasil Prediksi:")
    st.success(f"Kategori Pasien: {label}")
