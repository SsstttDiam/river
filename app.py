import streamlit as st
import numpy as np
import joblib

# Load model dan encoder
# Pastikan file 'river.pkl' dan 'label_encoder.pkl' ada di direktori yang sama
# dengan script Streamlit Anda, atau berikan path lengkap ke file tersebut.
model = joblib.load('river.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("Prediksi Kategori Pasien Hepatitis")

# Form input untuk data pasien
with st.form("form_hepatitis"):
    st.header("Masukkan data pasien:")

    # Input untuk setiap fitur
    age = st.number_input("Usia", min_value=1, max_value=120, help="Usia pasien dalam tahun.")
    sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], help="Pilih jenis kelamin pasien.")
    alb = st.number_input("Albumin (ALB)", format="%.2f", help="Nilai Albumin dalam g/L.")
    alp = st.number_input("Alkaline Phosphatase (ALP)", format="%.2f", help="Nilai Alkaline Phosphatase dalam U/L.")
    alt = st.number_input("Alanine Aminotransferase (ALT)", format="%.2f", help="Nilai Alanine Aminotransferase dalam U/L.")
    ast = st.number_input("Aspartate Aminotransferase (AST)", format="%.2f", help="Nilai Aspartate Aminotransferase dalam U/L.")
    bil = st.number_input("Bilirubin (BIL)", format="%.2f", help="Nilai Bilirubin dalam mg/dL.")
    che = st.number_input("Cholinesterase (CHE)", format="%.2f", help="Nilai Cholinesterase dalam U/L.")
    chol = st.number_input("Kolesterol (CHOL)", format="%.2f", help="Nilai Kolesterol dalam mg/dL.")
    crea = st.number_input("Creatinine (CREA)", format="%.2f", help="Nilai Creatinine dalam mg/dL.")
    ggt = st.number_input("GGT", format="%.2f", help="Nilai Gamma-Glutamyl Transferase (GGT) dalam U/L.")
    prot = st.number_input("Protein Total (PROT)", format="%.2f", help="Nilai Protein Total dalam g/L.")

    # Tombol submit form
    submitted = st.form_submit_button("Prediksi Kategori")

# Proses prediksi ketika tombol submit ditekan
if submitted:
    # Konversi jenis kelamin menjadi nilai numerik (Laki-laki=1, Perempuan=0)
    sex_val = 1 if sex == "Laki-laki" else 0
    
    # Kumpulkan semua nilai input ke dalam sebuah list
    input_features = [
        age, sex_val, alb, alp, alt, ast, bil, 
        che, chol, crea, ggt, prot
    ]
    
    # Ubah list menjadi numpy array dan pastikan dimensinya (1 baris, 12 kolom)
    # Ini sangat penting untuk memastikan model menerima input dalam format yang benar.
    data = np.array(input_features).reshape(1, -1) 
    
    try:
        # Lakukan prediksi menggunakan model
        prediction = model.predict(data)[0]
        
        # Ubah hasil prediksi numerik kembali ke label kategori yang mudah dimengerti
        label = label_encoder.inverse_transform([prediction])[0]

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        st.success(f"Kategori Pasien: **{label}**")
        st.write("Catatan: Prediksi ini adalah hasil dari model machine learning dan bukan diagnosis medis.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.write("Pastikan semua input sudah diisi dengan benar dan model/encoder dimuat dengan sukses.")
