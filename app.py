import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Misalkan Anda memuat data Anda ke DataFrame
# df = pd.read_csv('your_hepatitis_data.csv')

# Contoh DataFrame dengan 12 fitur (sesuaikan dengan nama kolom Anda)
data = {
    'Age': [30, 45, 55, 60],
    'Sex': ['Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan'],
    'ALB': [3.5, 4.0, 3.2, 4.1],
    'ALP': [100, 120, 90, 130],
    'ALT': [40, 50, 35, 60],
    'AST': [30, 45, 28, 55],
    'BIL': [0.8, 1.2, 0.7, 1.5],
    'CHE': [150, 180, 140, 190],
    'CHOL': [200, 220, 195, 230],
    'CREA': [0.9, 1.1, 0.8, 1.2],
    'GGT': [50, 65, 45, 70],
    'PROT': [7.0, 7.5, 6.8, 7.2],
    'Category': ['Normal', 'Hepatitis A', 'Normal', 'Hepatitis B'] # Target/Label
}
df = pd.DataFrame(data)

# Pra-pemrosesan yang sama seperti di Streamlit
df['Sex_encoded'] = df['Sex'].apply(lambda x: 1 if x == 'Laki-laki' else 0)

# Definisikan fitur (X) dan target (y)
# PASTIKAN SEMUA 12 FITUR INI ADA DI SINI
features = [
    'Age', 'Sex_encoded', 'ALB', 'ALP', 'ALT', 'AST', 'BIL',
    'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'
]
X = df[features]
y = df['Category']

# Encode label target
label_encoder_train = LabelEncoder()
y_encoded = label_encoder_train.fit_transform(y)

# Split data (opsional, tapi disarankan)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Inisialisasi dan latih model (misal: KNeighborsClassifier)
model_trained = KNeighborsClassifier(n_neighbors=5) # Sesuaikan hyperparameter Anda
model_trained.fit(X_train, y_train) # Atau model_trained.fit(X, y_encoded) jika tidak split

# Simpan model dan label_encoder
joblib.dump(model_trained, 'river.pkl')
joblib.dump(label_encoder_train, 'label_encoder.pkl')

print("Model dan label encoder berhasil disimpan.")
print(f"Model dilatih dengan {X.shape[1]} fitur.") # Ini harusnya 12
