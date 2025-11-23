🚚 Food Delivery Time Prediction
Memprediksi Waktu Pengantaran Makanan Menggunakan Machine Learning

✨ Streamlit App:
👉 https://food-delivery-time-prediction8.streamlit.app/

📌 Overview
Proyek ini bertujuan untuk membangun model Machine Learning yang dapat memprediksi waktu pengantaran makanan (dalam menit) berdasarkan faktor operasional seperti:

Jarak pengantaran

Waktu persiapan makanan

Cuaca

Tingkat lalu lintas

Waktu dalam sehari

Pengalaman kurir

Jenis kendaraan

Model ini ditujukan untuk meningkatkan akurasi estimasi waktu pengiriman, meningkatkan kepuasan pelanggan, dan membantu perusahaan logistik/food delivery dalam optimalisasi operasional.

🧠 Machine Learning Model
Beberapa model diuji untuk menemukan model terbaik:

Linear Regression (Model Terbaik)

Random Forest Regressor

XGBoost Regressor

Ridge & Lasso Regression (perbandingan)

Setelah evaluasi metrik dan cross-validation, Linear Regression dipilih sebagai model final karena:

Performa terbaik (MAE, RMSE, R²)

Interpretasi mudah

Stabil dan konsisten

Cocok untuk kebutuhan ETA operasional

📊 Model Performance (Linear Regression)
Metric	Score
MAE	± 6 menit
RMSE	± 9 menit
R²	± 0.82
📌 Artinya model mampu menjelaskan sekitar 82% variasi waktu pengantaran, dengan error rata-rata sekitar 6 menit.

🗂️ Dataset
Dataset berasal dari Kaggle:
Food Delivery Time Prediction Dataset
Berisi data historis pengantaran lengkap dengan fitur numerik dan kategorikal.

Dataset dimuat menggunakan KaggleHub.

🧹 Data Preparation
Imputasi missing values (median untuk numerik, "Unknown" untuk kategorikal)

One-hot encoding untuk variabel kategorikal

Train-test split (80:20)

Scaling tidak digunakan karena model Linear Regression tetap stabil tanpa scaling

Feature importance dianalisis melalui koefisien model

🧮 Cara Kerja Aplikasi Streamlit
Aplikasi menyediakan 3 halaman:

1. Overview & EDA
Ringkasan dataset

Statistik deskriptif

Distribusi waktu pengantaran

Feature importance (koefisien model)

2. Model Performance
MAE, RMSE, R²

Interpretasi performa model

3. Prediksi Waktu Pengantaran
Pengguna dapat memasukkan:

Jarak

Cuaca

Lalu lintas

Waktu

Pengalaman kurir

Jenis kendaraan

Aplikasi akan memberikan estimasi waktu pengantaran secara realtime.

🚀 Cara Menjalankan Project Secara Lokal
1. Clone Repository
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
2. Install Dependencies
pip install -r requirements.txt
3. Jalankan Streamlit
streamlit run streamlit_app.py
🛠️ Tech Stack
Python 3.9+

Streamlit (UI)

scikit-learn (Machine Learning)

Pandas & NumPy (data handling)

KaggleHub (load dataset)

Matplotlib/Altair (visualisasi)

🌟 Fitur Utama
✔ Prediksi waktu pengantaran real-time
✔ Data cleaning otomatis (imputasi NaN)
✔ Feature importance
✔ Evaluasi model lengkap
✔ Antarmuka Streamlit yang intuitif
✔ Integrasi langsung dataset dari KaggleHub

📬 Contact
Jika ingin kolaborasi, diskusi, atau saran:
Author: Andrianus Alvien
📧 Email: andrianusalvien008@gmail.com

