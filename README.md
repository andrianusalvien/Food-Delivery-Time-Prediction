# 🚚 Food Delivery Time Prediction  
### *Machine Learning untuk Prediksi Waktu Pengantaran Makanan*

---

## 📌 1. Overview

Proyek ini membangun model Machine Learning untuk **memprediksi waktu pengantaran makanan (ETA)** berdasarkan data historis.  
Model disajikan dalam bentuk **Streamlit App** yang interaktif dan mudah digunakan.

🔗 **Live App:**  
https://food-delivery-time-prediction8.streamlit.app/

---

## 📂 2. Features

### 📊 **Overview & EDA**
- Statistik dataset  
- Distribusi waktu pengantaran  
- Feature importance  

### 📈 **Model Performance**
Menampilkan:
- MAE  
- RMSE  
- R²  
- Interpretasi performa model  

### 🧮 **Real‑Time Prediction**
Input fitur meliputi:
- Jarak  
- Waktu persiapan  
- Cuaca  
- Lalu lintas  
- Waktu dalam sehari  
- Pengalaman kurir  
- Jenis kendaraan  

Output berupa **estimasi waktu pengantaran (menit)**.

---

## 🧠 3. Machine Learning Models

Model yang diuji:

| Model | Status |
|-------|--------|
| **Linear Regression** | ⭐ Terbaik |
| Random Forest | Pembanding |
| XGBoost | Pembanding |
| Ridge & Lasso | Pembanding |

### 📈 **Final Model Performance (Linear Regression)**

| Metric | Score |
|--------|--------|
| **MAE** | ~6 menit |
| **RMSE** | ~9 menit |
| **R²** | ~0.82 |

---

## 🗂️ 4. Dataset

Dataset berasal dari Kaggle:  
**Food Delivery Time Prediction Dataset**

Dimuat melalui **KaggleHub**.

### **Fitur utama:**
- Distance_km  
- Preparation_Time_min  
- Courier_Experience_yrs  
- Weather  
- Traffic_Level  
- Time_of_Day  
- Vehicle_Type  
- Delivery_Time_min (target)

---

## 🧹 5. Data Preparation

- Menghapus kolom tidak relevan  
- Imputasi missing value  
- One‑Hot Encoding  
- Train‑test split  
- Training model LinearRegression  

---

## 🖥️ 6. Installation & Run Locally

### Clone Repo
```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Streamlit
```bash
streamlit run streamlit_app.py
```

---

## 📦 7. Requirements

```
streamlit
pandas
numpy
scikit-learn
kagglehub==0.3.13
```

---

## 🧰 8. Tech Stack
- Python  
- Streamlit  
- Scikit‑Learn  
- Pandas & NumPy  
- KaggleHub  
- Altair / Matplotlib  

---

## 📁 9. Project Structure
```
📦 Food-Delivery-Time-Prediction
│── streamlit_app.py
│── requirements.txt
│── README.md
└── notebooks/
```

---

## 👨‍💻 10. Author
**Andrianus Alvien**

