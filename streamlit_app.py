import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import kagglehub
from kagglehub import KaggleDatasetAdapter  


# 
#  LOAD DATA DARI KAGGLE
# 

@st.cache_data
def load_data():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "denkuznetz/food-delivery-time-prediction",
        "Food_Delivery_Times.csv",
    )
    return df


# 
#  TRAIN MODEL (dengan imputasi NaN)
# 

@st.cache_resource
def train_model(df: pd.DataFrame):
    df = df.copy()

    # Drop ID jika ada
    if "Order_ID" in df.columns:
        df = df.drop(columns=["Order_ID"])

    # Pastikan target ada
    y = df["Delivery_Time_min"]

    # Kolom numerik & kategorikal
    num_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
    cat_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]

    # Imputasi Missing Values
    # Numerik -> median
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Kategorikal -> "Unknown"
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    X = df_encoded.drop(columns=["Delivery_Time_min"])

    feature_columns = X.columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Hitung performance (test set)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    performance = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return model, feature_columns, performance


# 
#  SETUP STREAMLIT
# 

st.set_page_config(
    page_title="Prediksi Waktu Pengantaran",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 Prediksi Waktu Pengantaran Makanan")

df_raw = load_data()
model, feature_cols, performance = train_model(df_raw)


# 
#  MENU HALAMAN
# 

page = st.sidebar.radio(
    "Menu",
    ["Overview & EDA", "Model Performance", "Prediksi Waktu Pengantaran"]
)


# 
#  HALAMAN OVERVIEW & EDA
# 

if page == "Overview & EDA":
    st.subheader("📊 Data Overview")

    st.write("**Jumlah baris & kolom:**")
    st.write(df_raw.shape)

    st.write("**Contoh data:**")
    st.dataframe(df_raw.head())

    st.write("**Ringkasan statistik:**")
    st.write(df_raw.describe())

    st.write("**Distribusi Delivery_Time_min:**")
    hist_data = df_raw["Delivery_Time_min"].value_counts().sort_index()
    st.bar_chart(hist_data)

    st.write("### Feature Importance (Linear Regression)")

    # Feature importance dari koefisien Linear Regression
    coefs = pd.Series(model.coef_, index=feature_cols)
    importance = coefs.abs().sort_values(ascending=False)

    st.write("Menampilkan 15 fitur terpenting berdasarkan |koefisien|:")
    st.bar_chart(importance.head(15))

    st.caption(
        "Catatan: Feature importance dihitung dari nilai absolut koefisien Linear Regression "
        "(tanpa standardisasi), sehingga terkait dengan skala fitur."
    )

    st.write("### Insight Singkat")
    st.write(
        "- Distribusi waktu pengantaran stabil, sebagian besar 40–70 menit.\n"
        "- Ada beberapa outlier di atas 100 menit.\n"
        "- Fitur dengan |koefisien| paling besar (misal Distance_km, Preparation_Time_min) "
        "memiliki pengaruh terbesar terhadap target."
    )


# 
#  HALAMAN MODEL PERFORMANCE
# 

elif page == "Model Performance":
    st.subheader("📈 Model Performance (Linear Regression)")

    st.write("Model yang digunakan sebagai model final adalah **Linear Regression**.")
    st.write("Berikut metrik evaluasi pada data test:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE (menit)", f"{performance['MAE']:.2f}")
    with col2:
        st.metric("RMSE (menit)", f"{performance['RMSE']:.2f}")
    with col3:
        st.metric("R²", f"{performance['R2']:.3f}")

    st.write("---")
    st.write("**Interpretasi singkat:**")
    st.write(
        f"- Rata-rata kesalahan absolut (MAE) sekitar **{performance['MAE']:.2f} menit**.\n"
        f"- RMSE sekitar **{performance['RMSE']:.2f} menit**, sensitif terhadap error besar.\n"
        f"- R² sebesar **{performance['R2']:.3f}**, menunjukkan seberapa besar variasi waktu pengantaran "
        "yang dapat dijelaskan oleh fitur-fitur dalam model."
    )

    st.caption(
        "Nilai metrik di atas dihitung dari data test (20% dari dataset) "
        "setelah train-test split."
    )


# 
#  HALAMAN PREDIKSI (INPUT DI MAIN PAGE)
# 

elif page == "Prediksi Waktu Pengantaran":
    st.subheader("🧮 Prediksi Waktu Pengantaran")

    st.write(
        "Isi parameter di bawah ini, lalu klik tombol **Prediksi** untuk mendapatkan "
        "estimasi waktu pengantaran (dalam menit)."
    )

    # Input di main page (bukan sidebar)
    col1, col2 = st.columns(2)

    with col1:
        distance_km = st.number_input(
            "Jarak Pengantaran (km)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.1
        )

        prep_time = st.number_input(
            "Waktu Persiapan (menit)",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )

        courier_exp = st.number_input(
            "Pengalaman Kurir (tahun)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        )

    with col2:
        weather = st.selectbox(
            "Cuaca",
            options=["Clear", "Rainy", "Snowy", "Foggy", "Windy"]
        )

        traffic = st.selectbox(
            "Tingkat Lalu Lintas",
            options=["Low", "Medium", "High"]
        )

        time_of_day = st.selectbox(
            "Waktu dalam Sehari",
            options=["Morning", "Afternoon", "Evening", "Night"]
        )

        vehicle = st.selectbox(
            "Jenis Kendaraan",
            options=["Bike", "Scooter", "Car"]
        )

    st.write("### Ringkasan Input")
    input_dict = {
        "Distance_km": [distance_km],
        "Preparation_Time_min": [prep_time],
        "Courier_Experience_yrs": [courier_exp],
        "Weather": [weather],
        "Traffic_Level": [traffic],
        "Time_of_Day": [time_of_day],
        "Vehicle_Type": [vehicle]
    }
    input_df = pd.DataFrame(input_dict)
    st.table(input_df)

    # One-hot encode input, samakan dengan training
    input_encoded = pd.get_dummies(
        input_df,
        columns=["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"],
        drop_first=False
    )
    input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

    if st.button("Prediksi Waktu Pengantaran"):
        try:
            pred = model.predict(input_encoded)[0]
            pred_rounded = np.round(pred, 2)

            st.success(f"📦 Estimasi Waktu Pengantaran: **{pred_rounded} menit**")
            st.caption(
                "Model ini menggunakan Linear Regression yang dilatih dari data historis. "
                "Hasil merupakan estimasi, bukan jaminan kondisi real-time."
            )
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
