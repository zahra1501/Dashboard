import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from joblib import load
from sklearn.inspection import permutation_importance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import geopandas as gpd
import json
import os
#import scikitplot as skplt

from lime import lime_tabular
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

## Load Data
df = pd.read_csv('/Users/fathimahaz-zahra/Documents/Dashboard/data_irigasi_revisi_lagi2.csv')
geo_df = pd.read_csv('/Users/fathimahaz-zahra/Documents/Dashboard/data_tahunan_baru.csv')

# Standarisasi
dset = df[['presipitasi','sm_smap','tmin','tmax','t','swr','et_era5']]
mean = dset.mean(axis=0)
std = dset.std(axis=0)
dset = (dset - mean) / std

# Split 
data_x = dset.iloc[:, 0:7].drop('t', axis=1)
data_y = df.iloc[:, 12]

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

# Load Model
rf_model = load("model_rf.model")

y_pred = rf_model.predict(X_test)

## Dashboard
st.title("Estimasi Penggunaan Air Irigasi di Indonesia")
st.markdown("Estimasi Air Irigasi Menggunakan Model Random Forest Berdasarkan Karakteristik Hidrologi dan Iklim")

tab1, tab2, tab3, tab4, tab5 = st.tabs(("Gambaran Umum", "Performa Model", "Hasil Penelitian", "Estimasi dengan Model", "Data"))

with tab1:
    st.header("Gambaran Umum Penelitian")
    st.markdown("""
    Dengan kebutuhan sekitar **900 km³ air per tahun** untuk tanaman pangan, irigasi menyumbang sekitar **70% dari total pengambilan air global**, menjadikannya elemen kunci dalam sektor pertanian. 

    Meskipun efisiensi penggunaan air meningkat, kebutuhan air irigasi diperkirakan **terus bertambah**. Oleh karena itu, **estimasi yang akurat sangat penting** untuk mendukung manajemen air berkelanjutan.

    Penelitian ini bertujuan **mengestimasi air irigasi yang digunakan di Indonesia** menggunakan pendekatan **machine learning** (Random Forest, XGBoost, SVR) dan **deep learning** (MLP, CNN), dengan memanfaatkan data iklim dan hidrologi seperti:
    - Evapotranspirasi
    - Kelembaban tanah
    - Presipitasi
    - Suhu
    - Shortwave Radiation

    Estimasi volume air irigasi dilakukan secara **tidak langsung** melalui model **SM2RAIN**, dan hasilnya digunakan sebagai **target model prediksi**.

    Hasil penelitian menunjukkan bahwa **Random Forest memberikan performa terbaik**. Pemetaan estimasi mengungkapkan distribusi air irigasi yang **tidak merata**, dengan:
    - **NTT** serta wilayah **tengah dan timur Jawa** menunjukkan angka tertinggi
    - **Sumatera, Sulawesi, dan Kalimantan** menunjukkan volume yang relatif lebih rendah.
    """)

with tab2:
    st.header("Heatmap Korelasi | Feature Importances")
    col1, col2 = st.columns(2)
    with col1:
        heatmap_fig = plt.figure(figsize=(6,6))
        corr = df[['irigasi', 'et_era5', 'presipitasi', 'tmin', 'tmax', 't', 'sm_smap', 'swr']].corr(method='pearson')
        sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, vmin=-1, vmax=1)
        st.pyplot(heatmap_fig, use_container_width=True)  
    
    with col2:
        # Hitung permutation importance
        result = permutation_importance(rf_model, X_test, y_test, n_repeats=30, random_state=42)
        importances = result.importances_mean

        # Urutkan berdasarkan importance
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = np.array(data_x.columns)[sorted_idx]
        sorted_importances = importances[sorted_idx]

        # Visualisasi vertikal
        feat_imp_fig = plt.figure(figsize=(6,6))
        ax = feat_imp_fig.add_subplot(111)

        plt.bar(sorted_features, sorted_importances)
        plt.xticks(rotation=45)
        plt.ylabel("Mean Importance Score")
        plt.title("Permutation Importance (Random Forest)")
        plt.tight_layout()

        st.pyplot(feat_imp_fig, use_container_width=True)

    st.divider()
    st.header("Hasil Evaluasi Model")
    st.markdown("<br>", unsafe_allow_html=True)

    # Hitung metrik evaluasi
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test,y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("RMSE", f"{rmse:.3f}")
    col2.metric("MAE", f"{mae:.3f}")
    col3.metric("MAPE (%)", f"{mape:.3f}")

with tab3:
    st.header("Peta Interaktif Volume Air Irigasi")

    @st.cache_data
    def load_geo():
        return gpd.read_parquet("batas_wilayah_simplified.parquet")
    
    gdf = load_geo()
    tahun = st.selectbox("Pilih Tahun", sorted(geo_df['tahun'].unique()))
    geo_filtered = geo_df[geo_df['tahun'] == tahun]

    # Gabung ke dataframe wilayah
    merged = gdf[['WADMPR', 'geometry']].rename(columns={'WADMPR': 'Provinsi'})
    merged = merged.merge(geo_filtered[['Provinsi', 'irigasi_pred']], on='Provinsi', how='left')

    # Pisahkan data dengan nilai dan tanpa nilai (0 atau NaN)
    merged['kategori'] = merged['irigasi_pred'].apply(lambda x: 'Tanpa Data' if pd.isna(x) or x == 0 else 'Ada Data')

    # Pecah ke dua dataframe
    gdf_valid = merged[merged['kategori'] == 'Ada Data']
    gdf_nodata = merged[merged['kategori'] == 'Tanpa Data']

    # Buat figure
    fig = go.Figure()

    # Layer 1: wilayah tanpa data (abu-abu)
    fig.add_trace(go.Choropleth(
        geojson=gdf.__geo_interface__,
        locations=gdf_nodata['Provinsi'],
        z=[0]*len(gdf_nodata),  # Dummy
        featureidkey="properties.WADMPR",
        colorscale=[[0, "#d3d3d3"], [1, "#d3d3d3"]],
        showscale=False,
        name=""
    ))

    # Layer 2: wilayah dengan data
    fig.add_trace(go.Choropleth(
        geojson=gdf.__geo_interface__,
        locations=gdf_valid['Provinsi'],
        z=gdf_valid['irigasi_pred'],
        featureidkey="properties.WADMPR",
        colorscale="YlGnBu",
        colorbar_title="Volume (mm/tahun)",
        name="Volume Air"
    ))

    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_layout(
        title=f"Estimasi Volume Air Irigasi Tahun {tahun}",
        height=700,
        margin={"r":0,"t":50,"l":0,"b":0}
    )

    st.plotly_chart(fig, use_container_width=True)

with tab4:
    sliders = []
    col1, col2 = st.columns(2)

    fitur_model = ['presipitasi','sm_smap','tmin','tmax','swr','et_era5']
    with col1:
        for var in fitur_model:
            var_slider = st.slider(
                label=var,
                min_value=float(df[var].min()),
                max_value=float(df[var].max()) 
            )
            sliders.append(var_slider)
    
    with col2:
        # Standardisasi input
        input_array = np.array(sliders)
        input_standardized = (input_array - mean.drop('t')) / std.drop('t')

        # Prediksi
        estimasi = rf_model.predict([input_standardized])
        konversi = estimasi[0] * (1 / 262.97) * (2629744)  # mm/bulan → liter/bulan/hektar

        st.markdown(
            f"""
            <div style='
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #2c7be5;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            '>
                <h4 style='margin:0; color:#000000;'>Estimasi Air Irigasi</h4>
                <p style='font-size:22px; font-weight:bold; color:#2c7be5; margin:0;'>
                    {estimasi[0]:.3f} mm/bulan
                </p>
                <hr style='margin:10px 0;'>
                <p style='margin:0; color: #555;'><b>Setara dengan:</b></p>
                <p style='font-size:18px; font-weight:600; color:#444; margin:0;'>
                    {konversi:.3f} liter/bulan/hektar
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


with tab5:
    st.header("Data Volume Air Irigasi 2021-2024")
    st.write(df)