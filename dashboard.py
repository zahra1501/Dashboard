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
import pickle
import os
#import scikitplot as skplt

# Mengatur environment variable untuk GeoPandas
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

# --- Konfigurasi Halaman Streamlit ---
# Mengatur judul, ikon, dan layout halaman menjadi "wide"
st.set_page_config(
    page_title="Dashboard Estimasi Air Irigasi",
    page_icon="💧",
    layout="wide"
)

# --- Judul dan Deskripsi Utama Dashboard ---
st.title("💧 Estimasi Volume Air Irigasi di Indonesia")
st.markdown("""
    Dashboard ini menggunakan Model Random Forest untuk mengestimasi volume air irigasi berdasarkan karakteristik hidrologi dan iklim.
""")

# --- Fungsi untuk memuat data dan model ---
# Menggunakan cache untuk menghindari pemuatan ulang data setiap kali interaksi
@st.cache_data
def load_data():
    """Memuat data dari file CSV."""
    df = pd.read_csv('data_irigasi_revisi_lagi2.csv')
    geo_df = pd.read_csv('data_tahunan_baru.csv')
    return df, geo_df

@st.cache_data
def load_model_and_scaler():
    """Memuat model dan scaler dari file."""
    # Standarisasi data
    df, _ = load_data()
    dset = df[['presipitasi','sm_smap','tmin','tmax','t','swr','et_era5']]
    mean = dset.mean(axis=0)
    std = dset.std(axis=0)
    
    # Load Model
    rf_model = load("model_rf.model")
    
    return rf_model, mean, std

# Panggil fungsi untuk memuat data dan model
df, geo_df = load_data()
rf_model, mean, std = load_model_and_scaler()

# --- Tabs untuk navigasi ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Gambaran Umum", 
    "Performa Model", 
    "Peta Interaktif", 
    "Estimasi dengan Model", 
    "Data"
])

# --- Konten Tab 1: Gambaran Umum ---
with tab1:
    st.header("🌾 Gambaran Umum Penelitian")

    st.markdown("""
    <style>
    .highlight {
        background-color: #e6f0ff;
        padding: 12px;
        border-left: 5px solid #1f77b4;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .subsection {
        margin-top: 1.5rem;
        font-weight: bold;
        font-size: 16px;
        color: #444;
    }
    ul li {
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight">
        💧 Sekitar <strong>900 km³ air per tahun</strong> dibutuhkan untuk tanaman pangan. 
        <br>🌍 Irigasi menyumbang sekitar <strong>70% dari total pengambilan air global</strong>, menjadikannya komponen penting dalam sektor pertanian.
    </div>

    <div class="highlight">
        🔍 Kebutuhan air irigasi <strong>diperkirakan terus meningkat</strong>. 
        Oleh karena itu, <strong>estimasi yang akurat</strong> sangat dibutuhkan untuk pengelolaan air yang berkelanjutan.
    </div>

    <div class="subsection">🎯 Tujuan Penelitian</div>
    <p>Penelitian ini bertujuan <strong>mengestimasi volume air irigasi di Indonesia</strong> menggunakan pendekatan:</p>
    <ul>
        <li><strong>Machine Learning</strong>: Random Forest, XGBoost, SVR</li>
        <li><strong>Deep Learning</strong>: MLP, CNN</li>
    </ul>

    <div class="subsection">🌦️ Data yang Digunakan</div>
    <ul>
        <li>Evapotranspirasi</li>
        <li>Kelembaban tanah</li>
        <li>Presipitasi</li>
        <li>Suhu (Tmin, Tmax)</li>
        <li>Shortwave Radiation</li>
    </ul>

    <div class="subsection">🧠 Metode Estimasi</div>
    <p>Estimasi dilakukan secara <strong>tidak langsung</strong> melalui model <strong>SM2RAIN</strong>, yang hasilnya dijadikan <strong>target prediksi</strong>.</p>

    <div class="subsection">🏆 Hasil Utama</div>
    <ul>
        <li><strong>Random Forest</strong> memberikan performa terbaik</li>
        <li>Pemetaan menunjukkan distribusi yang <strong>tidak merata</strong> di seluruh Indonesia</li>
        <li>Wilayah dengan estimasi tertinggi: <strong>NTT dan Jawa Tengah–Timur</strong></li>
        <li>Wilayah dengan estimasi lebih rendah: <strong>Sumatera, Sulawesi, Kalimantan</strong></li>
    </ul>
    """, unsafe_allow_html=True)


# --- Konten Tab 2: Performa Model ---
with tab2:
    st.header("Model Terbaik: Random Forest")

    # Data model
    models_data = {
        'Model': ['Random Forest', 'XGBoost', 'SVR', 'MLP', 'CNN'],
        'RMSE': [25.52, 26.15, 29.03, 27.63, 28.33],
        'MAE': [19.28, 20.28, 20.39, 21.51, 20.48],
        'MAPE': [16.68, 18.09, 16.22, 19.00, 17.56]
    }
    models_df = pd.DataFrame(models_data)

    # Pilih model terbaik (Random Forest)
    best_model_idx = 0
    best_model = models_df.loc[best_model_idx]

    # Tambahkan gaya CSS untuk KPI card
    st.markdown("""
        <style>
        .kpi-card {
            padding: 1rem;
            border-radius: 12px;
            background-color: #f9f9f9;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: 0.3s ease-in-out;
        }
        .kpi-card:hover {
            background-color: #e6f0ff;
            transform: scale(1.02);
        }
        .kpi-title {
            font-size: 18px;
            color: #333333;
            margin-bottom: 0.5rem;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: bold;
            color: #007BFF;
        }
        </style>
    """, unsafe_allow_html=True)

    # Tampilkan KPI cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">RMSE</div>
                <div class="kpi-value">{best_model["RMSE"]:.2f}</div>
            </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">MAE</div>
                <div class="kpi-value">{best_model["MAE"]:.2f}</div>
            </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">MAPE</div>
                <div class="kpi-value">{best_model["MAPE"]:.2f}%</div>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")

    # Menampilkan metrik semua model dalam tabel
    st.subheader("Hasil Evaluasi Seluruh Model")
    st.dataframe(models_df, use_container_width=True)

    st.markdown("---")
    
    st.subheader("Heatmap Korelasi dan Feature Importance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Heatmap Korelasi Antar Variabel")
        heatmap_fig = plt.figure(figsize=(6,6))
        corr = df[['irigasi', 'et_era5', 'presipitasi', 'tmin', 'tmax', 't', 'sm_smap', 'swr']].corr(method='pearson')
        sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, vmin=-1, vmax=1)
        st.pyplot(heatmap_fig, use_container_width=True)  

        with st.expander("Penjelasan Heatmap Korelasi"):
            st.markdown("""
            Heatmap menunjukkan bahwa volume air irigasi berkorelasi positif dengan evapotranspirasi, presipitasi, dan 
            kelembapan tanah, yang menandakan peran irigasi dalam menjaga suplai air bagi tanaman. Sebaliknya, terdapat 
            korelasi negatif dengan suhu, suhu maksimum, dan shortwave radiation, yang mengindikasikan efek pendinginan 
            irigasi melalui peningkatan kelembapan dan tutupan vegetasi.
            """)

    with col2:
        st.info("Permutation Importance untuk Model Random Forest")
        # Load permutation importance dari file
        with open("perm_importance.pkl", "rb") as f:
            result, feature_names = pickle.load(f)

        importances = result.importances_mean
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = np.array(feature_names)[sorted_idx]
        sorted_importances = importances[sorted_idx]

        # Visualisasi vertikal
        feat_imp_fig = plt.figure(figsize=(6,6))
        ax = feat_imp_fig.add_subplot(111)

        plt.bar(sorted_features, sorted_importances, color='teal')
        plt.xticks(rotation=45)
        plt.ylabel("Importance Score")
        plt.title("Permutation Importance (Random Forest)")
        plt.tight_layout()

        st.pyplot(feat_imp_fig, use_container_width=True)

        with st.expander("Penjelasan Feature Importance"):
            st.markdown("""
            Berdasarkan grafik di atas, variabel yang paling berpengaruh dalam estimasi volume air irigasi 
            adalah presipitasi dan kelembapan tanah, karena keduanya berperan langsung 
            dalam ketersediaan dan penyerapan air oleh tanah. Sementara itu, variabel lain seperti suhu, shortwave radiation, 
            dan evapotranspirasi memiliki pengaruh lebih kecil, kemungkinan karena efeknya bersifat tidak langsung 
            atau sudah tercakup oleh pengaruh presipitasi dan kelembapan tanah.
            """)

# --- Konten Tab 3: Peta Interaktif ---
with tab3:
    st.header("Peta Interaktif Volume Air Irigasi Tahunan")

    @st.cache_data
    def load_geo():
        return gpd.read_parquet("batas_wilayah_simplified.parquet")
    
    gdf = load_geo()
    tahun = st.selectbox("Pilih Tahun:", sorted(geo_df['tahun'].unique()))
    geo_filtered = geo_df[geo_df['tahun'] == tahun]

    merged = gdf[['WADMPR', 'geometry']].rename(columns={'WADMPR': 'Provinsi'})
    merged = merged.merge(geo_filtered[['Provinsi', 'irigasi_pred']], on='Provinsi', how='left')

    merged['kategori'] = merged['irigasi_pred'].apply(lambda x: 'Tanpa Data' if pd.isna(x) or x == 0 else 'Ada Data')
    gdf_valid = merged[merged['kategori'] == 'Ada Data']
    gdf_nodata = merged[merged['kategori'] == 'Tanpa Data']

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        geojson=gdf.__geo_interface__,
        locations=gdf_nodata['Provinsi'],
        z=[0]*len(gdf_nodata),
        featureidkey="properties.WADMPR",
        colorscale=[[0, "#d3d3d3"], [1, "#d3d3d3"]],
        showscale=False,
        name=""
    ))

    fig.add_trace(go.Choropleth(
        geojson=gdf.__geo_interface__,
        locations=gdf_valid['Provinsi'],
        z=gdf_valid['irigasi_pred'],
        featureidkey="properties.WADMPR",
        colorscale="YlGnBu",
        colorbar_title="Volume (mm/tahun)",
        name="Volume Air",
        marker_line_color='black',
        marker_line_width=0.5
    ))

    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_layout(
        title=f"Estimasi Volume Air Irigasi Tahun {tahun}",
        height=700,
        margin={"r":0,"t":50,"l":0,"b":0},
        geo=dict(
            bgcolor='rgba(0,0,0,0)'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Konten Tab 4: Estimasi dengan Model ---
with tab4:
    st.header("Estimasi Volume Air Irigasi")
    st.markdown("""
        Sesuaikan nilai variabel iklim dan hidrologi pada slider di sebelah kiri.
        Hasil estimasi akan langsung tampil di sebelah kanan.
    """)

    col_slider, col_result = st.columns([1.2, 1])  # Kolom kiri untuk slider, kanan untuk hasil

    with col_slider:
        sliders = []
        fitur_model = ['presipitasi', 'sm_smap', 'tmin', 'tmax', 'swr', 'et_era5']

        label_mapping = {
            'presipitasi': 'Presipitasi (mm/bulan)',
            'sm_smap': 'Kelembaban Tanah (m³/m³)',
            'tmin': 'Suhu Minimum (°C)',
            'tmax': 'Suhu Maksimum (°C)',
            'swr': 'Shortwave Radiation (J/m²)',
            'et_era5': 'Evapotranspirasi (mm/bulan)'
        }

        for var in fitur_model:
            var_slider = st.slider(
                label=label_mapping[var],
                min_value=float(df[var].min()),
                max_value=float(df[var].max()),
                value=float(df[var].mean())
            )
            sliders.append(var_slider)

    with col_result:
        input_array = np.array(sliders)
        input_standardized = (input_array - mean.drop('t')) / std.drop('t')
        estimasi = rf_model.predict([input_standardized])
        konversi = estimasi[0] * (1 / 262.97) * (2629744)  # mm/bulan -> liter/bulan/hektar

        st.markdown(
            f"""
            <div style='
                background-color: #f0f2f6;
                padding: 25px;
                border-radius: 12px;
                border-left: 6px solid #1a73e8;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            '>
                <h4 style='margin:0; color:#1a73e8; font-weight:600;'>Estimasi Volume Air Irigasi:</h4>
                <p style='font-size:32px; font-weight:bold; color:#1a73e8; margin:10px 0 5px 0;'>
                    {estimasi[0]:.3f} mm/bulan
                </p>
                <hr style='border-top: 1px dashed #ccc; margin:15px 0;'>
                <p style='margin:0; color: #555; font-size:16px;'>Setara dengan:</p>
                <p style='font-size:24px; font-weight:600; color:#444; margin:5px 0 0 0;'>
                    {konversi:.3f} liter/bulan/hektar
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.info("Catatan: Konversi dilakukan dari mm/bulan ke liter/bulan/hektar.")



# --- Konten Tab 5: Tampilan Data ---
with tab5:
    st.header("Data Volume Air Irigasi 2021-2024")
    st.dataframe(df, use_container_width=True)
