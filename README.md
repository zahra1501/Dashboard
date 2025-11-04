# Estimasi Volume Air Irigasi dengan Menggunakan Machine Learning dan Deep Learning

## Deskripsi Singkat
Penelitian ini mengestimasi volume air irigasi tahunan tahun 2021-2024 per provinsi di Indonesia menggunakan metode Machine Learning (Random Forest, XGBoost, SVR) dan Deep Learning (MLP, CNN) berbasis data lingkungan seperti evapotranspirasi, kelembaban tanah, dan presipitasi. Perhitungan awal dilakukan secara tidak langsung melalui model SM2RAIN. Hasil menunjukkan Random Forest memiliki performa terbaik, dengan distribusi volume irigasi tertinggi di NTT serta wilayah tengah dan timur Jawa, dan terendah di Sumatera, Sulawesi, dan Kalimantan. Penelitian ini juga menghasilkan dashboard web interaktif untuk visualisasi hasil estimasi.

Dashboard dapat diakses pada link berikut:
https://dashboard-volume-air-irigasi.streamlit.app

## Struktur Folder Project
- **dashboard/**
  - Kode dashboard interaktif.
  - Data yang digunakan pada dashboard.
- **data/**
  - Kode preprocessing
  - Seluruh data penelitian yang digunakan dalam proses pemodelan.
- **pemodelan/**
  - Kode estimasi volume air irigasi tahun 2021-2022 dengan SM2RAIN
  - Kode pemodelan menggunakan Machine Learning dan Deep Learning.

