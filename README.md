# PROJEK-ML-GIAN---GANESH
### Prediksi Dini Risiko Cacat Produk Injection Molding (QC Lantai Produksi)



Aplikasi web interaktif dan sistem *machine learning* untuk memprediksi probabilitas kecacatan komponen plastik (**tutup botol**, **rumah komponen**, dan **klip**) secara *real-time* berdasarkan parameter satu siklus cetak pada mesin *injection molding* (**Pabrik Komponen Plastik — Cikarang**).

---

## Daftar Isi
1. [Ringkasan Masalah & Tujuan Bisnis](#1-ringkasan-masalah--tujuan-bisnis)
2. [Arsitektur & Alur Kerja Sistem](#2-arsitektur--alur-kerja-sistem)
3. [Pipeline Data & Rekayasa Fitur (`src/features.py`)](#3-pipeline-data--rekayasa-fitur-srcfeaturespy)
4. [Hasil Pemodelan & Optimasi Threshold (`Tugas_ML_inject.ipynb`)](#4-hasil-pemodelan--optimasi-threshold-tugas_ml_injectipynb)
5. [Struktur Repositori](#5-struktur-repositori)
6. [Panduan Instalasi & Penggunaan](#6-panduan-instalasi--penggunaan)
7. [Kontributor](#7-kontributor)

---

## 1. Ringkasan Masalah & Tujuan Bisnis

### Latar Belakang Permasalahan
Pada lini produksi *injection molding*, cacat produk sebelumnya baru terdeteksi melalui inspeksi visual manual di akhir *shift*. Akibat keterlambatan deteksi tersebut, ratusan komponen terlanjur diproduksi ketika parameter mesin mengalami penyimpangan (*process drift*), sehingga meningkatkan tingkat *scrap* maupun risiko lolosnya produk cacat ke pelanggan.

### Tujuan Sistem Peringatan Dini (*Early Warning System*)
Proyek ini menghadirkan sistem pendukung keputusan bagi operator lantai produksi untuk:
1. **Prediksi Real-Time:** Menghitung probabilitas kecacatan komponen segera setelah parameter siklus cetak dimasukkan ke antarmuka aplikasi.
2. **Keputusan Penyisihan Berbasis Biaya (Cost-Sensitive Decision):** Menentukan rekomendasi tindakan (*Lolos* vs. *Sisihkan / Karantina*) berdasarkan optimasi total biaya bisnis dengan **rasio kerugian asimetris 20 : 1**:

| Jenis Kesalahan Prediksi | Skenario Operasional | Dampak Bisnis | Biaya per Unit |
| :--- | :--- | :--- | :---: |
| **False Negative (FN)** | Produk **cacat** diprediksi normal dan **lolos ke pelanggan** | Retur barang, klaim garansi, penalti kualitas, dan penurunan reputasi | **Rp500.000** |
| **False Positive (FP)** | Produk **bagus** diprediksi cacat dan **ikut disisihkan** | Biaya inspeksi ulang, *rework*, atau *scrap* material | **Rp25.000** |

---

## 2. Arsitektur & Alur Kerja Sistem

Seluruh alur pengolahan data dirancang secara modular agar transformasi fitur pada tahap pelatihan model identik dengan tahap inferensi pada aplikasi web (*mencegah training-serving skew*):

```text
[raw.csv (25.350 siklus)]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  src/features.py                                       │
│  1. clean_data()       : Deduplikasi, standar kategori,│
│                          koreksi °F->°C, validasi fisik│
│  2. build_features()   : 10 fitur asli + 10 turunan    │
│  3. prepare_features() : Seleksi & format input model  │
└─────────────────────────┬──────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
[Tugas_ML_inject.ipynb]         [models/model.joblib & metadata.json]
 • Stratified Split (70/15/15)   • Pipeline terlatih + Imputer median
 • Training & Evaluasi PR-AUC    • Threshold optimal biaya (0,0497)
 • Cost-Threshold Optimization             │
                                           ▼
                                  [app.py (Streamlit)]
                                   • Input parameter siklus operator
                                   • Prediksi probabilitas & keputusan QC


PROJEK-ML-GIAN---GANESH/
├── app.py                   # Aplikasi web interaktif Streamlit untuk operator QC
├── Tugas_ML_inject.ipynb    # Notebook EDA, pelatihan model, tuning threshold, & evaluasi
├── requirements.txt         # Daftar dependensi library Python
├── raw.csv                  # Dataset mentah historis mesin injection molding (25.350 baris)
├── data/
│   └── test.csv             # Dataset pengujian (15% stratified split — 3.750 baris)
├── models/
│   ├── model.joblib         # Artefak pipeline HistGradientBoostingClassifier terlatih
│   └── metadata.json        # Konfigurasi daftar fitur, threshold optimal (0.0497), & metrik
└── src/
    └── features.py          # Modul pembersihan & rekayasa fitur: clean_data(), build_features(), prepare_features()
