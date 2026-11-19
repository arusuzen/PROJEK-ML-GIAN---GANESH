# PROJEK-ML-GIAN---GANESH
### Prediksi Dini Risiko Cacat Produk *Injection Molding* (QC Lantai Produksi)

Aplikasi web interaktif dan model *machine learning* untuk memprediksi probabilitas kecacatan komponen plastik (**tutup botol**, **rumah komponen**, dan **klip**) secara *real-time* berdasarkan parameter satu siklus cetak pada mesin *injection molding* (**Pabrik Komponen Plastik — Cikarang**).

---

## Daftar Isi
1. [Ringkasan Masalah & Tujuan Bisnis](#1-ringkasan-masalah--tujuan-bisnis)
2. [Pipeline Data & Rekayasa Fitur (`src/features.py`)](#2-pipeline-data--rekayasa-fitur-srcfeaturespy)
3. [Hasil Pemodelan & Optimasi Threshold (`Tugas_ML_inject.ipynb`)](#3-hasil-pemodelan--optimasi-threshold-tugas_ml_injectipynb)
4. [Fitur Aplikasi Web Operator (`app.py`)](#4-fitur-aplikasi-web-operator-apppy)
5. [Struktur Repositori](#5-struktur-repositori)


---

## 1. Ringkasan Masalah & Tujuan Bisnis

Pada lini produksi *injection molding*, cacat produk sebelumnya baru terdeteksi melalui inspeksi visual di akhir *shift*, sehingga ratusan komponen terlanjur diproduksi ketika parameter mesin menyimpang. Proyek ini membangun sistem peringatan dini (*early warning system*) bagi operator mesin untuk:

1. **Menghitung Probabilitas Kecacatan Real-Time:** Memprediksi risiko cacat komponen segera setelah parameter satu siklus cetak dimasukkan ke antarmuka aplikasi (`app.py`).
2. **Menentukan Keputusan Penyisihan Berbasis Biaya:** Menetapkan rekomendasi penyisihan komponen berdasarkan optimasi biaya bisnis dengan **rasio kerugian asimetris 20 : 1**:

| Kategori Kesalahan | Kondisi Operasional | Biaya per Unit |
| :--- | :--- | :---: |
| **False Negative (FN)** | Komponen **cacat** diprediksi bagus dan **lolos ke pelanggan** | **Rp500.000** |
| **False Positive (FP)** | Komponen **bagus** diprediksi cacat dan **ikut disisihkan** | **Rp25.000** |

---

## 2. Pipeline Data & Rekayasa Fitur (`src/features.py`)

Dataset awal (`raw.csv`) berisi **25.350 rekaman** siklus cetak dari **4 mesin** (`IM-01` s.d. `IM-04`) periode **Maret–Juni 2026**. Seluruh logika prapemrosesan dienkapsulasi di dalam modul `src/features.py` agar transformasi data konsisten antara tahap pelatihan (`Tugas_ML_inject.ipynb`) dan tahap inferensi web (`app.py`).

### A. Pembersihan Data — `clean_data()`
* **Deduplikasi Data:** Menghapus **350 baris duplikat** sehingga tersisa **25.000 siklus unik** dengan tingkat cacat aktual sebesar **7,76%** (**1.940 unit cacat** dan **23.060 unit lolos QC**).
* **Standardisasi Kategori:** Menyeragamkan penulisan kategori `mesin`, `shift`, dan jenis `bahan` polimer (misalnya `'P.P'`, `'pp'`, `'PP '` dinormalisasi menjadi `'PP'`).
* **Koreksi Anomali Sensor Suhu:** Mengonversi **579 pembacaan anomali sensor suhu Fahrenheit** (`suhu_leleh_C > 380` pada mesin `IM-04`) ke skala Celsius ($^\circ\text{C}$) saat pembersihan dataset mentah (`convert_fahrenheit=True`).
* **Validasi Batasan Fisika:** Mengubah nilai fisik yang tidak valid secara mekanis (`suhu_leleh_C` di luar 100–400 °C, `suhu_cetakan_C` di luar 0–200 °C, `tekanan_injeksi_bar <= 0`, `kecepatan_injeksi_mm_s <= 0`, `waktu_pendinginan_s <= 0`, `waktu_siklus_s <= 0`, dan `waktu_siklus_s <= waktu_pendinginan_s`) menjadi `NaN` untuk diimputasi menggunakan **nilai median** di dalam *pipeline* pelatihan.

### B. Rekayasa Fitur — `build_features()` (10 Fitur Asli + 10 Fitur Turunan = 20 Fitur)
1. **Deviasi Suhu Standar Material (5 Fitur Turunan):**
   Menghitung selisih bertanda dan nilai absolut penyimpangan `suhu_leleh_C` serta `suhu_cetakan_C` terhadap standar masing-masing jenis plastik, ditambah gradien termal `selisih_suhu_leleh_cetakan`:

   | Jenis Bahan | Standar Suhu Leleh (`suhu_leleh_C`) | Standar Suhu Cetakan (`suhu_cetakan_C`) | Sifat Higroskopis |
   | :--- | :---: | :---: | :---: |
   | **PP** | 230 °C | 40 °C | Non-Higroskopis |
   | **ABS** | 240 °C | 60 °C | Higroskopis |
   | **PC** | 300 °C | 90 °C | Higroskopis |

2. **Risiko Kelembaban Higroskopis (2 Fitur Turunan):**
   Interaksi `kelembaban_bahan_pct` khusus pada bahan penyerap uap air (**ABS** dan **PC**) melalui fitur `is_higroskopis` dan `risiko_kelembaban_higro` untuk menangkap risiko cacat permukaan *silver streak*.
3. **Dinamika Siklus & Injeksi (3 Fitur Turunan):**
   * `rasio_waktu_pendinginan`: Proporsi waktu pendinginan terhadap total waktu siklus (`waktu_pendinginan_s / waktu_siklus_s`).
   * `waktu_isi_tahan_s`: Estimasi waktu fase pengisian dan penahanan (`waktu_siklus_s - waktu_pendinginan_s`).
   * `indeks_laju_tekanan`: Perkalian intensitas tekanan dan kecepatan injeksi (`tekanan_injeksi_bar * kecepatan_injeksi_mm_s`).

### C. Persiapan Input Model — `prepare_features()`
Mengonversi kamus input satu siklus cetak dari aplikasi web menjadi `DataFrame`, menjalankan `clean_data()` dan `build_features()`, lalu menghasilkan 20 kolom fitur siap prediksi untuk `models/model.joblib`.

---

## 3. Hasil Pemodelan & Optimasi Threshold (`Tugas_ML_inject.ipynb`)

Sebanyak **25.000 siklus bersih** dibagi secara *stratified* (`random_state=42`) menjadi **70% Train (17.500)**, **15% Validation (3.750)**, dan **15% Test (3.750)** dengan metrik evaluasi utama **PR-AUC** (*Precision-Recall Area Under Curve*).

### Perbandingan Performa Model

| Model | Jumlah Fitur | PR-AUC Validation | PR-AUC Test | Keterangan |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression (*Baseline*)** | 10 fitur asli | 0,5140 | — | Model acuan awal |
| **HistGradientBoostingClassifier (Final)** | 20 fitur | **0,6720** | **0,6669** | **+30,74%** peningkatan PR-AUC |

### Optimasi Threshold Biaya Bisnis
Berdasarkan pencarian *threshold* pada *validation set* untuk meminimalkan total biaya $\text{Total Biaya} = (\text{FN} \times \text{Rp500.000}) + (\text{FP} \times \text{Rp25.000})$:

* **Threshold Optimal:** **`0,0492`** (`4,92%` / tepatnya `0.049212`)
* **Performa pada *Test Set* (3.750 komponen; 291 cacat aktual & 3.459 lolos QC):**
  * **Confusion Matrix:** `TP = 252`, `FN = 39`, `FP = 574`, `TN = 2.885`
  * **Recall Kelas Cacat:** **86,60%**
  * **Estimasi Penghematan Biaya:** Menekan potensi kerugian dari **Rp145.500.000** (tanpa penyaringan model) menjadi **Rp33.850.000** (penghematan **Rp111.650.000** atau **76,74%**).

---

## 4. Fitur Aplikasi Web Operator (`app.py`)

* **Path Portabel Lintas Perangkat:** Memuat `models/model.joblib` dan `models/metadata.json` menggunakan `Path(__file__).resolve().parent` sehingga dapat dijalankan langsung di Windows, Linux, maupun Streamlit Community Cloud.
* **Validasi Input Real-Time (*Inline Guardrails*):** Menampilkan peringatan langsung pada kolom form jika nilai berada di luar batas wajar (`suhu_leleh` 100–400 °C, `suhu_cetakan` 0–200 °C, `tekanan` 0–1.500 bar, `kecepatan` 0–300 mm/detik, `waktu_pendinginan` dan `waktu_siklus` 0–100 detik, `waktu_siklus > waktu_pendinginan`, serta `kelembaban` 0–100%) dan menahan proses prediksi hingga masukan diperbaiki.

### Contoh Skenario Pengujian Aplikasi

| Parameter Input | Skenario 1: Berhasil QC (Normal) | Skenario 2: Gagal QC (Risiko Tinggi) |
| :--- | :---: | :---: |
| **Bahan / Mesin / Shift** | `PP` / `IM-01` / `pagi` | `PC` / `IM-03` / `malam` |
| **Suhu Leleh / Cetakan** | `230 °C` / `40 °C` | `275 °C` / `75 °C` |
| **Tekanan / Kecepatan Injeksi** | `1100 bar` / `45 mm/s` | `950 bar` / `35 mm/s` |
| **Waktu Pendinginan / Siklus** | `12 detik` / `22 detik` | `15 detik` / `26 detik` |
| **Kelembapan Bahan** | `0.02%` | `0.25%` |
| **Hasil Prediksi Sistem** | Risiko di bawah *threshold* (`< 4,92%`) — **Aman** | Risiko melewati *threshold* (`>= 4,92%`) — **Peringatan** |

---

## 5. Struktur Repositori

```text
PROJEK-ML-GIAN---GANESH/
├── app.py                   # Aplikasi web interaktif Streamlit untuk operator QC
├── Tugas_ML_inject.ipynb    # Notebook EDA, training, tuning threshold, dan evaluasi
├── requirements.txt         # Daftar dependensi Python (scikit-learn==1.9.1, dll.)
├── README.md                # Dokumentasi proyek
├── raw.csv                  # Dataset mentah dari sistem pabrik (25.350 baris)
├── models/
│   ├── model.joblib         # Pipeline model HistGradientBoostingClassifier terlatih
│   └── metadata.json        # Konfigurasi fitur, threshold optimal (0.0492), dan metrik
└── src/
    └── features.py          # Fungsi clean_data(), build_features(), dan prepare_features()
```

---


