# PROJEK-ML-GIAN---GANESH
# Prediksi Dini Risiko Cacat Produk *Injection Molding* (QC Lantai Produksi)

Aplikasi web interaktif dan model *machine learning* untuk memprediksi probabilitas kecacatan komponen plastik (tutup botol, rumah komponen, klip) secara *real-time* berdasarkan parameter satu siklus cetak pada mesin *injection molding* (Pabrik Komponen Plastik — Cikarang).



## 1. Ringkasan Masalah & Tujuan Bisnis

Pada lini produksi *injection molding*, cacat produk sebelumnya baru terdeteksi melalui inspeksi visual di akhir *shift*, sehingga ratusan komponen terlanjur diproduksi saat parameter mesin menyimpang. Proyek ini membangun sistem peringatan dini (*early warning*) bagi operator mesin untuk:
1. Menghitung **probabilitas kecacatan komponen** segera setelah parameter siklus cetak dimasukkan.
2. Menentukan keputusan penyisihan komponen berdasarkan **optimasi biaya bisnis** dengan rasio kerugian asimetris:
   - **False Negative (FN — cacat lolos ke pelanggan):** Rp500.000 / unit
   - **False Positive (FP — komponen bagus ikut disisihkan):** Rp25.000 / unit

---

## 2. Pipeline Data & Rekayasa Fitur (`src/features.py`)

Dataset awal (`raw.csv`) berisi **25.350 rekaman** dari 4 mesin (`IM-01` s.d. `IM-04`) periode Maret–Juni 2026. Seluruh prapemrosesan dienkapsulasi di dalam `src/features.py` agar konsisten antara tahap pelatihan dan inferensi web:

* **Pembersihan Data (`clean_data`):**
  * Menghapus **350 baris duplikat** sehingga tersisa **25.000 siklus unik** (tingkat cacat aktual **7,76%** atau 1.940 unit).
  * Menyeragamkan penulisan kategori `mesin`, `shift`, dan `bahan` (misalnya `'P.P'`, `'pp'`, `'PP '` menjadi `'PP'`).
  * Mengonversi **579 pembacaan anomali sensor suhu Fahrenheit** (`suhu_leleh_C > 380` pada mesin `IM-04`) ke skala Celsius.
  * Mengubah nilai fisik tidak valid (`tekanan_injeksi_bar <= 0`, `waktu_siklus_s <= 0`, dan `waktu_siklus_s <= waktu_pendinginan_s`) menjadi `NaN` untuk diimputasi dengan **nilai median** di dalam *pipeline* pelatihan.
* **Rekayasa Fitur (`build_features` — 10 fitur asli + 10 fitur turunan = 20 fitur):**
  * **Deviasi Suhu Standar Material:** Menghitung selisih dan nilai absolut penyimpangan `suhu_leleh_C` dan `suhu_cetakan_C` terhadap standar tiap jenis plastik (**PP:** 230/40 °C, **ABS:** 240/60 °C, **PC:** 300/90 °C), serta gradien `selisih_suhu_leleh_cetakan`.
  * **Risiko Kelembaban Higroskopis:** Interaksi `kelembaban_bahan_pct` khusus pada bahan penyerap air (**ABS** dan **PC**) untuk menangkap risiko cacat *silver streak*.
  * **Dinamika Siklus & Injeksi:** `rasio_waktu_pendinginan`, `waktu_isi_tahan_s` (`waktu_siklus_s - waktu_pendinginan_s`), dan `indeks_laju_tekanan` (`tekanan_injeksi_bar * kecepatan_injeksi_mm_s`).

---

## 3. Hasil Pemodelan & Optimasi Threshold (`Tugas_ML_inject.ipynb`)

Data dibagi secara *stratified* menjadi **70% Train (17.500)**, **15% Validation (3.750)**, dan **15% Test (3.750)** dengan metrik utama **PR-AUC**.

| Model | Jumlah Fitur | PR-AUC Validation | PR-AUC Test | Keterangan |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression (*Baseline*)** | 10 fitur asli | 0,5140 | — | Model acuan awal |
| **HistGradientBoostingClassifier (Final)** | 20 fitur | **0,6720** | **0,6669** | **+30,74% peningkatan PR-AUC** |

### Optimasi Threshold Biaya Bisnis
Berdasarkan pencarian *threshold* pada *validation set* untuk meminimalkan total biaya $\text{Total Biaya} = (\text{FN} \times \text{Rp500.000}) + (\text{FP} \times \text{Rp25.000})$:
* **Threshold Optimal:** **`0,0497`** (`4,97%`)
* **Performa pada *Test Set* (3.750 komponen; 291 cacat aktual):**
  * **Recall Kelas Cacat:** **86,25%** (`TP = 251`, `FN = 40`, `FP = 569`, `TN = 2.890`)
  * **Estimasi Penghematan Biaya:** Menekan potensi kerugian dari **Rp145.500.000** (tanpa model) menjadi **Rp34.225.000** (penghematan **Rp111.275.000** atau **76,48%**).

---

## 4. Struktur Repositori

```text
├── app.py                   # Aplikasi web interaktif Streamlit untuk operator
├── Tugas_ML_inject.ipynb    # Notebook EDA, training, tuning threshold, dan evaluasi
├── requirements.txt         # Daftar dependensi Python
├── raw.csv                  # Dataset mentah dari sistem pabrik
├── data/
│   └── test.csv             # Data uji (15% stratified split)
├── models/
│   ├── model.joblib         # Pipeline model HistGradientBoostingClassifier terlatih
│   └── metadata.json        # Konfigurasi fitur, threshold optimal (0.0497), dan metrik
└── src/
    └── features.py          # Fungsi clean_data(), build_features(), dan prepare_features()
