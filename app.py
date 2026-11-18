from pathlib import Path
import json
import joblib
import streamlit as st

from src.features import prepare_features

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "metadata.json"

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {METADATA_PATH}")

    model = joblib.load(MODEL_PATH)
    with METADATA_PATH.open(encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata

model, metadata = load_artifacts()
threshold = float(metadata["threshold"])

st.title("Pemeriksaan Risiko Cacat")
st.write("Masukkan parameter satu siklus cetak untuk melihat estimasi risiko.")

with st.form("prediction_form"):
    bahan = st.selectbox("Bahan", ["PP", "ABS", "PC"])
    mesin = st.selectbox("Mesin", ["IM-01", "IM-02", "IM-03", "IM-04"])
    shift = st.selectbox("Shift", ["pagi", "siang", "malam"])

    suhu_leleh = st.number_input("Suhu leleh (°C)", min_value=0.0, max_value=400.0)
    suhu_cetakan = st.number_input("Suhu cetakan (°C)", min_value=0.0, max_value=200.0)
    tekanan = st.number_input("Tekanan injeksi (bar)", min_value=0.0)
    kecepatan = st.number_input("Kecepatan injeksi (mm/detik)", min_value=0.0)
    pendinginan = st.number_input("Waktu pendinginan (detik)", min_value=0.0)
    siklus = st.number_input("Waktu siklus (detik)", min_value=0.0)
    kelembapan = st.number_input("Kelembapan bahan (%)", min_value=0.0, max_value=100.0)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    values = {
        "mesin": mesin,
        "shift": shift,
        "bahan": bahan,
        "suhu_leleh_C": suhu_leleh,
        "suhu_cetakan_C": suhu_cetakan,
        "tekanan_injeksi_bar": tekanan,
        "kecepatan_injeksi_mm_s": kecepatan,
        "waktu_pendinginan_s": pendinginan,
        "waktu_siklus_s": siklus,
        "kelembaban_bahan_pct": kelembapan,
    }

    try:
        X = prepare_features(values)
        probability = float(model.predict_proba(X)[0, 1])
        st.metric("Probabilitas cacat", f"{probability:.1%}")

        if probability >= threshold:
            st.warning("Risiko melewati threshold. Periksa parameter dan komponen.")
        else:
            st.success("Risiko di bawah threshold. Ikuti prosedur QC yang berlaku.")

    except Exception as error:
        st.error(f"Prediksi gagal. Periksa format masukan. Detail: {error}")