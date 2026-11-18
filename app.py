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


def show_inline_warning(condition: bool, message: str) -> None:
    if condition:
        st.warning(message)


with st.form("prediction_form"):
    bahan = st.selectbox("Bahan", ["PP", "ABS", "PC"])
    mesin = st.selectbox("Mesin", ["IM-01", "IM-02", "IM-03", "IM-04"])
    shift = st.selectbox("Shift", ["pagi", "siang", "malam"])

    suhu_leleh = st.number_input(
        "Suhu leleh (°C)",
        min_value=-100.0,
        max_value=1000.0,
        value=250.0,
        step=1.0,
    )
    show_inline_warning(
        suhu_leleh < 100 or suhu_leleh > 400,
        "Suhu leleh harus berada di antara 100°C dan 400°C.",
    )
    suhu_cetakan = st.number_input(
        "Suhu cetakan (°C)",
        min_value=-100.0,
        max_value=500.0,
        value=55.0,
        step=1.0,
    )
    show_inline_warning(
        suhu_cetakan <= 0 or suhu_cetakan > 200,
        "Suhu cetakan harus lebih dari 0°C dan tidak boleh melebihi 200°C.",
    )
    tekanan = st.number_input(
        "Tekanan injeksi (bar)",
        min_value=-100.0,
        max_value=2000.0,
        value=900.0,
        step=10.0,
    )
    show_inline_warning(
        tekanan <= 0 or tekanan > 1500,
        "Tekanan injeksi harus lebih dari 0 bar dan tidak boleh melebihi 1.500 bar.",
    )
    kecepatan = st.number_input(
        "Kecepatan injeksi (mm/detik)",
        min_value=-100.0,
        max_value=500.0,
        value=80.0,
        step=1.0,
    )
    show_inline_warning(
        kecepatan <= 0 or kecepatan > 300,
        "Kecepatan injeksi harus lebih dari 0 dan tidak boleh melebihi 300 mm/detik.",
    )
    pendinginan = st.number_input(
        "Waktu pendinginan (detik)",
        min_value=-100.0,
        max_value=500.0,
        value=18.0,
        step=1.0,
    )
    show_inline_warning(
        pendinginan <= 0 or pendinginan > 100,
        "Waktu pendinginan harus lebih dari 0 detik dan tidak boleh melebihi 100 detik.",
    )
    siklus = st.number_input(
        "Waktu siklus (detik)",
        min_value=-100.0,
        max_value=600.0,
        value=32.0,
        step=1.0,
    )
    siklus_message = None
    if siklus <= 0 or siklus > 100:
        siklus_message = "Waktu siklus harus lebih dari 0 detik dan tidak boleh melebihi 100 detik."
    elif siklus <= pendinginan:
        siklus_message = "Waktu siklus harus lebih lama daripada waktu pendinginan."
    show_inline_warning(siklus_message is not None, siklus_message or "")
    kelembapan = st.number_input(
        "Kelembapan bahan (%)",
        min_value=-10.0,
        max_value=110.0,
        value=5.0,
        step=0.1,
    )
    show_inline_warning(
        kelembapan < 0 or kelembapan > 100,
        "Kelembapan bahan harus berada di antara 0% dan 100%.",
    )

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

    has_invalid_value = (
        suhu_leleh < 100
        or suhu_leleh > 400
        or suhu_cetakan <= 0
        or suhu_cetakan > 200
        or tekanan <= 0
        or tekanan > 1500
        or kecepatan <= 0
        or kecepatan > 300
        or pendinginan <= 0
        or pendinginan > 100
        or siklus <= 0
        or siklus > 100
        or siklus <= pendinginan
        or kelembapan < 0
        or kelembapan > 100
    )

    if not has_invalid_value:
        try:
            X = prepare_features(values)
            probability = float(model.predict_proba(X)[0, 1])

            st.metric("Probabilitas cacat", f"{probability:.1%}")
            if probability >= threshold:
                st.warning("Risiko melewati threshold. Periksa parameter dan komponen.")
            else:
                st.success("Risiko di bawah threshold. Ikuti prosedur QC yang berlaku.")

        except Exception as error:
            st.error(f"Prediksi gagal. Periksa kembali masukan. Detail: {error}")