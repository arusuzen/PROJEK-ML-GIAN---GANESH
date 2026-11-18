import numpy as np
import pandas as pd
from collections.abc import Mapping

STD_SUHU_LELEH = {"PP": 230.0, "ABS": 240.0, "PC": 300.0}
STD_SUHU_CETAKAN = {"PP": 40.0, "ABS": 60.0, "PC": 90.0}

RAW_FEATURES = [
    "mesin",
    "shift",
    "bahan",
    "suhu_leleh_C",
    "suhu_cetakan_C",
    "tekanan_injeksi_bar",
    "kecepatan_injeksi_mm_s",
    "waktu_pendinginan_s",
    "waktu_siklus_s",
    "kelembaban_bahan_pct",
]

def clean_data(df: pd.DataFrame, *, convert_fahrenheit: bool = False) -> pd.DataFrame:
    df = df.copy()

    if "mesin" in df.columns:
        df["mesin"] = df["mesin"].astype(str).str.strip().str.upper()
        df.loc[~df["mesin"].isin(["IM-01", "IM-02", "IM-03", "IM-04"]), "mesin"] = np.nan

    if "shift" in df.columns:
        df["shift"] = df["shift"].astype(str).str.strip().str.lower()
        df.loc[~df["shift"].isin(["pagi", "siang", "malam"]), "shift"] = np.nan

    if "bahan" in df.columns:
        df["bahan"] = (
            df["bahan"].astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
        )
        df.loc[~df["bahan"].isin(["PP", "ABS", "PC"]), "bahan"] = np.nan

    num_cols = [
        "suhu_leleh_C", "suhu_cetakan_C", "tekanan_injeksi_bar",
        "kecepatan_injeksi_mm_s", "waktu_pendinginan_s", "waktu_siklus_s",
        "kelembaban_bahan_pct"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if convert_fahrenheit:
        f_mask = (
            df["mesin"].eq("IM-04")
            & df["suhu_leleh_C"].gt(380)
        )
        df.loc[f_mask, "suhu_leleh_C"] = (
            (df.loc[f_mask, "suhu_leleh_C"] - 32.0) * 5.0 / 9.0
        )

    df.loc[(df["suhu_leleh_C"] < 100) | (df["suhu_leleh_C"] > 400), "suhu_leleh_C"] = np.nan
    df.loc[(df["suhu_cetakan_C"] <= 0) | (df["suhu_cetakan_C"] > 200), "suhu_cetakan_C"] = np.nan
    df.loc[df["tekanan_injeksi_bar"] <= 0, "tekanan_injeksi_bar"] = np.nan
    df.loc[df["kecepatan_injeksi_mm_s"] <= 0, "kecepatan_injeksi_mm_s"] = np.nan
    df.loc[df["waktu_pendinginan_s"] <= 0, "waktu_pendinginan_s"] = np.nan
    df.loc[df["waktu_siklus_s"] <= 0, "waktu_siklus_s"] = np.nan
    df.loc[df["kelembaban_bahan_pct"] < 0, "kelembaban_bahan_pct"] = np.nan

    invalid_cycle = df["waktu_siklus_s"] <= df["waktu_pendinginan_s"]
    df.loc[invalid_cycle, ["waktu_siklus_s", "waktu_pendinginan_s"]] = np.nan

    return df

def build_features(df: pd.DataFrame, *, convert_fahrenheit: bool = False) -> pd.DataFrame:
    df = clean_data(df, convert_fahrenheit=convert_fahrenheit)

    std_leleh = df["bahan"].map(STD_SUHU_LELEH)
    std_cetakan = df["bahan"].map(STD_SUHU_CETAKAN)

    # Deviasi terhadap standar suhu tiap jenis plastik
    df["deviasi_suhu_leleh"] = df["suhu_leleh_C"] - std_leleh
    df["abs_deviasi_suhu_leleh"] = df["deviasi_suhu_leleh"].abs()
    df["deviasi_suhu_cetakan"] = df["suhu_cetakan_C"] - std_cetakan
    df["abs_deviasi_suhu_cetakan"] = df["deviasi_suhu_cetakan"].abs()
    df["selisih_suhu_leleh_cetakan"] = df["suhu_leleh_C"] - df["suhu_cetakan_C"]

    # Interaksi kelembaban pada bahan higroskopis (ABS & PC)
    df["is_higroskopis"] = df["bahan"].isin(["ABS", "PC"]).astype(float)
    df["risiko_kelembaban_higro"] = df["kelembaban_bahan_pct"] * df["is_higroskopis"]

    # Dinamika siklus & injeksi
    df["rasio_waktu_pendinginan"] = df["waktu_pendinginan_s"] / df["waktu_siklus_s"]
    df["waktu_isi_tahan_s"] = df["waktu_siklus_s"] - df["waktu_pendinginan_s"]
    df["indeks_laju_tekanan"] = df["tekanan_injeksi_bar"] * df["kecepatan_injeksi_mm_s"]

    return df

def prepare_features(values: Mapping[str, object]) -> pd.DataFrame:
    return build_features(pd.DataFrame([values]))
