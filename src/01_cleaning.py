"""
01_cleaning.py

Tahap pertama pipeline:
1. Membaca dataset berita NTT
2. Mengecek kolom dataset
3. Mengecek missing value
4. Menghapus baris yang judul/isi beritanya kosong
5. Mengecek duplikat tanpa menghapusnya
6. Menyimpan dataset hasil cleaning
"""

import sys
from pathlib import Path

import pandas as pd


# ==========================================================
# ROOT PROJECT
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ==========================================================
# PATH DATASET
# ==========================================================

RAW_DIR = ROOT_DIR / "data" / "raw"

RAW_DATA = RAW_DIR / "NTT_berita bersih.xlsx"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"

CLEAN_DATA = PROCESSED_DIR / "dataset_clean.xlsx"


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("DATA CLEANING DATASET NTT")
    print("=" * 60)

    # ======================================================
    # CEK FILE
    # ======================================================

    print("\nNama file :", RAW_DATA.name)

    # Aman untuk Windows/VS Code dengan encoding cp1252
    print(
        "Lokasi    :",
        str(RAW_DATA).encode(
            "ascii",
            "replace"
        ).decode()
    )

    print(
        "File ada? :",
        RAW_DATA.exists()
    )

    if not RAW_DATA.exists():

        raise FileNotFoundError(
            """
Dataset tidak ditemukan.

Pastikan file:
NTT_berita bersih.xlsx

berada di:
data/raw/
"""
        )

    # ======================================================
    # MEMBACA DATASET
    # ======================================================

    print("\nMembaca dataset...")

    df = pd.read_excel(
        RAW_DATA
    )

    print(
        "Dataset berhasil dibaca."
    )

    print(
        "Jumlah data awal :",
        len(df)
    )

    # ======================================================
    # NORMALISASI NAMA KOLOM
    # ======================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print("\nKolom dataset:")

    print(
        df.columns.tolist()
    )

    # ======================================================
    # VALIDASI KOLOM
    # ======================================================

    kolom_wajib = [
        "keyword",
        "judul",
        "tanggal",
        "media",
        "url",
        "isi_berita"
    ]

    kolom_hilang = [
        kolom
        for kolom in kolom_wajib
        if kolom not in df.columns
    ]

    if kolom_hilang:

        raise ValueError(
            f"""
Kolom berikut tidak ditemukan:

{kolom_hilang}

Kolom yang tersedia:

{df.columns.tolist()}
"""
        )

    print(
        "\nSemua kolom wajib tersedia."
    )

    # ======================================================
    # MISSING VALUE
    # ======================================================

    print(
        "\nMengecek Missing Value..."
    )

    missing = (
        df[kolom_wajib]
        .isnull()
        .sum()
    )

    print(
        missing
    )

    # ======================================================
    # HAPUS DATA TANPA JUDUL / ISI BERITA
    # ======================================================

    jumlah_awal = len(df)

    df["judul"] = (
        df["judul"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["isi_berita"] = (
        df["isi_berita"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["judul"] != "") &
        (df["isi_berita"] != "")
    ].copy()

    jumlah_terhapus = (
        jumlah_awal - len(df)
    )

    print(
        "\nData tanpa judul/isi yang dihapus :",
        jumlah_terhapus
    )

    # ======================================================
    # CEK DUPLIKAT
    # ======================================================

    print(
        "\nPengecekan duplikat..."
    )

    jumlah_duplikat_url = (
        df["url"]
        .duplicated()
        .sum()
    )

    jumlah_duplikat_judul = (
        df["judul"]
        .duplicated()
        .sum()
    )

    print(
        "Duplikat URL   :",
        jumlah_duplikat_url
    )

    print(
        "Duplikat judul :",
        jumlah_duplikat_judul
    )

    print(
        "\nPenghapusan duplikat dilewati."
    )

    # ======================================================
    # RESET INDEX
    # ======================================================

    df.reset_index(
        drop=True,
        inplace=True
    )

    # ======================================================
    # BUAT FOLDER OUTPUT
    # ======================================================

    CLEAN_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # SIMPAN DATASET
    # ======================================================

    print(
        "\nMenyimpan dataset cleaning..."
    )

    df.to_excel(
        CLEAN_DATA,
        index=False
    )

    print(
        "Dataset berhasil disimpan."
    )

    # ======================================================
    # HASIL AKHIR
    # ======================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "CLEANING SELESAI"
    )

    print(
        "=" * 60
    )

    print(
        "File input        :",
        RAW_DATA.name
    )

    print(
        "Jumlah data awal  :",
        jumlah_awal
    )

    print(
        "Jumlah data akhir :",
        len(df)
    )

    print(
        "Data dihapus      :",
        jumlah_terhapus
    )

    print(
        "Output            :",
        CLEAN_DATA.name
    )

    print(
        "Lokasi output     :",
        str(CLEAN_DATA).encode(
            "ascii",
            "replace"
        ).decode()
    )

    print(
        "=" * 60
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()