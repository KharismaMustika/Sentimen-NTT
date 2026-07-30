"""
02_preprocessing.py

Preprocessing Dataset Berita 

Tahapan:
1. Membaca dataset hasil cleaning
2. Menggabungkan judul + isi berita
3. Cleaning teks
4. Tokenisasi
5. Stopword removal
6. Stemming Bahasa Indonesia
7. Menyimpan hasil preprocessing

Catatan:
- Jumlah berita dipertahankan
- Tidak menghapus duplikat
- Dataset digunakan untuk tahap labeling dan TF-IDF
"""

import sys
import re
import time
from pathlib import Path

# ==========================================================
# ROOT PROJECT
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ==========================================================
# LIBRARY
# ==========================================================

import pandas as pd
from tqdm import tqdm

from config import CLEAN_DATA, PREPROCESS_DATA

from src.utils import (
    tokenize,
    remove_stopwords,
    stemming,
    join_tokens
)

tqdm.pandas()


# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # ------------------------------------------------------
    # lowercase
    # ------------------------------------------------------

    text = text.lower()

    # ------------------------------------------------------
    # hapus URL
    # ------------------------------------------------------

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # ------------------------------------------------------
    # hapus email
    # ------------------------------------------------------

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # ------------------------------------------------------
    # hapus HTML
    # ------------------------------------------------------

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # ------------------------------------------------------
    # hapus copyright
    # ------------------------------------------------------

    text = re.sub(
        r"copyright.*",
        " ",
        text
    )

    # ------------------------------------------------------
    # hapus angka
    # ------------------------------------------------------

    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # ------------------------------------------------------
    # hapus tanda baca / karakter selain huruf
    # ------------------------------------------------------

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # ------------------------------------------------------
    # hapus spasi berlebih
    # ------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# PREPROCESSING TEXT
# ==========================================================

def preprocess_text(text):

    # Cleaning
    text = clean_text(text)

    # Jika kosong
    if not text:
        return ""

    # Tokenisasi
    tokens = tokenize(text)

    # Stopword removal
    tokens = remove_stopwords(tokens)

    # Stemming
    tokens = stemming(tokens)

    # Gabungkan kembali
    text_final = join_tokens(tokens)

    return text_final


# ==========================================================
# MAIN
# ==========================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("PREPROCESSING DATASET NTT")
    print("=" * 60)

    # ======================================================
    # CEK FILE INPUT
    # ======================================================

    print("\nFile input:")
    print(
        str(CLEAN_DATA)
        .encode("ascii", "replace")
        .decode()
    )

    if not CLEAN_DATA.exists():

        raise FileNotFoundError(
            "\nFile dataset cleaning tidak ditemukan.\n"
            "Pastikan jalankan 01_cleaning.py terlebih dahulu.\n\n"
            + str(CLEAN_DATA)
        )

    print("File ditemukan.")

    # ======================================================
    # BACA DATASET
    # ======================================================

    print("\nMembaca dataset...")

    df = pd.read_excel(
        CLEAN_DATA
    )

    print(
        "Dataset berhasil dibaca."
    )

    print(
        "Jumlah data :",
        len(df)
    )

    jumlah_awal = len(df)

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
    # CEK KOLOM WAJIB
    # ======================================================

    kolom_wajib = [
        "judul",
        "isi_berita"
    ]

    kolom_hilang = [
        kolom
        for kolom in kolom_wajib
        if kolom not in df.columns
    ]

    if kolom_hilang:

        raise KeyError(
            "\nKolom berikut tidak ditemukan:\n"
            + str(kolom_hilang)
            + "\n\nKolom tersedia:\n"
            + str(df.columns.tolist())
        )

    print(
        "\nSemua kolom wajib tersedia."
    )

    # ======================================================
    # HANDLE MISSING VALUE
    # ======================================================

    print(
        "\nMengecek missing value..."
    )

    print(
        df[kolom_wajib].isnull().sum()
    )

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

    # ======================================================
    # GABUNG JUDUL + ISI BERITA
    # ======================================================

    print(
        "\nMenggabungkan judul dan isi berita..."
    )

    df["text"] = (
        df["judul"]
        + " "
        + df["isi_berita"]
    )

    # ======================================================
    # CLEANING
    # ======================================================

    print(
        "\nMembersihkan teks..."
    )

    df["text_clean"] = (
        df["text"]
        .progress_apply(clean_text)
    )

    # ======================================================
    # TOKENISASI
    # ======================================================

    print(
        "\nMelakukan tokenisasi..."
    )

    df["tokens"] = (
        df["text_clean"]
        .apply(tokenize)
    )

    # ======================================================
    # STOPWORD REMOVAL
    # ======================================================

    print(
        "\nMenghapus stopword..."
    )

    df["tokens_no_stopword"] = (
        df["tokens"]
        .apply(remove_stopwords)
    )

    # ======================================================
    # STEMMING
    # ======================================================

    print(
        "\nMelakukan stemming Bahasa Indonesia..."
    )

    df["tokens_stemmed"] = (
        df["tokens_no_stopword"]
        .progress_apply(stemming)
    )

    # ======================================================
    # TEXT FINAL
    # ======================================================

    print(
        "\nMembuat text_final..."
    )

    df["text_final"] = (
        df["tokens_stemmed"]
        .apply(join_tokens)
    )

    # ======================================================
    # DUPLIKAT
    # ======================================================

    print(
        "\nPengecekan jumlah data..."
    )

    print(
        "Penghapusan duplikat dilewati."
    )

    print(
        "Tidak ada baris yang dihapus pada tahap preprocessing."
    )

    # ======================================================
    # RESET INDEX
    # ======================================================

    df.reset_index(
        drop=True,
        inplace=True
    )

    # ======================================================
    # CEK JUMLAH DATA
    # ======================================================

    jumlah_akhir = len(df)

    print(
        "\n" + "-" * 60
    )

    print(
        "HASIL PREPROCESSING"
    )

    print(
        "-" * 60
    )

    print(
        "Jumlah data awal  :",
        jumlah_awal
    )

    print(
        "Jumlah data akhir :",
        jumlah_akhir
    )

    if jumlah_awal == jumlah_akhir:

        print(
            "Status            : JUMLAH DATA TETAP"
        )

    else:

        print(
            "Status            : PERINGATAN - JUMLAH DATA BERUBAH"
        )

    # ======================================================
    # CEK TEXT FINAL KOSONG
    # ======================================================

    kosong = (
        df["text_final"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        "Text final kosong :",
        kosong
    )

    # ======================================================
    # SIMPAN DATASET
    # ======================================================

    PREPROCESS_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nMenyimpan dataset preprocessing..."
    )

    df.to_excel(
        PREPROCESS_DATA,
        index=False
    )

    # ======================================================
    # HASIL AKHIR
    # ======================================================

    waktu = time.time() - start_time

    print(
        "\n" + "=" * 60
    )

    print(
        "PREPROCESSING SELESAI"
    )

    print(
        "=" * 60
    )

    print(
        "Jumlah data awal  :",
        jumlah_awal
    )

    print(
        "Jumlah data akhir :",
        jumlah_akhir
    )

    print(
        "Text final kosong :",
        kosong
    )

    print(
        "Output file       :",
        PREPROCESS_DATA.name
    )

    print(
        "Lokasi output     :",
        str(PREPROCESS_DATA)
        .encode("ascii", "replace")
        .decode()
    )

    print(
        "Waktu proses      : {:.2f} detik".format(
            waktu
        )
    )

    print(
        "=" * 60
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()