"""
03_labeling.py

Labeling Sentimen Dataset Berita NTT

Tahapan:
1. Membaca dataset hasil preprocessing
2. Membaca lexicon positif dan negatif
3. Menghitung skor sentimen setiap berita
4. Menentukan label:
   - positif
   - negatif
5. Menyimpan hasil labeling

Catatan:
- Jumlah berita dipertahankan
- Tidak menghapus duplikat
- Tidak menghapus berita
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

from config import (
    PREPROCESS_DATA,
    LABELED_DATA,
    POSITIVE_LEXICON,
    NEGATIVE_LEXICON
)


# ==========================================================
# LOAD LEXICON
# ==========================================================

def load_lexicon(path):

    if not path.exists():

        raise FileNotFoundError(
            "\nFile lexicon tidak ditemukan:\n"
            + str(path)
        )

    words = set()

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Lewati header jika ada
            # ------------------------------------------------

            lower_line = line.lower()

            if lower_line in [
                "word",
                "kata",
                "positive",
                "negative",
                "positif",
                "negatif"
            ]:

                continue

            # ------------------------------------------------
            # Jika TSV memiliki beberapa kolom,
            # ambil kolom pertama
            # ------------------------------------------------

            if "\t" in line:

                word = line.split("\t")[0].strip()

            else:

                word = line.strip()

            word = word.lower()

            if word:
                words.add(word)

    return words


# ==========================================================
# TOKENIZER SEDERHANA
# ==========================================================

def get_tokens(text):

    if pd.isna(text):
        return []

    text = str(text).lower()

    tokens = re.findall(
        r"[a-z]+",
        text
    )

    return tokens


# ==========================================================
# HITUNG SENTIMEN
# ==========================================================

def calculate_sentiment(
    text,
    positive_words,
    negative_words
):

    tokens = get_tokens(text)

    positive_count = 0
    negative_count = 0

    positive_terms = []
    negative_terms = []

    for token in tokens:

        if token in positive_words:

            positive_count += 1
            positive_terms.append(token)

        if token in negative_words:

            negative_count += 1
            negative_terms.append(token)

    score = (
        positive_count
        - negative_count
    )

    # ------------------------------------------------------
    # LABEL
    # ------------------------------------------------------

    if score > 0:

        label = "positif"

    elif score < 0:

        label = "negatif"

    else:

        # --------------------------------------------------
        # Jika skor 0, gunakan jumlah kata yang cocok.
        # Jika tetap tidak ada, label netral.
        # --------------------------------------------------

        if positive_count > negative_count:

            label = "positif"

        elif negative_count > positive_count:

            label = "negatif"

        else:

            label = "netral"

    return (
        positive_count,
        negative_count,
        score,
        label,
        positive_terms,
        negative_terms
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("LABELING SENTIMEN DATASET NTT")
    print("=" * 60)

    # ======================================================
    # CEK DATASET PREPROCESSING
    # ======================================================

    print("\nDataset preprocessing:")

    print(
        str(PREPROCESS_DATA)
        .encode("ascii", "replace")
        .decode()
    )

    if not PREPROCESS_DATA.exists():

        raise FileNotFoundError(
            "\nDataset preprocessing tidak ditemukan.\n"
            "Jalankan terlebih dahulu:\n\n"
            "python -m src.02_preprocessing\n\n"
            "File yang dibutuhkan:\n"
            + str(PREPROCESS_DATA)
        )

    print("File ditemukan.")

    # ======================================================
    # BACA DATASET
    # ======================================================

    print("\nMembaca dataset...")

    df = pd.read_excel(
        PREPROCESS_DATA
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
    # NORMALISASI KOLOM
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
    # CEK KOLOM TEXT_FINAL
    # ======================================================

    if "text_final" not in df.columns:

        raise KeyError(
            "\nKolom 'text_final' tidak ditemukan.\n"
            "Pastikan 02_preprocessing.py sudah berhasil dijalankan.\n\n"
            "Kolom tersedia:\n"
            + str(df.columns.tolist())
        )

    # ======================================================
    # LOAD LEXICON
    # ======================================================

    print("\nMembaca lexicon positif...")

    positive_words = load_lexicon(
        POSITIVE_LEXICON
    )

    print(
        "Jumlah kata positif :",
        len(positive_words)
    )

    print("\nMembaca lexicon negatif...")

    negative_words = load_lexicon(
        NEGATIVE_LEXICON
    )

    print(
        "Jumlah kata negatif :",
        len(negative_words)
    )

    # ======================================================
    # CEK LEXICON
    # ======================================================

    if len(positive_words) == 0:

        raise ValueError(
            "Lexicon positif kosong."
        )

    if len(negative_words) == 0:

        raise ValueError(
            "Lexicon negatif kosong."
        )

    # ======================================================
    # LABELING
    # ======================================================

    print("\n" + "=" * 60)
    print("MELAKUKAN LABELING")
    print("=" * 60)

    positive_counts = []
    negative_counts = []
    sentiment_scores = []
    labels = []
    positive_terms_all = []
    negative_terms_all = []

    for text in tqdm(
        df["text_final"].fillna(""),
        total=len(df),
        desc="Labeling"
    ):

        (
            positive_count,
            negative_count,
            score,
            label,
            positive_terms,
            negative_terms
        ) = calculate_sentiment(
            text,
            positive_words,
            negative_words
        )

        positive_counts.append(
            positive_count
        )

        negative_counts.append(
            negative_count
        )

        sentiment_scores.append(
            score
        )

        labels.append(
            label
        )

        positive_terms_all.append(
            ", ".join(positive_terms)
        )

        negative_terms_all.append(
            ", ".join(negative_terms)
        )

    # ======================================================
    # TAMBAHKAN KOLOM
    # ======================================================

    df["positive_count"] = (
        positive_counts
    )

    df["negative_count"] = (
        negative_counts
    )

    df["sentiment_score"] = (
        sentiment_scores
    )

    df["positive_terms"] = (
        positive_terms_all
    )

    df["negative_terms"] = (
        negative_terms_all
    )

    df["label"] = labels

    # ======================================================
    # JUMLAH DATA
    # ======================================================

    jumlah_akhir = len(df)

    print("\n" + "-" * 60)
    print("HASIL LABELING")
    print("-" * 60)

    print(
        "Jumlah data awal  :",
        jumlah_awal
    )

    print(
        "Jumlah data akhir :",
        jumlah_akhir
    )

    # ======================================================
    # DISTRIBUSI LABEL
    # ======================================================

    print("\nDistribusi label:")

    distribusi = (
        df["label"]
        .value_counts()
    )

    print(
        distribusi
    )

    print("\nPersentase label:")

    persentase = (
        df["label"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print(
        persentase
    )

    # ======================================================
    # CEK LABEL
    # ======================================================

    jumlah_positif = (
        (df["label"] == "positif")
        .sum()
    )

    jumlah_negatif = (
        (df["label"] == "negatif")
        .sum()
    )

    jumlah_netral = (
        (df["label"] == "netral")
        .sum()
    )

    print("\nRincian label:")

    print(
        "Positif :",
        jumlah_positif
    )

    print(
        "Negatif :",
        jumlah_negatif
    )

    print(
        "Netral  :",
        jumlah_netral
    )

    # ======================================================
    # CEK LABEL KOSONG
    # ======================================================

    label_kosong = (
        df["label"]
        .isna()
        .sum()
    )

    print(
        "\nLabel kosong :",
        label_kosong
    )

    # ======================================================
    # CEK JUMLAH DATA
    # ======================================================

    if jumlah_awal == jumlah_akhir:

        print(
            "Status data : JUMLAH DATA TETAP"
        )

    else:

        print(
            "PERINGATAN: jumlah data berubah!"
        )

    # ======================================================
    # RESET INDEX
    # ======================================================

    df.reset_index(
        drop=True,
        inplace=True
    )

    # ======================================================
    # SIMPAN
    # ======================================================

    LABELED_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nMenyimpan dataset hasil labeling..."
    )

    df.to_excel(
        LABELED_DATA,
        index=False
    )

    # ======================================================
    # HASIL AKHIR
    # ======================================================

    waktu = (
        time.time()
        - start_time
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "LABELING SELESAI"
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
        "Positif           :",
        jumlah_positif
    )

    print(
        "Negatif           :",
        jumlah_negatif
    )

    print(
        "Netral            :",
        jumlah_netral
    )

    print(
        "Output file       :",
        LABELED_DATA.name
    )

    print(
        "Lokasi output     :",
        str(LABELED_DATA)
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