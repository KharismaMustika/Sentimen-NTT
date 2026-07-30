"""
05_train.py

Training Model SVM untuk Analisis Sentimen Berita NTT

Tahapan:
1. Membaca dataset hasil labeling
2. Mengecek text_final dan label
3. Mengecek data kosong
4. Membagi data menjadi train dan test
5. Melakukan TF-IDF pada data train
6. Training Linear SVM
7. Menyimpan model SVM
8. Menyimpan TF-IDF vectorizer
"""

import sys
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

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from config import (
    LABELED_DATA,
    SVM_MODEL,
    TFIDF_MODEL,
    RANDOM_STATE,
    TEST_SIZE,
)


# ==========================================================
# FUNGSI PATH AMAN UNTUK WINDOWS
# ==========================================================

def safe_path(path):

    return str(path).encode(
        "ascii",
        "replace"
    ).decode()


# ==========================================================
# MAIN
# ==========================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("TRAINING MODEL SVM DATASET NTT")
    print("=" * 60)

    # ======================================================
    # CEK FILE DATASET
    # ======================================================

    print("\nMengecek file dataset...")

    if not LABELED_DATA.exists():

        raise FileNotFoundError(
            "Dataset labeling tidak ditemukan."
        )

    print("Dataset labeling ditemukan.")

    # ======================================================
    # MEMBACA DATASET
    # ======================================================

    print("\nMembaca dataset...")

    df = pd.read_excel(
        LABELED_DATA
    )

    print("Dataset berhasil dibaca.")
    print("Jumlah data :", len(df))

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
    print(df.columns.tolist())

    # ======================================================
    # CEK KOLOM WAJIB
    # ======================================================

    print("\nMengecek kolom dataset...")

    kolom_wajib = [
        "text_final",
        "label"
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

    print("Kolom text_final ditemukan.")
    print("Kolom label ditemukan.")

    # ======================================================
    # HANDLE MISSING VALUE
    # ======================================================

    print("\nMengecek missing value...")

    df["text_final"] = (
        df["text_final"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    print(
        "Text kosong :",
        df["text_final"].eq("").sum()
    )

    print(
        "Label kosong:",
        df["label"].eq("").sum()
    )

    # ======================================================
    # DATA VALID
    # ======================================================

    print("\nMengecek data yang dapat digunakan...")

    mask_valid = (
        (df["text_final"] != "") &
        (df["label"] != "")
    )

    df = df[
        mask_valid
    ].copy()

    jumlah_akhir = len(df)

    print(
        "Jumlah data awal    :",
        jumlah_awal
    )

    print(
        "Jumlah data valid   :",
        jumlah_akhir
    )

    print(
        "Data tidak digunakan:",
        jumlah_awal - jumlah_akhir
    )

    if jumlah_akhir == 0:

        raise ValueError(
            "Tidak ada data yang dapat digunakan untuk training."
        )

    # ======================================================
    # DISTRIBUSI LABEL
    # ======================================================

    print("\nDistribusi label:")
    print("-" * 40)

    distribusi = df[
        "label"
    ].value_counts()

    print(distribusi)

    print("-" * 40)

    # ======================================================
    # CEK JUMLAH KELAS
    # ======================================================

    jumlah_kelas = df[
        "label"
    ].nunique()

    print(
        "\nJumlah kelas sentimen:",
        jumlah_kelas
    )

    if jumlah_kelas < 2:

        raise ValueError(
            "Training SVM membutuhkan minimal 2 kelas."
        )

    # ======================================================
    # TRAIN TEST SPLIT
    # ======================================================

    print(
        "\nMembagi data train dan test..."
    )

    X = df[
        "text_final"
    ]

    y = df[
        "label"
    ]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )

    print("\nPembagian dataset:")
    print("-" * 40)

    print(
        "Total data :",
        len(df)
    )

    print(
        "Data train :",
        len(X_train)
    )

    print(
        "Data test  :",
        len(X_test)
    )

    print("-" * 40)

    # ======================================================
    # DISTRIBUSI TRAIN
    # ======================================================

    print(
        "\nDistribusi label data train:"
    )

    print(
        y_train.value_counts()
    )

    # ======================================================
    # DISTRIBUSI TEST
    # ======================================================

    print(
        "\nDistribusi label data test:"
    )

    print(
        y_test.value_counts()
    )

    # ======================================================
    # TF-IDF
    # ======================================================

    print("\n" + "=" * 60)
    print("MELAKUKAN TF-IDF")
    print("=" * 60)

    print(
        "\nMembuat TF-IDF Vectorizer..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=False
    )

    # ======================================================
    # FIT DATA TRAIN
    # ======================================================

    print(
        "\nFit TF-IDF pada data train..."
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    # ======================================================
    # TRANSFORM DATA TEST
    # ======================================================

    print(
        "Transform data test..."
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print(
        "TF-IDF selesai."
    )

    # ======================================================
    # INFORMASI TF-IDF
    # ======================================================

    print("\nInformasi TF-IDF:")
    print("-" * 40)

    print(
        "Jumlah dokumen train :",
        X_train_tfidf.shape[0]
    )

    print(
        "Jumlah dokumen test  :",
        X_test_tfidf.shape[0]
    )

    print(
        "Jumlah fitur         :",
        X_train_tfidf.shape[1]
    )

    print(
        "Bentuk train matrix  :",
        X_train_tfidf.shape
    )

    print(
        "Bentuk test matrix   :",
        X_test_tfidf.shape
    )

    print("-" * 40)

    # ======================================================
    # TRAINING SVM
    # ======================================================

    print("\n" + "=" * 60)
    print("TRAINING LINEAR SVM")
    print("=" * 60)

    model = LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    print(
        "\nMelatih model..."
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    print(
        "Training model selesai."
    )

    # ======================================================
    # PREDIKSI DATA TEST
    # ======================================================

    print(
        "\nMelakukan prediksi data test..."
    )

    y_pred = model.predict(
        X_test_tfidf
    )

    print(
        "Prediksi berhasil."
    )

    # ======================================================
    # SIMPAN MODEL
    # ======================================================

    print("\n" + "=" * 60)
    print("MENYIMPAN MODEL")
    print("=" * 60)

    SVM_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    TFIDF_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # SIMPAN SVM
    # ======================================================

    print(
        "\nMenyimpan model SVM..."
    )

    joblib.dump(
        model,
        SVM_MODEL
    )

    print(
        "Model SVM berhasil disimpan."
    )

    print(
        "Lokasi :",
        safe_path(SVM_MODEL)
    )

    # ======================================================
    # SIMPAN TF-IDF
    # ======================================================

    print(
        "\nMenyimpan TF-IDF vectorizer..."
    )

    joblib.dump(
        vectorizer,
        TFIDF_MODEL
    )

    print(
        "TF-IDF vectorizer berhasil disimpan."
    )

    print(
        "Lokasi :",
        safe_path(TFIDF_MODEL)
    )

    # ======================================================
    # HASIL AKHIR
    # ======================================================

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("TRAINING SVM SELESAI")
    print("=" * 60)

    print(
        "Jumlah data awal     :",
        jumlah_awal
    )

    print(
        "Jumlah data digunakan:",
        len(df)
    )

    print(
        "Data train           :",
        len(X_train)
    )

    print(
        "Data test            :",
        len(X_test)
    )

    print(
        "Jumlah fitur TF-IDF  :",
        X_train_tfidf.shape[1]
    )

    print(
        "Jumlah kelas         :",
        jumlah_kelas
    )

    print(
        "Model SVM            : BERHASIL"
    )

    print(
        "TF-IDF Vectorizer    : BERHASIL"
    )

    print(
        "Model SVM            :",
        safe_path(SVM_MODEL)
    )

    print(
        "TF-IDF Vectorizer    :",
        safe_path(TFIDF_MODEL)
    )

    print(
        "Waktu proses         : {:.2f} detik".format(
            elapsed
        )
    )

    print("=" * 60)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()