"""
04_tfidf.py

Tahap TF-IDF:
1. Membaca dataset hasil labeling
2. Memastikan text_final dan label tersedia
3. Mengecek data kosong
4. Melakukan TF-IDF Vectorization
5. Menambahkan label
6. Menyimpan hasil TF-IDF ke CSV

Catatan:
- Tidak menghapus duplikat
- Tidak mengubah jumlah data kecuali text_final kosong
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

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    LABELED_DATA,
    TFIDF_DATA
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("TF-IDF VECTORIZATION DATASET NTT")
    print("=" * 60)

    # ======================================================
    # CEK FILE
    # ======================================================

    print("\nMengecek file dataset labeling...")

    print(
        "Lokasi :",
        str(LABELED_DATA)
        .encode("ascii", "replace")
        .decode()
    )

    if not LABELED_DATA.exists():

        raise FileNotFoundError(
            "\nDataset labeling tidak ditemukan:\n"
            + str(LABELED_DATA)
            + "\n\n"
            "Pastikan 03_labeling.py sudah berhasil dijalankan."
        )

    print("File ditemukan.")

    # ======================================================
    # MEMBACA DATASET
    # ======================================================

    print("\nMembaca dataset labeling...")

    df = pd.read_excel(
        LABELED_DATA
    )

    print("Dataset berhasil dibaca.")

    print(
        "Jumlah data awal :",
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
            "\nKolom 'text_final' tidak ditemukan.\n\n"
            "Kolom tersedia:\n"
            + str(df.columns.tolist())
        )

    print(
        "\nKolom text_final ditemukan."
    )

    # ======================================================
    # CEK KOLOM LABEL
    # ======================================================

    if "label" not in df.columns:

        raise KeyError(
            "\nKolom 'label' tidak ditemukan.\n\n"
            "Kolom tersedia:\n"
            + str(df.columns.tolist())
        )

    print(
        "Kolom label ditemukan."
    )

    # ======================================================
    # CEK DATA KOSONG
    # ======================================================

    print(
        "\nMengecek text_final kosong..."
    )

    df["text_final"] = (
        df["text_final"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    kosong = (
        df["text_final"] == ""
    ).sum()

    print(
        "Text final kosong :",
        kosong
    )

    # ======================================================
    # HAPUS TEXT KOSONG
    # ======================================================

    if kosong > 0:

        print(
            "\nMenghapus baris dengan text_final kosong..."
        )

        df = df[
            df["text_final"] != ""
        ].copy()

    else:

        print(
            "\nTidak ada text_final kosong."
        )

    jumlah_setelah = len(df)

    print(
        "Data yang diproses :",
        jumlah_setelah
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
        "Label kosong       :",
        label_kosong
    )

    if label_kosong > 0:

        raise ValueError(
            "Terdapat label kosong. "
            "Periksa kembali hasil labeling."
        )

    # ======================================================
    # DISTRIBUSI LABEL SEBELUM TF-IDF
    # ======================================================

    print(
        "\nDistribusi label sebelum TF-IDF:"
    )

    print(
        df["label"].value_counts()
    )

    # ======================================================
    # TF-IDF
    # ======================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "MELAKUKAN TF-IDF VECTORIZATION"
    )

    print(
        "=" * 60
    )

    print(
        "\nMembuat TF-IDF Vectorizer..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True
    )

    print(
        "Melakukan transformasi TF-IDF..."
    )

    X = vectorizer.fit_transform(
        df["text_final"]
    )

    print(
        "TF-IDF berhasil."
    )

    # ======================================================
    # INFORMASI MATRIX
    # ======================================================

    jumlah_dokumen = X.shape[0]

    jumlah_fitur = X.shape[1]

    print(
        "\nInformasi TF-IDF:"
    )

    print(
        "Jumlah dokumen :",
        jumlah_dokumen
    )

    print(
        "Jumlah fitur   :",
        jumlah_fitur
    )

    print(
        "Bentuk matrix  :",
        X.shape
    )

    # ======================================================
    # FEATURE NAMES
    # ======================================================

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    print(
        "\nJumlah vocabulary :",
        len(feature_names)
    )

    # ======================================================
    # BUAT DATAFRAME
    # ======================================================

    print(
        "\nMembuat DataFrame TF-IDF..."
    )

    tfidf_df = pd.DataFrame(
        X.toarray(),
        columns=feature_names
    )

    # ======================================================
    # TAMBAHKAN LABEL
    # ======================================================

    tfidf_df["label"] = (
        df["label"]
        .values
    )

    # ======================================================
    # RESET INDEX
    # ======================================================

    tfidf_df.reset_index(
        drop=True,
        inplace=True
    )

    # ======================================================
    # CEK HASIL
    # ======================================================

    print(
        "\nUkuran dataset TF-IDF:"
    )

    print(
        tfidf_df.shape
    )

    print(
        "\nContoh 5 baris pertama:"
    )

    print(
        tfidf_df.head()
    )

    # ======================================================
    # DISTRIBUSI LABEL
    # ======================================================

    print(
        "\nDistribusi label hasil TF-IDF:"
    )

    print(
        tfidf_df["label"]
        .value_counts()
    )

    # ======================================================
    # FOLDER OUTPUT
    # ======================================================

    TFIDF_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # SIMPAN CSV
    # ======================================================

    print(
        "\nMenyimpan dataset TF-IDF..."
    )

    tfidf_df.to_csv(
        TFIDF_DATA,
        index=False
    )

    # ======================================================
    # CEK FILE
    # ======================================================

    if not TFIDF_DATA.exists():

        raise RuntimeError(
            "File TF-IDF gagal dibuat."
        )

    print(
        "File TF-IDF berhasil disimpan."
    )

    # ======================================================
    # CEK JUMLAH DATA
    # ======================================================

    jumlah_output = len(tfidf_df)

    print(
        "\nPengecekan jumlah data:"
    )

    print(
        "Jumlah awal      :",
        jumlah_awal
    )

    print(
        "Jumlah diproses  :",
        jumlah_setelah
    )

    print(
        "Jumlah output    :",
        jumlah_output
    )

    if jumlah_awal == jumlah_output:

        print(
            "Status : JUMLAH DATA TETAP"
        )

    else:

        print(
            "Status : JUMLAH DATA BERUBAH"
        )

    # ======================================================
    # HASIL AKHIR
    # ======================================================

    elapsed = (
        time.time()
        - start_time
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "TF-IDF SELESAI"
    )

    print(
        "=" * 60
    )

    print(
        "Jumlah dokumen :",
        jumlah_dokumen
    )

    print(
        "Jumlah fitur   :",
        jumlah_fitur
    )

    print(
        "Jumlah output  :",
        jumlah_output
    )

    print(
        "Output file    :",
        TFIDF_DATA.name
    )

    print(
        "Lokasi output  :",
        str(TFIDF_DATA)
        .encode("ascii", "replace")
        .decode()
    )

    print(
        "Waktu proses   : {:.2f} detik".format(
            elapsed
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