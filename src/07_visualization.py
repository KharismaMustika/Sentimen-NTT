"""
07_visualization.py

Visualisasi Analisis Sentimen Berita Pelayanan Publik NTT

Visualisasi:
1. Distribusi Sentimen
2. Performa Model Linear SVM
3. Confusion Matrix
4. Jumlah Berita Positif Berdasarkan Topik
5. Jumlah Berita Negatif Berdasarkan Topik

Dataset:
493 berita NTT
Positif : 246
Negatif : 247

Output:
results/
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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from config import (
    LABELED_DATA,
    SVM_MODEL,
    TFIDF_MODEL,
    RESULT_DIR,
    RANDOM_STATE,
    TEST_SIZE
)


# ==========================================================
# FOLDER OUTPUT
# ==========================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# 1. DISTRIBUSI SENTIMEN
# ==========================================================

def visualisasi_distribusi_sentimen(df):

    print("\n[1/5] Membuat distribusi sentimen...")

    positif = (
        df["label"] == "Positif"
    ).sum()

    negatif = (
        df["label"] == "Negatif"
    ).sum()

    labels = [
        "Positif",
        "Negatif"
    ]

    values = [
        positif,
        negatif
    ]

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        labels,
        values,
        width=0.55
    )

    plt.title(
        "Distribusi Sentimen Berita Pelayanan Publik NTT",
        fontsize=14,
        weight="bold"
    )

    plt.xlabel(
        "Sentimen"
    )

    plt.ylabel(
        "Jumlah Berita"
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 3,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold"
        )

    plt.ylim(
        0,
        max(values) + 35
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    output = (
        RESULT_DIR /
        "01_distribusi_sentimen.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "✓ Distribusi sentimen berhasil disimpan:"
    )

    print(output)


# ==========================================================
# 2. PERFORMA MODEL
# ==========================================================

def visualisasi_performa_model(
    y_test,
    y_pred
):

    print("\n[2/5] Membuat performa model...")

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Positif",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Positif",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Positif",
        zero_division=0
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]

    values = [
        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100
    ]

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        metrics,
        values,
        width=0.55
    )

    plt.title(
        "Performa Model Linear SVM",
        fontsize=14,
        weight="bold"
    )

    plt.xlabel(
        "Metrik Evaluasi"
    )

    plt.ylabel(
        "Nilai (%)"
    )

    plt.ylim(
        0,
        100
    )

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold"
        )

    plt.xticks(
        fontsize=10
    )

    plt.yticks(
        fontsize=9
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    output = (
        RESULT_DIR /
        "02_performa_model.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "✓ Performa model berhasil disimpan:"
    )

    print(output)

    print("\nHasil evaluasi model:")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )


# ==========================================================
# 3. CONFUSION MATRIX
# ==========================================================

def visualisasi_confusion_matrix(
    y_test,
    y_pred
):

    print("\n[3/5] Membuat confusion matrix...")

    labels = [
        "Negatif",
        "Positif"
    ]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        cm
    )

    plt.title(
        "Confusion Matrix Model Linear SVM",
        fontsize=14,
        weight="bold"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
        range(2),
        labels,
        fontsize=10
    )

    plt.yticks(
        range(2),
        labels,
        fontsize=10
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                weight="bold"
            )

    plt.colorbar()

    plt.tight_layout()

    output = (
        RESULT_DIR /
        "03_confusion_matrix.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "✓ Confusion matrix berhasil disimpan:"
    )

    print(output)

    print("\nConfusion Matrix:")

    print(
        pd.DataFrame(
            cm,
            index=[
                "Actual Negatif",
                "Actual Positif"
            ],
            columns=[
                "Predicted Negatif",
                "Predicted Positif"
            ]
        )
    )


# ==========================================================
# 4. KAMUS TOPIK PELAYANAN PUBLIK
# ==========================================================

TOPIC_DICT = {

    "Administrasi Kependudukan": [
        "ktp",
        "kartu tanda penduduk",
        "kk",
        "kartu keluarga",
        "akta",
        "dukcapil",
        "kependudukan",
        "nik",
        "dokumen kependudukan",
        "administrasi kependudukan",
        "catatan sipil"
    ],

    "Kesehatan": [
        "rumah sakit",
        "rsud",
        "puskesmas",
        "dokter",
        "pasien",
        "obat",
        "bpjs",
        "kesehatan",
        "rawat",
        "rawat inap",
        "ambulans",
        "vaksin",
        "tenaga kesehatan",
        "layanan kesehatan"
    ],

    "Pendidikan": [
        "sekolah",
        "guru",
        "siswa",
        "pendidikan",
        "kampus",
        "universitas",
        "beasiswa",
        "kelas",
        "murid",
        "sekolah dasar",
        "sekolah menengah",
        "madrasah"
    ],

    "Infrastruktur": [
        "jalan",
        "jembatan",
        "drainase",
        "trotoar",
        "aspal",
        "lampu jalan",
        "infrastruktur",
        "jalan rusak",
        "jembatan rusak",
        "pembangunan jalan",
        "irigasi",
        "air bersih"
    ],

    "Transportasi": [
        "terminal",
        "angkutan",
        "transportasi",
        "pelabuhan",
        "bandara",
        "bus",
        "angkutan umum",
        "jalan raya",
        "penyeberangan",
        "kapal"
    ],

    "Administrasi Pemerintahan": [
        "pemerintah",
        "pemda",
        "pemerintah daerah",
        "dinas",
        "pelayanan publik",
        "pelayanan masyarakat",
        "administrasi",
        "birokrasi",
        "aparatur",
        "asn",
        "pegawai negeri",
        "ombudsman",
        "pengaduan masyarakat"
    ],

    "Perizinan": [
        "izin",
        "perizinan",
        "oss",
        "legalitas",
        "nib",
        "investasi",
        "izin usaha",
        "pelayanan perizinan"
    ],

    "Lingkungan": [
        "sampah",
        "limbah",
        "banjir",
        "lingkungan",
        "kebersihan",
        "pencemaran",
        "air bersih",
        "pengelolaan sampah",
        "sanitasi"
    ],

    "Sosial": [
        "bansos",
        "bantuan sosial",
        "bantuan",
        "pkh",
        "kemiskinan",
        "disabilitas",
        "lansia",
        "penyandang disabilitas",
        "perlindungan sosial",
        "kesejahteraan sosial"
    ],

    "Keamanan": [
        "satpol",
        "satpol pp",
        "polisi",
        "keamanan",
        "kriminal",
        "ketertiban",
        "gangguan keamanan",
        "ketertiban umum"
    ],

    "Perpajakan & Retribusi": [
        "pajak",
        "retribusi",
        "pendapatan daerah",
        "pbb",
        "pajak daerah",
        "pajak kendaraan",
        "retribusi daerah"
    ],

    "Ketenagakerjaan": [
        "tenaga kerja",
        "pekerja",
        "buruh",
        "upah",
        "pelatihan kerja",
        "pengangguran",
        "lowongan kerja",
        "ketenagakerjaan",
        "pekerjaan"
    ],

    "Pertanian & Perikanan": [
        "petani",
        "pertanian",
        "nelayan",
        "perkebunan",
        "pupuk",
        "panen",
        "ikan",
        "perikanan",
        "peternakan",
        "pertanian rakyat"
    ],

    "UMKM & Ekonomi": [
        "umkm",
        "usaha mikro",
        "usaha kecil",
        "pasar",
        "pedagang",
        "ekonomi",
        "koperasi",
        "perdagangan",
        "usaha masyarakat"
    ],

    "Pariwisata": [
        "wisata",
        "pariwisata",
        "objek wisata",
        "destinasi",
        "hotel",
        "wisatawan",
        "tempat wisata",
        "sektor pariwisata"
    ]
}


# ==========================================================
# 5. KLASIFIKASI TOPIK BERDASARKAN KEYWORD
# ==========================================================

def analyze_topics(
    df_sentiment
):

    hasil = []

    for topic, keywords in TOPIC_DICT.items():

        jumlah_berita = 0

        for text in df_sentiment["text_final"].astype(str):

            text = text.lower()

            ditemukan = any(
                keyword.lower() in text
                for keyword in keywords
            )

            if ditemukan:

                jumlah_berita += 1

        hasil.append({

            "Topik": topic,

            "Jumlah Berita": jumlah_berita

        })

    topic_df = pd.DataFrame(
        hasil
    )

    topic_df = topic_df.sort_values(
        by="Jumlah Berita",
        ascending=False
    )

    topic_df.reset_index(
        drop=True,
        inplace=True
    )

    return topic_df


# ==========================================================
# 6. GRAFIK TOPIK POSITIF
# ==========================================================

def visualisasi_topik_positif(
    df
):

    print("\n[4/5] Membuat jumlah berita positif berdasarkan topik...")

    positive_df = df[
        df["label"] == "Positif"
    ].copy()

    positive_topic_df = analyze_topics(
        positive_df
    )

    plt.figure(
        figsize=(14, 7)
    )

    # ======================================================
    # WAJIB VERTIKAL
    # plt.bar = VERTIKAL
    # JANGAN DIGANTI plt.barh
    # ======================================================

    bars = plt.bar(
        positive_topic_df["Topik"],
        positive_topic_df["Jumlah Berita"],
        width=0.65
    )

    # ======================================================
    # JUDUL HANYA "JUMLAH BERITA POSITIF"
    # ======================================================

    plt.title(
        "Jumlah Berita Positif",
        fontsize=15,
        weight="bold"
    )

    plt.xlabel(
        "Topik",
        fontsize=12
    )

    plt.ylabel(
        "Jumlah Berita",
        fontsize=12
    )

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=9
    )

    nilai_maksimum = (
        positive_topic_df["Jumlah Berita"].max()
    )

    for bar in bars:

        height = bar.get_height()

        if height > 0:

            plt.text(
                bar.get_x() +
                bar.get_width() / 2,

                height +
                max(1, nilai_maksimum * 0.015),

                f"{int(height)}",

                ha="center",

                va="bottom",

                fontsize=9,

                weight="bold"
            )

    plt.ylim(
        0,
        max(
            1,
            nilai_maksimum * 1.15
        )
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    output = (
        RESULT_DIR /
        "04_topik_positif.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    csv_output = (
        RESULT_DIR /
        "04_topik_positif.csv"
    )

    positive_topic_df.to_csv(
        csv_output,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "✓ Jumlah berita positif berhasil disimpan:"
    )

    print(output)

    print(
        "✓ Data topik positif:"
    )

    print(csv_output)


# ==========================================================
# 7. GRAFIK TOPIK NEGATIF
# ==========================================================

def visualisasi_topik_negatif(
    df
):

    print("\n[5/5] Membuat jumlah berita negatif berdasarkan topik...")

    negative_df = df[
        df["label"] == "Negatif"
    ].copy()

    negative_topic_df = analyze_topics(
        negative_df
    )

    plt.figure(
        figsize=(14, 7)
    )

    # ======================================================
    # WAJIB VERTIKAL
    # plt.bar = VERTIKAL
    # JANGAN DIGANTI plt.barh
    # ======================================================

    bars = plt.bar(
        negative_topic_df["Topik"],
        negative_topic_df["Jumlah Berita"],
        width=0.65
    )

    # ======================================================
    # JUDUL HANYA "JUMLAH BERITA NEGATIF"
    # ======================================================

    plt.title(
        "Jumlah Berita Negatif",
        fontsize=15,
        weight="bold"
    )

    plt.xlabel(
        "Topik",
        fontsize=12
    )

    plt.ylabel(
        "Jumlah Berita",
        fontsize=12
    )

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=9
    )

    nilai_maksimum = (
        negative_topic_df["Jumlah Berita"].max()
    )

    for bar in bars:

        height = bar.get_height()

        if height > 0:

            plt.text(
                bar.get_x() +
                bar.get_width() / 2,

                height +
                max(1, nilai_maksimum * 0.015),

                f"{int(height)}",

                ha="center",

                va="bottom",

                fontsize=9,

                weight="bold"
            )

    plt.ylim(
        0,
        max(
            1,
            nilai_maksimum * 1.15
        )
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    output = (
        RESULT_DIR /
        "05_topik_negatif.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    csv_output = (
        RESULT_DIR /
        "05_topik_negatif.csv"
    )

    negative_topic_df.to_csv(
        csv_output,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "✓ Jumlah berita negatif berhasil disimpan:"
    )

    print(output)

    print(
        "✓ Data topik negatif:"
    )

    print(csv_output)


# ==========================================================
# MAIN
# ==========================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("VISUALISASI ANALISIS SENTIMEN DATASET NTT")
    print("=" * 60)

    # ======================================================
    # CEK FILE
    # ======================================================

    print("\nMengecek file...")

    if not LABELED_DATA.exists():

        raise FileNotFoundError(
            "Dataset labeling tidak ditemukan:\n"
            + str(LABELED_DATA)
        )

    if not SVM_MODEL.exists():

        raise FileNotFoundError(
            "Model SVM tidak ditemukan:\n"
            + str(SVM_MODEL)
        )

    if not TFIDF_MODEL.exists():

        raise FileNotFoundError(
            "TF-IDF vectorizer tidak ditemukan:\n"
            + str(TFIDF_MODEL)
        )

    print(
        "Semua file ditemukan."
    )

    # ======================================================
    # BACA DATASET
    # ======================================================

    print(
        "\nMembaca dataset labeling..."
    )

    df = pd.read_excel(
        LABELED_DATA
    )

    print(
        "Jumlah data awal:",
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

    # ======================================================
    # CEK KOLOM
    # ======================================================

    if "text_final" not in df.columns:

        raise KeyError(
            "Kolom 'text_final' tidak ditemukan."
        )

    if "label" not in df.columns:

        raise KeyError(
            "Kolom 'label' tidak ditemukan."
        )

    # ======================================================
    # CLEAN DATA
    # ======================================================

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

    df = df[
        (df["text_final"] != "") &
        (df["label"] != "")
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    # ======================================================
    # INFORMASI DATASET
    # ======================================================

    print(
        "\nJumlah data digunakan:",
        len(df)
    )

    print(
        "\nDistribusi label:"
    )

    print(
        df["label"].value_counts()
    )

    # ======================================================
    # VISUAL 1
    # ======================================================

    visualisasi_distribusi_sentimen(
        df
    )

    # ======================================================
    # TRAIN TEST SPLIT
    # ======================================================

    print(
        "\nMembagi data train dan test..."
    )

    X_train, X_test, y_train, y_test = train_test_split(

        df["text_final"],

        df["label"],

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=df["label"]
    )

    print(
        "Data train:",
        len(X_train)
    )

    print(
        "Data test :",
        len(X_test)
    )

    # ======================================================
    # LOAD MODEL SVM
    # ======================================================

    print(
        "\nMemuat model SVM..."
    )

    model = joblib.load(
        SVM_MODEL
    )

    print(
        "Model SVM berhasil dimuat."
    )

    # ======================================================
    # LOAD TF-IDF
    # ======================================================

    print(
        "\nMemuat TF-IDF vectorizer..."
    )

    vectorizer = joblib.load(
        TFIDF_MODEL
    )

    print(
        "TF-IDF vectorizer berhasil dimuat."
    )

    # ======================================================
    # TRANSFORM DATA TEST
    # ======================================================

    print(
        "\nMelakukan transformasi TF-IDF..."
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    # ======================================================
    # PREDIKSI
    # ======================================================

    print(
        "\nMelakukan prediksi..."
    )

    y_pred = model.predict(
        X_test_tfidf
    )

    print(
        "Prediksi berhasil."
    )

    # ======================================================
    # VISUAL 2
    # ======================================================

    visualisasi_performa_model(
        y_test,
        y_pred
    )

    # ======================================================
    # VISUAL 3
    # ======================================================

    visualisasi_confusion_matrix(
        y_test,
        y_pred
    )

    # ======================================================
    # VISUAL 4
    # ======================================================

    visualisasi_topik_positif(
        df
    )

    # ======================================================
    # VISUAL 5
    # ======================================================

    visualisasi_topik_negatif(
        df
    )

    # ======================================================
    # SELESAI
    # ======================================================

    elapsed = (
        time.time()
        - start_time
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "VISUALISASI SELESAI"
    )

    print(
        "=" * 60
    )

    print(
        "Jumlah data :",
        len(df)
    )

    print(
        "Positif     :",
        (df["label"] == "Positif").sum()
    )

    print(
        "Negatif     :",
        (df["label"] == "Negatif").sum()
    )

    print(
        "\n5 FILE VISUALISASI:"
    )

    print(
        "1. 01_distribusi_sentimen.png"
    )

    print(
        "2. 02_performa_model.png"
    )

    print(
        "3. 03_confusion_matrix.png"
    )

    print(
        "4. 04_topik_positif.png"
    )

    print(
        "5. 05_topik_negatif.png"
    )

    print(
        "\n2 FILE DATA TOPIK:"
    )

    print(
        "1. 04_topik_positif.csv"
    )

    print(
        "2. 05_topik_negatif.csv"
    )

    print(
        "\nFolder hasil:",
        RESULT_DIR
    )

    print(
        "Waktu proses: {:.2f} detik".format(
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