"""
06_evaluation.py

Evaluasi Model SVM Dataset Berita NTT

Tahapan:
1. Membaca dataset labeling
2. Membagi data train dan test dengan konfigurasi yang sama
3. Memuat model SVM dan TF-IDF
4. Melakukan prediksi pada data test
5. Menghitung Accuracy, Precision, Recall, F1-Score
6. Membuat Classification Report
7. Membuat Confusion Matrix
8. Membuat grafik performa model
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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split

from config import (
    LABELED_DATA,
    RESULT_DIR,
    SVM_MODEL,
    TFIDF_MODEL,
    RANDOM_STATE,
    TEST_SIZE,
    CLASSIFICATION_REPORT,
    CONFUSION_MATRIX,
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("EVALUASI MODEL SVM DATASET NTT")
    print("=" * 60)

    # ======================================================
    # CEK FILE
    # ======================================================

    print("\nMengecek file...")

    if not LABELED_DATA.exists():
        raise FileNotFoundError(
            "Dataset labeling tidak ditemukan."
        )

    if not SVM_MODEL.exists():
        raise FileNotFoundError(
            "Model SVM tidak ditemukan."
        )

    if not TFIDF_MODEL.exists():
        raise FileNotFoundError(
            "TF-IDF vectorizer tidak ditemukan."
        )

    print("Semua file ditemukan.")

    # ======================================================
    # MEMBACA DATASET
    # ======================================================

    print("\nMembaca dataset labeling...")

    df = pd.read_excel(LABELED_DATA)

    print("Dataset berhasil dibaca.")
    print("Jumlah data :", len(df))

    # ======================================================
    # CEK KOLOM
    # ======================================================

    print("\nMengecek kolom dataset...")

    if "text_final" not in df.columns:
        raise KeyError(
            "Kolom 'text_final' tidak ditemukan."
        )

    if "label" not in df.columns:
        raise KeyError(
            "Kolom 'label' tidak ditemukan."
        )

    print("Kolom text_final ditemukan.")
    print("Kolom label ditemukan.")

    # ======================================================
    # HANDLE DATA KOSONG
    # ======================================================

    df["text_final"] = (
        df["text_final"]
        .fillna("")
        .astype(str)
    )

    df["label"] = (
        df["label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    print("\nMengecek missing value...")

    print(
        "Text kosong :",
        df["text_final"].str.strip().eq("").sum()
    )

    print(
        "Label kosong:",
        df["label"].eq("").sum()
    )

    # ======================================================
    # DATA VALID
    # ======================================================

    print("\nMengecek data yang dapat digunakan...")

    jumlah_awal = len(df)

    df = df[
        (df["text_final"].str.strip() != "") &
        (df["label"] != "")
    ].copy()

    jumlah_akhir = len(df)

    print("Jumlah data awal    :", jumlah_awal)
    print("Jumlah data valid   :", jumlah_akhir)
    print(
        "Data tidak digunakan:",
        jumlah_awal - jumlah_akhir
    )

    # ======================================================
    # DISTRIBUSI LABEL
    # ======================================================

    print("\nDistribusi label:")
    print("-" * 40)

    print(
        df["label"].value_counts()
    )

    print("-" * 40)

    jumlah_kelas = df["label"].nunique()

    print(
        "\nJumlah kelas sentimen:",
        jumlah_kelas
    )

    if jumlah_kelas < 2:
        raise ValueError(
            "Jumlah kelas sentimen kurang dari 2."
        )

    # ======================================================
    # TRAIN TEST SPLIT
    # ======================================================

    print("\nMembagi data train dan test...")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_final"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    print("\nPembagian dataset:")
    print("-" * 40)
    print("Total data :", len(df))
    print("Data train :", len(X_train))
    print("Data test  :", len(X_test))
    print("-" * 40)

    # ======================================================
    # DISTRIBUSI TRAIN
    # ======================================================

    print("\nDistribusi label data train:")

    print(
        y_train.value_counts()
    )

    # ======================================================
    # DISTRIBUSI TEST
    # ======================================================

    print("\nDistribusi label data test:")

    print(
        y_test.value_counts()
    )

    # ======================================================
    # MEMUAT MODEL
    # ======================================================

    print("\nMemuat model SVM...")

    model = joblib.load(
        SVM_MODEL
    )

    print("Model SVM berhasil dimuat.")

    print("\nMemuat TF-IDF vectorizer...")

    vectorizer = joblib.load(
        TFIDF_MODEL
    )

    print("TF-IDF vectorizer berhasil dimuat.")

    # ======================================================
    # TRANSFORM TEST DATA
    # ======================================================

    print("\nMelakukan transformasi TF-IDF pada data test...")

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print("Transformasi TF-IDF selesai.")

    print(
        "Bentuk matrix test:",
        X_test_tfidf.shape
    )

    # ======================================================
    # PREDIKSI
    # ======================================================

    print("\nMelakukan prediksi...")

    y_pred = model.predict(
        X_test_tfidf
    )

    print("Prediksi selesai.")

    # ======================================================
    # METRIK EVALUASI
    # ======================================================

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

    # ======================================================
    # HASIL EVALUASI
    # ======================================================

    print("\n" + "=" * 60)
    print("HASIL EVALUASI MODEL SVM")
    print("=" * 60)

    print(
        "Accuracy  : {:.4f} ({:.2f}%)".format(
            accuracy,
            accuracy * 100
        )
    )

    print(
        "Precision : {:.4f} ({:.2f}%)".format(
            precision,
            precision * 100
        )
    )

    print(
        "Recall    : {:.4f} ({:.2f}%)".format(
            recall,
            recall * 100
        )
    )

    print(
        "F1-Score  : {:.4f} ({:.2f}%)".format(
            f1,
            f1 * 100
        )
    )

    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    print("\nClassification Report")
    print("-" * 60)

    report = classification_report(
        y_test,
        y_pred,
        labels=["Negatif", "Positif"],
        target_names=["Negatif", "Positif"],
        zero_division=0
    )

    print(report)

    # ======================================================
    # SIMPAN CLASSIFICATION REPORT
    # ======================================================

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=["Negatif", "Positif"],
        target_names=["Negatif", "Positif"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    report_df.to_csv(
        CLASSIFICATION_REPORT
    )

    print(
        "Classification report berhasil disimpan."
    )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    print("\nMembuat confusion matrix...")

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=["Negatif", "Positif"],
        display_labels=["Negatif", "Positif"],
        ax=ax
    )

    ax.set_title(
        "Confusion Matrix Model SVM"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Confusion matrix berhasil disimpan."
    )

    # ======================================================
    # VISUALISASI PERFORMA
    # ======================================================

    print("\nMembuat visualisasi performa model...")

    metrics = {
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-Score": f1 * 100
    }

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        metrics.keys(),
        metrics.values(),
        width=0.55,
        edgecolor="black",
        linewidth=0.8
    )

    # Nilai di atas batang
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.2,
            "{:.2f}%".format(height),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    plt.title(
        "Performa Model SVM",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel(
        "Metrik Evaluasi",
        fontsize=12
    )

    plt.ylabel(
        "Persentase (%)",
        fontsize=12
    )

    plt.ylim(
        0,
        100
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    performance_path = (
        RESULT_DIR
        / "9_performa_model.png"
    )

    plt.savefig(
        performance_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Visualisasi performa model berhasil disimpan."
    )

    # ======================================================
    # HASIL AKHIR
    # ======================================================

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("EVALUASI MODEL SELESAI")
    print("=" * 60)

    print(
        "Jumlah data         :",
        len(df)
    )

    print(
        "Data test           :",
        len(X_test)
    )

    print(
        "Accuracy            : {:.2f}%".format(
            accuracy * 100
        )
    )

    print(
        "Precision           : {:.2f}%".format(
            precision * 100
        )
    )

    print(
        "Recall              : {:.2f}%".format(
            recall * 100
        )
    )

    print(
        "F1-Score            : {:.2f}%".format(
            f1 * 100
        )
    )

    print(
        "Classification report : BERHASIL"
    )

    print(
        "Confusion matrix      : BERHASIL"
    )

    print(
        "Performa model        : BERHASIL"
    )

    print(
        "Waktu proses          : {:.2f} detik".format(
            elapsed
        )
    )

    print("=" * 60)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()