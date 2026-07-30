from pathlib import Path


# ==========================================================
# ROOT PROJECT
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent


# ==========================================================
# DATA
# ==========================================================

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

FINAL_DIR = DATA_DIR / "final"


# ==========================================================
# DATASET UTAMA
# ==========================================================

RAW_DATA = RAW_DIR / "NTT_berita bersih.xlsx"


# ==========================================================
# HASIL CLEANING
# ==========================================================

CLEAN_DATA = PROCESSED_DIR / "dataset_clean.xlsx"

PREPROCESS_DATA = PROCESSED_DIR / "dataset_preprocessed.xlsx"

LABELED_DATA = PROCESSED_DIR / "dataset_labeled.xlsx"


# ==========================================================
# TF-IDF
# ==========================================================

TFIDF_DATA = FINAL_DIR / "dataset_tfidf.csv"


# ==========================================================
# MODEL
# ==========================================================

MODEL_DIR = BASE_DIR / "models"

SVM_MODEL = MODEL_DIR / "svm_model.pkl"

TFIDF_MODEL = MODEL_DIR / "tfidf_vectorizer.pkl"


# ==========================================================
# RESULT
# ==========================================================

RESULT_DIR = BASE_DIR / "results"

CONFUSION_MATRIX = RESULT_DIR / "confusion_matrix.png"

CLASSIFICATION_REPORT = RESULT_DIR / "classification_report.csv"

WORDCLOUD_POSITIVE = RESULT_DIR / "wordcloud_positive.png"

WORDCLOUD_NEGATIVE = RESULT_DIR / "wordcloud_negative.png"


# ==========================================================
# PARAMETER
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.2


# ==========================================================
# LEXICON
# ==========================================================

ASSETS_DIR = BASE_DIR / "assets"

LEXICON_DIR = ASSETS_DIR / "lexicon"

POSITIVE_LEXICON = LEXICON_DIR / "positive.tsv"

NEGATIVE_LEXICON = LEXICON_DIR / "negative.tsv"