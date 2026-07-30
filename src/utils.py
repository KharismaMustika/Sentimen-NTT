"""
utils.py
"""

import re
from functools import lru_cache

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stemmer = StemmerFactory().create_stemmer()

stop_factory = StopWordRemoverFactory()
STOPWORDS = set(stop_factory.get_stop_words())


def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)

    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):

    return text.split()


def remove_stopwords(tokens):

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


# ==========================
# CACHE STEMMING
# ==========================

@lru_cache(maxsize=50000)
def stem_word(word):

    return stemmer.stem(word)


def stemming(tokens):

    return [
        stem_word(token)
        for token in tokens
    ]


def join_tokens(tokens):

    return " ".join(tokens)