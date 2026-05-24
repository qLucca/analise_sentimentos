from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
from unidecode import unidecode

PORTUGUESE_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "essa",
    "esse",
    "esta",
    "este",
    "eu",
    "foi",
    "ha",
    "isso",
    "ja",
    "mas",
    "me",
    "meu",
    "minha",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "que",
    "se",
    "sem",
    "sua",
    "um",
    "uma",
    "veja",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords_from_text(text: str, language: str = "portuguese") -> str:
    if language != "portuguese":
        raise ValueError(f"Unsupported stopword language: {language}")
    stop_words = PORTUGUESE_STOPWORDS
    tokens = [token for token in text.split() if token not in stop_words]
    return " ".join(tokens)


def tokenize_text(text: str, language: str = "portuguese") -> list[str]:
    _ = language
    return text.split()


def clean_text(text: str | None) -> str:
    if text is None or pd.isna(text):
        return ""
    normalized = normalize_text(str(text))
    return remove_stopwords_from_text(normalized)


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    processed["texto_original"] = processed["texto_original"].fillna("")
    processed["texto_limpo"] = processed["texto_original"].apply(clean_text)
    processed = processed[processed["texto_limpo"].str.len() > 0].copy()
    processed["data_processamento"] = datetime.now()
    return processed
