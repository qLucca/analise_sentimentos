from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
from unidecode import unidecode

# Stopwords conservadoras para sentimento: removemos ruído gramatical,
# mas preservamos negadores e intensificadores que carregam sinal.
PORTUGUESE_STOPWORDS = {
    "a",
    "acerca",
    "agora",
    "ainda",
    "alem",
    "algo",
    "algum",
    "alguma",
    "algumas",
    "alguns",
    "ali",
    "ambos",
    "ante",
    "antes",
    "ao",
    "aos",
    "apos",
    "aquela",
    "aquelas",
    "aquele",
    "aqueles",
    "aquilo",
    "as",
    "ate",
    "cada",
    "com",
    "como",
    "contra",
    "da",
    "das",
    "de",
    "dela",
    "delas",
    "dele",
    "deles",
    "depois",
    "desde",
    "desta",
    "deste",
    "disso",
    "disto",
    "do",
    "dos",
    "e",
    "ela",
    "elas",
    "ele",
    "eles",
    "em",
    "entre",
    "era",
    "eram",
    "essa",
    "essas",
    "esse",
    "esses",
    "esta",
    "estamos",
    "estao",
    "estas",
    "estava",
    "estavam",
    "este",
    "estes",
    "eu",
    "foi",
    "foram",
    "ha",
    "isso",
    "isto",
    "ja",
    "lhe",
    "lhes",
    "mas",
    "me",
    "mesmo",
    "meu",
    "minha",
    "na",
    "nas",
    "no",
    "nos",
    "nossa",
    "nossas",
    "nosso",
    "nossos",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "qual",
    "quais",
    "quando",
    "que",
    "quem",
    "se",
    "seja",
    "sendo",
    "ser",
    "seria",
    "seu",
    "seus",
    "sua",
    "suas",
    "tal",
    "tambem",
    "te",
    "tem",
    "tendo",
    "tenho",
    "ter",
    "teu",
    "teus",
    "tu",
    "um",
    "uma",
    "uns",
    "umas",
    "voce",
    "voces",
    "vou",
    "vao",
    "vai",
    "vai",
}

TOPIC_NEUTRAL_STOPWORDS = {
    "app",
    "aplicativo",
    "cliente",
    "clientes",
    "gente",
    "nubank",
    "nu",
    "pessoa",
    "pessoas",
    "pra",
    "pro",
    "pros",
    "q",
    "pq",
    "ta",
    "tava",
    "tb",
    "tbm",
    "to",
    "vc",
    "vcs",
    "voce",
    "voces",
}

TOPIC_STOPWORDS = PORTUGUESE_STOPWORDS | TOPIC_NEUTRAL_STOPWORDS | {
    "nao",
    "nem",
    "nunca",
    "jamais",
    "nada",
    "nenhum",
    "nenhuma",
    "sem",
}

TOPIC_CLUSTER_STOPWORDS = TOPIC_STOPWORDS | {
    "bom",
    "boa",
    "bons",
    "boas",
    "melhor",
    "melhores",
    "pior",
    "piores",
    "ruim",
    "ruins",
    "otimo",
    "otima",
    "otimos",
    "otimas",
    "excelente",
    "excelentes",
    "horrivel",
    "horriveis",
    "terrivel",
    "terriveis",
    "top",
    "super",
    "mega",
    "muito",
    "pouco",
    "bastante",
    "mais",
    "menos",
    "bem",
    "mal",
    "banco",
    "bancos",
    "nubank",
    "nu",
    "so",
    "dinheiro",
    "gostei",
    "gosto",
    "adorei",
    "adoro",
    "amo",
    "recomendo",
    "legal",
    "show",
    "sensacional",
    "fantastico",
    "incrivel",
    "maravilhoso",
    "maravilhosa",
    "perfeito",
    "perfeita",
    "top",
    "rapido",
    "rapida",
    "estavel",
    "estavel",
    "excelente",
    "excelentes",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords_from_text(
    text: str,
    language: str = "portuguese",
    stop_words: set[str] | None = None,
) -> str:
    if language != "portuguese":
        raise ValueError(f"Unsupported stopword language: {language}")
    stop_words = stop_words or PORTUGUESE_STOPWORDS
    tokens = [token for token in tokenize_text(text, language=language) if token not in stop_words]
    return " ".join(tokens)


def tokenize_text(text: str, language: str = "portuguese") -> list[str]:
    _ = language
    return re.findall(r"[a-z0-9]+", text.lower())


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
