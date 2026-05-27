import pandas as pd

from src.preprocessing.text_preprocessing import clean_text, preprocess_reviews
from src.preprocessing.text_preprocessing import tokenize_text


def test_clean_text_remove_links_and_lowercase():
    text = "Atendimento HORRIVEL!!! Veja https://exemplo.com"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_preserves_negation_signal():
    text = "Nao gostei sem suporte e nunca resolve"
    cleaned = clean_text(text)
    assert "nao" in cleaned
    assert "sem" in cleaned
    assert "nunca" in cleaned


def test_clean_text_preserves_intensity_signal():
    text = "Nao gostei mais do app muito demorado demais"
    cleaned = clean_text(text)
    assert "nao" in cleaned
    assert "mais" in cleaned
    assert "muito" in cleaned
    assert "demais" in cleaned


def test_tokenize_text_uses_word_boundaries():
    tokens = tokenize_text("Nao gostei do app travando 24h")
    assert tokens == ["nao", "gostei", "do", "app", "travando", "24h"]


def test_preprocess_reviews_remove_empty_rows():
    df = pd.DataFrame(
        [
            {"fonte": "google_play", "texto_original": "App bom"},
            {"fonte": "google_play", "texto_original": None},
        ]
    )
    result = preprocess_reviews(df)
    assert "texto_limpo" in result.columns
    assert len(result) == 1
