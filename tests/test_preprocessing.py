import pandas as pd

from src.preprocessing.text_preprocessing import clean_text, preprocess_reviews


def test_clean_text_remove_links_and_lowercase():
    text = "Atendimento HORRIVEL!!! Veja https://exemplo.com"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert cleaned == cleaned.lower()


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
