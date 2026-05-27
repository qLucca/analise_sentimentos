from __future__ import annotations

import re

import numpy as np
import pandas as pd

BUSINESS_THEME_RULES = [
    ("Conta bloqueada e acesso", r"bloquead|desbloq|acesso|login|senha|biometr|cadastro|recuperar"),
    ("Empréstimo e crédito", r"emprest|credito|consign|limite|parcel|financi"),
    ("Cartão e fatura", r"cartao|fatura|chip|virtual|compra|maquin|anuidade"),
    ("Pix e transferências", r"\bpix\b|transfer|ted|doc|transferencia|enviad"),
    ("Cobrança e contestação", r"cobranc|juros|taxa|estorno|reembolso|iof|duplic"),
    ("Atendimento e suporte", r"atendimento|suporte|chat|sac|protocol|resposta|demora"),
    ("Privacidade e segurança", r"fraude|seguran|privacidade|dados|clon|golpe|verificacao"),
    ("Conta e cadastro", r"\bconta\b|abrir conta|documento|registro|perfil"),
    ("App e estabilidade", r"\bapp\b|aplicativ|trav|erro|bug|instal|lent"),
]


def predict_sentiment(df: pd.DataFrame, model, vectorizer) -> pd.DataFrame:
    predicted = df.copy()
    predicted["texto_limpo"] = predicted["texto_limpo"].fillna("")
    non_empty_mask = predicted["texto_limpo"].str.len() > 0

    predicted["sentimento_previsto"] = None
    predicted["confianca_modelo"] = None
    predicted["predicao_incerta"] = None
    theme_source = predicted.get("texto_original")
    if theme_source is None:
        theme_source = predicted["texto_limpo"]
    predicted["tema_negocio"] = theme_source.fillna("").map(_classify_business_theme)
    if non_empty_mask.any():
        if vectorizer is None:
            text_batch = predicted.loc[non_empty_mask, "texto_limpo"]
            predicted_values = model.predict(text_batch)
            predicted.loc[non_empty_mask, "sentimento_previsto"] = predicted_values
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(text_batch)
                _attach_probability_columns(predicted, non_empty_mask, model.classes_, proba)
        else:
            transformed = vectorizer.transform(predicted.loc[non_empty_mask, "texto_limpo"])
            predicted_values = model.predict(transformed)
            predicted.loc[non_empty_mask, "sentimento_previsto"] = predicted_values
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(transformed)
                _attach_probability_columns(predicted, non_empty_mask, model.classes_, proba)

    labeled_mask = predicted["sentimento_real"].isin(["Negativo", "Neutro", "Positivo"])
    predicted.loc[labeled_mask, "sentimento_previsto"] = predicted.loc[labeled_mask, "sentimento_real"]
    return predicted


def _attach_probability_columns(
    predicted: pd.DataFrame,
    mask: pd.Series,
    classes_,
    probabilities,
) -> None:
    class_scores = {str(label): probabilities[:, idx] for idx, label in enumerate(classes_)}
    predicted.loc[mask, "score_negativo"] = class_scores.get("Negativo")
    predicted.loc[mask, "score_neutro"] = class_scores.get("Neutro")
    predicted.loc[mask, "score_positivo"] = class_scores.get("Positivo")

    max_probabilities = probabilities.max(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    uncertain_mask = (max_probabilities < 0.55) | (margins < 0.15)

    predicted.loc[mask, "confianca_modelo"] = max_probabilities
    predicted.loc[mask, "predicao_incerta"] = uncertain_mask


def _classify_business_theme(text: str | None) -> str:
    if text is None or pd.isna(text):
        return "Sem texto"

    normalized_text = str(text).lower()
    for theme_name, pattern in BUSINESS_THEME_RULES:
        if re.search(pattern, normalized_text):
            return theme_name
    return "Outros temas"
