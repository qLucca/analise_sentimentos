from __future__ import annotations

from datetime import datetime
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

from src.models.model_summary import choose_primary_model
from src.preprocessing.text_preprocessing import tokenize_text


SENTIMENT_LABELS = ["Negativo", "Neutro", "Positivo"]
POSITIVE_NEUTRAL_NOISE_TERMS = [
    "otimo",
    "excelente",
    "bom",
    "amei",
    "perfeito",
    "maravilhoso",
    "gostei",
    "muito boa",
    "muito bom",
    "top",
]
NEGATIVE_NEUTRAL_NOISE_TERMS = [
    "horrivel",
    "pessimo",
    "ruim",
    "demorado",
    "problema",
    "travando",
    "erro",
    "lixo",
    "nao abre",
    "nao funciona",
]


def _prepare_labeled_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    labeled_df = df[df["sentimento_real"].notna()].copy()
    labeled_df = labeled_df[
        labeled_df["sentimento_real"].isin(SENTIMENT_LABELS)
    ].copy()
    if labeled_df.empty:
        raise ValueError("Nenhum registro rotulado encontrado para treino do modelo.")

    labeled_df["texto_limpo"] = labeled_df["texto_limpo"].fillna("")
    labeled_df = labeled_df[labeled_df["texto_limpo"].str.len() > 0].copy()
    if labeled_df.empty:
        raise ValueError("Nenhum texto limpo disponivel para treino do modelo.")

    if labeled_df["sentimento_real"].nunique() < 2:
        raise ValueError("Treino do modelo ficou com apenas uma classe apos o split.")

    return labeled_df


def _filter_noisy_neutral_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    positive_pattern = re.compile("|".join(re.escape(term) for term in POSITIVE_NEUTRAL_NOISE_TERMS))
    negative_pattern = re.compile("|".join(re.escape(term) for term in NEGATIVE_NEUTRAL_NOISE_TERMS))
    neutral_mask = df["sentimento_real"].eq("Neutro")
    positive_hits = df["texto_limpo"].str.contains(positive_pattern, regex=True, na=False)
    negative_hits = df["texto_limpo"].str.contains(negative_pattern, regex=True, na=False)
    suspicious_neutral_mask = neutral_mask & (positive_hits ^ negative_hits)
    return df.loc[~suspicious_neutral_mask].copy(), int(suspicious_neutral_mask.sum())


def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=tokenize_text,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        max_features=25000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def _build_candidate_pipelines() -> list[dict]:
    return [
        {
            "name": "LogisticRegression",
            "pipeline": Pipeline(
                [
                    ("vectorizer", _build_vectorizer()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=42,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            "score_method": "predict_proba",
        },
        {
            "name": "LinearSVC",
            "pipeline": Pipeline(
                [
                    ("vectorizer", _build_vectorizer()),
                    (
                        "classifier",
                        LinearSVC(
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "score_method": "decision_function",
        },
        {
            "name": "MultinomialNB",
            "pipeline": Pipeline(
                [
                    ("vectorizer", _build_vectorizer()),
                    ("classifier", MultinomialNB(alpha=0.35)),
                ]
            ),
            "score_method": "predict_proba",
        },
    ]


def _compute_roc_auc(y_true: pd.Series, y_score, labels: list[str]) -> tuple[float | None, float | None]:
    if y_score is None:
        return None, None

    y_true_binary = label_binarize(y_true, classes=labels)
    if getattr(y_score, "ndim", 1) == 1:
        return None, None
    if getattr(y_score, "shape", (0, 0))[1] != len(labels):
        return None, None

    return (
        roc_auc_score(y_true_binary, y_score, multi_class="ovr", average="macro"),
        roc_auc_score(y_true_binary, y_score, multi_class="ovr", average="weighted"),
    )


def _build_metric_row(
    model_name: str,
    y_true: pd.Series,
    y_pred,
    y_score,
) -> dict:
    roc_auc_macro, roc_auc_weighted = _compute_roc_auc(y_true, y_score, SENTIMENT_LABELS)
    f1_by_class = f1_score(
        y_true,
        y_pred,
        labels=SENTIMENT_LABELS,
        average=None,
        zero_division=0,
    )

    return {
        "modelo": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "f1_negativo": f1_by_class[0],
        "f1_neutro": f1_by_class[1],
        "f1_positivo": f1_by_class[2],
    }


def _select_cv_splits(y: pd.Series) -> int:
    min_class_count = int(y.value_counts().min())
    return max(2, min(5, min_class_count))


def _evaluate_candidates_with_cross_validation(X: pd.Series, y: pd.Series) -> list[dict]:
    cv = StratifiedKFold(
        n_splits=_select_cv_splits(y),
        shuffle=True,
        random_state=42,
    )
    rows = []

    for candidate in _build_candidate_pipelines():
        y_pred = cross_val_predict(
            candidate["pipeline"],
            X,
            y,
            cv=cv,
            method="predict",
            n_jobs=1,
        )
        try:
            y_score = cross_val_predict(
                candidate["pipeline"],
                X,
                y,
                cv=cv,
                method=candidate["score_method"],
                n_jobs=1,
            )
        except (AttributeError, ValueError):
            y_score = None
        rows.append(_build_metric_row(candidate["name"], y, y_pred, y_score))

    return rows


def _fit_candidate_on_full_dataset(model_name: str, X: pd.Series, y: pd.Series):
    candidate_map = {candidate["name"]: candidate["pipeline"] for candidate in _build_candidate_pipelines()}
    pipeline = candidate_map[model_name]
    pipeline.fit(X, y)
    return pipeline


def _fallback_train_for_small_dataset(X: pd.Series, y: pd.Series):
    stratify = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2 if len(X) > 5 else 0.5,
        random_state=42,
        stratify=stratify,
    )
    if y_train.nunique() < 2:
        raise ValueError("Treino do modelo ficou com apenas uma classe apos o split.")

    pipeline = Pipeline(
        [
            ("vectorizer", _build_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metric_row = _build_metric_row("LogisticRegression", y_test, y_pred, None)
    final_pipeline = Pipeline(
        [
            ("vectorizer", _build_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    final_pipeline.fit(X, y)
    return final_pipeline, metric_row, [metric_row]


def train_sentiment_model(df: pd.DataFrame):
    labeled_df = _prepare_labeled_dataframe(df)
    labeled_df, noisy_neutral_removed = _filter_noisy_neutral_labels(labeled_df)
    X = labeled_df["texto_limpo"]
    y = labeled_df["sentimento_real"]

    if y.value_counts().min() < 2 or len(labeled_df) < 18:
        final_pipeline, winner_row, candidate_rows = _fallback_train_for_small_dataset(X, y)
    else:
        candidate_rows = _evaluate_candidates_with_cross_validation(X, y)
        winner_row = choose_primary_model(candidate_rows)
        final_pipeline = _fit_candidate_on_full_dataset(winner_row["modelo"], X, y)

    metrics = {
        "modelo": winner_row["modelo"].lower(),
        "model_name": winner_row["modelo"],
        "vetorizador": "tfidf",
        "accuracy": winner_row["accuracy"],
        "precision_macro": winner_row["precision_macro"],
        "recall_macro": winner_row["recall_macro"],
        "f1_macro": winner_row["f1_macro"],
        "roc_auc_macro": winner_row.get("roc_auc_macro"),
        "roc_auc_weighted": winner_row.get("roc_auc_weighted"),
        "f1_negativo": winner_row.get("f1_negativo"),
        "f1_neutro": winner_row.get("f1_neutro"),
        "f1_positivo": winner_row.get("f1_positivo"),
        "classic_candidate_rows": candidate_rows,
        "neutral_noise_removed": noisy_neutral_removed,
        "data_treinamento": datetime.now(),
        "observacoes": (
            f"Selecao classica via validacao cruzada estratificada com {len(candidate_rows)} candidatos "
            f"e {len(labeled_df)} registros rotulados; {noisy_neutral_removed} neutros ruidosos removidos."
        ),
    }
    return final_pipeline, metrics, None
