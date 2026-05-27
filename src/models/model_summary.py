import math


def build_model_summary_row(
    model_name: str,
    metrics: dict,
    is_primary: bool,
    insight_summary: str,
    provenance: str,
) -> dict:
    return {
        "modelo": model_name,
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "roc_auc_macro": metrics.get("roc_auc_macro"),
        "roc_auc_weighted": metrics.get("roc_auc_weighted"),
        "f1_negativo": metrics.get("f1_negativo"),
        "f1_neutro": metrics.get("f1_neutro"),
        "f1_positivo": metrics.get("f1_positivo"),
        "modelo_principal": is_primary,
        "insight_resumo": insight_summary,
        "proveniencia": provenance,
    }


def choose_primary_model(rows: list[dict]) -> dict:
    def metric_value(row: dict, key: str) -> float:
        value = row.get(key)
        if value is None or value is math.nan:
            return float("-inf")
        try:
            if math.isnan(value):
                return float("-inf")
        except (TypeError, ValueError):
            pass
        try:
            if value is not None and __import__("pandas").isna(value):
                return float("-inf")
        except (TypeError, ValueError, AttributeError):
            pass
        return value

    def neutral_guard(row: dict) -> int:
        if "f1_neutro" not in row:
            return 1
        return int(metric_value(row, "f1_neutro") >= 0.05)

    return sorted(
        rows,
        key=lambda row: (
            neutral_guard(row),
            metric_value(row, "roc_auc_macro"),
            metric_value(row, "f1_macro"),
            metric_value(row, "recall_macro"),
            metric_value(row, "f1_neutro"),
            metric_value(row, "accuracy"),
        ),
        reverse=True,
    )[0]


def merge_model_summaries(classic_rows: list[dict], bert_row: dict | None) -> list[dict]:
    rows = [dict(row) for row in classic_rows]
    if bert_row is not None:
        rows.append(dict(bert_row))
    if not rows:
        return []

    comparable_rows = [row for row in rows if row.get("proveniencia") != "benchmark"]
    winner = choose_primary_model(comparable_rows or rows)
    winner_index = rows.index(winner)

    for index, row in enumerate(rows):
        row["modelo_principal"] = index == winner_index

    return rows
