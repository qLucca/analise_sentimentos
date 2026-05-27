from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_summary import build_model_summary_row
from src.utils.paths import (
    GOLD_DASHBOARD_BERTIMBAU_FULL_DATASET_PATH,
    GOLD_DASHBOARD_YOUTUBE_BERT_DATASET_PATH,
    MODELS_DIR,
)


BERTIMBAU_EXPERIMENT_PATH = Path(__file__).resolve().parents[1] / "training" / "bertimbau_experiment.py"
REQUIRED_BERTIMBAU_BENCHMARK_FIELDS = {"accuracy", "f1_macro", "roc_auc_macro"}
BERTIMBAU_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
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


def _load_bertimbau_benchmark_metrics() -> tuple[dict | None, str]:
    if not BERTIMBAU_EXPERIMENT_PATH.exists():
        return None, f"Arquivo de benchmark ausente em {BERTIMBAU_EXPERIMENT_PATH}."

    try:
        source = BERTIMBAU_EXPERIMENT_PATH.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(BERTIMBAU_EXPERIMENT_PATH))
    except OSError:
        return None, f"Falha ao ler arquivo de benchmark em {BERTIMBAU_EXPERIMENT_PATH}."
    except UnicodeDecodeError:
        return None, f"Arquivo de benchmark ilegivel em {BERTIMBAU_EXPERIMENT_PATH}."
    except SyntaxError:
        return None, f"Arquivo de benchmark com sintaxe invalida em {BERTIMBAU_EXPERIMENT_PATH}."

    last_valid_rows: list[dict] | None = None
    last_invalid_reason = ""
    found_comparison_rows = False

    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue

        for target in targets:
            if isinstance(target, ast.Name) and target.id == "comparison_rows":
                found_comparison_rows = True
                try:
                    rows = ast.literal_eval(value_node)
                except (ValueError, TypeError, SyntaxError):
                    last_invalid_reason = (
                        "comparison_rows encontrado, mas com estrutura nao-literal/programatica em "
                        f"{BERTIMBAU_EXPERIMENT_PATH}."
                    )
                    continue
                if not isinstance(rows, list):
                    last_invalid_reason = (
                        "comparison_rows encontrado como literal, mas com estrutura malformada em "
                        f"{BERTIMBAU_EXPERIMENT_PATH}."
                    )
                    continue
                malformed_rows = False
                for row in rows:
                    if not isinstance(row, dict):
                        last_invalid_reason = (
                            "comparison_rows encontrado como literal, mas com itens malformados em "
                            f"{BERTIMBAU_EXPERIMENT_PATH}."
                        )
                        malformed_rows = True
                        break
                if malformed_rows:
                    continue
                last_valid_rows = rows

    if last_valid_rows is not None:
        for row in last_valid_rows:
            if row.get("modelo") == "BERTimbau":
                return row, ""
        return None, f"Linha BERTimbau ausente em comparison_rows de {BERTIMBAU_EXPERIMENT_PATH}."

    if found_comparison_rows:
        return None, last_invalid_reason

    return None, f"comparison_rows ausente em {BERTIMBAU_EXPERIMENT_PATH}."


def _should_run_runtime_bertimbau() -> bool:
    return os.getenv("BERTIMBAU_ENABLE_TRAINING", "0").strip().lower() in {"1", "true", "yes", "on"}


def _build_runtime_text_column(df: pd.DataFrame) -> pd.Series:
    text_column = df.get("texto_original")
    if text_column is None:
        text_column = df.get("texto_limpo")
    if text_column is None:
        return pd.Series(dtype=str)
    return text_column.fillna("").astype(str)


def _normalize_context_value(value) -> str:
    if value is None or pd.isna(value):
        return ""

    normalized_value = re.sub(r"\s+", " ", str(value).strip())
    if normalized_value.lower() in {"", "nan", "none", "sem informacao", "nao informado"}:
        return ""
    return normalized_value


def _format_source_label(source_name: str) -> str:
    source_labels = {
        "google_play": "Google Play",
        "youtube": "YouTube",
        "consumidor_gov": "Consumidor.gov",
    }
    return source_labels.get(source_name, source_name.replace("_", " ").title())


def _build_bert_context_text(row: pd.Series) -> str:
    parts: list[str] = []

    source = _normalize_context_value(row.get("fonte"))
    if source:
        parts.append(f"fonte: {_format_source_label(source)}")

    category = _normalize_context_value(row.get("categoria_problema"))
    if not category:
        category = _normalize_context_value(row.get("tema_negocio"))
    if category:
        parts.append(f"categoria: {category}")

    status = _normalize_context_value(row.get("status_reclamacao"))
    if status:
        parts.append(f"status: {status}")

    text = _normalize_context_value(row.get("texto_original"))
    if not text:
        text = _normalize_context_value(row.get("texto_limpo"))
    if text:
        parts.append(f"texto: {text}")

    return " | ".join(parts)


def _prepare_runtime_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()
    working_df["texto_base"] = _build_runtime_text_column(working_df)
    working_df["texto_modelo"] = working_df.apply(_build_bert_context_text, axis=1)
    fallback_mask = working_df["texto_modelo"].str.len().eq(0)
    working_df.loc[fallback_mask, "texto_modelo"] = working_df.loc[fallback_mask, "texto_base"]
    return working_df


def _classify_business_theme(text: str | None) -> str:
    if text is None or pd.isna(text):
        return "Sem texto"

    normalized_text = str(text).lower()
    for theme_name, pattern in BUSINESS_THEME_RULES:
        if re.search(pattern, normalized_text):
            return theme_name
    return "Outros temas"


def _build_prediction_frame(working_df: pd.DataFrame) -> pd.DataFrame:
    prediction_df = working_df.loc[working_df["texto_modelo"].str.len().gt(0)].copy()
    theme_source = prediction_df.get("texto_base")
    if theme_source is None:
        theme_source = prediction_df["texto_modelo"]
    prediction_df["tema_negocio"] = theme_source.map(_classify_business_theme)
    return prediction_df


def _annotate_bert_predictions(
    prediction_df: pd.DataFrame,
    pred_output,
    id2label: dict[int, str],
) -> pd.DataFrame:
    logits = np.asarray(pred_output.predictions)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predicted_labels = probabilities.argmax(axis=-1)
    predicted_df = prediction_df.copy()
    predicted_df["sentimento_previsto_bert"] = [id2label[label] for label in predicted_labels]
    predicted_df["confianca_bert"] = probabilities.max(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    predicted_df["predicao_incerta_bert"] = (predicted_df["confianca_bert"] < 0.55) | (margins < 0.15)
    predicted_df["score_negativo_bert"] = probabilities[:, 0]
    predicted_df["score_neutro_bert"] = probabilities[:, 1]
    predicted_df["score_positivo_bert"] = probabilities[:, 2]
    return predicted_df


def _run_bertimbau_runtime(df: pd.DataFrame) -> tuple[dict | None, dict]:
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import label_binarize
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        return None, {
            "available": False,
            "reason": f"Dependencias do BERTimbau indisponiveis para treino em runtime: {type(exc).__name__}: {exc}",
        }

    if not torch.cuda.is_available():
        return None, {
            "available": False,
            "reason": "CUDA indisponivel; treino BERTimbau em runtime requer GPU para este fluxo.",
        }

    working_df = _prepare_runtime_text_columns(df)
    labeled_df = working_df.loc[
        working_df["sentimento_real"].notna() & working_df["texto_modelo"].str.len().gt(0)
    ].copy()
    if labeled_df.empty or labeled_df["sentimento_real"].nunique() < 2:
        return None, {
            "available": False,
            "reason": "Base rotulada insuficiente para treino runtime do BERTimbau.",
        }

    label2id = {"Negativo": 0, "Neutro": 1, "Positivo": 2}
    id2label = {value: key for key, value in label2id.items()}
    labeled_df["label"] = labeled_df["sentimento_real"].map(label2id)
    if labeled_df["label"].isna().any():
        labeled_df = labeled_df.dropna(subset=["label"]).copy()
        labeled_df["label"] = labeled_df["label"].astype(int)

    max_train_samples = int(os.getenv("BERTIMBAU_MAX_TRAIN_SAMPLES", "18000"))
    if len(labeled_df) > max_train_samples:
        total_labeled = len(labeled_df)
        sampled_frames = []
        for sentiment_name, frame in labeled_df.groupby("sentimento_real"):
            sample_size = min(
                len(frame),
                max(1, round(max_train_samples * len(frame) / total_labeled)),
            )
            sampled_frames.append(frame.sample(n=sample_size, random_state=42))
        labeled_df = pd.concat(sampled_frames, ignore_index=True)

    train_df, temp_df = train_test_split(
        labeled_df[["texto_modelo", "label", "sentimento_real"]].copy(),
        test_size=0.30,
        random_state=42,
        stratify=labeled_df["label"],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(BERTIMBAU_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERTIMBAU_MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )
    from transformers import DataCollatorWithPadding

    def tokenize_function(examples):
        return tokenizer(
            examples["texto_modelo"],
            truncation=True,
            max_length=int(os.getenv("BERTIMBAU_MAX_LENGTH", "160")),
        )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize_function, batched=True)
    valid_dataset = Dataset.from_pandas(valid_df.reset_index(drop=True)).map(tokenize_function, batched=True)
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize_function, batched=True)

    prediction_df = _build_prediction_frame(working_df)
    pred_dataset = None
    if not prediction_df.empty:
        pred_dataset = Dataset.from_pandas(prediction_df[["texto_modelo"]].reset_index(drop=True)).map(
            tokenize_function,
            batched=True,
        )

    dataset_columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in train_dataset.column_names:
        dataset_columns.append("token_type_ids")
    train_dataset = train_dataset.rename_column("label", "labels")
    valid_dataset = valid_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")
    train_dataset.set_format(type="torch", columns=dataset_columns + ["labels"])
    valid_dataset.set_format(type="torch", columns=dataset_columns + ["labels"])
    test_dataset.set_format(type="torch", columns=dataset_columns + ["labels"])
    if pred_dataset is not None:
        pred_columns = ["input_ids", "attention_mask"]
        if "token_type_ids" in pred_dataset.column_names:
            pred_columns.append("token_type_ids")
        pred_dataset.set_format(type="torch", columns=pred_columns)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        }

    output_dir = MODELS_DIR / "bertimbau_runtime"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        num_train_epochs=float(os.getenv("BERTIMBAU_NUM_EPOCHS", "2")),
        per_device_train_batch_size=int(os.getenv("BERTIMBAU_BATCH_SIZE", "12")),
        per_device_eval_batch_size=int(os.getenv("BERTIMBAU_BATCH_SIZE", "12")),
        learning_rate=float(os.getenv("BERTIMBAU_LEARNING_RATE", "1.8e-5")),
        weight_decay=float(os.getenv("BERTIMBAU_WEIGHT_DECAY", "0.02")),
        warmup_ratio=float(os.getenv("BERTIMBAU_WARMUP_RATIO", "0.08")),
        label_smoothing_factor=float(os.getenv("BERTIMBAU_LABEL_SMOOTHING", "0.05")),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()

    pred_output = trainer.predict(test_dataset)
    y_pred = pred_output.predictions.argmax(axis=-1)
    y_true = pred_output.label_ids
    y_true_binary = label_binarize(y_true, classes=[0, 1, 2])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc_macro": roc_auc_score(y_true_binary, pred_output.predictions, multi_class="ovr", average="macro"),
        "roc_auc_weighted": roc_auc_score(
            y_true_binary,
            pred_output.predictions,
            multi_class="ovr",
            average="weighted",
        ),
    }
    f1_by_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    metrics["f1_negativo"] = f1_by_class[0]
    metrics["f1_neutro"] = f1_by_class[1]
    metrics["f1_positivo"] = f1_by_class[2]

    if pred_dataset is not None and not prediction_df.empty:
        full_pred_output = trainer.predict(pred_dataset)
        full_predictions = _annotate_bert_predictions(prediction_df, full_pred_output, id2label)
        full_predictions.to_csv(GOLD_DASHBOARD_BERTIMBAU_FULL_DATASET_PATH, index=False)
        youtube_predictions = full_predictions.loc[
            full_predictions["fonte"].fillna("").eq("youtube")
        ].copy()
        youtube_predictions.to_csv(GOLD_DASHBOARD_YOUTUBE_BERT_DATASET_PATH, index=False)

    summary_row = build_model_summary_row(
        model_name="BERTimbau",
        metrics=metrics,
        is_primary=False,
        insight_summary=(
            "Treino runtime em GPU com validacao dedicada, early stopping e melhor checkpoint carregado."
        ),
        provenance="bertimbau_runtime",
    )
    return summary_row, {
        "available": True,
        "source": "runtime_gpu",
        "reason": (
            "BERTimbau reexecutado em GPU com split estratificado train/validation/test "
            "e export de previsoes para o dashboard."
        ),
    }


def run_bertimbau_evaluation(df: pd.DataFrame) -> tuple[dict | None, dict]:
    missing_columns = {"texto_limpo", "sentimento_real"} - set(df.columns)
    if missing_columns:
        return None, {
            "available": False,
            "reason": f"Dataset sem colunas obrigatorias para avaliacao BERTimbau: {sorted(missing_columns)}.",
        }

    runtime_info = None
    if _should_run_runtime_bertimbau():
        runtime_row, runtime_info = _run_bertimbau_runtime(df)
        if runtime_row is not None:
            return runtime_row, runtime_info

    benchmark_metrics, benchmark_reason = _load_bertimbau_benchmark_metrics()
    if benchmark_metrics is None:
        runtime_reason = f" Tentativa runtime: {runtime_info['reason']}" if runtime_info else ""
        return None, {
            "available": False,
            "reason": benchmark_reason + runtime_reason,
        }

    missing_benchmark_fields = sorted(
        key
        for key in REQUIRED_BERTIMBAU_BENCHMARK_FIELDS
        if benchmark_metrics.get(key) is None or pd.isna(benchmark_metrics.get(key))
    )
    if missing_benchmark_fields:
        return None, {
            "available": False,
            "reason": (
                "Schema minimo do benchmark BERTimbau incompleto; "
                f"campos ausentes: {missing_benchmark_fields}."
            ),
        }

    summary_row = build_model_summary_row(
        model_name="BERTimbau",
        metrics=benchmark_metrics,
        is_primary=False,
        insight_summary=(
            "Benchmark historico reaproveitado de src/training/bertimbau_experiment.py; "
            "metricas nao foram reexecutadas neste pipeline."
        ),
        provenance="benchmark",
    )

    return summary_row, {
        "available": True,
        "source": "benchmark",
        "reason": (
            "Linha comparativa do BERTimbau carregada de benchmark versionado no repositorio; "
            "sem downloads reais e sem treino pesado."
        ),
    }
