from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.preprocessing.text_preprocessing import TOPIC_CLUSTER_STOPWORDS, tokenize_text


TOPIC_EMBEDDING_MODEL = os.getenv(
    "TOPIC_EMBEDDING_MODEL",
    "neuralmind/bert-base-portuguese-cased",
)
TOPIC_EMBEDDING_BATCH_SIZE = int(os.getenv("TOPIC_EMBEDDING_BATCH_SIZE", "24"))
TOPIC_MAX_LENGTH = int(os.getenv("TOPIC_MAX_LENGTH", "96"))
DEFAULT_TOPIC_COUNT = int(os.getenv("TOPIC_CLUSTER_COUNT", "8"))
LAST_TOPIC_EMBEDDING_METHOD = "bert_embeddings"


@lru_cache(maxsize=1)
def _load_transformer_bundle(model_name: str) -> tuple[object, object, object]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _build_fallback_embeddings(texts: list[str]) -> np.ndarray:
    global LAST_TOPIC_EMBEDDING_METHOD
    LAST_TOPIC_EMBEDDING_METHOD = "tfidf_fallback"
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectorizer = TfidfVectorizer(
        max_features=4096,
        ngram_range=(1, 2),
        stop_words=list(TOPIC_CLUSTER_STOPWORDS),
        tokenizer=tokenize_text,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray().astype(np.float32)


def build_text_embeddings(texts: list[str], model_name: str | None = None) -> np.ndarray:
    global LAST_TOPIC_EMBEDDING_METHOD
    cleaned_texts = [str(text).strip() for text in texts if str(text).strip()]
    if not cleaned_texts:
        return np.empty((0, 0), dtype=np.float32)

    embedding_model_name = model_name or TOPIC_EMBEDDING_MODEL
    try:
        tokenizer, model, device = _load_transformer_bundle(embedding_model_name)
        import torch
    except Exception:
        return _build_fallback_embeddings(cleaned_texts)

    batches: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start_index in range(0, len(cleaned_texts), TOPIC_EMBEDDING_BATCH_SIZE):
                batch_texts = cleaned_texts[start_index : start_index + TOPIC_EMBEDDING_BATCH_SIZE]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=TOPIC_MAX_LENGTH,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                outputs = model(**encoded)
                pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                batches.append(pooled.detach().cpu().numpy())
    except Exception:
        return _build_fallback_embeddings(cleaned_texts)

    LAST_TOPIC_EMBEDDING_METHOD = "bert_embeddings"
    embeddings = np.vstack(batches).astype(np.float32)
    embeddings = normalize(embeddings)
    return embeddings


def _select_text_column(df: pd.DataFrame) -> pd.Series:
    if "texto_limpo" in df.columns:
        text_series = df["texto_limpo"].fillna("").astype(str)
    elif "texto_original" in df.columns:
        text_series = df["texto_original"].fillna("").astype(str)
    else:
        text_series = pd.Series("", index=df.index, dtype=str)
    return text_series.str.strip()


def _clean_keyword_candidates(texts: pd.Series) -> list[str]:
    candidate_tokens = (
        texts.fillna("")
        .astype(str)
        .map(tokenize_text)
        .explode()
        .dropna()
        .astype(str)
    )
    candidate_tokens = candidate_tokens[
        candidate_tokens.str.len().ge(3) & ~candidate_tokens.str.fullmatch(r"\d+", na=False)
    ]
    candidate_tokens = candidate_tokens[~candidate_tokens.isin(TOPIC_CLUSTER_STOPWORDS)]
    return candidate_tokens.tolist()


def _build_cluster_keywords(cluster_texts: pd.Series, top_n: int = 5) -> list[str]:
    cleaned_texts = cluster_texts.fillna("").astype(str).str.strip()
    cleaned_texts = cleaned_texts.loc[cleaned_texts.ne("")]
    if cleaned_texts.empty:
        return []

    try:
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words=list(TOPIC_CLUSTER_STOPWORDS),
            tokenizer=tokenize_text,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(cleaned_texts)
        feature_names = vectorizer.get_feature_names_out()
        if matrix.shape[1] == 0:
            raise ValueError("cluster without usable terms")
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        top_indices = scores.argsort()[-top_n:][::-1]
        keywords = [str(feature_names[index]).strip() for index in top_indices if scores[index] > 0]
        keywords = [keyword for keyword in keywords if keyword]
        if keywords:
            return keywords[:top_n]
    except Exception:
        pass

    tokens = _clean_keyword_candidates(cleaned_texts)
    if not tokens:
        return []
    token_counts = pd.Series(tokens).value_counts().head(top_n)
    return token_counts.index.astype(str).tolist()


def _format_keyword_phrase(keyword: str) -> str:
    cleaned = str(keyword).replace("_", " ").strip()
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def _format_topic_label(keywords: list[str]) -> str:
    keyword_text = " ".join(str(keyword).lower() for keyword in keywords if str(keyword).strip())

    business_rules = [
        (["bloque", "acesso", "login", "senha", "cadastro", "recuper"], "Conta bloqueada / acesso"),
        (["emprest", "credito", "limite", "parcel", "financi"], "Empréstimo / crédito"),
        (["pix", "transfer", "ted", "doc"], "Pix / transferências"),
        (["cobranc", "juros", "taxa", "estorno", "reembolso", "contest"], "Cobrança / contestação"),
        (["atendimento", "suporte", "chat", "sac", "protoc"], "Atendimento / suporte"),
        (["fraud", "seguran", "privacidad", "golpe", "clon"], "Segurança / fraude"),
        (["app", "erro", "trav", "instal", "lent"], "App / estabilidade"),
    ]
    for markers, label in business_rules:
        if any(marker in keyword_text for marker in markers):
            return label
    if "cartao" in keyword_text:
        if any(marker in keyword_text for marker in ["fatura", "chip", "virtual", "compra", "anuidade"]):
            return "Cartão / fatura"
        return "Cartão / uso"
    if "conta" in keyword_text and any(marker in keyword_text for marker in ["saldo", "extrato", "mov", "pix", "valor"]):
        return "Conta / movimentação"

    meaningful_keywords: list[str] = []
    seen_tokens: set[str] = set()
    for keyword in keywords:
        cleaned = str(keyword).replace("_", " ").strip()
        if not cleaned:
            continue
        first_token = cleaned.split()[0].lower()
        if cleaned.lower() in {item.lower() for item in meaningful_keywords}:
            continue
        if first_token in seen_tokens:
            continue
        seen_tokens.add(first_token)
        meaningful_keywords.append(_format_keyword_phrase(cleaned))
        if len(meaningful_keywords) == 2:
            break
    if not meaningful_keywords:
        return "Tema sem rotulo"
    return " / ".join(meaningful_keywords[:2])


def _refine_topic_label(topic_name: str, cluster_signal: str, keywords: list[str]) -> str:
    normalized_topic = str(topic_name).strip()
    keyword_text = " ".join(str(keyword).lower() for keyword in keywords if str(keyword).strip())

    if "emprest" in keyword_text or "credito" in keyword_text or normalized_topic.startswith("Empréstimo"):
        if cluster_signal == "Positivo":
            return "Crédito / oferta"
        if cluster_signal == "Negativo":
            return "Empréstimo / crédito"
        return "Crédito / elegibilidade"

    if "cartao" in keyword_text or normalized_topic.startswith("Cartão"):
        if cluster_signal == "Negativo":
            return "Cartão / fatura"
        if cluster_signal == "Positivo":
            return "Cartão / uso"
        return "Cartão / suporte"

    if any(marker in keyword_text for marker in ["bloque", "acess", "login", "senha", "consig", "entrar"]):
        return "Conta bloqueada / acesso"

    if "conta" in keyword_text and cluster_signal == "Negativo":
        return "Conta bloqueada / acesso"

    return normalized_topic


def _suggest_cluster_action(topic_name: str, cluster_signal: str) -> str:
    topic = topic_name.lower()
    if cluster_signal == "Misto":
        return "Revisar jornada e validar com amostras antes de escalar correcao."
    if "bloque" in topic or "acesso" in topic or "login" in topic or "senha" in topic:
        return "Reduzir friccao de acesso e simplificar desbloqueio."
    if "emprest" in topic or "credito" in topic or "limite" in topic:
        return "Ajustar comunicacao de credito, limite e elegibilidade."
    if "cartao" in topic or "fatura" in topic:
        return "Melhorar suporte de cartao, fatura e recorrencia de uso."
    if "pix" in topic or "transfer" in topic:
        return "Reforcar status de transacoes e confiabilidade das transferencias."
    if "cobranc" in topic or "contest" in topic or "estorno" in topic:
        return "Acelerar contestacao, estorno e explicacao de cobrancas."
    if "atendimento" in topic or "suporte" in topic or "chat" in topic:
        return "Diminuir tempo de resposta e reforcar resolucao no primeiro contato."
    if "seguran" in topic or "fraud" in topic or "privacidad" in topic:
        return "Refinar validacao de identidade e comunicacao de seguranca."
    if "app" in topic or "estabil" in topic or "erro" in topic or "trav" in topic:
        return "Priorizar estabilidade, velocidade e reducao de erros no app."
    return "Investigar a causa raiz e confirmar com amostras textuais."


def _infer_cluster_signal(positive_share: float, negative_share: float) -> str:
    if positive_share >= 0.55 and positive_share - negative_share >= 0.15:
        return "Positivo"
    if negative_share >= 0.55 and negative_share - positive_share >= 0.15:
        return "Negativo"
    return "Misto"


def _select_representative_example(
    cluster_embeddings: np.ndarray,
    cluster_texts: pd.Series,
) -> str:
    if cluster_embeddings.size == 0 or cluster_texts.empty:
        return ""
    centroid = cluster_embeddings.mean(axis=0, keepdims=True)
    similarities = cluster_embeddings @ centroid.T
    best_index = int(np.argmax(similarities.ravel()))
    return str(cluster_texts.iloc[best_index]).strip()


def _build_legacy_summary(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame(
            columns=[
                "id_topico",
                "nome_topico",
                "palavras_chave",
                "quantidade_registros",
                "sentimento_predominante",
                "fonte_predominante",
                "data_processamento",
            ]
        )

    legacy_summary = summary_frame[
        [
            "id_topico",
            "nome_topico",
            "palavras_chave",
            "quantidade_registros",
            "sentimento_predominante",
            "fonte_predominante",
            "data_processamento",
        ]
    ].copy()
    return legacy_summary


def extract_topics(df: pd.DataFrame, n_clusters: int = DEFAULT_TOPIC_COUNT) -> tuple[pd.DataFrame, pd.DataFrame]:
    working_df = df.copy()
    text_series = _select_text_column(working_df)
    working_df["topico"] = None
    working_df["cluster_id"] = pd.NA
    working_df["cluster_label"] = None
    working_df["cluster_keywords"] = None
    working_df["cluster_signal"] = None
    working_df["cluster_action"] = None
    working_df["cluster_example"] = None
    working_df["cluster_share_positivo"] = pd.NA
    working_df["cluster_share_negativo"] = pd.NA
    working_df["cluster_share_neutro"] = pd.NA

    cluster_mask = text_series.ne("")
    cluster_df = working_df.loc[cluster_mask].copy()
    if cluster_df.empty:
        return working_df, _build_legacy_summary(pd.DataFrame())

    cluster_texts = text_series.loc[cluster_df.index].copy()
    embeddings = build_text_embeddings(cluster_texts.tolist())
    if embeddings.size == 0:
        return working_df, _build_legacy_summary(pd.DataFrame())

    working_embeddings = embeddings
    if len(cluster_texts) > 2 and embeddings.shape[1] > 2:
        n_components = min(24, embeddings.shape[1], len(cluster_texts) - 1)
        if n_components >= 2:
            working_embeddings = PCA(n_components=n_components, random_state=42).fit_transform(embeddings)
    working_embeddings = normalize(working_embeddings)

    cluster_count = min(max(1, n_clusters), len(cluster_texts))
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=42,
        batch_size=max(32, min(1024, len(cluster_texts))),
        n_init="auto",
    )
    labels = model.fit_predict(working_embeddings)
    cluster_df["cluster_id"] = labels

    sentiment_column = "sentimento_previsto" if "sentimento_previsto" in working_df.columns else None
    if sentiment_column is None and "sentimento_previsto_bert" in working_df.columns:
        sentiment_column = "sentimento_previsto_bert"

    summary_rows: list[dict] = []
    for cluster_id in sorted(np.unique(labels).tolist()):
        cluster_records = cluster_df.loc[cluster_df["cluster_id"] == cluster_id].copy()
        cluster_embeddings = working_embeddings[labels == cluster_id]
        cluster_text_slice = cluster_texts.loc[cluster_records.index]
        cluster_keywords = _build_cluster_keywords(cluster_text_slice, top_n=5)
        cluster_signal = "Misto"
        positive_share = negative_share = neutral_share = 0.0
        if sentiment_column is not None and not cluster_records.empty:
            sentiment_distribution = (
                cluster_records[sentiment_column]
                .fillna("Sem informacao")
                .value_counts(normalize=True)
            )
            positive_share = float(sentiment_distribution.get("Positivo", 0.0))
            negative_share = float(sentiment_distribution.get("Negativo", 0.0))
            neutral_share = float(sentiment_distribution.get("Neutro", 0.0))
            cluster_signal = _infer_cluster_signal(positive_share, negative_share)
        topic_name = _refine_topic_label(_format_topic_label(cluster_keywords), cluster_signal, cluster_keywords)
        dominant_source = (
            cluster_records["fonte"].fillna("desconhecido").astype(str).value_counts().index[0]
            if "fonte" in cluster_records.columns and not cluster_records.empty
            else "desconhecido"
        )
        representative_example = _select_representative_example(
            cluster_embeddings,
            cluster_text_slice,
        )
        cluster_records = cluster_records.copy()
        cluster_records["cluster_id"] = int(cluster_id)
        cluster_records["cluster_label"] = topic_name
        cluster_records["cluster_keywords"] = ", ".join(cluster_keywords)
        cluster_records["cluster_signal"] = cluster_signal
        cluster_records["cluster_action"] = _suggest_cluster_action(topic_name, cluster_signal)
        cluster_records["cluster_example"] = representative_example
        cluster_records["cluster_share_positivo"] = positive_share
        cluster_records["cluster_share_negativo"] = negative_share
        cluster_records["cluster_share_neutro"] = neutral_share
        cluster_records["topico"] = topic_name
        working_df.loc[cluster_records.index, [
            "cluster_id",
            "cluster_label",
            "cluster_keywords",
            "cluster_signal",
            "cluster_action",
            "cluster_example",
            "cluster_share_positivo",
            "cluster_share_negativo",
            "cluster_share_neutro",
            "topico",
        ]] = cluster_records[
            [
                "cluster_id",
                "cluster_label",
                "cluster_keywords",
                "cluster_signal",
                "cluster_action",
                "cluster_example",
                "cluster_share_positivo",
                "cluster_share_negativo",
                "cluster_share_neutro",
                "topico",
            ]
        ]
        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "nome_topico": topic_name,
                "palavras_chave": ", ".join(cluster_keywords),
                "quantidade_registros": int(len(cluster_records)),
                "sentimento_predominante": (
                    "Positivo"
                    if positive_share >= max(negative_share, neutral_share)
                    else "Negativo"
                    if negative_share >= max(positive_share, neutral_share)
                    else "Neutro"
                ),
                "fonte_predominante": dominant_source,
                "cluster_signal": cluster_signal,
                "share_positivo": positive_share,
                "share_negativo": negative_share,
                "share_neutro": neutral_share,
                "acao_sugerida": _suggest_cluster_action(topic_name, cluster_signal),
                "exemplo_representativo": representative_example,
                "metodo": LAST_TOPIC_EMBEDDING_METHOD,
                "data_processamento": pd.Timestamp.now(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["quantidade_registros", "cluster_signal", "nome_topico"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        summary_df["id_topico"] = range(1, len(summary_df) + 1)
        summary_df = summary_df[
            [
                "id_topico",
                "cluster_id",
                "nome_topico",
                "palavras_chave",
                "quantidade_registros",
                "sentimento_predominante",
                "fonte_predominante",
                "cluster_signal",
                "share_positivo",
                "share_negativo",
                "share_neutro",
                "acao_sugerida",
                "exemplo_representativo",
                "metodo",
                "data_processamento",
            ]
        ]

    return working_df, summary_df
