from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_topics(df: pd.DataFrame, n_clusters: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    negative_df = df[df["sentimento_previsto"] == "Negativo"].copy()
    if negative_df.empty:
        return negative_df.assign(topico=None), pd.DataFrame(
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

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(negative_df["texto_limpo"])
    model = KMeans(n_clusters=min(n_clusters, len(negative_df)), random_state=42, n_init="auto")
    negative_df["cluster_id"] = model.fit_predict(matrix)

    terms = vectorizer.get_feature_names_out()
    topic_rows = []
    for cluster_id in sorted(negative_df["cluster_id"].unique()):
        center = model.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-8:][::-1]
        keywords = ", ".join(terms[index] for index in top_indices)
        cluster_records = negative_df[negative_df["cluster_id"] == cluster_id]
        topic_name = f"Topico {cluster_id + 1}"
        topic_rows.append(
            {
                "id_topico": int(cluster_id + 1),
                "nome_topico": topic_name,
                "palavras_chave": keywords,
                "quantidade_registros": int(len(cluster_records)),
                "sentimento_predominante": "Negativo",
                "fonte_predominante": cluster_records["fonte"].mode().iloc[0],
                "data_processamento": pd.Timestamp.now(),
            }
        )
        negative_df.loc[negative_df["cluster_id"] == cluster_id, "topico"] = topic_name

    final_df = df.copy()
    final_df["topico"] = final_df.get("topico")
    final_df.loc[negative_df.index, "topico"] = negative_df["topico"]
    return final_df, pd.DataFrame(topic_rows)
