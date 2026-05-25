# %%
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.paths import (
    GOLD_DASHBOARD_YOUTUBE_BERT_DATASET_PATH,
    NOTEBOOK_FIGURES_DIR,
    NOTEBOOK_REPORTS_DIR,
)

# %%
df_yt_bert = pd.read_csv(
    GOLD_DASHBOARD_YOUTUBE_BERT_DATASET_PATH,
    encoding="utf-8",
)

print(df_yt_bert.shape)
print(df_yt_bert.columns.tolist())
print(df_yt_bert.head())

# %%
sent_dist = df_yt_bert["sentimento_previsto_bert"].value_counts()
sent_dist_pct = df_yt_bert["sentimento_previsto_bert"].value_counts(normalize=True) * 100

print(sent_dist)
print(sent_dist_pct.round(2))

# %%
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df_yt_bert,
    x="sentimento_previsto_bert",
    order=["Negativo", "Neutro", "Positivo"],
)
plt.title("Distribuição de Sentimentos no YouTube - BERTimbau")
plt.xlabel("Sentimento previsto")
plt.ylabel("Quantidade")
plt.grid(axis="y", alpha=0.3)
plt.show()

# %%
for sentimento in ["Negativo", "Neutro", "Positivo"]:
    print(f"\n### {sentimento}")
    exemplos = df_yt_bert.loc[
        df_yt_bert["sentimento_previsto_bert"] == sentimento,
        ["texto_original", "texto_limpo"],
    ].head(10)
    print(exemplos.to_string(index=False))

# %%
for sentimento in ["Negativo", "Neutro", "Positivo"]:
    textos = df_yt_bert.loc[
        df_yt_bert["sentimento_previsto_bert"] == sentimento,
        "texto_limpo",
    ].dropna()

    tokens = " ".join(textos.astype(str)).split()
    freq = Counter(tokens)

    print(f"\n### {sentimento}")
    print(freq.most_common(20))

# %%
df_yt_bert["data_publicacao"] = pd.to_datetime(df_yt_bert["data_publicacao"], errors="coerce")
df_yt_bert["ano_mes"] = df_yt_bert["data_publicacao"].dt.to_period("M").astype(str)

yt_time = (
    df_yt_bert.groupby(["ano_mes", "sentimento_previsto_bert"])
    .size()
    .reset_index(name="quantidade")
)

print(yt_time.head())

# %%
plt.figure(figsize=(10, 5))
sns.lineplot(
    data=yt_time,
    x="ano_mes",
    y="quantidade",
    hue="sentimento_previsto_bert",
    hue_order=["Negativo", "Neutro", "Positivo"],
    marker="o",
)
plt.title("Evolução Mensal dos Sentimentos no YouTube - BERTimbau")
plt.xlabel("Ano-mês")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.show()

# %%
summary_table = pd.DataFrame(
    {
        "quantidade": sent_dist,
        "percentual": sent_dist_pct.round(2),
    }
).reset_index()

summary_table.columns = ["sentimento", "quantidade", "percentual"]
summary_table

# %%
summary_output = NOTEBOOK_REPORTS_DIR / "youtube_bert_sentiment_summary.csv"
summary_table.to_csv(summary_output, index=False, encoding="utf-8-sig")
print(summary_output)

# %%
plt.figure(figsize=(10, 5))
sns.lineplot(
    data=yt_time,
    x="ano_mes",
    y="quantidade",
    hue="sentimento_previsto_bert",
    hue_order=["Negativo", "Neutro", "Positivo"],
    marker="o",
)

plt.title("Evolução Mensal dos Sentimentos no YouTube - BERTimbau")
plt.xlabel("Ano-mês")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()

chart_output = NOTEBOOK_FIGURES_DIR / "youtube_bert_sentiment_over_time.png"
plt.savefig(chart_output, dpi=300, bbox_inches="tight")
plt.show()

print(chart_output)
