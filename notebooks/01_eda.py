# %%
import pandas as pd

unified_path = r"../data/gold/dashboard/unified_dataset.csv"
df_unified = pd.read_csv(unified_path, encoding="utf-8")

df_unified["data_publicacao"] = pd.to_datetime(df_unified["data_publicacao"], errors="coerce")

print(df_unified.shape)
print(df_unified.head())

# %%
print(df_unified["fonte"].value_counts())
print(df_unified["fonte"].value_counts(normalize=True) * 100)

# %%
print(
    df_unified.groupby("fonte")["data_publicacao"]
    .agg(["min", "max", "count"])
)

# %%
print((df_unified.notna().mean() * 100).sort_values(ascending=False))

# %%
print(df_unified["sentimento_real"].value_counts(dropna=False))

# %%
print(df_unified["nota"].describe())
print(df_unified["nota"].value_counts(dropna=False).sort_index())

# %%
textual_path = r"../data/sandbox/notebooks/textual_dataset.csv"
df_textual = pd.read_csv(textual_path, encoding="utf-8")

df_textual["data_publicacao"] = pd.to_datetime(df_textual["data_publicacao"], errors="coerce")

print(df_textual.shape)
print(df_textual.head())

# %%
print(df_textual["fonte"].value_counts())
print(df_textual["fonte"].value_counts(normalize=True) * 100)

# %%
df_textual["tamanho_texto"] = df_textual["texto_original"].astype(str).str.len()

print(df_textual["tamanho_texto"].describe())

# %%
print(
    df_textual.groupby("fonte")["tamanho_texto"]
    .agg(["mean", "median", "min", "max"])
)

# %%
df_textual["ano_mes"] = df_textual["data_publicacao"].dt.to_period("M")

print(df_textual["ano_mes"].value_counts().sort_index())

# %%
print(df_textual["sentimento_real"].value_counts(dropna=False))

# %%
print(df_textual["texto_original"].sample(10, random_state=42).tolist())

# %%
df_gp = df_unified[df_unified["fonte"] == "google_play"].copy()

print(pd.crosstab(df_gp["nota"], df_gp["sentimento_real"]))

# %%
print(
    df_unified.groupby("fonte")["texto_original"]
    .apply(lambda s: s.notna().sum())
)

# %%
df_youtube = pd.read_csv(r'../data/raw/youtube/nubank_youtube_comments.csv', encoding="utf-8")
print(df_youtube[["data_publicacao"]].head(10))
print(df_youtube["data_publicacao"].dtype)

# %%
print(
    df_textual.groupby("fonte")["data_publicacao"]
    .agg(["min", "max", "count"])
)

# %%
print(
    df_textual.groupby(["fonte", "ano_mes"])
    .size()
)

# %%
df_cg = df_unified[df_unified["fonte"] == "consumidor_gov"].copy()
print(df_cg["categoria"].value_counts().head(20))

# %%
print(
    df_unified.groupby("fonte")["data_publicacao"]
    .agg(["min", "max", "count"])
)

print(
    df_textual.groupby("fonte")["data_publicacao"]
    .agg(["min", "max", "count"])
)

# %%
from collections import Counter

df = pd.read_csv(
    r"../data/silver/preprocessing/textual_dataset_preprocessed.csv",
    encoding="utf-8"
)

df["qtd_palavras"] = df["texto_limpo"].astype(str).str.split().str.len()

print(df.shape)
print(df["fonte"].value_counts())
print(df["qtd_palavras"].describe())
print(df["sentimento_real"].value_counts(dropna=False))
print(df["texto_limpo"].sample(10, random_state=42).tolist())

# %%
tokens = " ".join(df["texto_limpo"].astype(str)).split()
freq = Counter(tokens)
print(freq.most_common(30))

# %%
for fonte in df["fonte"].unique():
    tokens_fonte = " ".join(df.loc[df["fonte"] == fonte, "texto_limpo"].astype(str)).split()
    freq_fonte = Counter(tokens_fonte)
    print(f"\nFonte: {fonte}")
    print(freq_fonte.most_common(20))


