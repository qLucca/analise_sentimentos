# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

df_gov = pd.read_csv(
    r"../data/raw/consumidor_gov/consumidor_gov_processed.csv",
    encoding="utf-8"
)

print(df_gov.shape)
print(df_gov.columns.tolist())
print(df_gov.head())

# %%
print("Total de registros:", len(df_gov))
print(df_gov["fonte"].value_counts())
print(df_gov["data_publicacao"].min(), "até", df_gov["data_publicacao"].max())

# %%
top_categorias = df_gov["categoria"].value_counts().head(10)
print(top_categorias)

# %%
plt.figure(figsize=(10, 5))
sns.barplot(
    x=top_categorias.values,
    y=top_categorias.index
)
plt.title("Top 10 Categorias de Reclamação - Consumidor.gov")
plt.xlabel("Quantidade")
plt.ylabel("Categoria")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "gov_top_categorias.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
top_problemas = df_gov["Problema"].value_counts().head(10)
print(top_problemas)

# %%
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_problemas.values,
    y=top_problemas.index
)
plt.title("Top 10 Problemas - Consumidor.gov")
plt.xlabel("Quantidade")
plt.ylabel("Problema")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "gov_top_problemas.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
status_dist = df_gov["status"].value_counts(dropna=False)
status_pct = (df_gov["status"].value_counts(normalize=True, dropna=False) * 100).round(2)

print(status_dist)
print(status_pct)

# %%
print(df_gov["nota"].describe())
print(df_gov["nota"].value_counts(dropna=False).sort_index())

# %%
plt.figure(figsize=(8, 5))
sns.histplot(df_gov["nota"].dropna(), bins=5)
plt.title("Distribuição da Nota do Consumidor - Consumidor.gov")
plt.xlabel("Nota")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "gov_distribuicao_nota.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
df_gov["data_publicacao"] = pd.to_datetime(df_gov["data_publicacao"], errors="coerce")
df_gov["ano_mes"] = df_gov["data_publicacao"].dt.to_period("M").astype(str)

gov_time = df_gov["ano_mes"].value_counts().sort_index()
print(gov_time)

# %%
plt.figure(figsize=(10, 5))
gov_time.plot(marker="o")
plt.title("Evolução Mensal das Reclamações - Consumidor.gov")
plt.xlabel("Ano-mês")
plt.ylabel("Quantidade")
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(IMAGES_DIR / "gov_evolucao_mensal.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
top_ufs = df_gov["UF"].value_counts().head(10)
print(top_ufs)

# %%
plt.figure(figsize=(8, 5))
sns.barplot(
    x=top_ufs.index,
    y=top_ufs.values
)
plt.title("Top 10 UFs com Reclamações - Consumidor.gov")
plt.xlabel("UF")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "gov_top_ufs.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
top_categorias_df = top_categorias.reset_index()
top_categorias_df.columns = ["categoria", "quantidade"]

# %%
status_table = pd.DataFrame({
    "quantidade": status_dist,
    "percentual": status_pct
}).reset_index()
status_table.columns = ["status", "quantidade", "percentual"]

# %%
gov_time_df = gov_time.reset_index()
gov_time_df.columns = ["ano_mes", "quantidade"]

# %%
top_categorias_df.to_csv(
    r"../artifacts/reports/notebooks/gov_top_categorias.csv",
    index=False,
    encoding="utf-8-sig"
)

status_table.to_csv(
    r"../artifacts/reports/notebooks/gov_status_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

gov_time_df.to_csv(
    r"../artifacts/reports/notebooks/gov_time_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Arquivos salvos.")


