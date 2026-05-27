from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.database.connection import get_engine
from src.utils.paths import RAW_DIR

SILVER_COLUMNS = [
    "id_registro",
    "fonte",
    "data_publicacao",
    "texto_original",
    "texto_limpo",
    "nota",
    "status_reclamacao",
    "categoria_problema",
    "uf",
    "versao_app",
    "sentimento_real",
    "data_processamento",
]
EXTRA_COLUMNS = [
    "assunto",
    "problema_detalhado",
    "cidade",
    "empresa",
    "arquivo_origem",
]


def clean_text(text: str | None) -> str:
    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zà-ú0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_texto_original(dataframe: pd.DataFrame) -> pd.Series:
    assunto = dataframe.get("Assunto", pd.Series("", index=dataframe.index)).fillna("")
    problema = dataframe.get("Problema", pd.Series("", index=dataframe.index)).fillna("")
    categoria = dataframe.get("Grupo Problema", pd.Series("", index=dataframe.index)).fillna("")

    text_blocks = pd.DataFrame(
        {
            "assunto": assunto.astype(str).str.strip(),
            "problema": problema.astype(str).str.strip(),
            "categoria": categoria.astype(str).str.strip(),
        }
    )

    return text_blocks.apply(
        lambda row: " | ".join(
            [
                f"Assunto: {row['assunto']}" if row["assunto"] else "",
                f"Problema: {row['problema']}" if row["problema"] else "",
                f"Categoria: {row['categoria']}" if row["categoria"] else "",
            ]
        ).strip(" |"),
        axis=1,
    )


def _load_raw_consumidor_gov() -> tuple[pd.DataFrame, str]:
    input_dir = RAW_DIR / "consumidor_gov"
    output_path = input_dir / "consumidor_gov_processed.csv"
    input_files = [
        path for path in sorted(input_dir.glob("*.csv")) if path.name != output_path.name
    ]
    if not input_files:
        sql_dataframe = _load_consumidor_gov_from_sql()
        if sql_dataframe.empty:
            raise FileNotFoundError(
                f"Nenhum CSV bruto encontrado em {input_dir} e nenhum fallback valido no SQL Server. "
                "Adicione os arquivos do Consumidor.gov antes de processar a base."
            )
        return sql_dataframe, str(output_path)

    dataframes = []
    for file_path in input_files:
        dataframe = pd.read_csv(file_path, sep=";", encoding="utf-8")
        dataframe["arquivo_origem"] = file_path.name
        dataframes.append(dataframe)

    return pd.concat(dataframes, ignore_index=True), str(output_path)


def _load_consumidor_gov_from_sql() -> pd.DataFrame:
    query = """
        SELECT
            id_registro,
            fonte,
            data_publicacao,
            texto_original,
            texto_limpo,
            nota,
            status_reclamacao,
            categoria_problema,
            uf,
            versao_app,
            sentimento_real,
            data_processamento
        FROM silver.reviews_cleaned
        WHERE fonte = 'consumidor_gov'
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            dataframe = pd.read_sql_query(query, connection)
    except Exception:
        return pd.DataFrame()

    if dataframe.empty:
        return dataframe

    for column in EXTRA_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = ""
    return dataframe[SILVER_COLUMNS + EXTRA_COLUMNS].copy()


def _transform_consumidor_gov_to_silver(df_consumidor: pd.DataFrame) -> pd.DataFrame:
    filtered = df_consumidor[
        df_consumidor["Nome Fantasia"].astype(str).str.contains("nubank", case=False, na=False)
    ].copy()

    filtered["Data Abertura"] = pd.to_datetime(
        filtered.get("Data Abertura"),
        errors="coerce",
    )
    filtered["Nota do Consumidor"] = pd.to_numeric(
        filtered.get("Nota do Consumidor"),
        errors="coerce",
    )

    filtered["id_registro"] = "cg_" + filtered.index.astype(str)
    filtered["fonte"] = "consumidor_gov"
    filtered["data_publicacao"] = filtered["Data Abertura"]
    filtered["texto_original"] = _build_texto_original(filtered)
    filtered["texto_limpo"] = filtered["texto_original"].apply(clean_text)
    filtered["nota"] = filtered["Nota do Consumidor"]
    filtered["status_reclamacao"] = (
        filtered.get("Situação", pd.Series("Não informado", index=filtered.index))
        .fillna("Não informado")
        .astype(str)
        .str.strip()
        .replace("", "Não informado")
    )
    filtered["categoria_problema"] = (
        filtered.get("Grupo Problema", pd.Series("Não informado", index=filtered.index))
        .fillna("Não informado")
        .astype(str)
        .str.strip()
        .replace("", "Não informado")
    )
    filtered["uf"] = (
        filtered.get("UF", pd.Series("Não informado", index=filtered.index))
        .fillna("Não informado")
        .astype(str)
        .str.strip()
        .replace("", "Não informado")
    )
    filtered["versao_app"] = "Não aplicável"
    filtered["sentimento_real"] = None
    filtered["data_processamento"] = datetime.now()
    filtered["assunto"] = filtered.get("Assunto", pd.Series("", index=filtered.index)).fillna("")
    filtered["problema_detalhado"] = (
        filtered.get("Problema", pd.Series("", index=filtered.index)).fillna("")
    )
    filtered["cidade"] = filtered.get("Cidade", pd.Series("", index=filtered.index)).fillna("")
    filtered["empresa"] = filtered.get("Nome Fantasia", pd.Series("", index=filtered.index)).fillna("")

    processed = filtered[SILVER_COLUMNS + EXTRA_COLUMNS].copy()
    processed = processed.dropna(subset=["data_publicacao"])
    processed = processed[processed["texto_limpo"].str.len() > 0].copy()
    processed = processed.drop_duplicates(subset=["texto_original"])
    return processed.reset_index(drop=True)


def main() -> None:
    df_consumidor, output_path = _load_raw_consumidor_gov()
    if set(SILVER_COLUMNS).issubset(df_consumidor.columns):
        processed = df_consumidor.copy()
        for column in EXTRA_COLUMNS:
            if column not in processed.columns:
                processed[column] = ""
        processed["texto_limpo"] = processed["texto_original"].apply(clean_text)
        processed = processed[SILVER_COLUMNS + EXTRA_COLUMNS].copy()
    else:
        processed = _transform_consumidor_gov_to_silver(df_consumidor)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)


if __name__ == "__main__":
    main()
