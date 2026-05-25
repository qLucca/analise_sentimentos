from __future__ import annotations

import pandas as pd

from src.utils.paths import RAW_DIR


def main() -> None:
    input_dir = RAW_DIR / "consumidor_gov"
    output_path = input_dir / "consumidor_gov_processed.csv"
    input_files = [
        path for path in sorted(input_dir.glob("*.csv")) if path.name != output_path.name
    ]

    print(input_files)
    print(len(input_files))

    dataframes = []
    for file_path in input_files:
        dataframe = pd.read_csv(file_path, sep=";", encoding="utf-8")
        dataframe["arquivo_origem"] = file_path.name
        dataframes.append(dataframe)

    df_consumidor = pd.concat(dataframes, ignore_index=True)

    print(df_consumidor.shape)
    print(df_consumidor.columns.tolist())
    print(df_consumidor.head())

    df_consumidor = df_consumidor[
        df_consumidor["Nome Fantasia"].str.contains("nubank", case=False, na=False)
    ].copy()

    print(df_consumidor.shape)
    print(df_consumidor["Nome Fantasia"].value_counts())

    df_consumidor["Data Abertura"] = pd.to_datetime(
        df_consumidor["Data Abertura"],
        errors="coerce",
    )
    df_consumidor["Nota do Consumidor"] = pd.to_numeric(
        df_consumidor["Nota do Consumidor"],
        errors="coerce",
    )

    df_consumidor["id_registro"] = "cg_" + df_consumidor.index.astype(str)
    df_consumidor["fonte"] = "consumidor_gov"
    df_consumidor["data_publicacao"] = df_consumidor["Data Abertura"]
    df_consumidor["titulo"] = df_consumidor["Problema"]
    df_consumidor["texto_original"] = None
    df_consumidor["nota"] = df_consumidor["Nota do Consumidor"]
    df_consumidor["usuario"] = None
    df_consumidor["categoria"] = df_consumidor["Grupo Problema"]
    df_consumidor["status"] = df_consumidor["Situação"]
    df_consumidor["sentimento_real"] = None

    df_processed = df_consumidor[
        [
            "id_registro",
            "fonte",
            "data_publicacao",
            "titulo",
            "texto_original",
            "nota",
            "usuario",
            "categoria",
            "status",
            "sentimento_real",
            "Nome Fantasia",
            "Segmento de Mercado",
            "Área",
            "Assunto",
            "Problema",
            "Respondida",
            "Avaliação Reclamação",
            "UF",
            "Cidade",
            "arquivo_origem",
        ]
    ].copy()

    print(df_processed.shape)
    print(df_processed.columns.tolist())
    print(df_processed.head())

    df_processed.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)


if __name__ == "__main__":
    main()
