from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas.io.formats.style import Styler

st.set_page_config(
    page_title="Dashboard de Sentimentos do Nubank",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONSUMIDOR_GOV_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "consumidor_gov"

UNIFIED_DATASET_PATH = PROCESSED_DATA_DIR / "unified_dataset.csv"
MODEL_COMPARISON_PATH = PROCESSED_DATA_DIR / "model_comparison_summary.csv"
YOUTUBE_BERT_DATASET_PATH = (
    PROCESSED_DATA_DIR / "youtube_with_predicted_sentiment_bertimbau.csv"
)
CONSUMIDOR_GOV_DATASET_PATH = CONSUMIDOR_GOV_DATA_DIR / "consumidor_gov_processed.csv"

SECTION_OPTIONS = (
    "Visao Geral",
    "YouTube + BERTimbau",
    "Consumidor.gov",
    "Modelos",
)
FILTERABLE_DATASET_NAMES = (
    "Base unificada",
    "YouTube + BERTimbau",
    "Consumidor.gov",
)
SENTIMENT_ORDER = ["Positivo", "Negativo", "Neutro"]
SENTIMENT_COLORS = {
    "Positivo": "#2E8B57",
    "Negativo": "#C0392B",
    "Neutro": "#7F8C8D",
}


def get_file_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


@st.cache_data(show_spinner=False)
def load_csv(path: Path, file_mtime_ns: int) -> pd.DataFrame:
    _ = file_mtime_ns
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_unified_dataset(file_mtime_ns: int) -> pd.DataFrame:
    return load_csv(UNIFIED_DATASET_PATH, file_mtime_ns)


@st.cache_data(show_spinner=False)
def load_model_comparison_summary(file_mtime_ns: int) -> pd.DataFrame:
    return load_csv(MODEL_COMPARISON_PATH, file_mtime_ns)


@st.cache_data(show_spinner=False)
def load_youtube_bert_dataset(file_mtime_ns: int) -> pd.DataFrame:
    return load_csv(YOUTUBE_BERT_DATASET_PATH, file_mtime_ns)


@st.cache_data(show_spinner=False)
def load_consumidor_gov_dataset(file_mtime_ns: int) -> pd.DataFrame:
    return load_csv(CONSUMIDOR_GOV_DATASET_PATH, file_mtime_ns)


def load_app_data() -> dict[str, pd.DataFrame]:
    return {
        "Base unificada": load_unified_dataset(get_file_mtime_ns(UNIFIED_DATASET_PATH)),
        "Resumo de modelos": load_model_comparison_summary(
            get_file_mtime_ns(MODEL_COMPARISON_PATH)
        ),
        "YouTube + BERTimbau": load_youtube_bert_dataset(
            get_file_mtime_ns(YOUTUBE_BERT_DATASET_PATH)
        ),
        "Consumidor.gov": load_consumidor_gov_dataset(
            get_file_mtime_ns(CONSUMIDOR_GOV_DATASET_PATH)
        ),
    }


def prepare_filterable_datasets(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    prepared_datasets = datasets.copy()

    for dataset_name in FILTERABLE_DATASET_NAMES:
        dataframe = prepared_datasets[dataset_name].copy()
        dataframe["data_publicacao"] = pd.to_datetime(
            dataframe["data_publicacao"], errors="coerce"
        )
        prepared_datasets[dataset_name] = dataframe

    return prepared_datasets


def get_global_date_bounds(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    combined_dates = pd.concat(
        [datasets[name]["data_publicacao"].dropna() for name in FILTERABLE_DATASET_NAMES],
        ignore_index=True,
    )

    if combined_dates.empty:
        raise ValueError("Nenhuma data valida encontrada nas bases filtraveis.")

    return combined_dates.min(), combined_dates.max()


def render_sidebar(
    datasets: dict[str, pd.DataFrame],
) -> tuple[str, tuple[pd.Timestamp, pd.Timestamp], str]:
    min_date, max_date = get_global_date_bounds(datasets)
    youtube_sentiment_options = ["Todos"] + sorted(
        datasets["YouTube + BERTimbau"]["sentimento_previsto_bert"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    st.sidebar.header("Filtros")
    selected_section = st.sidebar.radio("Secao", SECTION_OPTIONS)
    selected_date_range = st.sidebar.date_input(
        "Intervalo de datas",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    selected_youtube_sentiment = st.sidebar.selectbox(
        "Sentimento do YouTube",
        youtube_sentiment_options,
        index=0,
    )

    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
    else:
        start_date = end_date = selected_date_range

    return (
        selected_section,
        (pd.Timestamp(start_date), pd.Timestamp(end_date)),
        selected_youtube_sentiment,
    )


def filter_by_date_range(
    dataframe: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if "data_publicacao" not in dataframe:
        return dataframe

    normalized_start = start_date.normalize()
    next_day_after_end = end_date.normalize() + pd.Timedelta(days=1)
    date_mask = (
        dataframe["data_publicacao"] >= normalized_start
    ) & (dataframe["data_publicacao"] < next_day_after_end)
    return dataframe.loc[date_mask].copy()


def build_filtered_datasets(
    datasets: dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    filtered_datasets = datasets.copy()

    for dataset_name in FILTERABLE_DATASET_NAMES:
        filtered_datasets[dataset_name] = filter_by_date_range(
            filtered_datasets[dataset_name], start_date, end_date
        )

    return filtered_datasets


def filter_youtube_by_sentiment(
    youtube_dataframe: pd.DataFrame,
    youtube_sentiment: str,
) -> pd.DataFrame:
    if youtube_sentiment != "Todos":
        return youtube_dataframe.loc[
            youtube_dataframe["sentimento_previsto_bert"] == youtube_sentiment
        ].copy()

    return youtube_dataframe


def format_source_label(source_name: str) -> str:
    source_labels = {
        "google_play": "Google Play",
        "youtube": "YouTube",
        "consumidor_gov": "Consumidor.gov",
    }
    return source_labels.get(source_name, source_name.replace("_", " ").title())


def build_model_comparison_table(modelos_dataframe: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "f1_macro",
        "roc_auc_macro",
        "roc_auc_weighted",
        "f1_negativo",
        "f1_neutro",
        "f1_positivo",
    ]
    comparison_table = modelos_dataframe.copy()

    for column_name in metric_columns:
        if column_name in comparison_table:
            comparison_table[column_name] = comparison_table[column_name].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else "-"
            )

    return comparison_table


def format_integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def build_sentiment_summary(youtube_dataframe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        youtube_dataframe["sentimento_previsto_bert"]
        .fillna("Sem informacao")
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("sentimento")
        .reset_index(name="quantidade")
    )
    total_comments = summary["quantidade"].sum()
    summary["participacao"] = summary["quantidade"].map(
        lambda value: (value / total_comments) if total_comments else 0
    )
    summary["participacao"] = summary["participacao"].map(lambda value: f"{value:.1%}")
    return summary


def build_monthly_sentiment_series(
    youtube_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    monthly_series = youtube_dataframe.dropna(subset=["data_publicacao"]).copy()
    monthly_series["mes"] = monthly_series["data_publicacao"].dt.to_period("M").dt.to_timestamp()
    aggregated = (
        monthly_series.groupby(["mes", "sentimento_previsto_bert"])
        .size()
        .reset_index(name="quantidade")
    )
    return aggregated.sort_values(["mes", "sentimento_previsto_bert"])


def build_comment_examples(
    youtube_dataframe: pd.DataFrame,
    examples_per_sentiment: int = 3,
) -> pd.DataFrame:
    example_rows = []

    for sentiment in SENTIMENT_ORDER:
        sentiment_rows = youtube_dataframe.loc[
            youtube_dataframe["sentimento_previsto_bert"] == sentiment
        ].copy()
        sentiment_rows = sentiment_rows.sort_values(
            "data_publicacao", ascending=False, na_position="last"
        ).head(examples_per_sentiment)

        if sentiment_rows.empty:
            continue

        example_rows.append(
            sentiment_rows.assign(
                comentario=lambda dataframe: dataframe["texto_original"]
                .fillna("")
                .astype(str)
                .str.slice(0, 220),
                data_publicacao=lambda dataframe: dataframe["data_publicacao"].dt.strftime(
                    "%d/%m/%Y"
                ),
            )[
                [
                    "sentimento_previsto_bert",
                    "data_publicacao",
                    "usuario",
                    "titulo",
                    "comentario",
                ]
            ]
        )

    if not example_rows:
        return pd.DataFrame(
            columns=["sentimento_previsto_bert", "data_publicacao", "usuario", "titulo", "comentario"]
        )

    return pd.concat(example_rows, ignore_index=True).rename(
        columns={"sentimento_previsto_bert": "sentimento"}
    )


def build_frequent_words_table(
    youtube_dataframe: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    word_tables = []

    for sentiment in SENTIMENT_ORDER:
        sentiment_texts = youtube_dataframe.loc[
            youtube_dataframe["sentimento_previsto_bert"] == sentiment,
            "texto_limpo",
        ].dropna()
        words = sentiment_texts.astype(str).str.split().explode()

        if words.empty:
            continue

        words = words[
            words.str.len().ge(3) & ~words.str.fullmatch(r"\d+", na=False)
        ]
        top_words = words.value_counts().head(top_n).reset_index()
        top_words.columns = ["palavra", "frequencia"]
        top_words.insert(0, "sentimento", sentiment)
        word_tables.append(top_words)

    if not word_tables:
        return pd.DataFrame(columns=["sentimento", "palavra", "frequencia"])

    return pd.concat(word_tables, ignore_index=True)


def build_top_n_counts(
    dataframe: pd.DataFrame,
    column_name: str,
    top_n: int = 10,
    fallback_label: str = "Sem informacao",
) -> pd.DataFrame:
    counts = (
        dataframe[column_name]
        .fillna(fallback_label)
        .astype(str)
        .str.strip()
        .replace("", fallback_label)
        .value_counts()
        .head(top_n)
        .rename_axis(column_name)
        .reset_index(name="quantidade")
    )
    return counts


def build_monthly_volume(dataframe: pd.DataFrame) -> pd.DataFrame:
    monthly_volume = dataframe.dropna(subset=["data_publicacao"]).copy()
    monthly_volume["mes"] = monthly_volume["data_publicacao"].dt.to_period("M").dt.to_timestamp()
    return (
        monthly_volume.groupby("mes")
        .size()
        .reset_index(name="quantidade")
        .sort_values("mes")
    )


def style_model_comparison_table(modelos_dataframe: pd.DataFrame) -> Styler:
    numeric_columns = [
        "accuracy",
        "f1_macro",
        "roc_auc_macro",
        "roc_auc_weighted",
        "f1_negativo",
        "f1_neutro",
        "f1_positivo",
    ]
    best_model_index = modelos_dataframe.sort_values(
        ["accuracy", "roc_auc_macro", "f1_macro"],
        ascending=False,
    ).index[0]
    best_values = {
        column_name: modelos_dataframe[column_name].max()
        for column_name in numeric_columns
        if column_name in modelos_dataframe
    }

    def highlight_best_row(row: pd.Series) -> list[str]:
        if row.name == best_model_index:
            return ["background-color: #E8F8F0; font-weight: 600;"] * len(row)
        return [""] * len(row)

    def highlight_best_metric(column: pd.Series) -> list[str]:
        best_value = best_values.get(column.name)
        if best_value is None:
            return [""] * len(column)
        return [
            "background-color: #FCF3CF; font-weight: 700;"
            if pd.notna(value) and value == best_value
            else ""
            for value in column
        ]

    format_map = {
        column_name: "{:.2%}"
        for column_name in numeric_columns
        if column_name in modelos_dataframe
    }
    return (
        modelos_dataframe.style.format(format_map)
        .apply(highlight_best_row, axis=1)
        .apply(highlight_best_metric, subset=list(best_values.keys()))
    )


def build_model_interpretation(modelos_dataframe: pd.DataFrame) -> str:
    bert_row = modelos_dataframe.loc[modelos_dataframe["modelo"] == "BERTimbau"].iloc[0]
    neutral_leader = modelos_dataframe.sort_values("f1_neutro", ascending=False).iloc[0]
    return (
        f"O BERTimbau lidera em accuracy ({bert_row['accuracy']:.2%}) e ROC AUC macro "
        f"({bert_row['roc_auc_macro']:.2%}), indicando melhor desempenho geral. "
        f"Entre os baselines classicos, {neutral_leader['modelo']} entrega o melhor equilibrio "
        f"na classe Neutro com F1 de {neutral_leader['f1_neutro']:.2%}, mesmo sem superar o "
        "transformer nas metricas globais."
    )


def render_horizontal_bar_chart(
    dataframe: pd.DataFrame,
    category_column: str,
    value_column: str,
    title: str,
    color: str,
    x_label: str,
) -> None:
    figure_height = max(4.0, 0.55 * len(dataframe))
    figure, axis = plt.subplots(figsize=(10, figure_height))
    sns.barplot(
        data=dataframe,
        x=value_column,
        y=category_column,
        color=color,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.0f", padding=3)
    sns.despine(ax=axis)
    st.pyplot(figure, clear_figure=True, use_container_width=True)


def render_visao_geral(
    unified_dataframe: pd.DataFrame,
    modelos_dataframe: pd.DataFrame,
    selected_date_range: tuple[pd.Timestamp, pd.Timestamp],
) -> None:
    st.subheader("Visao Geral")
    total_registros = len(unified_dataframe)
    total_textual = unified_dataframe["texto_original"].fillna("").str.strip().ne("").sum()
    total_youtube = unified_dataframe["fonte"].eq("youtube").sum()
    total_consumidor_gov = unified_dataframe["fonte"].eq("consumidor_gov").sum()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total de registros", f"{total_registros:,}".replace(",", "."))
    metric_columns[1].metric("Total textual", f"{total_textual:,}".replace(",", "."))
    metric_columns[2].metric(
        "Registros do YouTube", f"{total_youtube:,}".replace(",", ".")
    )
    metric_columns[3].metric(
        "Registros do Consumidor.gov",
        f"{total_consumidor_gov:,}".replace(",", "."),
    )

    filtered_dates = unified_dataframe["data_publicacao"].dropna()
    if filtered_dates.empty:
        start_date, end_date = selected_date_range
    else:
        start_date, end_date = filtered_dates.min(), filtered_dates.max()

    st.caption(
        "Periodo coberto pelos dados filtrados: "
        f"{start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}"
    )

    source_volume = (
        unified_dataframe["fonte"]
        .fillna("desconhecido")
        .value_counts()
        .rename_axis("fonte")
        .reset_index(name="volume")
    )
    source_volume["fonte"] = source_volume["fonte"].map(format_source_label)

    chart_column, table_column = st.columns((1.4, 1), gap="large")

    with chart_column:
        st.markdown("**Volume por fonte**")
        figure, axis = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=source_volume, x="fonte", y="volume", palette="Blues_d", ax=axis)
        axis.set_xlabel("Fonte")
        axis.set_ylabel("Quantidade de registros")
        axis.set_title("Distribuicao de registros por fonte")
        axis.tick_params(axis="x", rotation=10)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.0f", padding=3)
        sns.despine(ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

    with table_column:
        st.markdown("**Resumo por fonte**")
        st.dataframe(source_volume, use_container_width=True, hide_index=True)

    st.markdown("**Comparacao de modelos**")
    st.dataframe(
        build_model_comparison_table(modelos_dataframe),
        use_container_width=True,
        hide_index=True,
    )


def render_youtube_section(youtube_dataframe: pd.DataFrame) -> None:
    st.subheader("YouTube + BERTimbau")
    st.metric("Total de comentarios do YouTube", format_integer(len(youtube_dataframe)))

    sentiment_summary = build_sentiment_summary(youtube_dataframe)
    monthly_sentiment_series = build_monthly_sentiment_series(youtube_dataframe)
    comment_examples = build_comment_examples(youtube_dataframe)
    frequent_words = build_frequent_words_table(youtube_dataframe)

    summary_column, chart_column = st.columns((0.9, 1.4), gap="large")

    with summary_column:
        st.markdown("**Resumo de sentimentos previstos**")
        st.dataframe(sentiment_summary, use_container_width=True, hide_index=True)

    with chart_column:
        st.markdown("**Distribuicao de sentimentos previstos**")
        figure, axis = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            data=sentiment_summary,
            x="sentimento",
            y="quantidade",
            order=SENTIMENT_ORDER,
            palette=[SENTIMENT_COLORS[sentiment] for sentiment in SENTIMENT_ORDER],
            ax=axis,
        )
        axis.set_xlabel("Sentimento previsto")
        axis.set_ylabel("Quantidade de comentarios")
        axis.set_title("Distribuicao de sentimentos do BERTimbau")
        for container in axis.containers:
            axis.bar_label(container, fmt="%.0f", padding=3)
        sns.despine(ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

    st.markdown("**Evolucao mensal por sentimento previsto**")
    if monthly_sentiment_series.empty:
        st.info("Nao ha dados suficientes para montar a serie temporal do YouTube.")
    else:
        figure, axis = plt.subplots(figsize=(10, 4.5))
        sns.lineplot(
            data=monthly_sentiment_series,
            x="mes",
            y="quantidade",
            hue="sentimento_previsto_bert",
            hue_order=SENTIMENT_ORDER,
            palette=SENTIMENT_COLORS,
            marker="o",
            ax=axis,
        )
        axis.set_xlabel("Mes")
        axis.set_ylabel("Quantidade de comentarios")
        axis.set_title("Comentarios por mes e sentimento previsto")
        axis.tick_params(axis="x", rotation=25)
        sns.despine(ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

    examples_column, words_column = st.columns((1.4, 1), gap="large")

    with examples_column:
        st.markdown("**Exemplos de comentarios por sentimento**")
        st.dataframe(comment_examples, use_container_width=True, hide_index=True)

    with words_column:
        st.markdown("**Palavras mais frequentes por sentimento**")
        st.dataframe(frequent_words, use_container_width=True, hide_index=True)


def render_consumidor_section(consumidor_dataframe: pd.DataFrame) -> None:
    st.subheader("Consumidor.gov")
    st.metric("Total de registros", format_integer(len(consumidor_dataframe)))

    top_categories = build_top_n_counts(consumidor_dataframe, "categoria", top_n=8)
    top_problems = build_top_n_counts(consumidor_dataframe, "Problema", top_n=8)
    status_distribution = build_top_n_counts(consumidor_dataframe, "status", top_n=10)
    top_ufs = build_top_n_counts(consumidor_dataframe, "UF", top_n=10)
    monthly_volume = build_monthly_volume(consumidor_dataframe)
    notas_series = pd.to_numeric(consumidor_dataframe["nota"], errors="coerce").dropna()

    category_column, problem_column = st.columns(2, gap="large")

    with category_column:
        st.markdown("**Principais categorias**")
        render_horizontal_bar_chart(
            top_categories,
            "categoria",
            "quantidade",
            "Categorias com mais reclamacoes",
            "#1f77b4",
            "Quantidade de registros",
        )

    with problem_column:
        st.markdown("**Principais problemas**")
        render_horizontal_bar_chart(
            top_problems,
            "Problema",
            "quantidade",
            "Problemas mais recorrentes",
            "#d35400",
            "Quantidade de registros",
        )

    status_column, notes_column = st.columns((1.1, 1), gap="large")

    with status_column:
        st.markdown("**Distribuicao de status**")
        figure, axis = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            data=status_distribution,
            x="status",
            y="quantidade",
            color="#16A085",
            ax=axis,
        )
        axis.set_xlabel("Status")
        axis.set_ylabel("Quantidade de registros")
        axis.set_title("Status das reclamacoes")
        axis.tick_params(axis="x", rotation=15)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.0f", padding=3)
        sns.despine(ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

    with notes_column:
        st.markdown("**Distribuicao da nota do consumidor**")
        if notas_series.empty:
            st.info("Nao ha notas validas para exibir.")
        else:
            figure, axis = plt.subplots(figsize=(8, 4.5))
            bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
            axis.hist(notas_series, bins=bins, color="#8E44AD", edgecolor="white")
            axis.set_xticks([1, 2, 3, 4, 5])
            axis.set_xlabel("Nota")
            axis.set_ylabel("Quantidade de registros")
            axis.set_title("Histograma das notas atribuidas")
            sns.despine(ax=axis)
            st.pyplot(figure, clear_figure=True, use_container_width=True)

    timeline_column, uf_column = st.columns((1.3, 1), gap="large")

    with timeline_column:
        st.markdown("**Evolucao mensal das reclamacoes**")
        if monthly_volume.empty:
            st.info("Nao ha datas validas para a serie temporal do Consumidor.gov.")
        else:
            figure, axis = plt.subplots(figsize=(10, 4.5))
            sns.lineplot(
                data=monthly_volume,
                x="mes",
                y="quantidade",
                color="#2C3E50",
                marker="o",
                ax=axis,
            )
            axis.set_xlabel("Mes")
            axis.set_ylabel("Quantidade de reclamacoes")
            axis.set_title("Volume mensal de reclamacoes")
            axis.tick_params(axis="x", rotation=25)
            sns.despine(ax=axis)
            st.pyplot(figure, clear_figure=True, use_container_width=True)

    with uf_column:
        st.markdown("**Principais UFs**")
        render_horizontal_bar_chart(
            top_ufs,
            "UF",
            "quantidade",
            "Estados com maior volume",
            "#7D3C98",
            "Quantidade de registros",
        )


def render_modelos_section(
    modelos_dataframe: pd.DataFrame,
    youtube_dataframe: pd.DataFrame,
) -> None:
    st.subheader("Modelos")
    st.markdown("**Comparacao final de modelos**")
    st.dataframe(
        style_model_comparison_table(modelos_dataframe),
        use_container_width=True,
        hide_index=True,
    )
    st.info(build_model_interpretation(modelos_dataframe))
    st.markdown("**Preview do dataset do YouTube previsto pelo BERTimbau**")
    preview_columns = [
        "data_publicacao",
        "titulo",
        "texto_original",
        "sentimento_previsto_bert",
    ]
    preview_dataframe = youtube_dataframe[preview_columns].head(15).copy()
    preview_dataframe["data_publicacao"] = preview_dataframe["data_publicacao"].dt.strftime(
        "%d/%m/%Y"
    )
    st.dataframe(preview_dataframe, use_container_width=True, hide_index=True)


def render_home(
    filtered_datasets: dict[str, pd.DataFrame],
    filtered_youtube_dataframe: pd.DataFrame,
    selected_section: str,
    selected_date_range: tuple[pd.Timestamp, pd.Timestamp],
    selected_youtube_sentiment: str,
) -> None:
    st.title("Dashboard de Sentimentos do Nubank")
    st.caption("Estrutura base do dashboard com filtros globais e navegacao lateral.")

    start_date, end_date = selected_date_range
    st.write(
        f"Secao atual: **{selected_section}** | "
        f"Periodo global: **{start_date.date()}** ate **{end_date.date()}**"
    )

    if selected_youtube_sentiment == "Todos":
        st.write("Filtro de sentimento do YouTube: **Todos**")
    else:
        st.write(
            "Filtro de sentimento do YouTube aplicado apenas na secao do YouTube: "
            f"**{selected_youtube_sentiment}**"
        )

    status_rows = [
        {
            "base": "Base unificada",
            "registros_filtrados": len(filtered_datasets["Base unificada"]),
            "registros_exibidos_na_secao": len(filtered_datasets["Base unificada"]),
        },
        {
            "base": "YouTube + BERTimbau",
            "registros_filtrados": len(filtered_datasets["YouTube + BERTimbau"]),
            "registros_exibidos_na_secao": len(filtered_youtube_dataframe),
        },
        {
            "base": "Consumidor.gov",
            "registros_filtrados": len(filtered_datasets["Consumidor.gov"]),
            "registros_exibidos_na_secao": len(filtered_datasets["Consumidor.gov"]),
        },
        {
            "base": "Resumo de modelos",
            "registros_filtrados": len(filtered_datasets["Resumo de modelos"]),
            "registros_exibidos_na_secao": len(filtered_datasets["Resumo de modelos"]),
        },
    ]

    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    if selected_section == "Visao Geral":
        render_visao_geral(
            filtered_datasets["Base unificada"],
            filtered_datasets["Resumo de modelos"],
            selected_date_range,
        )
    elif selected_section == "YouTube + BERTimbau":
        render_youtube_section(filtered_youtube_dataframe)
    elif selected_section == "Consumidor.gov":
        render_consumidor_section(filtered_datasets["Consumidor.gov"])
    elif selected_section == "Modelos":
        render_modelos_section(
            filtered_datasets["Resumo de modelos"],
            filtered_datasets["YouTube + BERTimbau"],
        )


def main() -> None:
    try:
        datasets = prepare_filterable_datasets(load_app_data())
        selected_section, selected_date_range, selected_youtube_sentiment = render_sidebar(
            datasets
        )
        filtered_datasets = build_filtered_datasets(
            datasets,
            selected_date_range[0],
            selected_date_range[1],
        )
        filtered_youtube_dataframe = filter_youtube_by_sentiment(
            filtered_datasets["YouTube + BERTimbau"],
            selected_youtube_sentiment,
        )
        render_home(
            filtered_datasets,
            filtered_youtube_dataframe,
            selected_section,
            selected_date_range,
            selected_youtube_sentiment,
        )
    except FileNotFoundError as exc:
        st.error(f"Arquivo de dados nao encontrado: {exc.filename}")
        st.stop()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Falha ao carregar os dados ou aplicar os filtros: {exc}")
        st.stop()


if __name__ == "__main__":
    main()
