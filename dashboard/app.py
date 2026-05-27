from pathlib import Path
import sys
import re
from string import Template

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas.io.formats.style import Styler
from src.utils.paths import get_dashboard_dataset_paths

st.set_page_config(
    page_title="Dashboard de Sentimentos do Nubank",
    layout="wide",
)

DASHBOARD_DATASET_PATHS = get_dashboard_dataset_paths()

SECTION_OPTIONS = (
    "Visão Geral",
    "O que está bom",
    "O que está ruim",
    "O que melhorar",
    "Tendência",
)
FILTERABLE_DATASET_NAMES = (
    "Base unificada",
    "BERTimbau na base completa",
    "YouTube + BERTimbau",
    "Consumidor.gov",
)
SENTIMENT_ORDER = ["Positivo", "Negativo", "Neutro"]
SENTIMENT_COLORS = {
    "Positivo": "#2E8B57",
    "Negativo": "#C0392B",
    "Neutro": "#7F8C8D",
}
SOURCE_COLORS = {
    "Google Play": "#0F766E",
    "YouTube": "#D9485F",
    "Consumidor.gov": "#1D4ED8",
}
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
OPTIONAL_DATASET_SCHEMAS = {
    "BERTimbau na base completa": [
        "id_registro",
        "fonte",
        "data_publicacao",
        "texto_original",
        "nota",
        "status_reclamacao",
        "categoria_problema",
        "uf",
        "versao_app",
        "sentimento_real",
        "texto_limpo",
        "data_processamento",
        "sentimento_previsto_bert",
        "confianca_bert",
        "predicao_incerta_bert",
        "tema_negocio",
        "score_negativo_bert",
        "score_neutro_bert",
        "score_positivo_bert",
    ],
    "Resumo de clusters": [
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
    ],
    "YouTube + BERTimbau": [
        "data_publicacao",
        "titulo",
        "texto_original",
        "texto_limpo",
        "usuario",
        "sentimento_previsto_bert",
    ],
    "Consumidor.gov": [
        "data_publicacao",
        "categoria_problema",
        "texto_original",
        "status_reclamacao",
        "uf",
        "nota",
    ],
}


def apply_custom_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "axes.edgecolor": "#d7e3dc",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#17342f",
            "axes.labelcolor": "#17342f",
            "xtick.color": "#17342f",
            "ytick.color": "#17342f",
        }
    )

    bg_soft = "#f5f7f4"
    card_bg = "linear-gradient(135deg, #ffffff 0%, #f7fbf8 100%)"
    accent = "#0f766e"
    accent_soft = "#dff3ef"
    text_main = "#17342f"
    text_muted = "#5f6f69"
    border_soft = "rgba(23, 52, 47, 0.08)"
    shadow_soft = "0 18px 45px rgba(18, 52, 45, 0.08)"
    hero_bg = "linear-gradient(135deg, #113b35 0%, #1f6f67 58%, #d9485f 150%)"

    css = Template(
        """
        <style>
            :root {
                --bg-soft: $bg_soft;
                --card-bg: $card_bg;
                --accent: $accent;
                --accent-soft: $accent_soft;
                --text-main: $text_main;
                --text-muted: $text_muted;
                --border-soft: $border_soft;
                --shadow-soft: $shadow_soft;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 32%),
                    radial-gradient(circle at top right, rgba(217, 72, 95, 0.10), transparent 28%),
                    linear-gradient(180deg, $bg_soft 0%, $bg_soft 100%);
            }

            .block-container {
                padding-top: 2.2rem;
                padding-bottom: 2rem;
            }

            .hero-panel {
                background: $hero_bg;
                color: white;
                padding: 1.8rem 1.8rem 1.4rem 1.8rem;
                border-radius: 24px;
                box-shadow: 0 22px 55px rgba(17, 59, 53, 0.22);
                margin-bottom: 1.2rem;
            }

            .hero-eyebrow {
                display: inline-block;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.15);
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin-bottom: 0.85rem;
            }

            .hero-title {
                font-size: 2.2rem;
                font-weight: 800;
                line-height: 1.08;
                margin: 0;
            }

            .hero-subtitle {
                margin-top: 0.8rem;
                color: rgba(255, 255, 255, 0.84);
                font-size: 1rem;
                max-width: 760px;
            }

            .hero-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin-top: 1rem;
            }

            .filter-pill {
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 999px;
                padding: 0.45rem 0.8rem;
                font-size: 0.86rem;
            }

            .metric-card {
                background: var(--card-bg);
                border: 1px solid var(--border-soft);
                border-radius: 20px;
                padding: 1.05rem 1rem 1rem 1rem;
                box-shadow: var(--shadow-soft);
                min-height: 118px;
            }

            .metric-label {
                color: var(--text-muted);
                font-size: 0.88rem;
                font-weight: 600;
                margin-bottom: 0.45rem;
            }

            .metric-value {
                color: var(--text-main);
                font-size: 1.9rem;
                font-weight: 800;
                line-height: 1.05;
            }

            .metric-caption {
                color: var(--text-muted);
                font-size: 0.82rem;
                margin-top: 0.35rem;
            }

            .section-card {
                background: rgba(255, 255, 255, 0.84);
                border: 1px solid var(--border-soft);
                border-radius: 22px;
                padding: 1.1rem 1.1rem 0.55rem 1.1rem;
                box-shadow: var(--shadow-soft);
                margin-bottom: 1.25rem;
                backdrop-filter: blur(8px);
            }

            .section-heading {
                color: var(--text-main);
                font-size: 1.08rem;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }

            .section-subheading {
                color: var(--text-muted);
                font-size: 0.87rem;
                margin-bottom: 0.9rem;
            }

            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f7fbf9 0%, #eef5f1 100%);
                border-right: 1px solid rgba(23, 52, 47, 0.08);
            }
        </style>
        """
    ).substitute(
        bg_soft=bg_soft,
        card_bg=card_bg,
        accent=accent,
        accent_soft=accent_soft,
        text_main=text_main,
        text_muted=text_muted,
        border_soft=border_soft,
        shadow_soft=shadow_soft,
        hero_bg=hero_bg,
    )
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-heading">{title}</div>
            <div class="section-subheading">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_file_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


@st.cache_data(show_spinner=False)
def load_csv(path: Path, file_mtime_ns: int) -> pd.DataFrame:
    _ = file_mtime_ns
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def load_unified_dataset(file_mtime_ns: int) -> pd.DataFrame:
    return load_csv(DASHBOARD_DATASET_PATHS["Base unificada"], file_mtime_ns)


@st.cache_data(show_spinner=False)
def load_model_comparison_summary(file_mtime_ns: int) -> pd.DataFrame:
    return load_csv(DASHBOARD_DATASET_PATHS["Resumo de modelos"], file_mtime_ns)


@st.cache_data(show_spinner=False)
def load_primary_model_dataset(file_mtime_ns: int) -> pd.DataFrame:
    return load_csv(DASHBOARD_DATASET_PATHS["Modelo principal"], file_mtime_ns)


@st.cache_data(show_spinner=False)
def load_youtube_bert_dataset(file_mtime_ns: int) -> pd.DataFrame:
    return load_csv(DASHBOARD_DATASET_PATHS["YouTube + BERTimbau"], file_mtime_ns)


@st.cache_data(show_spinner=False)
def load_bertimbau_full_dataset(file_mtime_ns: int) -> pd.DataFrame:
    return load_csv(DASHBOARD_DATASET_PATHS["BERTimbau na base completa"], file_mtime_ns)


@st.cache_data(show_spinner=False)
def load_consumidor_gov_dataset(file_mtime_ns: int) -> pd.DataFrame:
    return load_csv(DASHBOARD_DATASET_PATHS["Consumidor.gov"], file_mtime_ns)


def load_app_data() -> dict[str, pd.DataFrame]:
    datasets = {
        "Base unificada": load_unified_dataset(
            get_file_mtime_ns(DASHBOARD_DATASET_PATHS["Base unificada"])
        ),
        "Resumo de modelos": load_model_comparison_summary(
            get_file_mtime_ns(DASHBOARD_DATASET_PATHS["Resumo de modelos"])
        ),
        "Modelo principal": load_primary_model_dataset(
            get_file_mtime_ns(DASHBOARD_DATASET_PATHS["Modelo principal"])
        ),
    }

    try:
        datasets["BERTimbau na base completa"] = load_bertimbau_full_dataset(
            get_file_mtime_ns(DASHBOARD_DATASET_PATHS["BERTimbau na base completa"])
        )
    except KeyError:
        datasets["BERTimbau na base completa"] = pd.DataFrame(
            columns=OPTIONAL_DATASET_SCHEMAS["BERTimbau na base completa"]
        )
    except FileNotFoundError:
        datasets["BERTimbau na base completa"] = pd.DataFrame(
            columns=OPTIONAL_DATASET_SCHEMAS["BERTimbau na base completa"]
        )

    for dataset_name, columns in OPTIONAL_DATASET_SCHEMAS.items():
        if dataset_name not in DASHBOARD_DATASET_PATHS:
            datasets[dataset_name] = pd.DataFrame(columns=columns)
            continue
        try:
            dataset_path = Path(DASHBOARD_DATASET_PATHS[dataset_name])
            if not dataset_path.exists():
                raise FileNotFoundError(str(dataset_path))
            datasets[dataset_name] = load_csv(
                dataset_path,
                get_file_mtime_ns(dataset_path),
            )
        except FileNotFoundError:
            datasets[dataset_name] = pd.DataFrame(columns=columns)

    return datasets


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

    st.sidebar.markdown("## Painel de filtros")
    st.sidebar.caption(
        "Controle o período e navegue entre as leituras de negócio do painel."
    )
    selected_section = st.sidebar.radio("Seção", SECTION_OPTIONS)
    selected_date_range = st.sidebar.date_input(
        "Intervalo de datas",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    selected_youtube_sentiment = "Todos"

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


def format_model_name(model_name: str | None) -> str:
    if model_name is None or pd.isna(model_name):
        return "Modelo desconhecido"

    normalized_name = str(model_name).strip()
    model_labels = {
        "LogisticRegression": "Logistic Regression",
        "LinearSVC": "Linear SVC",
        "MultinomialNB": "Multinomial NB",
        "BERTimbau": "BERTimbau",
    }
    return model_labels.get(normalized_name, normalized_name.replace("_", " "))


def classify_business_theme(text: str | None) -> str:
    if text is None or pd.isna(text):
        return "Sem texto"

    normalized_text = str(text).lower()
    for theme_name, pattern in BUSINESS_THEME_RULES:
        if re.search(pattern, normalized_text):
            return theme_name
    return "Outros temas"


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


def format_percentage(value: float) -> str:
    return f"{value:.1%}"


def get_date_span_label(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> str:
    return f"{start_date.strftime('%d/%m/%Y')} ate {end_date.strftime('%d/%m/%Y')}"


def build_source_volume_summary(unified_dataframe: pd.DataFrame) -> pd.DataFrame:
    source_volume = (
        unified_dataframe["fonte"]
        .fillna("desconhecido")
        .value_counts()
        .rename_axis("fonte")
        .reset_index(name="volume")
    )
    source_volume["fonte"] = source_volume["fonte"].map(format_source_label)
    total_volume = source_volume["volume"].sum()
    source_volume["participacao"] = source_volume["volume"].map(
        lambda value: value / total_volume if total_volume else 0
    )
    return source_volume


def build_unified_monthly_volume(unified_dataframe: pd.DataFrame) -> pd.DataFrame:
    monthly_volume = unified_dataframe.dropna(subset=["data_publicacao"]).copy()
    if monthly_volume.empty:
        return pd.DataFrame(columns=["mes", "quantidade"])

    monthly_volume["mes"] = (
        monthly_volume["data_publicacao"].dt.to_period("M").dt.to_timestamp()
    )
    return (
        monthly_volume.groupby("mes")
        .size()
        .reset_index(name="quantidade")
        .sort_values("mes")
    )


def build_sentiment_summary(
    youtube_dataframe: pd.DataFrame,
    sentiment_column: str = "sentimento_previsto",
) -> pd.DataFrame:
    effective_sentiment_column = _resolve_sentiment_column(youtube_dataframe, sentiment_column)
    if effective_sentiment_column is None or youtube_dataframe.empty:
        return pd.DataFrame(columns=["sentimento", "quantidade", "participacao"])

    summary = (
        youtube_dataframe[effective_sentiment_column]
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
    sentiment_column: str = "sentimento_previsto",
) -> pd.DataFrame:
    effective_sentiment_column = _resolve_sentiment_column(youtube_dataframe, sentiment_column)
    if effective_sentiment_column is None or youtube_dataframe.empty:
        return pd.DataFrame(columns=["mes", "sentimento_previsto", "quantidade"])

    monthly_series = youtube_dataframe.copy()
    monthly_series["data_publicacao"] = pd.to_datetime(
        monthly_series["data_publicacao"], errors="coerce"
    )
    monthly_series = monthly_series.dropna(subset=["data_publicacao"]).copy()
    if monthly_series.empty:
        return pd.DataFrame(columns=["mes", "sentimento_previsto", "quantidade"])

    monthly_series["mes"] = monthly_series["data_publicacao"].dt.to_period("M").dt.to_timestamp()
    aggregated = (
        monthly_series.groupby(["mes", effective_sentiment_column])
        .size()
        .reset_index(name="quantidade")
    )
    return aggregated.sort_values(["mes", effective_sentiment_column]).reset_index(drop=True)


def build_google_play_insights(
    unified_dataframe: pd.DataFrame,
    primary_model_dataframe: pd.DataFrame,
    sentiment_column: str = "sentimento_previsto",
) -> dict[str, object]:
    google_play_mask = primary_model_dataframe.get("fonte", pd.Series(dtype=str)).fillna("").eq("google_play")
    google_play_dataframe = primary_model_dataframe.loc[google_play_mask].copy()
    if google_play_dataframe.empty:
        return {
            "total_registros": 0,
            "participacao_base": 0.0,
            "sentiment_summary": pd.DataFrame(columns=["sentimento", "quantidade", "participacao"]),
            "top_theme": "Sem dados",
            "insights": [],
        }

    snapshot = build_primary_model_snapshot(google_play_dataframe, sentiment_column=sentiment_column)
    top_theme_table = build_theme_priority_table(
        google_play_dataframe,
        top_n=3,
        sentiment_column=sentiment_column,
        sentiment_value="Negativo",
    )
    uncertain_column = _resolve_sentiment_column(google_play_dataframe, "predicao_incerta")
    uncertain_share = (
        google_play_dataframe[uncertain_column].fillna(False).astype(bool).mean()
        if uncertain_column is not None and not google_play_dataframe.empty
        else 0
    )
    positive_share = (
        snapshot["sentiment_summary"].set_index("sentimento")["participacao"].get("Positivo", 0.0)
        if not snapshot["sentiment_summary"].empty
        else 0.0
    )
    negative_share = (
        snapshot["sentiment_summary"].set_index("sentimento")["participacao"].get("Negativo", 0.0)
        if not snapshot["sentiment_summary"].empty
        else 0.0
    )
    total_base = len(unified_dataframe) if not unified_dataframe.empty else 0
    base_share = len(google_play_dataframe) / total_base if total_base else 0
    top_theme = top_theme_table.iloc[0]["tema"] if not top_theme_table.empty else "Sem tema predominante"
    top_action = top_theme_table.iloc[0]["acao_sugerida"] if not top_theme_table.empty else "Investigar a jornada mais recorrente."

    insights = [
        f"Google Play concentra {base_share:.1%} da base analisada, então o humor desse canal pesa de forma relevante no agregado.",
        f"No Google Play, a leitura positiva chega a {positive_share:.1%} e a negativa a {negative_share:.1%}, o que mostra se a experiência está sustentando confiança ou friccao.",
    ]
    if top_theme != "Sem tema predominante":
        insights.append(
            f"O tema mais sensível no canal é {top_theme}, com ação sugerida: {top_action.lower()}"
        )
    if uncertain_share > 0:
        insights.append(
            f"{uncertain_share:.1%} das previsões no Google Play ficaram em baixa confiança, então vale tratar os casos cinzentos com amostragem."
        )

    return {
        "total_registros": len(google_play_dataframe),
        "participacao_base": base_share,
        "sentiment_summary": snapshot["sentiment_summary"],
        "top_theme": top_theme,
        "top_action": top_action,
        "insights": insights[:4],
    }


def build_trend_insights(
    primary_model_dataframe: pd.DataFrame,
    sentiment_column: str = "sentimento_previsto",
) -> list[str]:
    effective_sentiment_column = _resolve_sentiment_column(primary_model_dataframe, sentiment_column)
    if effective_sentiment_column is None or primary_model_dataframe.empty:
        return []

    monthly = primary_model_dataframe.copy()
    monthly["data_publicacao"] = pd.to_datetime(monthly["data_publicacao"], errors="coerce")
    monthly = monthly.dropna(subset=["data_publicacao"]).copy()
    if monthly.empty:
        return []

    monthly["mes"] = monthly["data_publicacao"].dt.to_period("M").dt.to_timestamp()
    monthly_mix = (
        monthly.groupby(["mes", effective_sentiment_column])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
        .sort_index()
    )
    monthly_mix["total"] = monthly_mix.sum(axis=1)
    monthly_mix["neg_share"] = monthly_mix["Negativo"] / monthly_mix["total"].replace(0, pd.NA)
    monthly_mix["pos_share"] = monthly_mix["Positivo"] / monthly_mix["total"].replace(0, pd.NA)

    insights: list[str] = []
    if len(monthly_mix) >= 2:
        first_month = monthly_mix.iloc[0]
        last_month = monthly_mix.iloc[-1]
        neg_delta = float(last_month["neg_share"] - first_month["neg_share"])
        pos_delta = float(last_month["pos_share"] - first_month["pos_share"])
        if neg_delta > 0:
            insights.append(
                f"A participação negativa subiu {format_percentage(abs(neg_delta))} entre o primeiro e o último mês observado."
            )
        elif neg_delta < 0:
            insights.append(
                f"A participação negativa caiu {format_percentage(abs(neg_delta))} entre o primeiro e o último mês observado."
            )
        else:
            insights.append("A participação negativa ficou estável entre o primeiro e o último mês observado.")

        if pos_delta > 0:
            insights.append(
                f"A leitura positiva avançou {format_percentage(abs(pos_delta))} na ponta mais recente da série."
            )
        elif pos_delta < 0:
            insights.append(
                f"A leitura positiva perdeu {format_percentage(abs(pos_delta))} na ponta mais recente da série."
            )

    if len(monthly_mix) >= 3:
        first_window = monthly_mix.head(3)
        last_window = monthly_mix.tail(3)
        first_neg_mean = float(first_window["neg_share"].mean())
        last_neg_mean = float(last_window["neg_share"].mean())
        if last_neg_mean > first_neg_mean:
            insights.append(
                f"A média móvel dos últimos 3 meses mostra piora de {format_percentage(last_neg_mean - first_neg_mean)} em pressão negativa."
            )
        elif last_neg_mean < first_neg_mean:
            insights.append(
                f"A média móvel dos últimos 3 meses mostra melhora de {format_percentage(first_neg_mean - last_neg_mean)} em pressão negativa."
            )

    google_play_rows = monthly.loc[monthly["fonte"].fillna("").eq("google_play")].copy()
    if not google_play_rows.empty:
        google_play_monthly = (
            google_play_rows.groupby(["mes", effective_sentiment_column])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=SENTIMENT_ORDER, fill_value=0)
            .sort_index()
        )
        if len(google_play_monthly) >= 2:
            first_gp = google_play_monthly.iloc[0]
            last_gp = google_play_monthly.iloc[-1]
            first_gp_total = float(first_gp.sum())
            last_gp_total = float(last_gp.sum())
            first_gp_neg = float(first_gp.get("Negativo", 0))
            last_gp_neg = float(last_gp.get("Negativo", 0))
            if first_gp_total and last_gp_total:
                gp_neg_delta = (last_gp_neg / last_gp_total) - (first_gp_neg / first_gp_total)
                insights.append(
                    f"No Google Play, a pressão negativa variou {format_percentage(abs(gp_neg_delta))} entre o primeiro e o último mês com dados."
                )

    return insights[:4]


def build_comment_examples(
    youtube_dataframe: pd.DataFrame,
    examples_per_sentiment: int = 3,
    sentiment_column: str = "sentimento_previsto",
) -> pd.DataFrame:
    example_rows = []
    effective_sentiment_column = _resolve_sentiment_column(youtube_dataframe, sentiment_column)
    if effective_sentiment_column is None or youtube_dataframe.empty:
        return pd.DataFrame(columns=["sentimento", "data_publicacao", "usuario", "titulo", "comentario"])

    working_dataframe = youtube_dataframe.copy()
    working_dataframe["data_publicacao"] = pd.to_datetime(
        working_dataframe["data_publicacao"], errors="coerce"
    )

    for sentiment in SENTIMENT_ORDER:
        sentiment_rows = working_dataframe.loc[
            working_dataframe[effective_sentiment_column] == sentiment
        ].copy()
        sentiment_rows = sentiment_rows.sort_values(
            "data_publicacao", ascending=False, na_position="last"
        ).head(examples_per_sentiment)

        if sentiment_rows.empty:
            continue

        if "usuario" not in sentiment_rows.columns:
            sentiment_rows["usuario"] = "Nao informado"
        if "titulo" not in sentiment_rows.columns:
            sentiment_rows["titulo"] = sentiment_rows.get("tema_negocio", "Sem titulo")
        if effective_sentiment_column not in sentiment_rows.columns:
            sentiment_rows[effective_sentiment_column] = sentiment

        example_rows.append(
            sentiment_rows.assign(
                comentario=lambda dataframe: dataframe["texto_original"]
                .fillna("")
                .astype(str)
                .str.slice(0, 220),
                data_publicacao=lambda dataframe: pd.to_datetime(
                    dataframe["data_publicacao"], errors="coerce"
                ).dt.strftime("%d/%m/%Y"),
            )[
                [
                    effective_sentiment_column,
                    "data_publicacao",
                    "usuario",
                    "titulo",
                    "comentario",
                ]
            ]
        )

    if not example_rows:
        return pd.DataFrame(
            columns=["sentimento", "data_publicacao", "usuario", "titulo", "comentario"]
        )

    return pd.concat(example_rows, ignore_index=True).rename(columns={effective_sentiment_column: "sentimento"})


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
    primary_rows = get_primary_model_rows(modelos_dataframe)
    if not primary_rows.empty:
        best_model_index = primary_rows.index[0]
    else:
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
    if modelos_dataframe.empty or "modelo" not in modelos_dataframe:
        return "Ainda nao ha comparacao de modelos disponivel para interpretacao."

    primary_rows = get_primary_model_rows(modelos_dataframe)
    best_model = primary_rows.iloc[0] if not primary_rows.empty else modelos_dataframe.iloc[0]
    benchmark_rows = get_benchmark_rows(modelos_dataframe)
    critical_points = []

    if pd.notna(best_model.get("f1_neutro")) and best_model.get("f1_neutro", 0) < 0.5:
        critical_points.append(
            f"a classe Neutro ainda e fragil, com F1 de {best_model['f1_neutro']:.2%}"
        )
    if pd.isna(best_model.get("roc_auc_macro")):
        critical_points.append("o resumo atual nao traz ROC AUC macro reproduzido para este baseline")
    if not critical_points:
        critical_points.append("o baseline atual mostra leitura global consistente para uso no dashboard")

    benchmark_note = ""
    if not benchmark_rows.empty:
        benchmark_row = benchmark_rows.iloc[0]
        benchmark_note = (
            f" O BERTimbau permanece como benchmark historico do painel "
            f"({benchmark_row['accuracy']:.2%} de accuracy"
            if pd.notna(benchmark_row.get("accuracy"))
            else " O BERTimbau permanece como benchmark historico do painel"
        )
        if pd.notna(benchmark_row.get("roc_auc_macro")):
            benchmark_note += f", {benchmark_row['roc_auc_macro']:.2%} de ROC AUC macro"
        benchmark_note += "), sem reexecucao completa neste fluxo."

    return (
        f"O modelo principal do fluxo atual e {best_model['modelo']}. "
        f"Ele foi promovido para a leitura operacional do dashboard porque e o melhor resultado "
        f"reproduzivel desta execucao. Em senso critico, {critical_points[0]}."
        f"{benchmark_note}"
    )


def build_primary_model_snapshot(
    primary_model_dataframe: pd.DataFrame,
    sentiment_column: str = "sentimento_previsto",
) -> dict[str, object]:
    if primary_model_dataframe.empty:
        return {
            "total_registros": 0,
            "sentiment_summary": pd.DataFrame(columns=["sentimento", "quantidade", "participacao"]),
            "negative_categories": pd.DataFrame(columns=["categoria", "quantidade"]),
            "source_mix": pd.DataFrame(columns=["fonte", "quantidade"]),
        }

    working_dataframe = primary_model_dataframe.copy()
    summary = (
        working_dataframe[sentiment_column]
        .fillna("Sem informacao")
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("sentimento")
        .reset_index(name="quantidade")
    )
    total = summary["quantidade"].sum()
    summary["participacao"] = summary["quantidade"].map(
        lambda value: value / total if total else 0
    )

    negative_dataframe = working_dataframe.loc[
        working_dataframe[sentiment_column] == "Negativo"
    ].copy()
    category_column = _resolve_topic_column(negative_dataframe) or "fonte"
    negative_categories = build_top_n_counts(
        negative_dataframe if not negative_dataframe.empty else working_dataframe,
        category_column,
        top_n=8,
    ).rename(columns={category_column: "categoria"})
    source_mix = build_top_n_counts(working_dataframe, "fonte", top_n=5).rename(
        columns={"fonte": "fonte", "quantidade": "quantidade"}
    )
    source_mix["fonte"] = source_mix["fonte"].map(format_source_label)

    return {
        "total_registros": len(working_dataframe),
        "sentiment_summary": summary,
        "negative_categories": negative_categories,
        "source_mix": source_mix,
    }


def build_theme_priority_table(
    primary_model_dataframe: pd.DataFrame,
    top_n: int = 8,
    sentiment_column: str = "sentimento_previsto",
    sentiment_value: str = "Negativo",
    excluded_themes: set[str] | None = None,
) -> pd.DataFrame:
    empty_frame = pd.DataFrame(
        columns=[
            "tema",
            "quantidade",
            "participacao",
            "canal_dominante",
            "sentimento_dominante",
            "cluster_sinal",
            "acao_sugerida",
            "exemplo",
        ]
    )
    if primary_model_dataframe.empty:
        return empty_frame

    working_dataframe = primary_model_dataframe.copy()
    effective_sentiment_column = _resolve_sentiment_column(working_dataframe, sentiment_column)
    if effective_sentiment_column is None:
        return empty_frame

    has_semantic_clusters = "cluster_label" in working_dataframe.columns or "topico" in working_dataframe.columns
    if has_semantic_clusters:
        cluster_label_column = "cluster_label" if "cluster_label" in working_dataframe.columns else "topico"
        cluster_signal_column = "cluster_signal" if "cluster_signal" in working_dataframe.columns else None
        cluster_example_column = "cluster_example" if "cluster_example" in working_dataframe.columns else None
        cluster_action_column = "cluster_action" if "cluster_action" in working_dataframe.columns else None
        cluster_keywords_column = "cluster_keywords" if "cluster_keywords" in working_dataframe.columns else None

        working_dataframe["tema_prioritario"] = (
            working_dataframe[cluster_label_column]
            .fillna("Sem informacao")
            .astype(str)
            .str.strip()
            .replace({"": "Sem informacao", "nan": "Sem informacao"})
        )
        if sentiment_value in {"Positivo", "Negativo", "Misto"} and cluster_signal_column is not None:
            relevant_dataframe = working_dataframe.loc[
                working_dataframe[cluster_signal_column].fillna("Misto").astype(str).eq(sentiment_value)
            ].copy()
        elif sentiment_value in {"Positivo", "Negativo"}:
            relevant_dataframe = working_dataframe.loc[
                working_dataframe[effective_sentiment_column].eq(sentiment_value)
            ].copy()
        else:
            relevant_dataframe = working_dataframe.copy()
        if relevant_dataframe.empty:
            return empty_frame

        topic_groups = relevant_dataframe.groupby("tema_prioritario")
        theme_counts = topic_groups.size().reset_index(name="quantidade").sort_values("quantidade", ascending=False)
        if excluded_themes:
            theme_counts = theme_counts.loc[~theme_counts["tema_prioritario"].isin(excluded_themes)].copy()
        theme_counts = theme_counts.loc[
            ~theme_counts["tema_prioritario"].isin(["Outros temas", "Sem texto", "Sem informacao"])
        ].copy()
        if theme_counts.empty:
            return empty_frame
        theme_counts = theme_counts.head(top_n)
        total_rows = int(theme_counts["quantidade"].sum()) if not theme_counts.empty else 0
        theme_counts["participacao"] = theme_counts["quantidade"].map(
            lambda value: value / total_rows if total_rows else 0
        )

        dominant_source = (
            relevant_dataframe.groupby("tema_prioritario")["fonte"]
            .agg(lambda series: series.fillna("desconhecido").astype(str).value_counts().index[0])
            .reset_index(name="canal_dominante")
        )
        sentiment_mode = (
            relevant_dataframe.groupby("tema_prioritario")[effective_sentiment_column]
            .agg(lambda series: series.fillna("Sem informacao").astype(str).value_counts().index[0])
            .reset_index(name="sentimento_dominante")
        )
        if cluster_signal_column is not None:
            signal_mode = (
                relevant_dataframe.groupby("tema_prioritario")[cluster_signal_column]
                .agg(lambda series: series.fillna("Misto").astype(str).value_counts().index[0])
                .reset_index(name="cluster_sinal")
            )
        else:
            signal_mode = theme_counts[["tema_prioritario"]].copy()
            signal_mode["cluster_sinal"] = sentiment_value
        if cluster_example_column is not None:
            example_text = (
                relevant_dataframe.groupby("tema_prioritario")[cluster_example_column]
                .agg(
                    lambda series: series.dropna().astype(str).replace("", pd.NA).dropna().head(1).iloc[0]
                    if not series.dropna().astype(str).replace("", pd.NA).dropna().empty
                    else ""
                )
                .reset_index(name="exemplo")
            )
        elif "texto_original" in relevant_dataframe.columns:
            example_text = (
                relevant_dataframe.groupby("tema_prioritario")["texto_original"]
                .agg(
                    lambda series: series.dropna().astype(str).head(1).iloc[0]
                    if not series.dropna().empty
                    else ""
                )
                .reset_index(name="exemplo")
            )
        else:
            example_text = theme_counts[["tema_prioritario"]].copy()
            example_text["exemplo"] = ""
        if cluster_action_column is not None:
            action_text = (
                relevant_dataframe.groupby("tema_prioritario")[cluster_action_column]
                .agg(lambda series: series.dropna().astype(str).head(1).iloc[0] if not series.dropna().empty else "")
                .reset_index(name="acao_sugerida")
            )
        else:
            action_text = theme_counts[["tema_prioritario"]].copy()
            action_text["acao_sugerida"] = action_text["tema_prioritario"].map(_suggest_theme_action)

        if cluster_keywords_column is not None:
            keyword_text = (
                relevant_dataframe.groupby("tema_prioritario")[cluster_keywords_column]
                .agg(lambda series: series.dropna().astype(str).head(1).iloc[0] if not series.dropna().empty else "")
                .reset_index(name="palavras_chave")
            )
        else:
            keyword_text = theme_counts[["tema_prioritario"]].copy()
            keyword_text["palavras_chave"] = ""

        theme_counts = (
            theme_counts.merge(dominant_source, on="tema_prioritario", how="left")
            .merge(sentiment_mode, on="tema_prioritario", how="left")
            .merge(signal_mode, on="tema_prioritario", how="left")
            .merge(action_text, on="tema_prioritario", how="left")
            .merge(example_text, on="tema_prioritario", how="left")
            .merge(keyword_text, on="tema_prioritario", how="left")
        )
        theme_counts = theme_counts.rename(columns={"tema_prioritario": "tema"})
        theme_counts["canal_dominante"] = theme_counts["canal_dominante"].map(format_source_label)
        return theme_counts[
            [
                "tema",
                "quantidade",
                "participacao",
                "canal_dominante",
                "sentimento_dominante",
                "cluster_sinal",
                "acao_sugerida",
                "exemplo",
                "palavras_chave",
            ]
        ]

    if "tema_negocio" in working_dataframe.columns:
        theme_source = working_dataframe["tema_negocio"]
    elif "categoria_problema" in working_dataframe.columns:
        theme_source = working_dataframe["categoria_problema"]
    else:
        theme_source = pd.Series("Sem informacao", index=working_dataframe.index)
    working_dataframe["tema_prioritario"] = theme_source.fillna("Sem informacao").astype(str).str.strip()
    working_dataframe["tema_prioritario"] = working_dataframe["tema_prioritario"].replace(
        {"": "Sem informacao", "nan": "Sem informacao"}
    )
    relevant_dataframe = working_dataframe.loc[
        working_dataframe[effective_sentiment_column] == sentiment_value
    ].copy()

    if relevant_dataframe.empty:
        return empty_frame

    theme_counts = (
        relevant_dataframe.groupby("tema_prioritario")
        .size()
        .reset_index(name="quantidade")
        .sort_values("quantidade", ascending=False)
    )
    meaningful_themes = theme_counts.loc[
        ~theme_counts["tema_prioritario"].isin(["Outros temas", "Sem texto", "Sem informacao"])
    ].copy()
    if excluded_themes:
        meaningful_themes = meaningful_themes.loc[
            ~meaningful_themes["tema_prioritario"].isin(excluded_themes)
        ].copy()
    theme_counts = meaningful_themes if not meaningful_themes.empty else theme_counts
    theme_counts = theme_counts.head(top_n)
    total_count = int(theme_counts["quantidade"].sum()) if not theme_counts.empty else 0
    theme_counts["participacao"] = theme_counts["quantidade"].map(
        lambda value: value / total_count if total_count else 0
    )

    dominant_source = (
        relevant_dataframe.groupby("tema_prioritario")["fonte"]
        .agg(lambda series: series.fillna("desconhecido").astype(str).value_counts().index[0])
        .reset_index(name="canal_dominante")
    )
    sentiment_mode = (
        relevant_dataframe.groupby("tema_prioritario")[effective_sentiment_column]
        .agg(lambda series: series.fillna("Sem informacao").astype(str).value_counts().index[0])
        .reset_index(name="sentimento_dominante")
    )
    if "texto_original" in relevant_dataframe.columns:
        example_text = (
            relevant_dataframe.groupby("tema_prioritario")["texto_original"]
            .agg(
                lambda series: series.dropna().astype(str).head(1).iloc[0]
                if not series.dropna().empty
                else ""
            )
            .reset_index(name="exemplo")
        )
    else:
        example_text = theme_counts[["tema_prioritario"]].copy()
        example_text["exemplo"] = ""

    theme_counts = theme_counts.merge(dominant_source, on="tema_prioritario", how="left")
    theme_counts = theme_counts.merge(sentiment_mode, on="tema_prioritario", how="left")
    theme_counts = theme_counts.merge(example_text, on="tema_prioritario", how="left")
    theme_counts["acao_sugerida"] = theme_counts["tema_prioritario"].map(_suggest_theme_action)
    theme_counts = theme_counts.rename(columns={"tema_prioritario": "tema"})
    theme_counts["canal_dominante"] = theme_counts["canal_dominante"].map(format_source_label)
    theme_counts["cluster_sinal"] = sentiment_value
    theme_counts["palavras_chave"] = ""
    return theme_counts[
        [
            "tema",
            "quantidade",
            "participacao",
            "canal_dominante",
            "sentimento_dominante",
            "cluster_sinal",
            "acao_sugerida",
            "exemplo",
            "palavras_chave",
        ]
    ]


def _suggest_theme_action(theme_name: str) -> str:
    theme = str(theme_name).lower()
    if "bloqueada" in theme or "acesso" in theme or "login" in theme:
        return "Reduzir friccao de acesso e revisar desbloqueio."
    if "emprest" in theme or "credito" in theme:
        return "Ajustar jornada de oferta, limite e comunicacao do credito."
    if "cartao" in theme or "fatura" in theme:
        return "Melhorar gestao de cartao, fatura e suporte a uso recorrente."
    if "pix" in theme or "transfer" in theme:
        return "Reforcar status de transacoes e confiabilidade das transferencias."
    if "cobranca" in theme or "contestacao" in theme:
        return "Acelerar contestacao, estorno e explicacao de cobranças."
    if "atendimento" in theme or "suporte" in theme:
        return "Diminuir tempo de resposta e reforcar resolucao no primeiro contato."
    if "privacidade" in theme or "seguranca" in theme:
        return "Refinar validacao de identidade e comunicacao de seguranca."
    if "app" in theme:
        return "Priorizar estabilidade, velocidade e reducao de erros no aplicativo."
    return "Investigar a causa raiz e confirmar com amostras textuais."


def _resolve_sentiment_column(
    dataframe: pd.DataFrame,
    requested_column: str = "sentimento_previsto",
) -> str | None:
    if requested_column in dataframe.columns:
        return requested_column
    if requested_column == "sentimento_previsto" and "sentimento_previsto_bert" in dataframe.columns:
        return "sentimento_previsto_bert"
    if requested_column == "sentimento_previsto_bert" and "sentimento_previsto" in dataframe.columns:
        return "sentimento_previsto"
    if "sentimento_previsto" in dataframe.columns:
        return "sentimento_previsto"
    if "sentimento_previsto_bert" in dataframe.columns:
        return "sentimento_previsto_bert"
    return None


def _resolve_topic_column(dataframe: pd.DataFrame) -> str | None:
    for candidate in (
        "cluster_label",
        "topico",
        "tema_negocio",
        "categoria_problema",
    ):
        if candidate in dataframe.columns:
            return candidate
    return None


def build_audience_insights(
    unified_dataframe: pd.DataFrame,
    primary_model_dataframe: pd.DataFrame,
    sentiment_column: str = "sentimento_previsto",
) -> dict[str, list[str]]:
    snapshot = build_primary_model_snapshot(primary_model_dataframe, sentiment_column=sentiment_column)
    sentiment_summary = snapshot["sentiment_summary"]
    theme_table = build_theme_priority_table(
        primary_model_dataframe,
        top_n=5,
        sentiment_column=sentiment_column,
    )
    uncertain_column = _resolve_sentiment_column(primary_model_dataframe, "predicao_incerta")
    uncertain_share = (
        primary_model_dataframe[uncertain_column].fillna(False).astype(bool).mean()
        if uncertain_column is not None and not primary_model_dataframe.empty
        else 0
    )
    source_mix = (
        unified_dataframe["fonte"].fillna("desconhecido").value_counts(normalize=True)
        if not unified_dataframe.empty and "fonte" in unified_dataframe.columns
        else pd.Series(dtype=float)
    )

    investor = []
    user = []
    internal = []

    if not sentiment_summary.empty:
        sentiment_indexed = sentiment_summary.set_index("sentimento")
        negative_share = float(sentiment_indexed["participacao"].get("Negativo", 0.0))
        positive_share = float(sentiment_indexed["participacao"].get("Positivo", 0.0))
        investor.append(
            f"O agregado ainda mostra {positive_share:.1%} de leitura positiva, mas {negative_share:.1%} da base já representa risco de reputacao e retenção."
        )

    if not theme_table.empty:
        top_theme = theme_table.iloc[0]
        user.append(
            f"Para o usuario, o tema que mais aparece no negativo é {top_theme['tema']}, normalmente visto no {top_theme['canal_dominante']}."
        )
        internal.append(
            f"Prioridade operacional imediata: {top_theme['acao_sugerida']} O tema lidera com {int(top_theme['quantidade'])} casos negativos."
        )
        if len(theme_table) > 1:
            second_theme = theme_table.iloc[1]
            internal.append(
                f"Depois de {top_theme['tema']}, o segundo foco é {second_theme['tema']}, com concentração em {second_theme['canal_dominante']}."
            )

    if uncertain_share > 0:
        investor.append(
            f"{uncertain_share:.1%} das previsões ficaram em baixa confianca, então a leitura da parte mais ambígua da base precisa ser tratada como indício, não como certeza."
        )
        internal.append(
            f"A zona de baixa confianca continua em {uncertain_share:.1%} das linhas, o que recomenda revisão manual amostral nos casos mais sensíveis."
        )

    if not source_mix.empty:
        dominant_source = format_source_label(str(source_mix.index[0]))
        investor.append(
            f"A composição da base é puxada por {dominant_source}, por isso o resultado final responde mais a esse canal do que aos demais."
        )
        user.append(
            f"Para o cliente, o canal mais presente no volume segue sendo {dominant_source}, o que ajuda a entender onde o usuário mais conversa com a marca."
        )

    if not theme_table.empty:
        if "conta" in theme_table.iloc[0]["tema"].lower() or "acesso" in theme_table.iloc[0]["tema"].lower():
            user.append(
                "Quando o tema dominante é acesso ou conta, o impacto para o cliente tende a ser imediato porque afeta uso diário e continuidade do serviço."
            )

    return {
        "investidor": investor[:3],
        "usuario": user[:3],
        "interno": internal[:4],
    }


def build_business_insights(
    unified_dataframe: pd.DataFrame,
    primary_model_dataframe: pd.DataFrame,
    modelos_dataframe: pd.DataFrame,
    sentiment_column: str = "sentimento_previsto",
) -> list[str]:
    insights = []
    snapshot = build_primary_model_snapshot(primary_model_dataframe, sentiment_column=sentiment_column)
    total_registros = snapshot["total_registros"]
    sentiment_summary = snapshot["sentiment_summary"]
    negative_categories = snapshot["negative_categories"]
    theme_table = build_theme_priority_table(
        primary_model_dataframe,
        top_n=5,
        sentiment_column=sentiment_column,
    )

    negative_share = 0.0
    positive_share = 0.0
    neutral_share = 0.0
    if not sentiment_summary.empty:
        sentiment_indexed = sentiment_summary.set_index("sentimento")
        negative_share = float(sentiment_indexed["participacao"].get("Negativo", 0.0))
        positive_share = float(sentiment_indexed["participacao"].get("Positivo", 0.0))
        neutral_share = float(sentiment_indexed["participacao"].get("Neutro", 0.0))

    if total_registros:
        insights.append(
            f"A leitura geral mostra {positive_share:.1%} de registros positivos e {negative_share:.1%} negativos, "
            "o que aponta uma experiência majoritariamente favorável, porém com bolsões relevantes de atrito."
        )
    if neutral_share > 0:
        insights.append(
            f"A faixa neutra ainda representa {neutral_share:.1%} da leitura, então parte da conversa continua ambígua e exige interpretação cuidadosa."
        )

    if not negative_categories.empty:
        top_negative_category = negative_categories.iloc[0]
        insights.append(
            f"O maior foco de fricção aparece em {top_negative_category['categoria']}, "
            f"com {int(top_negative_category['quantidade'])} ocorrências negativas no recorte previsto."
        )

    if not theme_table.empty:
        top_theme = theme_table.iloc[0]
        insights.append(
            f"O tema de negócio mais sensível é {top_theme['tema']}, com {int(top_theme['quantidade'])} casos negativos e canal dominante {top_theme['canal_dominante']}."
        )

    if not primary_model_dataframe.empty and "fonte" in primary_model_dataframe.columns:
        effective_sentiment_column = _resolve_sentiment_column(primary_model_dataframe, sentiment_column)
        if effective_sentiment_column is not None:
            negative_mask = primary_model_dataframe[effective_sentiment_column] == "Negativo"
        else:
            negative_mask = pd.Series(False, index=primary_model_dataframe.index)
        negative_by_source = (
            primary_model_dataframe.loc[
                negative_mask,
                "fonte",
            ]
            .fillna("desconhecido")
            .value_counts()
        )
        if not negative_by_source.empty:
            top_source = format_source_label(str(negative_by_source.index[0]))
            insights.append(
                f"O canal com maior concentração de sinais negativos é {top_source}, "
                "sugerindo prioridade de resposta nesse ponto de contato."
            )

    primary_rows = get_primary_model_rows(modelos_dataframe)
    if not primary_rows.empty and pd.notna(primary_rows.iloc[0].get("f1_neutro")):
        neutral_f1 = float(primary_rows.iloc[0]["f1_neutro"])
        if neutral_f1 < 0.4:
            insights.append(
                f"A classe Neutro ainda é um ponto cego do modelo principal, com F1 de {neutral_f1:.1%}; "
                "comentarios ambiguos podem estar sendo puxados para polos positivos ou negativos."
            )

    if len(unified_dataframe) and "fonte" in unified_dataframe.columns:
        source_mix = unified_dataframe["fonte"].fillna("desconhecido").value_counts(normalize=True)
        if not source_mix.empty:
            dominant_source = format_source_label(str(source_mix.index[0]))
            insights.append(
                f"A base consolidada é puxada principalmente por {dominant_source}, "
                "então a leitura executiva do painel reflete mais fortemente esse canal."
            )

    if "predicao_incerta" in primary_model_dataframe.columns:
        uncertain_share = primary_model_dataframe["predicao_incerta"].fillna(False).astype(bool).mean()
    elif "predicao_incerta_bert" in primary_model_dataframe.columns:
        uncertain_share = primary_model_dataframe["predicao_incerta_bert"].fillna(False).astype(bool).mean()
    else:
        uncertain_share = 0
    if uncertain_share > 0:
            insights.append(
                f"{uncertain_share:.1%} das previsões ficaram em zona de baixa confiança, "
                "o que ajuda a separar sinal forte de casos mais ambíguos para revisão."
            )

    return insights[:5]


def build_business_storyline(
    primary_model_dataframe: pd.DataFrame,
    sentiment_column: str = "sentimento_previsto",
) -> dict[str, object]:
    if primary_model_dataframe.empty:
        return {
            "headline": "Ainda nao ha base prevista suficiente para montar storytelling de negocio.",
            "subheadline": "Execute a pipeline de previsao para popular esta secao.",
            "business_risks": [],
        }

    working_dataframe = primary_model_dataframe.copy()
    effective_sentiment_column = _resolve_sentiment_column(working_dataframe, sentiment_column)
    if effective_sentiment_column is None:
        return {
            "headline": "Ainda nao ha base prevista suficiente para montar storytelling de negocio.",
            "subheadline": "Execute a pipeline de previsao para popular esta secao.",
            "business_risks": [],
        }

    sentiment_share = (
        working_dataframe[effective_sentiment_column]
        .fillna("Sem informacao")
        .value_counts(normalize=True)
    )
    negative_share = float(sentiment_share.get("Negativo", 0.0))
    positive_share = float(sentiment_share.get("Positivo", 0.0))
    neutral_share = float(sentiment_share.get("Neutro", 0.0))
    uncertain_share = float(
        working_dataframe.get(
            "predicao_incerta",
            working_dataframe.get("predicao_incerta_bert", pd.Series(False, index=working_dataframe.index)),
        )
        .fillna(False)
        .astype(bool)
        .mean()
    )

    negative_by_source = (
        working_dataframe.loc[working_dataframe[effective_sentiment_column] == "Negativo", "fonte"]
        .fillna("desconhecido")
        .value_counts()
    )
    theme_table = build_theme_priority_table(
        working_dataframe,
        top_n=1,
        sentiment_column=sentiment_column,
    )
    top_negative_source = (
        format_source_label(str(negative_by_source.index[0])) if not negative_by_source.empty else "Sem fonte dominante"
    )
    top_theme = theme_table.iloc[0]["tema"] if not theme_table.empty else "tema nao identificado"

    business_risks = [
        (
            f"A conversa ainda favorece o Nubank no agregado, com {positive_share:.1%} de previsões positivas, "
            f"mas {negative_share:.1%} da base já representa risco e um bolsão relevante de insatisfação."
        ),
        (
            f"O principal foco de pressão aparece em {top_negative_source}, com destaque para o tema {top_theme}, "
            "onde o volume negativo pede prioridade de resposta."
        ),
    ]
    if neutral_share > 0:
        business_risks.append(
            f"{neutral_share:.1%} do conteúdo ficou em zona neutra, mostrando uma faixa de experiência sem encantamento claro."
        )
    if uncertain_share > 0:
        business_risks.append(
            f"{uncertain_share:.1%} das previsões ficaram em baixa confianca, o que sinaliza mensagens de alta ambiguidade e pede leitura cuidadosa."
        )

    return {
        "headline": (
            f"A base prevista mostra satisfação majoritária, mas a friccao já é material e o tema {top_theme} se concentra no canal {top_negative_source}."
        ),
        "subheadline": (
            "Esta seção lê o comportamento da base prevista para entender onde a experiência do cliente desliza, "
            "quais temas puxam o negativo e onde a leitura exige mais cautela."
        ),
        "business_risks": business_risks[:4],
    }


def build_sentiment_by_source(primary_model_dataframe: pd.DataFrame) -> pd.DataFrame:
    if primary_model_dataframe.empty:
        return pd.DataFrame(columns=["fonte", "sentimento_previsto", "quantidade", "participacao_fonte"])

    grouped = (
        primary_model_dataframe.groupby(["fonte", "sentimento_previsto"])
        .size()
        .reset_index(name="quantidade")
    )
    grouped["fonte"] = grouped["fonte"].fillna("desconhecido").map(format_source_label)
    total_by_source = grouped.groupby("fonte")["quantidade"].transform("sum")
    grouped["participacao_fonte"] = grouped["quantidade"] / total_by_source
    return grouped


def build_uncertainty_by_source(primary_model_dataframe: pd.DataFrame) -> pd.DataFrame:
    if primary_model_dataframe.empty or "predicao_incerta" not in primary_model_dataframe.columns:
        return pd.DataFrame(columns=["fonte", "participacao_incerta"])

    uncertainty = (
        primary_model_dataframe.assign(
            predicao_incerta=primary_model_dataframe["predicao_incerta"].fillna(False).astype(bool)
        )
        .groupby("fonte")["predicao_incerta"]
        .mean()
        .reset_index(name="participacao_incerta")
    )
    uncertainty["fonte"] = uncertainty["fonte"].fillna("desconhecido").map(format_source_label)
    return uncertainty.sort_values("participacao_incerta", ascending=False)


def get_primary_model_rows(modelos_dataframe: pd.DataFrame) -> pd.DataFrame:
    if "modelo_principal" not in modelos_dataframe.columns:
        return modelos_dataframe.iloc[0:0]
    return modelos_dataframe.loc[modelos_dataframe["modelo_principal"].fillna(False).astype(bool)]


def get_benchmark_rows(modelos_dataframe: pd.DataFrame) -> pd.DataFrame:
    if "proveniencia" not in modelos_dataframe.columns:
        return modelos_dataframe.iloc[0:0]
    return modelos_dataframe.loc[modelos_dataframe["proveniencia"].fillna("").eq("benchmark")]


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
    if dataframe[value_column].max() <= 1:
        axis.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    for container in axis.containers:
        label_format = "%.1f%%" if dataframe[value_column].max() <= 1 else "%.0f"
        axis.bar_label(container, fmt=label_format, padding=3)
    sns.despine(ax=axis)
    st.pyplot(figure, clear_figure=True, use_container_width=True)


def render_insight_list(title: str, insights: list[str]) -> None:
    st.markdown(f"**{title}**")
    if not insights:
        st.info("Ainda nao ha sinais suficientes para montar insights de negocio.")
        return
    for insight in insights:
        st.markdown(f"- {insight}")


def render_audience_cards(audience_insights: dict[str, list[str]]) -> None:
    labels = {
        "investidor": ("Leitura do investidor", "#113b35"),
        "usuario": ("Leitura do usuario", "#d9485f"),
        "interno": ("Leitura interna", "#0f4c81"),
    }
    columns = st.columns(3, gap="large")
    for column, key in zip(columns, labels):
        title, color = labels[key]
        insights = audience_insights.get(key, [])
        with column:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 4px solid {color}; min-height: 220px;">
                    <div class="metric-label">{title}</div>
                    <div class="metric-caption" style="line-height:1.45;">
                        {'<br>'.join(f'• {insight}' for insight in insights) if insights else 'Ainda nao ha sinais suficientes para este recorte.'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_visao_geral(
    unified_dataframe: pd.DataFrame,
    modelos_dataframe: pd.DataFrame,
    primary_model_dataframe: pd.DataFrame,
    selected_date_range: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
    render_section_intro(
        "Visão geral do ecossistema de dados",
        "Panorama consolidado para investidor, usuário e time interno, conectando volume, fontes, temas e risco percebido.",
    )
    total_registros = len(unified_dataframe)
    total_textual = unified_dataframe["texto_original"].fillna("").str.strip().ne("").sum()
    total_google_play = unified_dataframe["fonte"].eq("google_play").sum()
    total_youtube = unified_dataframe["fonte"].eq("youtube").sum()
    total_consumidor_gov = unified_dataframe["fonte"].eq("consumidor_gov").sum()

    metric_columns = st.columns(5)
    with metric_columns[0]:
        render_metric_card(
            "Total de registros",
            format_integer(total_registros),
            "Base completa considerada pelo recorte atual.",
        )
    with metric_columns[1]:
        render_metric_card(
            "Conteúdo textual",
            format_integer(total_textual),
            "Linhas com texto disponível para NLP e exploração.",
        )
    with metric_columns[2]:
        render_metric_card(
            "Google Play",
            format_integer(total_google_play),
            "Avaliações do app que mais pesam na leitura operacional.",
        )
    with metric_columns[3]:
        render_metric_card(
            "YouTube",
            format_integer(total_youtube),
            "Comentários e interações trazidos dos vídeos filtrados.",
        )
    with metric_columns[4]:
        render_metric_card(
            "Consumidor.gov",
            format_integer(total_consumidor_gov),
            "Reclamações estruturadas do canal público.",
        )

    filtered_dates = unified_dataframe["data_publicacao"].dropna()
    if filtered_dates.empty:
        start_date, end_date = selected_date_range
    else:
        start_date, end_date = filtered_dates.min(), filtered_dates.max()

    st.caption(
        "Período coberto pelos dados filtrados: "
        f"{start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}"
    )

    source_volume = build_source_volume_summary(unified_dataframe)
    monthly_volume = build_unified_monthly_volume(unified_dataframe)

    chart_column, trend_column = st.columns((1.15, 1), gap="large")

    with chart_column:
        st.markdown("**Mix de fontes**")
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.pie(
            source_volume["volume"],
            labels=source_volume["fonte"],
            autopct="%1.1f%%",
            startangle=90,
            colors=[
                SOURCE_COLORS.get(label, "#7F8C8D")
                for label in source_volume["fonte"]
            ],
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        axis.set_title("Participacao relativa por fonte")
        st.pyplot(figure, clear_figure=True, use_container_width=True)

    with trend_column:
        st.markdown("**Evolucao mensal da base**")
        if monthly_volume.empty:
            st.info("Não há datas suficientes para construir a serie temporal da base.")
        else:
            figure, axis = plt.subplots(figsize=(8, 4.5))
            sns.lineplot(
                data=monthly_volume,
                x="mes",
                y="quantidade",
                color="#0F766E",
                linewidth=2.6,
                marker="o",
                ax=axis,
            )
            axis.set_xlabel("Mes")
            axis.set_ylabel("Quantidade de registros")
            axis.set_title("Crescimento mensal do volume consolidado")
            axis.tick_params(axis="x", rotation=25)
            sns.despine(ax=axis)
            st.pyplot(figure, clear_figure=True, use_container_width=True)

    st.markdown("**Resumo por fonte**")
    source_table = source_volume.copy()
    source_table["participacao"] = source_table["participacao"].map(format_percentage)
    st.dataframe(source_table, use_container_width=True, hide_index=True)

    render_insight_list(
        "Leitura executiva",
        build_business_insights(
            unified_dataframe,
            primary_model_dataframe,
            modelos_dataframe,
        ),
    )

    audience_insights = build_audience_insights(unified_dataframe, primary_model_dataframe)
    render_audience_cards(audience_insights)

    st.markdown("**Como ler o modelo**")
    st.info(
        build_model_interpretation(modelos_dataframe)
        + " A utilidade prática dele no painel é transformar texto disperso em uma leitura comparável de risco, satisfação e prioridade operacional."
    )

    st.markdown("**Temas de negocio priorizados**")
    theme_table = build_theme_priority_table(primary_model_dataframe, top_n=6)
    if theme_table.empty:
        st.info("Ainda nao ha volume suficiente para priorizar temas de negocio.")
    else:
        theme_table_display = theme_table.copy()
        theme_table_display["participacao"] = theme_table_display["participacao"].map(format_percentage)
        st.dataframe(
            theme_table_display,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Comparacao de modelos**")
    st.dataframe(
        build_model_comparison_table(modelos_dataframe),
        use_container_width=True,
        hide_index=True,
    )


def render_bertimbau_full_section(bert_dataframe: pd.DataFrame) -> None:
    render_section_intro(
        "BERTimbau na base completa",
        "Leitura do modelo neural aplicada a todos os registros textuais da base consolidada, com foco em risco, tema e recorte por canal.",
    )
    if bert_dataframe.empty:
        st.info("Ainda nao ha previsoes do BERTimbau para a base completa.")
        return

    snapshot = build_primary_model_snapshot(
        bert_dataframe,
        sentiment_column="sentimento_previsto_bert",
    )
    theme_table = build_theme_priority_table(
        bert_dataframe,
        top_n=6,
        sentiment_column="sentimento_previsto_bert",
    )
    audience_insights = build_audience_insights(
        bert_dataframe,
        bert_dataframe,
        sentiment_column="sentimento_previsto_bert",
    )
    business_insights = build_business_insights(
        bert_dataframe,
        bert_dataframe,
        pd.DataFrame(),
        sentiment_column="sentimento_previsto_bert",
    )
    youtube_dataframe = bert_dataframe.loc[bert_dataframe["fonte"].fillna("").eq("youtube")].copy()
    youtube_summary = build_sentiment_summary(youtube_dataframe) if not youtube_dataframe.empty else pd.DataFrame(columns=["sentimento", "quantidade", "participacao"])
    youtube_monthly_series = build_monthly_sentiment_series(youtube_dataframe)
    youtube_examples = build_comment_examples(youtube_dataframe)
    youtube_words = build_frequent_words_table(youtube_dataframe)
    source_sentiment = build_sentiment_by_source(
        bert_dataframe.rename(columns={"sentimento_previsto_bert": "sentimento_previsto"})
    )

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card(
            "Registros com BERT",
            format_integer(snapshot["total_registros"]),
            "Total de linhas textuais avaliadas pelo BERTimbau na base inteira.",
        )
    with metric_columns[1]:
        positive_share = (
            snapshot["sentiment_summary"].set_index("sentimento")["participacao"].get("Positivo", 0)
            if not snapshot["sentiment_summary"].empty
            else 0
        )
        render_metric_card(
            "Fatia positiva",
            format_percentage(float(positive_share)) if snapshot["total_registros"] else "0.0%",
            "Volume positivo no agregado; nao confunda com saude integral do negocio.",
        )
    with metric_columns[2]:
        negative_share = (
            snapshot["sentiment_summary"].set_index("sentimento")["participacao"].get("Negativo", 0)
            if not snapshot["sentiment_summary"].empty
            else 0
        )
        render_metric_card(
            "Leitura negativa",
            format_percentage(float(negative_share)) if snapshot["total_registros"] else "0.0%",
            "Parcela que sinaliza dor, friccao e risco de reputacao.",
        )
    with metric_columns[3]:
        top_theme = (
            theme_table.iloc[0]["tema"]
            if not theme_table.empty
            else "Sem tema predominante"
        )
        render_metric_card(
            "Tema predominante",
            str(top_theme),
            "Tema de negocio mais recorrente entre as previsoes negativas do BERT.",
        )

    st.markdown("**Leitura executiva do BERTimbau na base completa**")
    st.caption(
        "A fatia positiva e maior no agregado porque o volume de Google Play domina a base. "
        "A decisao de negocio continua dependendo de onde o negativo se concentra e de como cada canal se comporta."
    )
    render_insight_list(
        "Narrativa por publico",
        business_insights,
    )
    render_audience_cards(audience_insights)

    summary_column, chart_column = st.columns((0.9, 1.4), gap="large")
    with summary_column:
        st.markdown("**Resumo geral do BERT**")
        st.dataframe(snapshot["sentiment_summary"], use_container_width=True, hide_index=True)
    with chart_column:
        st.markdown("**Distribuicao do BERT na base completa**")
        figure, axis = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            data=snapshot["sentiment_summary"],
            x="sentimento",
            y="quantidade",
            order=SENTIMENT_ORDER,
            palette=[SENTIMENT_COLORS[sentiment] for sentiment in SENTIMENT_ORDER],
            ax=axis,
        )
        axis.set_xlabel("Sentimento previsto pelo BERT")
        axis.set_ylabel("Quantidade de registros")
        axis.set_title("BERTimbau aplicado a toda a base textual")
        for container in axis.containers:
            axis.bar_label(container, fmt="%.0f", padding=3)
        sns.despine(ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

    st.markdown("**Sentimento por fonte no BERT**")
    if source_sentiment.empty:
        st.info("Não há dados suficientes para comparar sentimento por fonte.")
    else:
        source_pivot = (
            source_sentiment.pivot_table(
                index="fonte",
                columns="sentimento_previsto",
                values="participacao_fonte",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(columns=SENTIMENT_ORDER, fill_value=0)
            .reset_index()
        )
        source_order = (
            source_sentiment.groupby("fonte")["quantidade"].sum().sort_values(ascending=False).index.tolist()
        )
        source_pivot["fonte"] = pd.Categorical(
            source_pivot["fonte"],
            categories=source_order,
            ordered=True,
        )
        source_pivot = source_pivot.sort_values("fonte")
        figure, axis = plt.subplots(figsize=(9.5, 4.5))
        bottom = np.zeros(len(source_pivot))
        for sentiment in SENTIMENT_ORDER:
            values = source_pivot[sentiment].fillna(0).to_numpy()
            axis.bar(
                source_pivot["fonte"].astype(str),
                values,
                bottom=bottom,
                label=sentiment,
                color=SENTIMENT_COLORS[sentiment],
            )
            bottom = bottom + values
        axis.set_ylabel("Participacao dentro de cada fonte")
        axis.set_xlabel("Fonte")
        axis.set_ylim(0, 1)
        axis.set_title("Mix de sentimento por fonte na base completa")
        axis.legend(title="Sentimento")
        axis.tick_params(axis="x", rotation=20)
        sns.despine(ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

    st.markdown("**Temas prioritarios do BERTimbau**")
    if theme_table.empty:
        st.info("Não foi possivel localizar temas relevantes no recorte atual.")
    else:
        theme_display = theme_table.copy()
        theme_display["participacao"] = theme_display["participacao"].map(format_percentage)
        st.dataframe(theme_display, use_container_width=True, hide_index=True)

    st.markdown("**Recorte do YouTube dentro da base completa**")
    if youtube_dataframe.empty:
        st.info("Não há registros de YouTube nesta base completa.")
    else:
        youtube_metric_columns = st.columns(3)
        with youtube_metric_columns[0]:
            render_metric_card(
                "YouTube avaliados",
                format_integer(len(youtube_dataframe)),
                "Comentários e relatos do canal aberto do ecossistema.",
            )
        with youtube_metric_columns[1]:
            render_metric_card(
                "Positivo no YouTube",
                youtube_summary.set_index("sentimento")["participacao"].get("Positivo", "0.0%"),
                "Participação favorável no recorte de YouTube.",
            )
        with youtube_metric_columns[2]:
            render_metric_card(
                "Negativo no YouTube",
                youtube_summary.set_index("sentimento")["participacao"].get("Negativo", "0.0%"),
                "Participação negativa no recorte de YouTube.",
            )

        summary_column, chart_column = st.columns((0.9, 1.4), gap="large")
        with summary_column:
            st.markdown("**Resumo do YouTube**")
            st.dataframe(youtube_summary, use_container_width=True, hide_index=True)
        with chart_column:
            st.markdown("**Distribuicao mensal no YouTube**")
            if youtube_monthly_series.empty:
                st.info("Não há dados suficientes para a serie temporal do YouTube.")
            else:
                figure, axis = plt.subplots(figsize=(10, 4.5))
                sns.lineplot(
                    data=youtube_monthly_series,
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
                axis.set_title("YouTube dentro da base completa")
                axis.tick_params(axis="x", rotation=25)
                sns.despine(ax=axis)
                st.pyplot(figure, clear_figure=True, use_container_width=True)

        examples_column, words_column = st.columns((1.4, 1), gap="large")
        with examples_column:
            st.markdown("**Exemplos do YouTube**")
            st.dataframe(youtube_examples, use_container_width=True, hide_index=True)
        with words_column:
            st.markdown("**Palavras mais frequentes no YouTube**")
            st.dataframe(youtube_words, use_container_width=True, hide_index=True)


def render_youtube_section(youtube_dataframe: pd.DataFrame) -> None:
    render_section_intro(
        "YouTube + BERTimbau",
        "Leitura do que os usuarios falam em escala: sentimento, temas, cautela da previsao e sinais de friccao ou satisfacao.",
    )
    sentiment_summary = build_sentiment_summary(youtube_dataframe)
    total_comments = len(youtube_dataframe)
    positive_share = (
        sentiment_summary.loc[sentiment_summary["sentimento"] == "Positivo", "quantidade"].sum()
        / total_comments
        if total_comments
        else 0
    )
    negative_share = (
        sentiment_summary.loc[sentiment_summary["sentimento"] == "Negativo", "quantidade"].sum()
        / total_comments
        if total_comments
        else 0
    )
    neutral_share = (
        sentiment_summary.loc[sentiment_summary["sentimento"] == "Neutro", "quantidade"].sum()
        / total_comments
        if total_comments
        else 0
    )

    monthly_sentiment_series = build_monthly_sentiment_series(youtube_dataframe)
    comment_examples = build_comment_examples(youtube_dataframe)
    frequent_words = build_frequent_words_table(youtube_dataframe)

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card(
            "Comentarios analisados",
            format_integer(total_comments),
            "Volume exibido apos filtros globais e do YouTube.",
        )
    with metric_columns[1]:
        render_metric_card(
            "Participacao positiva",
            format_percentage(positive_share),
            "Parcela dos comentarios com sinal favoravel.",
        )
    with metric_columns[2]:
        render_metric_card(
            "Participacao negativa",
            format_percentage(negative_share),
            "Parcela com maior potencial de friccao.",
        )
    with metric_columns[3]:
        render_metric_card(
            "Participacao neutra",
            format_percentage(neutral_share),
            "Comentarios informativos ou pouco polarizados.",
        )

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
        st.info("Não há dados suficientes para montar a serie temporal do YouTube.")
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
    render_section_intro(
        "Consumidor.gov",
        "Painel das reclamacoes formais, traduzindo dados estruturados em prioridades de atuacao para experiencia e operacao.",
    )
    notas_series = pd.to_numeric(consumidor_dataframe["nota"], errors="coerce").dropna()
    mean_score = notas_series.mean() if not notas_series.empty else 0
    solved_share = (
        consumidor_dataframe["status_reclamacao"]
        .fillna("")
        .astype(str)
        .str.contains("Resolvida|Finalizada", case=False, regex=True)
        .mean()
        if "status_reclamacao" in consumidor_dataframe
        else 0
    )
    monthly_volume = build_monthly_volume(consumidor_dataframe)
    negative_like_share = (
        notas_series.le(2).mean()
        if not notas_series.empty
        else 0
    )

    top_categories = build_top_n_counts(
        consumidor_dataframe,
        "categoria_problema",
        top_n=8,
    )
    top_problems = build_top_n_counts(
        consumidor_dataframe,
        "texto_original",
        top_n=8,
        fallback_label="Sem contexto textual",
    )
    status_distribution = build_top_n_counts(
        consumidor_dataframe,
        "status_reclamacao",
        top_n=10,
    )
    top_ufs = build_top_n_counts(consumidor_dataframe, "uf", top_n=10)
    storyline = [
        (
            f"O Consumidor.gov traz {format_percentage(negative_like_share)} de notas ate 2, "
            "sinalizando atrito mais forte e formal que os canais de comentario aberto."
        )
        if not notas_series.empty
        else "O recorte atual nao traz notas suficientes para medir intensidade do atrito formal."
    ]
    if not top_categories.empty:
        storyline.append(
            f"A categoria mais recorrente e {top_categories.iloc[0]['categoria_problema']}, "
            "o que ajuda a priorizar a dor operacional dominante."
        )
    if not status_distribution.empty:
        storyline.append(
            f"O status mais frequente e {status_distribution.iloc[0]['status_reclamacao']}, "
            "mostrando como o encerramento das reclamacoes aparece para o cliente."
        )

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card(
            "Total de registros",
            format_integer(len(consumidor_dataframe)),
            "Volume estrutural do Consumidor.gov no recorte aplicado.",
        )
    with metric_columns[1]:
        render_metric_card(
            "Nota media",
            f"{mean_score:.2f}" if mean_score else "-",
            "Media das notas validas informadas pelos consumidores.",
        )
    with metric_columns[2]:
        render_metric_card(
            "Status resolvidos",
            format_percentage(solved_share) if solved_share else "0.0%",
            "Proporcao de registros com fechamento favoravel no status.",
        )
    with metric_columns[3]:
        render_metric_card(
            "Notas ate 2",
            format_percentage(negative_like_share) if not notas_series.empty else "0.0%",
            "Proxy de insatisfacao mais intensa nas reclamacoes formais.",
        )

    render_insight_list(
        "Leitura executiva do Consumidor.gov",
        storyline,
    )

    category_column, problem_column = st.columns(2, gap="large")

    with category_column:
        st.markdown("**Principais categorias**")
        render_horizontal_bar_chart(
            top_categories,
            "categoria_problema",
            "quantidade",
            "Categorias com mais reclamacoes",
            "#1f77b4",
            "Quantidade de registros",
        )

    with problem_column:
        st.markdown("**Principais relatos**")
        render_horizontal_bar_chart(
            top_problems,
            "texto_original",
            "quantidade",
            "Relatos textuais mais recorrentes",
            "#d35400",
            "Quantidade de registros",
        )

    status_column, notes_column = st.columns((1.1, 1), gap="large")

    with status_column:
        st.markdown("**Distribuicao de status**")
        figure, axis = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            data=status_distribution,
            x="status_reclamacao",
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
            st.info("Não há notas validas para exibir.")
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
            st.info("Não há datas validas para a serie temporal do Consumidor.gov.")
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
            "uf",
            "quantidade",
            "Estados com maior volume",
            "#7D3C98",
            "Quantidade de registros",
        )


def render_modelos_section(
    modelos_dataframe: pd.DataFrame,
    primary_model_dataframe: pd.DataFrame,
) -> None:
    render_section_intro(
        "O que melhorar",
        "Storytelling da base prevista: o que a base diz para mercado, cliente e operação, com foco em prioridade e risco.",
    )
    snapshot = build_primary_model_snapshot(primary_model_dataframe)
    story = build_business_storyline(primary_model_dataframe)
    interpretation = build_model_interpretation(modelos_dataframe)
    primary_rows = get_primary_model_rows(modelos_dataframe)
    sentiment_by_source = build_sentiment_by_source(primary_model_dataframe)
    uncertainty_by_source = build_uncertainty_by_source(primary_model_dataframe)

    metric_columns = st.columns(4)
    with metric_columns[0]:
        positive_share = (
            snapshot["sentiment_summary"].set_index("sentimento")["participacao"].get("Positivo", 0)
            if not snapshot["sentiment_summary"].empty
            else 0
        )
        render_metric_card(
            "Sinal favoravel",
            format_percentage(float(positive_share)) if snapshot["total_registros"] else "0.0%",
            "Parcela da base prevista com leitura favoravel ao relacionamento.",
        )
    with metric_columns[1]:
        negative_share = (
            snapshot["sentiment_summary"].set_index("sentimento")["participacao"].get("Negativo", 0)
            if not snapshot["sentiment_summary"].empty
            else 0
        )
        render_metric_card(
            "Sinal de atrito",
            format_percentage(float(negative_share)) if snapshot["total_registros"] else "0.0%",
            "Volume previsto de experiencia negativa que pede acao.",
        )
    with metric_columns[2]:
        top_theme_table = build_theme_priority_table(
            primary_model_dataframe,
            top_n=1,
            sentiment_value="Negativo",
        )
        top_negative_category = (
            top_theme_table.iloc[0]["tema"]
            if not top_theme_table.empty
            else (
                snapshot["negative_categories"].iloc[0]["categoria"]
                if not snapshot["negative_categories"].empty
                else "-"
            )
        )
        render_metric_card(
            "Tema critico",
            str(top_negative_category),
            "Assunto que mais concentra previsoes negativas no recorte atual.",
        )
    with metric_columns[3]:
        uncertain_share = (
            primary_model_dataframe["predicao_incerta"].fillna(False).astype(bool).mean()
            if "predicao_incerta" in primary_model_dataframe.columns and not primary_model_dataframe.empty
            else 0
        )
        render_metric_card(
            "Zona cinzenta",
            format_percentage(float(uncertain_share)) if snapshot["total_registros"] else "0.0%",
            "Faixa de previsoes em que a mensagem do cliente aparece mais ambigua.",
        )

    st.markdown(f"### {story['headline']}")
    st.caption(story["subheadline"])
    st.caption(
        "A leitura técnica dos modelos segue disponível ao final como apoio metodológico, mas a decisão começa pelos sinais de negócio."
    )
    render_insight_list(
        "Leitura executiva do recorte",
        story["business_risks"],
    )

    audience_insights = build_audience_insights(primary_model_dataframe, primary_model_dataframe)
    render_audience_cards(audience_insights)

    composition_column, uncertainty_column = st.columns((1.2, 1), gap="large")

    with composition_column:
        st.markdown("**Mix de sentimento por canal**")
        if sentiment_by_source.empty:
            st.info("Não há base suficiente para comparar o mix de sentimento por canal.")
        else:
            figure, axis = plt.subplots(figsize=(9, 4.8))
            sns.barplot(
                data=sentiment_by_source,
                x="fonte",
                y="participacao_fonte",
                hue="sentimento_previsto",
                hue_order=SENTIMENT_ORDER,
                palette=SENTIMENT_COLORS,
                ax=axis,
            )
            axis.set_xlabel("Canal")
            axis.set_ylabel("Participacao dentro do canal")
            axis.set_title("Cada canal conta uma historia diferente de satisfacao e friccao")
            axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            axis.legend(title="Sentimento", frameon=False)
            sns.despine(ax=axis)
            st.pyplot(figure, clear_figure=True, use_container_width=True)

    with uncertainty_column:
        st.markdown("**Onde a leitura pede mais cautela**")
        if uncertainty_by_source.empty:
            st.info("A base atual nao traz marcacao de incerteza por previsao.")
        else:
            render_horizontal_bar_chart(
                uncertainty_by_source.sort_values("participacao_incerta", ascending=True),
                "fonte",
                "participacao_incerta",
                "Participacao de previsoes incertas por canal",
                "#7F8C8D",
                "Participacao incerta",
            )

    pain_column, preview_column = st.columns((1.15, 1), gap="large")

    with pain_column:
        render_insight_list(
            "Insights de negocio orientados pela base prevista",
            build_business_insights(primary_model_dataframe, primary_model_dataframe, modelos_dataframe),
        )
        st.markdown("**Temas que pedem refinamento**")
        theme_table = build_theme_priority_table(
            primary_model_dataframe,
            top_n=8,
            sentiment_value="Misto",
        )
        if theme_table.empty:
            st.info("Nao ha volume suficiente para localizar temas mistos relevantes.")
        else:
            render_horizontal_bar_chart(
                theme_table.sort_values("quantidade", ascending=True),
                "tema",
                "quantidade",
                "Temas que pedem mais refinamento na jornada",
                "#C0392B",
                "Quantidade de registros",
            )

    with preview_column:
        st.markdown("**Exemplos recentes para leitura contextual**")
        preview_columns = [
            column_name
            for column_name in [
                "data_publicacao",
                "fonte",
                "categoria_problema",
                "texto_original",
                "sentimento_previsto",
                "confianca_modelo",
            ]
            if column_name in primary_model_dataframe.columns
        ]
        preview_dataframe = primary_model_dataframe.sort_values(
            ["sentimento_previsto", "confianca_modelo"] if "confianca_modelo" in primary_model_dataframe.columns else ["sentimento_previsto"],
            ascending=[True, True] if "confianca_modelo" in primary_model_dataframe.columns else [True],
        )[preview_columns].head(15).copy()
        if "data_publicacao" in preview_dataframe:
            preview_dataframe["data_publicacao"] = pd.to_datetime(
                preview_dataframe["data_publicacao"], errors="coerce"
            ).dt.strftime("%d/%m/%Y")
        if "confianca_modelo" in preview_dataframe.columns:
            preview_dataframe["confianca_modelo"] = preview_dataframe["confianca_modelo"].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "-"
            )
        st.dataframe(preview_dataframe, use_container_width=True, hide_index=True)

    with st.expander("Anexo técnico: comparação de modelos", expanded=False):
        st.caption(
            "Esta tabela fica em segundo plano. Ela serve para dar transparência metodológica, "
            "mas a leitura principal do painel parte da base prevista e de suas implicações de negócio."
        )
        st.dataframe(
            style_model_comparison_table(modelos_dataframe),
            use_container_width=True,
            hide_index=True,
        )
        if interpretation:
            st.write(interpretation)


def render_positive_section(primary_model_dataframe: pd.DataFrame) -> None:
    render_section_intro(
        "O que está bom",
        "Onde a experiência já aparece mais sólida, o que está funcionando melhor e quais temas sustentam a percepção positiva.",
    )
    if primary_model_dataframe.empty:
        st.info("Ainda não há base prevista suficiente para destacar sinais positivos.")
        return

    positive_rows = primary_model_dataframe.loc[
        primary_model_dataframe["sentimento_previsto"].eq("Positivo")
    ].copy()
    positive_summary = build_sentiment_summary(positive_rows)
    positive_themes = build_theme_priority_table(
        primary_model_dataframe,
        top_n=6,
        sentiment_column="sentimento_previsto",
        sentiment_value="Positivo",
    )
    source_mix = build_sentiment_by_source(primary_model_dataframe)
    positive_source_mix = source_mix.loc[source_mix["sentimento_previsto"].eq("Positivo")].copy()

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card(
            "Registros positivos",
            format_integer(len(positive_rows)),
            "Base prevista que já aparece com percepção favorável.",
        )
    with metric_columns[1]:
        positive_share = (len(positive_rows) / len(primary_model_dataframe)) if len(primary_model_dataframe) else 0
        render_metric_card(
            "Participação positiva",
            format_percentage(float(positive_share)) if len(positive_rows) else "0.0%",
            "Parcela do recorte que ajuda a sustentar a experiência da marca.",
        )
    with metric_columns[2]:
        top_theme = positive_themes.iloc[0]["tema"] if not positive_themes.empty else "Sem tema predominante"
        render_metric_card(
            "Tema forte",
            str(top_theme),
            "Tema que mais aparece entre os sinais positivos.",
        )
    with metric_columns[3]:
        top_source = (
            format_source_label(str(positive_source_mix.sort_values("participacao_fonte", ascending=False).iloc[0]["fonte"]))
            if not positive_source_mix.empty
            else "Sem canal dominante"
        )
        render_metric_card(
            "Canal forte",
            top_source,
            "Canal que mais concentra a leitura favorável.",
        )

    st.markdown("**Resumo do que está bom**")
    render_insight_list(
        "Sinais positivos",
        [
            "A base ainda mostra predominância favorável quando olhamos o agregado.",
            "Os sinais positivos ajudam a sustentar confiança, engajamento e retenção.",
            "Os temas fortes indicam onde a experiência já funciona bem e merece ser mantida.",
        ],
    )

    if positive_source_mix.empty:
        st.info("Não há canal suficiente para detalhar o que está funcionando melhor.")
    else:
        figure, axis = plt.subplots(figsize=(9, 4.8))
        sns.barplot(
            data=positive_source_mix.sort_values("participacao_fonte", ascending=False),
            x="fonte",
            y="participacao_fonte",
            color=SENTIMENT_COLORS["Positivo"],
            ax=axis,
        )
        axis.set_xlabel("Canal")
        axis.set_ylabel("Participação positiva")
        axis.set_title("Onde a percepção positiva se concentra")
        axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        sns.despine(ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

    st.markdown("**Temas que sustentam o bom resultado**")
    if positive_themes.empty:
        st.info("Não foi possivel localizar temas positivos relevantes no recorte atual.")
    else:
        positive_themes_display = positive_themes.copy()
        positive_themes_display["participacao"] = positive_themes_display["participacao"].map(format_percentage)
        st.dataframe(positive_themes_display, use_container_width=True, hide_index=True)


def render_negative_section(primary_model_dataframe: pd.DataFrame) -> None:
    render_section_intro(
        "O que está ruim",
        "Onde a experiência degrada, quais são os temas mais críticos e em quais canais o atrito ganha força.",
    )
    if primary_model_dataframe.empty:
        st.info("Ainda não há base prevista suficiente para destacar os pontos de dor.")
        return

    negative_rows = primary_model_dataframe.loc[
        primary_model_dataframe["sentimento_previsto"].eq("Negativo")
    ].copy()
    negative_summary = build_sentiment_summary(negative_rows)
    negative_themes = build_theme_priority_table(
        primary_model_dataframe,
        top_n=6,
        sentiment_value="Negativo",
    )
    source_mix = build_sentiment_by_source(primary_model_dataframe)
    negative_source_mix = source_mix.loc[source_mix["sentimento_previsto"].eq("Negativo")].copy()

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card(
            "Registros negativos",
            format_integer(len(negative_rows)),
            "Base prevista que já pede resposta e contenção.",
        )
    with metric_columns[1]:
        negative_share = (len(negative_rows) / len(primary_model_dataframe)) if len(primary_model_dataframe) else 0
        render_metric_card(
            "Participação negativa",
            format_percentage(float(negative_share)) if len(negative_rows) else "0.0%",
            "Parcela do recorte que sinaliza fricção e risco reputacional.",
        )
    with metric_columns[2]:
        top_theme = negative_themes.iloc[0]["tema"] if not negative_themes.empty else "Sem tema predominante"
        render_metric_card(
            "Tema crítico",
            str(top_theme),
            "Tema que mais aparece entre os sinais negativos.",
        )
    with metric_columns[3]:
        top_source = (
            format_source_label(str(negative_source_mix.sort_values("participacao_fonte", ascending=False).iloc[0]["fonte"]))
            if not negative_source_mix.empty
            else "Sem canal dominante"
        )
        render_metric_card(
            "Canal crítico",
            top_source,
            "Canal com maior concentração de atrito.",
        )

    render_insight_list(
        "Leitura de risco",
        build_business_insights(primary_model_dataframe, primary_model_dataframe, pd.DataFrame()),
    )

    if negative_source_mix.empty:
        st.info("Não há canal suficiente para detalhar o que está ruim.")
    else:
        figure, axis = plt.subplots(figsize=(9, 4.8))
        sns.barplot(
            data=negative_source_mix.sort_values("participacao_fonte", ascending=False),
            x="fonte",
            y="participacao_fonte",
            color=SENTIMENT_COLORS["Negativo"],
            ax=axis,
        )
        axis.set_xlabel("Canal")
        axis.set_ylabel("Participação negativa")
        axis.set_title("Onde a percepção negativa se concentra")
        axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        sns.despine(ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

    st.markdown("**Temas que mais machucam a experiência**")
    if negative_themes.empty:
        st.info("Não foi possivel localizar temas negativos relevantes no recorte atual.")
    else:
        negative_themes_display = negative_themes.copy()
        negative_themes_display["participacao"] = negative_themes_display["participacao"].map(format_percentage)
        st.dataframe(negative_themes_display, use_container_width=True, hide_index=True)


def render_tendency_section(primary_model_dataframe: pd.DataFrame) -> None:
    render_section_intro(
        "Tendência",
        "Como a percepção evolui ao longo do tempo e se a base está melhorando, piorando ou ficando estável.",
    )
    if primary_model_dataframe.empty:
        st.info("Ainda não há base prevista suficiente para analisar tendência.")
        return

    monthly_series = build_monthly_sentiment_series(primary_model_dataframe)
    if monthly_series.empty:
        st.info("Não há datas suficientes para construir a tendência temporal.")
        return
    trend_insights = build_trend_insights(primary_model_dataframe)

    trend_summary = (
        monthly_series.groupby("sentimento_previsto")["quantidade"].sum().reset_index(name="quantidade_total")
    )
    total_records = int(trend_summary["quantidade_total"].sum()) if not trend_summary.empty else 0
    negative_total = int(trend_summary.loc[trend_summary["sentimento_previsto"].eq("Negativo"), "quantidade_total"].sum())
    positive_total = int(trend_summary.loc[trend_summary["sentimento_previsto"].eq("Positivo"), "quantidade_total"].sum())
    neutral_total = int(trend_summary.loc[trend_summary["sentimento_previsto"].eq("Neutro"), "quantidade_total"].sum())

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card("Total analisado", format_integer(total_records), "Volume com data suficiente para série temporal.")
    with metric_columns[1]:
        render_metric_card("Positivos", format_integer(positive_total), "Acumulado positivo na janela observada.")
    with metric_columns[2]:
        render_metric_card("Negativos", format_integer(negative_total), "Acumulado negativo na janela observada.")
    with metric_columns[3]:
        render_metric_card("Neutros", format_integer(neutral_total), "Acumulado neutro na janela observada.")

    st.markdown("**Evolução mensal do sentimento**")
    figure, axis = plt.subplots(figsize=(10, 4.8))
    sns.lineplot(
        data=monthly_series,
        x="mes",
        y="quantidade",
        hue="sentimento_previsto",
        hue_order=SENTIMENT_ORDER,
        palette=SENTIMENT_COLORS,
        marker="o",
        ax=axis,
    )
    axis.set_xlabel("Mês")
    axis.set_ylabel("Quantidade de registros")
    axis.set_title("Tendência de sentimento ao longo do tempo")
    axis.tick_params(axis="x", rotation=25)
    sns.despine(ax=axis)
    st.pyplot(figure, clear_figure=True, use_container_width=True)

    render_insight_list(
        "Leitura executiva da tendência",
        trend_insights
        or [
            "A base ainda não tem um histórico longo o suficiente para estimar direção com confiança.",
        ],
    )

    if len(monthly_series["mes"].dropna().unique()) >= 2:
        ordered_months = sorted(monthly_series["mes"].dropna().unique())
        first_month = monthly_series.loc[monthly_series["mes"].eq(ordered_months[0])]
        last_month = monthly_series.loc[monthly_series["mes"].eq(ordered_months[-1])]
        first_negative = first_month.loc[first_month["sentimento_previsto"].eq("Negativo"), "quantidade"].sum()
        first_total = first_month["quantidade"].sum()
        last_negative = last_month.loc[last_month["sentimento_previsto"].eq("Negativo"), "quantidade"].sum()
        last_total = last_month["quantidade"].sum()
        first_share = (first_negative / first_total) if first_total else 0
        last_share = (last_negative / last_total) if last_total else 0
        trend_delta = last_share - first_share
        if trend_delta > 0:
            st.warning(
                f"A participação negativa subiu {format_percentage(abs(trend_delta))} entre o primeiro e o último mês da série."
            )
        elif trend_delta < 0:
            st.success(
                f"A participação negativa caiu {format_percentage(abs(trend_delta))} entre o primeiro e o último mês da série."
            )
        else:
            st.info("A participação negativa ficou estável entre o primeiro e o último mês observados.")


def render_home(
    filtered_datasets: dict[str, pd.DataFrame],
    filtered_youtube_dataframe: pd.DataFrame,
    selected_section: str,
    selected_date_range: tuple[pd.Timestamp, pd.Timestamp],
    selected_youtube_sentiment: str,
) -> None:
    start_date, end_date = selected_date_range
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-eyebrow">Monitor de percepção do cliente</div>
            <h1 class="hero-title">Dashboard de Sentimentos do Nubank</h1>
            <div class="hero-subtitle">
                Uma leitura consolidada para entender o que clientes, usuários e reclamantes estão dizendo,
                onde estão os temas que importam e o que isso sinaliza para receita, reputação e operação.
            </div>
            <div class="hero-meta">
                <div class="filter-pill">Seção: {selected_section}</div>
                <div class="filter-pill">Período: {get_date_span_label(start_date, end_date)}</div>
                <div class="filter-pill">Filtro YouTube: {selected_youtube_sentiment}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_rows = pd.DataFrame(
        [
        {
            "base": "Base unificada",
            "registros_filtrados": len(filtered_datasets["Base unificada"]),
            "registros_exibidos_na_secao": len(filtered_datasets["Base unificada"]),
        },
        {
            "base": "BERTimbau na base completa",
            "registros_filtrados": len(filtered_datasets["BERTimbau na base completa"]),
            "registros_exibidos_na_secao": len(filtered_datasets["BERTimbau na base completa"]),
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
        {
            "base": "Modelo principal",
            "registros_filtrados": len(filtered_datasets["Modelo principal"]),
            "registros_exibidos_na_secao": len(filtered_datasets["Modelo principal"]),
        },
        ]
    )

    if selected_section == "Visão Geral":
        render_section_intro(
            "Status das bases carregadas",
            "Comparativo rápido entre o volume filtrado no dataset e o volume efetivamente mostrado na seção ativa.",
        )
        st.dataframe(status_rows, use_container_width=True, hide_index=True)
        render_visao_geral(
            filtered_datasets["Base unificada"],
            filtered_datasets["Resumo de modelos"],
            filtered_datasets["Modelo principal"],
            selected_date_range,
        )
    elif selected_section == "O que está bom":
        render_positive_section(filtered_datasets["Modelo principal"])
    elif selected_section == "O que está ruim":
        render_negative_section(filtered_datasets["Modelo principal"])
    elif selected_section == "O que melhorar":
        render_modelos_section(
            filtered_datasets["Resumo de modelos"],
            filtered_datasets["Modelo principal"],
        )
    elif selected_section == "Tendência":
        render_tendency_section(filtered_datasets["Modelo principal"])


def main() -> None:
    try:
        apply_custom_theme()
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

