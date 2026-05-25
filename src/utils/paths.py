from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
SANDBOX_DIR = DATA_DIR / "sandbox"
NOTEBOOK_DATA_DIR = SANDBOX_DIR / "notebooks"

BRONZE_UNIFIED_DIR = BRONZE_DIR / "unified"
SILVER_PREPROCESSING_DIR = SILVER_DIR / "preprocessing"
SILVER_UNIFIED_DIR = SILVER_DIR / "unified"
SILVER_SENTIMENT_DIR = SILVER_DIR / "sentiment"
GOLD_ANALYTICS_DIR = GOLD_DIR / "analytics"
GOLD_TOPICS_DIR = GOLD_DIR / "topics"
GOLD_DASHBOARD_DIR = GOLD_DIR / "dashboard"
GOLD_SQLSERVER_DIR = GOLD_DIR / "sqlserver"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
NOTEBOOK_FIGURES_DIR = FIGURES_DIR / "notebooks"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
NOTEBOOK_REPORTS_DIR = REPORTS_DIR / "notebooks"
SQL_DIR = PROJECT_ROOT / "sql"
DOCS_DIR = PROJECT_ROOT / "docs"

BRONZE_UNIFIED_DATASET_PATH = BRONZE_UNIFIED_DIR / "dados_unificados_bronze.csv"
SILVER_UNIFIED_DATASET_PATH = SILVER_UNIFIED_DIR / "dados_unificados_silver.csv"
SILVER_TEXTUAL_DATASET_PATH = (
    SILVER_PREPROCESSING_DIR / "textual_dataset_preprocessed.csv"
)
GOLD_ANALYTICS_SENTIMENT_PATH = GOLD_ANALYTICS_DIR / "gold_sentiment_analysis.csv"
GOLD_TOPICS_ANALYSIS_PATH = GOLD_TOPICS_DIR / "gold_topic_analysis.csv"
GOLD_DASHBOARD_UNIFIED_DATASET_PATH = GOLD_DASHBOARD_DIR / "unified_dataset.csv"
GOLD_DASHBOARD_MODEL_COMPARISON_PATH = GOLD_DASHBOARD_DIR / "model_comparison_summary.csv"
GOLD_DASHBOARD_YOUTUBE_BERT_DATASET_PATH = (
    GOLD_DASHBOARD_DIR / "youtube_with_predicted_sentiment_bertimbau.csv"
)
GOLD_DASHBOARD_CLASSIC_SENTIMENT_PATH = (
    GOLD_DASHBOARD_DIR / "dados_com_sentimento_previsto.csv"
)
RAW_CONSUMIDOR_GOV_PROCESSED_PATH = (
    RAW_DIR / "consumidor_gov" / "consumidor_gov_processed.csv"
)


def get_dashboard_dataset_paths() -> dict[str, Path]:
    return {
        "Base unificada": GOLD_DASHBOARD_UNIFIED_DATASET_PATH,
        "Resumo de modelos": GOLD_DASHBOARD_MODEL_COMPARISON_PATH,
        "YouTube + BERTimbau": GOLD_DASHBOARD_YOUTUBE_BERT_DATASET_PATH,
        "Consumidor.gov": RAW_CONSUMIDOR_GOV_PROCESSED_PATH,
    }


def ensure_runtime_directories() -> None:
    for path in [
        RAW_DIR,
        BRONZE_DIR,
        SILVER_DIR,
        GOLD_DIR,
        SANDBOX_DIR,
        NOTEBOOK_DATA_DIR,
        BRONZE_UNIFIED_DIR,
        SILVER_PREPROCESSING_DIR,
        SILVER_UNIFIED_DIR,
        SILVER_SENTIMENT_DIR,
        GOLD_ANALYTICS_DIR,
        GOLD_TOPICS_DIR,
        GOLD_DASHBOARD_DIR,
        GOLD_SQLSERVER_DIR,
        ARTIFACTS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        NOTEBOOK_FIGURES_DIR,
        REPORTS_DIR,
        NOTEBOOK_REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
