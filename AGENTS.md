# AGENTS

## Purpose
This repository implements a Python data pipeline for sentiment analysis of Nubank-related content, with data ingestion, preprocessing, ML training, SQL Server loading, and a Streamlit dashboard.

## What agents should know first
- Python 3.11 is the recommended interpreter.
- Dependencies are installed from `requirements.txt`.
- The project is organized into:
  - `src/collectors/` for data collection
  - `src/preprocessing/` for text cleaning and dataset preparation
  - `src/features/` for feature engineering
  - `src/pipelines/` for end-to-end script entrypoints
  - `src/database/` for SQL Server integration and load logic
  - `dashboard/` for the Streamlit app
  - `docs/` for technical documentation and architecture notes

## Setup and execution
- Install dependencies:
  - `pip install -r requirements.txt`
- Run the pipeline entrypoints:
  - `python -m src.pipelines.run_ingestion`
  - `python -m src.pipelines.run_preprocessing`
  - `python -m src.pipelines.run_training`
  - `python -m src.pipelines.run_topics`
  - `python -m src.pipelines.run_load_sqlserver`
  - `python -m src.pipelines.run_full_pipeline`
- Start the dashboard:
  - `streamlit run dashboard/app.py`
- Tests use `pytest`.

## Important conventions
- Prefer the existing `python -m src...` execution pattern when adding CLI entrypoints.
- Keep raw/bronze/gold semantics clear: ingestion produces raw/bronze-layer artifacts, preprocessing and modeling produce silver/gold-level outputs.
- Use the repository structure consistently and avoid inventing new top-level components without explicit user direction.
- When editing existing logic, preserve naming and style conventions already present in the repository.

## Data and database notes
- SQL Server is the analytic target for silver/gold data.
- `src/database/` contains connection and load helpers.
- The dashboard is built to query `gold` schema views.
- Do not assume external data beyond repository sample files and documented data directories unless the user provides details.

## Documentation links
Use these documents as the source of truth rather than copying large sections:
- [README.md](README.md)
- `docs/01_visao_geral.md`
- `docs/04_pipeline.md`
- `docs/06_sql_server.md`
- `docs/07_dashboard.md`

## Testing guidance
- Run `pytest` for regression checks.
- Existing tests are in `tests/test_pipeline.py` and `tests/test_preprocessing.py`.
- Add tests for any new pipeline logic, preprocessing rules, or SQL load behavior.

## Codex-specific guidance
- Keep generated changes targeted and consistent with the current architecture.
- Avoid major refactors unless explicitly requested.
- Maintain Portuguese-English naming patterns used in the repository.
- When introducing new code, add a brief comment explaining the intention and update docs if the change affects user-visible workflow.
