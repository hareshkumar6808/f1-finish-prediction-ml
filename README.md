# F1 Finish Prediction — learning project

An end-to-end Formula 1 machine-learning experiment built while learning the
fundamentals of supervised ML and time-aware evaluation.

The project collects historical race results and FastF1 qualifying data,
engineers pre-race features, trains regression and ranking models, and evaluates
them chronologically against the strong baseline of finishing in grid order.

## Honest result

The final random-forest regression experiment trained on 2018–2024 and tested
on the held-out 2025 season:

| Metric | MAE |
|---|---:|
| Expected finishing-position model | **3.250** |
| Grid-position baseline | 3.344 |
| Forced 1–20 ranking | 3.436 |

The expected-position model improved MAE by about 2.8% over the grid baseline.
The forced ranking was worse, showing that independent regression scores are
not automatically a good full-order prediction. This repository preserves that
negative result rather than presenting the model as more reliable than it is.

## What it demonstrates

- Data ingestion from Jolpica and FastF1
- API caching, pagination, rate-limit handling, and resumable downloads
- Driver/race-level feature engineering
- Prevention of future-data leakage with shifted rolling features
- Chronological train/validation/test splits
- Linear regression, random forest, CatBoost, and CatBoost ranking experiments
- Baseline comparison, ablation tests, MAE, and saved inference artifacts

## Data flow

```text
Jolpica race results ─┐
                      ├─> merged driver-race table ─> model ─> evaluation
FastF1 qualifying ────┘
```

Downloaded API responses, FastF1 caches, processed tables, virtual environments,
and trained binaries are excluded from Git. Small prediction/evaluation outputs
are retained under `outputs/`.

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create the local generated-data folders if needed:

```powershell
New-Item -ItemType Directory -Force data\raw,data\fastf1_cache,data\processed,models,outputs
```

## Reproduce the pipeline

The scripts reflect the project's learning history as well as the final path:

1. `src/f1_model.py` downloads Jolpica results and builds the base table.
2. `build_fastf1_features.py` builds resumable qualifying features by season.
3. `combine_qualifying.py` combines the completed season tables.
4. `merge_data.py` joins qualifying features to driver-race results.
5. `train_fastf1_model.py` performs the 2024 validation experiment.
6. `final_test_2025.py` trains through 2024 and evaluates the held-out 2025 season.
7. `predict_race.py` demonstrates loading a saved model for inference.

The FastF1 downloader respects upstream rate limits and saves progress after
each completed round. Availability and upstream API behavior can change.

## Limitations

- Exact race outcomes contain crashes, failures, penalties, strategy, and safety
  cars that are not represented by the current feature set.
- The model's useful output is an expected finishing-position score, not a
  trustworthy exact finishing order.
- The 2026 regulation reset would require drift-aware retraining and evaluation.
- This is an educational project, not a betting or decision system.

## Next experiments

- Separate DNF classification from pace prediction
- Practice long-run pace and tyre-degradation features
- Weather and grid-penalty information available before the start
- Walk-forward evaluation across multiple seasons
- A ranking-specific or probabilistic simulation model
