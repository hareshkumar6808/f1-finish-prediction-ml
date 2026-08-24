"""Beginner-friendly F1 finishing-position ML pipeline.

Read this file from top to bottom. Each function corresponds to one step:
download -> table -> features -> split -> train -> evaluate -> save.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

START_SEASON = 2018
TEST_SEASON = 2025
API = "https://api.jolpi.ca/ergast/f1"
USER_AGENT = "BeginnerF1ML/0.1"


def download_season(season: int) -> dict:
    """Download one season's classified race results, or use our local cache."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"results_{season}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    url = f"{API}/{season}/results.json"
    # Jolpica returns at most 100 results per request, so collect every page.
    offset = 0
    races_by_round: dict[str, dict] = {}
    template: dict | None = None
    while True:
        response = requests.get(
            url,
            params={"limit": 100, "offset": offset},
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        response.raise_for_status()
        page = response.json()
        if template is None:
            template = page
        mrdata = page["MRData"]
        for race in mrdata["RaceTable"]["Races"]:
            key = race["round"]
            if key not in races_by_round:
                races_by_round[key] = {**race, "Results": []}
            races_by_round[key]["Results"].extend(race["Results"])
        offset += int(mrdata["limit"])
        if offset >= int(mrdata["total"]):
            break

    assert template is not None
    template["MRData"]["RaceTable"]["Races"] = list(races_by_round.values())
    payload = template
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def result_rows(payload: dict) -> list[dict]:
    """Flatten deeply nested API JSON into one simple row per driver/race."""
    races = payload["MRData"]["RaceTable"]["Races"]
    rows: list[dict] = []
    for race in races:
        for result in race["Results"]:
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "race_date": race["date"],
                    "circuit": race["Circuit"]["circuitId"],
                    "driver": result["Driver"]["driverId"],
                    "constructor": result["Constructor"]["constructorId"],
                    "grid": int(result["grid"]),
                    "finish_position": int(result["position"]),
                    "status": result["status"],
                }
            )
    return rows


def build_dataset() -> pd.DataFrame:
    """Download seasons and engineer rolling features without future leakage."""
    rows: list[dict] = []
    for season in range(START_SEASON, TEST_SEASON + 1):
        print(f"Loading {season}...")
        rows.extend(result_rows(download_season(season)))

    df = pd.DataFrame(rows).sort_values(["race_date", "round", "driver"])

    # shift(1) is crucial: the current race result must not help predict itself.
    df["driver_form_5"] = df.groupby("driver")["finish_position"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )
    df["constructor_form_5"] = df.groupby("constructor")["finish_position"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "driver_races.csv", index=False)
    return df


def train_and_evaluate(df: pd.DataFrame) -> None:
    """Train on past seasons, test on 2025, and compare with grid order."""
    numeric = ["season", "round", "grid", "driver_form_5", "constructor_form_5"]
    categorical = ["driver", "constructor", "circuit"]
    features = numeric + categorical

    train = df[df["season"] < TEST_SEASON].copy()
    test = df[df["season"] == TEST_SEASON].copy()

    # Preprocessing converts names to numbers and fills early missing form values.
    preprocessing = ColumnTransformer(
        [
            ("numbers", SimpleImputer(strategy="median"), numeric),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )

    model = Pipeline(
        [
            ("prepare", preprocessing),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=350,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train[features], train["finish_position"])
    test["predicted_finish"] = model.predict(test[features])

    model_mae = mean_absolute_error(test["finish_position"], test["predicted_finish"])
    grid_mae = mean_absolute_error(test["finish_position"], test["grid"])

    # Ranking each race converts raw scores into valid positions 1, 2, 3, ...
    test["predicted_rank"] = test.groupby(["season", "round"])[
        "predicted_finish"
    ].rank(method="first").astype(int)
    rank_mae = mean_absolute_error(test["finish_position"], test["predicted_rank"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "finish_predictor.joblib")
    output_columns = [
        "season",
        "round",
        "race_name",
        "driver",
        "constructor",
        "grid",
        "finish_position",
        "predicted_finish",
        "predicted_rank",
    ]
    test[output_columns].to_csv(OUTPUT_DIR / "predictions_2025.csv", index=False)

    metrics = {
        "test_season": TEST_SEASON,
        "training_rows": len(train),
        "test_rows": len(test),
        "model_raw_mae": round(model_mae, 3),
        "model_rank_mae": round(rank_mae, 3),
        "grid_baseline_mae": round(grid_mae, 3),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nLesson result")
    print(json.dumps(metrics, indent=2))
    if rank_mae < grid_mae:
        print("The model beat the grid-order baseline. Nice first checkpoint!")
    else:
        print("The baseline won. That is useful evidence: our next features must add signal.")


if __name__ == "__main__":
    train_and_evaluate(build_dataset())

