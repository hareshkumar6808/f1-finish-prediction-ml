import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Load race results merged with FastF1 qualifying features
df = pd.read_csv(
    "data/processed/f1_with_qualifying.csv"
)


features = [
    "grid",
    "constructor_form_5",
    "qualifying_gap",
]

target = "finish_position"


# Train on everything through 2024
train = df[
    df["season"] <= 2024
].copy()


# Keep 2025 unseen until evaluation
test = df[
    df["season"] == 2025
].copy()


print("Training rows:", len(train))
print("Test rows:", len(test))


# Learn missing-value replacements from training data
fill_values = {}

for column in features:
    fill_values[column] = train[column].median()

    train[column] = train[column].fillna(
        fill_values[column]
    )

    test[column] = test[column].fillna(
        fill_values[column]
    )


X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# Use the settings selected with 2024 validation
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=40,
    random_state=42,
    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,
)


test["prediction_score"] = model.predict(
    X_test
)


# Convert scores into positions within each race
test["predicted_position"] = (
    test
    .groupby(["season", "round"])["prediction_score"]
    .rank(
        method="first",
        ascending=True,
    )
    .astype(int)
)


raw_mae = mean_absolute_error(
    y_test,
    test["prediction_score"],
)

ranked_mae = mean_absolute_error(
    y_test,
    test["predicted_position"],
)

baseline_mae = mean_absolute_error(
    y_test,
    test["grid"],
)


print("\nFinal 2025 results")
print("Raw model MAE:", round(raw_mae, 3))
print("Ranked model MAE:", round(ranked_mae, 3))
print("Grid baseline MAE:", round(baseline_mae, 3))


# Save the final model
joblib.dump(
    {
        "model": model,
        "features": features,
        "fill_values": fill_values,
    },
    "models/final_f1_model.joblib",
)


# Save test predictions
test[
    [
        "season",
        "round",
        "race_name",
        "driver",
        "constructor",
        "grid",
        "qualifying_gap",
        "predicted_position",
        "finish_position",
    ]
].to_csv(
    "outputs/final_predictions_2025.csv",
    index=False,
)


with open(
    "outputs/final_metrics_2025.json",
    "w",
    encoding="utf-8",
) as metrics_file:
    json.dump(
        {
            "test_season": 2025,
            "training_rows": len(train),
            "test_rows": len(test),
            "raw_model_mae": round(raw_mae, 3),
            "ranked_model_mae": round(ranked_mae, 3),
            "grid_baseline_mae": round(baseline_mae, 3),
        },
        metrics_file,
        indent=2,
    )


print("\nFinal model, metrics, and predictions saved.")

