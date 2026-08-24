import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Load the merged race and FastF1 data
df = pd.read_csv(
    "data/processed/f1_with_qualifying.csv"
)


# These are the clues supplied to the model
features = [
    "grid",
    "constructor_form_5",
    "qualifying_gap",
]


# This is the answer the model learns to predict
target = "finish_position"


# Use older races for training
train = df[
    df["season"] <= 2023
].copy()


# Use 2024 as unseen validation data
validation = df[
    df["season"] == 2024
].copy()


print("Training rows:", len(train))
print("Validation rows:", len(validation))

print("\nMissing training values:")

print(
    train[features].isna().sum()
)


# Fill missing values using only training information
for column in features:
    median_value = train[column].median()

    train[column] = train[column].fillna(
        median_value
    )

    validation[column] = validation[column].fillna(
        median_value
    )


# Separate model inputs from the correct answers
X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]


# Create the model
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=40,
    random_state=42,
    n_jobs=-1,
)


# Train using races from 2018 through 2023
model.fit(
    X_train,
    y_train,
)


# Predict the 2024 finishing positions
predictions = model.predict(
    X_validation
)


# Measure model error
model_mae = mean_absolute_error(
    y_validation,
    predictions,
)


# Simple baseline: finish where you started
baseline_predictions = validation["grid"]

baseline_mae = mean_absolute_error(
    y_validation,
    baseline_predictions,
)


print("\nValidation results")

print(
    "FastF1 model MAE:",
    round(model_mae, 3),
)

print(
    "Grid baseline MAE:",
    round(baseline_mae, 3),
)


if model_mae < baseline_mae:
    print("The model beat the baseline!")
else:
    print("The baseline is still better.")

# Ablation test: train the same model without FastF1
basic_features = [
    "grid",
    "constructor_form_5",
]


basic_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=40,
    random_state=42,
    n_jobs=-1,
)


basic_model.fit(
    train[basic_features],
    y_train,
)


basic_predictions = basic_model.predict(
    validation[basic_features]
)


basic_mae = mean_absolute_error(
    y_validation,
    basic_predictions,
)


print("\nFeature comparison")

print(
    "Without qualifying gap:",
    round(basic_mae, 3),
)

print(
    "With qualifying gap:",
    round(model_mae, 3),
)

print(
    "Grid baseline:",
    round(baseline_mae, 3),
)



importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
})

importance = importance.sort_values(
    "importance",
    ascending=False,
)


print("\nFeature importance")

print(
    importance.to_string(index=False)
)


results = validation[
    [
        "season",
        "round",
        "race_name",
        "driver",
        "grid",
        "finish_position",
    ]
].copy()


results["prediction"] = predictions

results["absolute_error"] = (
    results["finish_position"]
    - results["prediction"]
).abs()


print("\nFirst race predictions")

print(
    results[
        results["round"] == 1
    ]
    .sort_values("prediction")
    .to_string(index=False)
)

results["predicted_rank"] = (
    results
    .groupby(["season", "round"])["prediction"]
    .rank(
        method="first",
        ascending=True,
    )
    .astype(int)
)

ranked_mae = mean_absolute_error(
    results["finish_position"],
    results["predicted_rank"],
)

print("\nRaw versus ranked predictions")

print(
    "Raw prediction MAE:",
    round(model_mae, 3),
)

print(
    "Ranked prediction MAE:",
    round(ranked_mae, 3),
)

print(
    "Grid baseline MAE:",
    round(baseline_mae, 3),
)

print("\nRound 1 finishing-order prediction")

print(
    results[
        results["round"] == 1
    ][
        [
            "driver",
            "grid",
            "predicted_rank",
            "finish_position",
        ]
    ]
    .sort_values("predicted_rank")
    .to_string(index=False)
)

# Save everything needed to reuse the model
model_package = {
    "model": model,
    "features": features,
    "fill_values": {
        column: train[column].median()
        for column in features
    },
}


joblib.dump(
    model_package,
    "models/fastf1_finish_model.joblib",
)


# Save validation predictions for inspection
results.to_csv(
    "outputs/validation_predictions_2024.csv",
    index=False,
)


print(
    "\nSaved model: "
    "models/fastf1_finish_model.joblib"
)

print(
    "Saved predictions: "
    "outputs/validation_predictions_2024.csv"
)
