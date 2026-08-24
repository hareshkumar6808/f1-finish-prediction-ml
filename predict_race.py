import joblib
import pandas as pd


# Load the saved model package
package = joblib.load(
    "models/fastf1_finish_model.joblib"
)

model = package["model"]
features = package["features"]
fill_values = package["fill_values"]


# Load the merged F1 dataset
df = pd.read_csv(
    "data/processed/f1_with_qualifying.csv"
)


# Select one race to predict
race = df[
    (df["season"] == 2024)
    & (df["round"] == 24)
].copy()


# Fill missing values exactly as during training
for column in features:
    race[column] = race[column].fillna(
        fill_values[column]
    )


# Use the saved model
race["prediction_score"] = model.predict(
    race[features]
)


# Convert scores into finishing positions
race["predicted_position"] = (
    race["prediction_score"]
    .rank(
        method="first",
        ascending=True,
    )
    .astype(int)
)


# Display the predicted finishing order
prediction_table = race[
    [
        "predicted_position",
        "driver",
        "constructor",
        "grid",
        "qualifying_gap",
        "finish_position",
    ]
].sort_values("predicted_position")


print("\nPredicted finishing order")

print(
    prediction_table.to_string(index=False)
)
