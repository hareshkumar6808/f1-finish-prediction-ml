import pandas as pd


# Load the original race-result dataset
races = pd.read_csv(
    "data/processed/driver_races.csv"
)


# Load the combined FastF1 qualifying dataset
qualifying = pd.read_csv(
    "data/qualifying_features_2018_2025.csv"
)


print("Race rows:", len(races))
print("Qualifying rows:", len(qualifying))


# Keep only the FastF1 columns needed by the model
qualifying = qualifying[
    [
        "season",
        "round",
        "driver",
        "qualifying_seconds",
        "qualifying_gap",
    ]
]


# Match the two tables using the race and driver identifiers
merged = races.merge(
    qualifying,
    on=["season", "round", "driver"],
    how="left",
)


print("Merged rows:", len(merged))


# Check how many rows successfully received FastF1 data
merged["has_qualifying_data"] = (
    merged["qualifying_gap"].notna()
)


print("\nFastF1 matches by season:")

print(
    merged[
        merged["season"] <= 2024
    ]
    .groupby("season")[
        "has_qualifying_data"
    ]
    .agg(["sum", "count"])
)


# Save the new model-ready table
merged.to_csv(
    "data/processed/f1_with_qualifying.csv",
    index=False,
)


print(
    "\nSaved: "
    "data/processed/f1_with_qualifying.csv"
)
