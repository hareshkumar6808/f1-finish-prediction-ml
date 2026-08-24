from pathlib import Path

import pandas as pd


# Find the individual season files
files = sorted(
    Path("data").glob(
        "qualifying_features_20??.csv"
    )
)


# Display the files that were found
print("Season files:")

for file in files:
    print(file.name)


# Load every season file
season_tables = []

for file in files:
    season_data = pd.read_csv(file)

    season_tables.append(season_data)


# Stack all seasons into one table
combined = pd.concat(
    season_tables,
    ignore_index=True,
)


# Save the combined table
combined.to_csv(
    "data/qualifying_features_2018_2025.csv",
    index=False,
)


print("\nCombined rows:", len(combined))

print("\nRows from each season:")

print(
    combined.groupby("season").size()
)
