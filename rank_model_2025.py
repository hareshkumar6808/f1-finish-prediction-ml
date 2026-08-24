import pandas as pd

from catboost import CatBoostRanker
from sklearn.metrics import mean_absolute_error


df = pd.read_csv(
    "data/processed/f1_with_qualifying.csv"
)

# Rows from each race must stay together
df = df.sort_values(
    ["season", "round", "driver"]
).copy()


numeric_features = [
    "grid",
    "constructor_form_5",
    "driver_form_5",
    "qualifying_gap",
]

categorical_features = [
    "driver",
    "constructor",
    "circuit",
]

features = (
    numeric_features
    + categorical_features
)


train = df[
    df["season"] <= 2024
].copy()

test = df[
    df["season"] == 2025
].copy()


# Fill missing numeric features
for column in numeric_features:
    median_value = train[column].median()

    train[column] = train[column].fillna(
        median_value
    )

    test[column] = test[column].fillna(
        median_value
    )


# Prepare categorical features
for column in categorical_features:
    train[column] = (
        train[column]
        .fillna("unknown")
        .astype(str)
    )

    test[column] = (
        test[column]
        .fillna("unknown")
        .astype(str)
    )


# Every race is one ranking group
train["race_group"] = (
    train["season"].astype(str)
    + "_"
    + train["round"].astype(str)
)

test["race_group"] = (
    test["season"].astype(str)
    + "_"
    + test["round"].astype(str)
)


# Higher target means a better finish
train["ranking_target"] = (
    100 - train["finish_position"]
)


model = CatBoostRanker(
    loss_function="YetiRank",
    iterations=800,
    depth=6,
    learning_rate=0.03,
    random_seed=42,
    verbose=False,
)


model.fit(
    train[features],
    train["ranking_target"],
    group_id=train["race_group"],
    cat_features=categorical_features,
)


# Higher scores represent better predicted finishes
test["ranking_score"] = model.predict(
    test[features]
)


test["predicted_position"] = (
    test
    .groupby("race_group")["ranking_score"]
    .rank(
        ascending=False,
        method="first",
    )
    .astype(int)
)


ranker_mae = mean_absolute_error(
    test["finish_position"],
    test["predicted_position"],
)

baseline_mae = mean_absolute_error(
    test["finish_position"],
    test["grid"],
)


print("\n2025 ranking results")
print("Ranking model MAE:", round(ranker_mae, 3))
print("Grid baseline MAE:", round(baseline_mae, 3))


test[
    [
        "season",
        "round",
        "race_name",
        "driver",
        "grid",
        "qualifying_gap",
        "predicted_position",
        "finish_position",
    ]
].to_csv(
    "outputs/ranking_predictions_2025.csv",
    index=False,
)
