import pandas as pd
from catboost import CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Load the historical F1 dataset
df = pd.read_csv("data/processed/driver_races.csv")

# Put races in chronological order before calculating history
df = df.sort_values(
    ["race_date", "round", "driver"]
).copy()


# Positive means the driver gained places during a race
df["positions_gained"] = (
    df["grid"] - df["finish_position"]
)


# Average positions gained over the driver's previous 10 races
df["driver_gained_avg_10"] = (
    df.groupby("driver")["positions_gained"]
      .transform(
          lambda values:
              values.shift(1)
                    .rolling(10, min_periods=1)
                    .mean()
      )
)


print("Dataset shape:", df.shape)
print(df.head())

# Features are the clues given to the model
numeric_features = [
    "grid",
    "driver_form_5",
    "constructor_form_5",
]

categorical_features = [
    "driver",
    "constructor",
    "circuit",
]

features = numeric_features + categorical_features
# The target is what we want to predict
target = "finish_position"


# Split races chronologically
train = df[df["season"] <= 2023].copy()
validation = df[df["season"] == 2024].copy()
test = df[df["season"] == 2025].copy()


# Fill missing values using only training information
for column in numeric_features:
    median_value = train[column].median()

    train[column] = train[column].fillna(median_value)
    validation[column] = validation[column].fillna(median_value)
    test[column] = test[column].fillna(median_value)


# Separate features from targets
X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]


print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Test rows:", len(test))



# Create the random forest
model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="MAE",
    random_seed=42,
    verbose=False,
)


# Train the model using races through 2023
model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
)


# Predict the unseen 2024 races
validation_predictions = model.predict(X_validation)


# Measure the model's average error
model_mae = mean_absolute_error(
    y_validation,
    validation_predictions,
)


# Baseline: assume every driver finishes where they started
baseline_predictions = validation["grid"]

baseline_mae = mean_absolute_error(
    y_validation,
    baseline_predictions,
)


print("\nValidation results")
print("Model MAE:", round(model_mae, 3))
print("Grid baseline MAE:", round(baseline_mae, 3))

if model_mae < baseline_mae:
    print("The model beat the baseline!")
else:
    print("The grid baseline is still better.")


















