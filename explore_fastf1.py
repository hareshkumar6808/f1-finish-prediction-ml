import fastf1


# Cache downloaded sessions so future runs are faster
fastf1.Cache.enable_cache("data/fastf1_cache")


# Load qualifying from the first race of 2024
session = fastf1.get_session(2024, 1, "Q")

session.load(
    telemetry=False,
    weather=False,
    messages=False,
)


print("Event:", session.event["EventName"])
print("Session:", session.name)
print("Number of laps:", len(session.laps))

print(
    session.laps[
        ["Driver", "LapNumber", "LapTime", "Compound"]
    ].head(20).to_string(index=False)
)


# Keep laps that have a recorded lap time
valid_laps = session.laps[
    session.laps["LapTime"].notna()
].copy()


# Find each driver's fastest qualifying lap
fastest_laps = (
    valid_laps.groupby("Driver", as_index=False)["LapTime"]
    .min()
)


# Convert lap time into seconds
fastest_laps["qualifying_seconds"] = (
    fastest_laps["LapTime"].dt.total_seconds()
)


# Find the fastest time achieved by anyone
pole_time = fastest_laps["qualifying_seconds"].min()


# Calculate every driver's gap to pole
fastest_laps["qualifying_gap"] = (
    fastest_laps["qualifying_seconds"] - pole_time
)


# Sort from fastest to slowest
fastest_laps = fastest_laps.sort_values(
    "qualifying_seconds"
)


print("\nFastest qualifying laps")
print(
    fastest_laps[
        ["Driver", "qualifying_seconds", "qualifying_gap"]
    ].to_string(index=False)
)



qualifying_features = fastest_laps[
    ["Driver", "qualifying_seconds", "qualifying_gap"]
].copy()


qualifying_features["season"] = 2024
qualifying_features["round"] = 1

qualifying_features = qualifying_features[
    [
        "season",
        "round",
        "Driver",
        "qualifying_seconds",
        "qualifying_gap",
    ]
]


qualifying_features.to_csv(
    "data/qualifying_features_2024_round_1.csv",
    index=False,
)

print("\nSaved qualifying features:")
print(qualifying_features.to_string(index=False))



