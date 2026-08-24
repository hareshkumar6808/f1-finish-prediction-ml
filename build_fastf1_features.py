import fastf1
import pandas as pd
from pathlib import Path


fastf1.Cache.enable_cache("data/fastf1_cache")


def get_qualifying_features(season, round_number):
    print(f"Loading {season} round {round_number}...")

    # Load the qualifying session
    session = fastf1.get_session(
        season,
        round_number,
        "Q",
    )

    session.load(
        telemetry=False,
        weather=False,
        messages=False,
    )

    # Keep only laps with recorded lap times
    valid_laps = session.laps[
        session.laps["LapTime"].notna()
    ].copy()

    # Find each driver's fastest qualifying lap
    fastest_laps = (
        valid_laps.groupby(
            "Driver",
            as_index=False,
        )["LapTime"]
        .min()
    )

    # Convert pandas time values into seconds
    fastest_laps["qualifying_seconds"] = (
        fastest_laps["LapTime"]
        .dt.total_seconds()
    )

    # Find the fastest qualifying time
    pole_time = fastest_laps[
        "qualifying_seconds"
    ].min()

    # Calculate each driver's gap to pole
    fastest_laps["qualifying_gap"] = (
        fastest_laps["qualifying_seconds"]
        - pole_time
    )

    # Connect codes such as VER to IDs such as max_verstappen
    driver_ids = session.results[
        ["Abbreviation", "DriverId"]
    ].copy()

    driver_ids = driver_ids.rename(
        columns={
            "Abbreviation": "Driver",
            "DriverId": "driver",
        }
    )

    fastest_laps = fastest_laps.merge(
        driver_ids,
        on="Driver",
        how="left",
    )

    # Add identifiers for the race
    fastest_laps["season"] = season
    fastest_laps["round"] = round_number

    return fastest_laps[
        [
            "season",
            "round",
            "Driver",
            "driver",
            "qualifying_seconds",
            "qualifying_gap",
        ]
    ]




seasons_to_download = [2025]

for season in seasons_to_download:
    season_file = Path(
        f"data/qualifying_features_{season}.csv"
    )

    schedule = fastf1.get_event_schedule(season)

    schedule = schedule[
        schedule["RoundNumber"] > 0
    ]

    if season_file.exists():
        season_features = pd.read_csv(season_file)

        completed_rounds = set(
            season_features["round"].unique()
        )

        print(
            f"{season}: already have "
            f"{len(completed_rounds)} rounds"
        )
    else:
        season_features = pd.DataFrame()

        completed_rounds = set()

    for round_number in schedule["RoundNumber"]:
        round_number = int(round_number)

        if round_number in completed_rounds:
            print(
                f"Skipping cached {season} "
                f"round {round_number}"
            )
            continue

        try:
            race_features = get_qualifying_features(
                season,
                round_number,
            )

        except Exception as error:
            print(
                f"Failed: {season} round "
                f"{round_number}: {error}"
            )

            if "500 calls/h" in str(error):
                print("Rate limit reached. Stop and resume later.")
                break

            continue

        season_features = pd.concat(
            [season_features, race_features],
            ignore_index=True,
        )

        season_features = (
            season_features
            .drop_duplicates(
                subset=["season", "round", "driver"]
            )
            .sort_values(
                ["season", "round", "Driver"]
            )
        )

        # Save after every race so progress cannot be lost
        season_features.to_csv(
            season_file,
            index=False,
        )

        completed_rounds.add(round_number)

        print(
            f"Saved {season} round {round_number}"
        )


print("\nDownload run finished")





