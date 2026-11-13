# =====================================
# Urban Ride Time Prediction
# Complete Data Mining Project
# =====================================

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Place rides.csv next to dm.py or update the file path."
        )

    df = pd.read_csv(path)
    expected = {
        "start_time",
        "end_xcoord",
        "start_xcoord",
        "end_ycoord",
        "start_ycoord",
        "elapsed_seconds",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates()

    if "elapsed_seconds" in df.columns:
        df = df[
            (df["elapsed_seconds"] > 60)
            &
            (df["elapsed_seconds"] < 7200)
        ]

    if "rider_total" in df.columns:
        df = df[
            (df["rider_total"] >= 1)
            &
            (df["rider_total"] <= 8)
        ]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["hour"] = df["start_time"].dt.hour
    df["weekday"] = df["start_time"].dt.dayofweek
    df["month"] = df["start_time"].dt.month
    df["weekend"] = (df["weekday"] >= 5).astype(int)
    df["rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 9))
        |
        ((df["hour"] >= 16) & (df["hour"] <= 19))
    ).astype(int)
    df["distance"] = np.sqrt(
        (df["end_xcoord"] - df["start_xcoord"]) ** 2
        +
        (df["end_ycoord"] - df["start_ycoord"]) ** 2
    )

    categorical = ["carrier_code", "save_forward_marker"]
    encoder = LabelEncoder()
    for col in categorical:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col].astype(str))

    drop_cols = ["record_key", "start_time", "end_time"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    return df


def train_and_evaluate(df: pd.DataFrame) -> None:
    X = df.drop("elapsed_seconds", axis=1)
    y = df["elapsed_seconds"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("\nRESULTS")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 2))


def main() -> None:
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "rides.csv"

    print("Loading data from:", data_path)
    df = load_data(data_path)

    print("Dataset Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nFirst Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    df = clean_data(df)
    print("\nShape after cleaning:", df.shape)

    df = engineer_features(df)
    print("\nDistance created.")

    train_and_evaluate(df)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        print(error)
    except ValueError as error:
        print(error)
    except Exception as error:
        print("An unexpected error occurred:", error)
