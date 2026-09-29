from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

FEATURES_PATH = DATA_DIR / "features.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "entity_matcher.pkl"
)


def main():

    print("=" * 70)
    print("MODEL PROBABILITY DISTRIBUTION")
    print("=" * 70)

    df = pd.read_csv(
        FEATURES_PATH
    )

    feature_columns = [
        "name_similarity",
        "address_similarity",
        "city_similarity",
        "state_similarity",
        "postal_similarity",
        "phone_similarity",
        "email_similarity",
        "website_similarity",
    ]

    X = df[feature_columns]
    y = df["label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    saved = joblib.load(
        MODEL_PATH
    )

    model = saved["model"]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    results = pd.DataFrame(
        {
            "probability": probabilities,
            "actual": y_test.values,
        }
    )

    positive = results[
        results["actual"] == 1
    ]["probability"]

    negative = results[
        results["actual"] == 0
    ]["probability"]

    print("\nPOSITIVE PAIRS")
    print("-" * 50)

    print(
        f"Min    : {positive.min():.6f}"
    )

    print(
        f"25%    : {positive.quantile(.25):.6f}"
    )

    print(
        f"Median : {positive.median():.6f}"
    )

    print(
        f"75%    : {positive.quantile(.75):.6f}"
    )

    print(
        f"Max    : {positive.max():.6f}"
    )

    print("\nNEGATIVE PAIRS")
    print("-" * 50)

    print(
        f"Min    : {negative.min():.6f}"
    )

    print(
        f"25%    : {negative.quantile(.25):.6f}"
    )

    print(
        f"Median : {negative.median():.6f}"
    )

    print(
        f"75%    : {negative.quantile(.75):.6f}"
    )

    print(
        f"Max    : {negative.max():.6f}"
    )

    print("\nSAMPLE PREDICTIONS")
    print("-" * 70)

    sample = results.sort_values(
        "probability"
    )

    print(
        sample.head(10).to_string(
            index=False
        )
    )

    print()

    print(
        sample.tail(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()