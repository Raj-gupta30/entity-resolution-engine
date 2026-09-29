from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
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
    print("ENTITY RESOLUTION THRESHOLD ANALYSIS")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(
        FEATURES_PATH
    )

    feature_columns = [
         "name_similarity",
        "name_fuzzy",
        "name_token_similarity",
        "name_token_sort",
        "address_similarity",
        "address_token_similarity",
        "city_similarity",
        "state_similarity",
        "postal_similarity",
        "phone_similarity",
        "email_similarity",
        "website_similarity",
        "website_domain_exact",
        "missing_field_count",
        "strong_identifier_count",
        "identity_evidence_score",
        "geographic_evidence_score",
        "geographic_only_match",
        "name_contact_agreement",
        "strong_conflict_count",
    ]

    X = df[feature_columns]
    y = df["label"]

    # Use the same split as training.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Load trained model
    saved = joblib.load(
        MODEL_PATH
    )

    model = saved["model"]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print(
        f"\nTesting {len(y_test):,} pairs"
    )

    print("\n" + "-" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1':<15}"
        f"{'False Pos.':<15}"
    )

    print("-" * 70)

    results = []

    for threshold in [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.99,
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        false_positives = (
            (
                (predictions == 1)
                & (y_test == 0)
            )
            .sum()
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": false_positives,
            }
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<15.4f}"
            f"{recall:<15.4f}"
            f"{f1:<15.4f}"
            f"{false_positives:<15}"
        )

    results_df = pd.DataFrame(
        results
    )

    best = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST F1 THRESHOLD")
    print("=" * 70)

    print(
        f"\nThreshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best['f1']:.4f}"
    )

    print(
        f"False positives: "
        f"{int(best['false_positives'])}"
    )

    output_path = (
        DATA_DIR
        / "threshold_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults saved to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()