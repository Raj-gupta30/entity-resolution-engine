from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

DATASET_A_PATH = DATA_DIR / "dataset_a.csv"
DATASET_B_PATH = DATA_DIR / "dataset_b.csv"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.csv"
NEGATIVE_PAIRS_PATH = DATA_DIR / "negative_pairs.csv"


def main():

    print("Loading datasets...")

    dataset_a = pd.read_csv(
        DATASET_A_PATH
    )

    dataset_b = pd.read_csv(
        DATASET_B_PATH
    )

    positive_pairs = pd.read_csv(
        GROUND_TRUTH_PATH
    )

    negative_pairs = pd.read_csv(
        NEGATIVE_PAIRS_PATH
    )

    evaluation_pairs = pd.concat(
        [
            positive_pairs,
            negative_pairs,
        ],
        ignore_index=True,
    )

    # Create lookup dictionaries.
    records_a = dataset_a.set_index(
        "record_id"
    ).to_dict("index")

    records_b = dataset_b.set_index(
        "record_id"
    ).to_dict("index")

    from matcher import match_records

    predictions = []

    print(
        f"\nEvaluating {len(evaluation_pairs):,} pairs..."
    )

    for _, pair in evaluation_pairs.iterrows():

        record_a = records_a[
            pair["record_id_a"]
        ]

        record_b = records_b[
            pair["record_id_b"]
        ]

        result = match_records(
            record_a,
            record_b,
        )

        predictions.append(
            {
                "record_id_a": pair[
                    "record_id_a"
                ],
                "record_id_b": pair[
                    "record_id_b"
                ],
                "actual_match": pair[
                    "same_entity"
                ],
                "match_score": result[
                    "match_score"
                ],
                "decision": result[
                    "decision"
                ],
            }
        )

    results = pd.DataFrame(
        predictions
    )

    y_true = results[
        "actual_match"
    ]

    # For this baseline:
    # >= 0.90 = predicted match
    y_pred = (
        results["match_score"]
        >= 0.90
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    print("\n" + "=" * 60)
    print("BASELINE ENTITY RESOLUTION EVALUATION")
    print("=" * 60)

    print(
        f"\nEvaluation pairs: {len(results):,}"
    )

    print(
        f"Positive pairs: "
        f"{y_true.sum():,}"
    )

    print(
        f"Negative pairs: "
        f"{(y_true == 0).sum():,}"
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        """
                 Predicted
                 Non-Match  Match
Actual
Non-Match
Match
        """
    )

    print(matrix)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Non-Match",
                "Match",
            ],
            zero_division=0,
        )
    )

    results.to_csv(
        DATA_DIR / "evaluation_results.csv",
        index=False,
    )

    print(
        "\nDetailed results saved to:"
    )

    print(
        "data/evaluation_results.csv"
    )


if __name__ == "__main__":
    main()