import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "entity_matcher.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "data",
    "threshold_results.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "error_analysis.csv"
)


# ============================================================
# LOAD BEST THRESHOLD
# ============================================================

def load_best_threshold():

    if not os.path.exists(THRESHOLD_PATH):

        print(
            "WARNING: threshold_results.csv not found."
        )

        print(
            "Using default threshold: 0.50"
        )

        return 0.50

    results = pd.read_csv(
        THRESHOLD_PATH
    )

    # Find maximum F1
    best_row = results.loc[
        results["f1"].idxmax()
    ]

    threshold = float(
        best_row["threshold"]
    )

    print(
        f"Using best threshold from threshold analysis: "
        f"{threshold:.2f}"
    )

    return threshold


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ENTITY RESOLUTION ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        FEATURES_PATH
    )

    print()
    print(
        f"Dataset size: {len(df):,}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_data = joblib.load(MODEL_PATH)

    if isinstance(model_data, dict):
        print()
        print("Model package detected.")
        print(
            "Stored keys:",
            list(model_data.keys())
        )

        # Try the common model key names
        if "model" in model_data:
            model = model_data["model"]

        elif "pipeline" in model_data:
            model = model_data["pipeline"]

        elif "classifier" in model_data:
            model = model_data["classifier"]

        else:
            raise ValueError(
                "Could not find the trained model inside entity_matcher.pkl. "
                f"Available keys: {list(model_data.keys())}"
            )

    else:
        model = model_data

    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    threshold = load_best_threshold()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    TARGET_COLUMN = "label"

    # --------------------------------------------------------
    # Columns that must NOT be sent to model
    # --------------------------------------------------------

    ID_COLUMNS = [
        "record_id_a",
        "record_id_b",
        TARGET_COLUMN
    ]

    # --------------------------------------------------------
    # Build X and y
    # --------------------------------------------------------

    X = df.drop(
        columns=ID_COLUMNS
    )

    y = df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Same test split used during training
    # --------------------------------------------------------

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Keep IDs for displaying errors
    # --------------------------------------------------------

    # X_test retains original pandas indexes.
    test_metadata = df.loc[
        X_test.index,
        [
            "record_id_a",
            "record_id_b"
        ]
    ].copy()

    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Apply best threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Build results table
    # --------------------------------------------------------

    results = test_metadata.copy()

    results["probability"] = probabilities

    results["actual"] = y_test.values

    results["prediction"] = predictions

    # Add feature values for analysis
    feature_values = X_test.copy()

    feature_values.index = results.index

    results = pd.concat(
        [
            results,
            feature_values
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Identify errors
    # --------------------------------------------------------

    false_positives = results[
        (results["actual"] == 0) &
        (results["prediction"] == 1)
    ].copy()

    false_negatives = results[
        (results["actual"] == 1) &
        (results["prediction"] == 0)
    ].copy()

    # ========================================================
    # ERROR SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("ERROR SUMMARY")
    print("=" * 70)

    print(
        f"Threshold:       {threshold:.2f}"
    )

    print(
        f"False positives: {len(false_positives)}"
    )

    print(
        f"False negatives: {len(false_negatives)}"
    )

    # ========================================================
    # FEATURE LIST
    # ========================================================

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
        "strong_conflict_count"
    ]

    feature_columns = [
        col
        for col in feature_columns
        if col in results.columns
    ]

    # ========================================================
    # FALSE POSITIVE ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("FALSE POSITIVE FEATURE ANALYSIS")
    print("=" * 70)

    if len(false_positives) > 0:

        print()
        print(
            "Average feature values for false positives:"
        )

        print(
            false_positives[
                feature_columns
            ]
            .mean()
            .round(4)
            .to_string()
        )

        print()
        print(
            "Highest-confidence false positives:"
        )

        display_columns = [
            "record_id_a",
            "record_id_b",
            "probability"
        ] + feature_columns

        print(
            false_positives
            .sort_values(
                "probability",
                ascending=False
            )
            [
                display_columns
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No false positives."
        )

    # ========================================================
    # FALSE NEGATIVE ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("FALSE NEGATIVE FEATURE ANALYSIS")
    print("=" * 70)

    if len(false_negatives) > 0:

        print()
        print(
            "Average feature values for false negatives:"
        )

        print(
            false_negatives[
                feature_columns
            ]
            .mean()
            .round(4)
            .to_string()
        )

        print()
        print(
            "Lowest-confidence false negatives:"
        )

        display_columns = [
            "record_id_a",
            "record_id_b",
            "probability"
        ] + feature_columns

        print(
            false_negatives
            .sort_values(
                "probability",
                ascending=True
            )
            [
                display_columns
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No false negatives."
        )

    # ========================================================
    # ERROR TYPE
    # ========================================================

    results["error_type"] = np.select(
        [
            (
                (results["actual"] == 0) &
                (results["prediction"] == 1)
            ),

            (
                (results["actual"] == 1) &
                (results["prediction"] == 0)
            )
        ],

        [
            "false_positive",
            "false_negative"
        ],

        default="correct"
    )

    # ========================================================
    # SAVE
    # ========================================================

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Threshold used: {threshold:.2f}"
    )

    print()
    print(
        "Detailed errors saved to:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()