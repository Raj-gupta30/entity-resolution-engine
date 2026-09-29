from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

FEATURES_PATH = (
    DATA_DIR / "features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "entity_matcher.pkl"
)


def main():

    print("=" * 60)
    print("TRAINING ENTITY RESOLUTION MODEL")
    print("=" * 60)

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

    X = df[
        feature_columns
    ]

    y = df["label"]

    print(
        f"\nDataset size: {len(df):,}"
    )

    print(
        f"Positive examples: "
        f"{y.sum():,}"
    )

    print(
        f"Negative examples: "
        f"{(y == 0).sum():,}"
    )

    # Stratified split keeps the same
    # positive/negative ratio in train/test.
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print(
        f"\nTraining examples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing examples: "
        f"{len(X_test):,}"
    )

    # Scaling + Logistic Regression
    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train,
    )

    # Predictions
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # Metrics
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

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"\nPrecision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Non-Match",
                "Match",
            ],
            zero_division=0,
        )
    )

    # Save model
    MODEL_PATH.parent.mkdir(
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
        },
        MODEL_PATH,
    )

    print(
        f"\nModel saved to:"
    )

    print(MODEL_PATH)


if __name__ == "__main__":
    main()