import os
import joblib
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_PATH = os.path.join(BASE_DIR, "data", "features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "entity_matcher.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "final_matches.csv")

DEFAULT_THRESHOLD = 0.70


def load_model():
    package = joblib.load(MODEL_PATH)

    if isinstance(package, dict):
        model = package.get("model")
        features = package.get("features")

        if model is None:
            raise ValueError(
                f"Model package does not contain 'model'. "
                f"Keys: {list(package.keys())}"
            )

        return model, features

    return package, None


def get_probability(model, X):
    probabilities = model.predict_proba(X)

    if probabilities.shape[1] == 2:
        return probabilities[:, 1]

    return probabilities[:, 0]


def evidence_decision(row, ml_probability, threshold):
    """
    Combines ML probability with evidence-aware business rules.

    Important:
    Geographic similarity alone should not establish identity.
    """

    strong_ids = row["strong_identifier_count"]
    identity = row["identity_evidence_score"]
    geographic_only = row["geographic_only_match"]
    conflicts = row["strong_conflict_count"]

    name_similarity = row["name_similarity"]
    name_fuzzy = row["name_fuzzy"]

    website_exact = row["website_domain_exact"]

    # ---------------------------------------------------------
    # RULE 1: Strong identity evidence
    # ---------------------------------------------------------
    if strong_ids >= 2 and conflicts <= 1:
        if ml_probability >= 0.40:
            return True, "Strong identity evidence"

    # Exact website/domain is particularly strong.
    if website_exact == 1 and conflicts == 0:
        if ml_probability >= 0.35:
            return True, "Exact website/domain match"

    # ---------------------------------------------------------
    # RULE 2: Geographic-only matches are dangerous
    # ---------------------------------------------------------
    if geographic_only == 1 and strong_ids == 0:
        if identity < 0.45:
            return False, "Geographic similarity without identity evidence"

    # ---------------------------------------------------------
    # RULE 3: Multiple strong conflicts
    # ---------------------------------------------------------
    if conflicts >= 3 and strong_ids == 0:
        return False, "Multiple conflicting identity signals"

    # ---------------------------------------------------------
    # RULE 4: Very weak identity evidence
    # ---------------------------------------------------------
    if (
        identity < 0.20
        and
        name_similarity < 0.35
        and
        website_exact == 0
    ):
        return False, "Insufficient identity evidence"

    # ---------------------------------------------------------
    # RULE 5: ML decision
    # ---------------------------------------------------------
    if ml_probability >= threshold:
        return True, "ML probability above threshold"

    return False, "ML probability below threshold"


def build_explanation(row, probability, decision, reason):
    evidence = []

    if row["website_domain_exact"] == 1:
        evidence.append("exact website")

    if row["strong_identifier_count"] > 0:
        evidence.append(
            f"{int(row['strong_identifier_count'])} strong identifier(s)"
        )

    if row["name_similarity"] >= 0.70:
        evidence.append("high name similarity")

    if row["name_fuzzy"] >= 0.70:
        evidence.append("high fuzzy-name similarity")

    if row["city_similarity"] == 1:
        evidence.append("same city")

    if row["state_similarity"] == 1:
        evidence.append("same state")

    if row["geographic_only_match"] == 1:
        evidence.append("geographic-only evidence")

    if row["strong_conflict_count"] > 0:
        evidence.append(
            f"{int(row['strong_conflict_count'])} strong conflict(s)"
        )

    if not evidence:
        evidence.append("limited evidence")

    return "; ".join(evidence)


def main():

    print("=" * 70)
    print("FINAL ENTITY RESOLUTION ENGINE")
    print("=" * 70)

    print("\nLoading feature dataset...")

    df = pd.read_csv(FEATURES_PATH)

    print(f"Records: {len(df):,}")

    target_column = "label"

    feature_columns = [
        c for c in df.columns
        if c not in [
            "record_id_a",
            "record_id_b",
            target_column
        ]
    ]

    model, stored_features = load_model()

    # Prefer the exact feature list saved with the model.
    if stored_features:
        feature_columns = stored_features

    missing = [
        c for c in feature_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = df[feature_columns].copy()

    print(f"Features used: {len(feature_columns)}")

    print("\nLoading trained model...")

    probabilities = get_probability(model, X)

    # ---------------------------------------------------------
    # Threshold
    # ---------------------------------------------------------

    threshold = DEFAULT_THRESHOLD

    threshold_path = os.path.join(
        BASE_DIR,
        "data",
        "threshold_results.csv"
    )

    if os.path.exists(threshold_path):

        thresholds = pd.read_csv(threshold_path)

        if "f1" in thresholds.columns:
            best = thresholds.loc[
                thresholds["f1"].idxmax()
            ]

            threshold = float(best["threshold"])

        elif "F1" in thresholds.columns:
            best = thresholds.loc[
                thresholds["F1"].idxmax()
            ]

            threshold = float(best["Threshold"])

    print(f"Decision threshold: {threshold:.2f}")

    # ---------------------------------------------------------
    # Apply evidence-aware decision layer
    # ---------------------------------------------------------

    decisions = []
    reasons = []
    explanations = []

    for i, row in df.iterrows():

        probability = probabilities[i]

        decision, reason = evidence_decision(
            row,
            probability,
            threshold
        )

        decisions.append(int(decision))
        reasons.append(reason)

        explanations.append(
            build_explanation(
                row,
                probability,
                decision,
                reason
            )
        )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    results = df[
        ["record_id_a", "record_id_b"]
    ].copy()

    results["match_probability"] = probabilities
    results["threshold"] = threshold
    results["match"] = decisions
    results["decision_reason"] = reasons
    results["evidence"] = explanations

    results["confidence"] = np.where(
        probabilities >= 0.90,
        "HIGH",
        np.where(
            probabilities >= 0.70,
            "MEDIUM",
            "LOW"
        )
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    matches = results["match"].sum()
    non_matches = len(results) - matches

    print("\n" + "=" * 70)
    print("FINAL MATCHING RESULTS")
    print("=" * 70)

    print(f"\nTotal pairs:       {len(results):,}")
    print(f"Matches:           {matches:,}")
    print(f"Non-matches:       {non_matches:,}")

    print("\nDecision reasons:")

    print(
        results["decision_reason"]
        .value_counts()
        .to_string()
    )

    print("\nConfidence distribution:")

    print(
        results["confidence"]
        .value_counts()
        .to_string()
    )

    print("\nTop matches:")

    print(
        results[
            results["match"] == 1
        ]
        .sort_values(
            "match_probability",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved final results to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("ENTITY RESOLUTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()