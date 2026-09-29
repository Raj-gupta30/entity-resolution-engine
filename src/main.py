from pathlib import Path

import pandas as pd

from matcher import match_records


# Project root:
# F:\entity-resolution-engine
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

DATASET_A_PATH = DATA_DIR / "dataset_a.csv"
DATASET_B_PATH = DATA_DIR / "dataset_b.csv"
RESULTS_PATH = DATA_DIR / "matching_results.csv"


def main():
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION ENGINE")
    print("=" * 60)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Dataset A: {DATASET_A_PATH}")
    print(f"Dataset B: {DATASET_B_PATH}")

    # Check that files exist before trying to load them.
    if not DATASET_A_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset A was not found:\n{DATASET_A_PATH}\n\n"
            "Run this first:\n"
            "python src\\generate_data.py"
        )

    if not DATASET_B_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset B was not found:\n{DATASET_B_PATH}\n\n"
            "Run this first:\n"
            "python src\\generate_data.py"
        )

    # Load datasets
    df_a = pd.read_csv(DATASET_A_PATH)
    df_b = pd.read_csv(DATASET_B_PATH)

    print(f"\nDataset A records: {len(df_a)}")
    print(f"Dataset B records: {len(df_b)}")

    results = []

    print("\nMatching records...\n")

    for i, (_, record_a) in enumerate(df_a.iterrows(), start=1):

        best_match = None
        best_score = -1

        for _, record_b in df_b.iterrows():

            result = match_records(
                record_a,
                record_b,
            )

            if result["match_score"] > best_score:

                best_score = result["match_score"]

                best_match = {
                    "record_id_a": record_a["record_id"],
                    "record_id_b": record_b["record_id"],
                    "match_score": result["match_score"],
                    "decision": result["decision"],
                }

        results.append(best_match)

        # Progress display
        if i % 100 == 0 or i == len(df_a):
            print(f"Processed {i}/{len(df_a)} records")

    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\n" + "=" * 60)
    print("MATCHING COMPLETE")
    print("=" * 60)

    print(f"\nResults saved to:")
    print(RESULTS_PATH)

    print("\nDecision counts:")

    print(
        results_df["decision"].value_counts()
    )

    print("\nTop matches:")

    print(
        results_df[
            [
                "record_id_a",
                "record_id_b",
                "match_score",
                "decision",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
