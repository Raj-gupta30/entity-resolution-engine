import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "final_matches.csv")


def main():

    df = pd.read_csv(RESULTS_PATH)

    print("=" * 75)
    print("BUSINESS ENTITY RESOLUTION ENGINE")
    print("=" * 75)

    print(f"\nTotal candidate pairs : {len(df):,}")
    print(
        f"Predicted matches     : "
        f"{df['match'].sum():,}"
    )
    print(
        f"Predicted non-matches : "
        f"{(df['match'] == 0).sum():,}"
    )

    print("\n" + "-" * 75)
    print("TOP MATCHES")
    print("-" * 75)

    matches = (
        df[df["match"] == 1]
        .sort_values(
            "match_probability",
            ascending=False
        )
        .head(10)
    )

    for _, row in matches.iterrows():

        print(
            f"\n{row['record_id_a']}  <-->  "
            f"{row['record_id_b']}"
        )

        print(
            f"Probability : "
            f"{row['match_probability']:.4f}"
        )

        print(
            f"Confidence  : "
            f"{row['confidence']}"
        )

        print(
            f"Reason      : "
            f"{row['decision_reason']}"
        )

        print(
            f"Evidence    : "
            f"{row['evidence']}"
        )

    print("\n" + "-" * 75)
    print("DECISION BREAKDOWN")
    print("-" * 75)

    print(
        df["decision_reason"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 75)
    print("DEMO COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()