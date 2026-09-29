import pandas as pd

from matcher import prepare_dataframe
from blocking import generate_candidates


df_a = pd.read_csv("data/dataset_a.csv")
df_b = pd.read_csv("data/dataset_b.csv")


df_a = prepare_dataframe(df_a)
df_b = prepare_dataframe(df_b)


candidates = generate_candidates(
    df_a,
    df_b,
)


print("=" * 60)
print("BLOCKING TEST")
print("=" * 60)

print(f"Dataset A records: {len(df_a)}")
print(f"Dataset B records: {len(df_b)}")

possible_pairs = len(df_a) * len(df_b)

candidate_pairs = len(candidates)

reduction_ratio = (
    1 - candidate_pairs / possible_pairs
)

print(f"\nPossible pairs: {possible_pairs:,}")

print(
    f"Candidate pairs: {candidate_pairs:,}"
)

print(
    f"Reduction ratio: {reduction_ratio:.2%}"
)

print("\nSample candidates:")

print(
    candidates.head(20).to_string(
        index=False
    )
)