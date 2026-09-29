import re
from pathlib import Path

import pandas as pd

from rapidfuzz.fuzz import (
    ratio,
    WRatio,
    token_set_ratio,
    token_sort_ratio,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

A_PATH = DATA_DIR / "dataset_a.csv"
B_PATH = DATA_DIR / "dataset_b.csv"

OUTPUT_PATH = DATA_DIR / "features.csv"


def clean(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize(value):
    value = clean(value)

    return re.sub(
        r"[^a-z0-9]",
        "",
        value.lower(),
    )


def normalized_tokens(value):
    value = clean(value).lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value,
    )

    return set(
        value.split()
    )


def similarity(a, b):

    a = normalize(a)
    b = normalize(b)

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


def jaro_winkler_similarity(a, b):

    a = clean(a)
    b = clean(b)

    if not a or not b:
        return 0.0

    # RapidFuzz WRatio gives us a robust
    # fuzzy similarity for noisy names.
    return WRatio(a, b) / 100.0


def token_similarity(a, b):

    a = clean(a)
    b = clean(b)

    if not a or not b:
        return 0.0

    return token_set_ratio(
        a,
        b,
    ) / 100.0


def token_sort_similarity(a, b):

    a = clean(a)
    b = clean(b)

    if not a or not b:
        return 0.0

    return token_sort_ratio(
        a,
        b,
    ) / 100.0


def exact_match(a, b):

    a = normalize(a)
    b = normalize(b)

    if not a or not b:
        return 0.0

    return float(a == b)


def field_available(a, b):

    return float(
        bool(clean(a))
        and bool(clean(b))
    )


def missing_both_or_one(a, b):

    return float(
        not clean(a)
        or not clean(b)
    )


def phone_similarity(a, b):

    a = re.sub(
        r"\D",
        "",
        clean(a),
    )

    b = re.sub(
        r"\D",
        "",
        clean(b),
    )

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    # Partial phone match.
    if len(a) >= 7 and len(b) >= 7:

        if a[-7:] == b[-7:]:
            return 0.85

    return ratio(a, b) / 100.0


def email_similarity(a, b):

    a = clean(a).lower()
    b = clean(b).lower()

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    # Compare local part and domain separately.
    a_parts = a.split("@")
    b_parts = b.split("@")

    if len(a_parts) != 2 or len(b_parts) != 2:
        return ratio(a, b) / 100.0

    local_score = ratio(
        a_parts[0],
        b_parts[0],
    ) / 100.0

    domain_score = ratio(
        a_parts[1],
        b_parts[1],
    ) / 100.0

    return (
        0.5 * local_score
        + 0.5 * domain_score
    )


def website_similarity(a, b):

    a = clean(a).lower()
    b = clean(b).lower()

    if not a or not b:
        return 0.0

    # Remove protocol and www.
    for prefix in [
        "https://",
        "http://",
        "www.",
    ]:
        a = a.replace(prefix, "")
        b = b.replace(prefix, "")

    if a == b:
        return 1.0

    # Compare domain-like strings.
    return ratio(a, b) / 100.0


def domain_exact(a, b):

    def extract_domain(value):

        value = clean(value).lower()

        value = value.replace(
            "https://",
            "",
        )

        value = value.replace(
            "http://",
            "",
        )

        value = value.replace(
            "www.",
            "",
        )

        return value.split("/")[0]

    a_domain = extract_domain(a)
    b_domain = extract_domain(b)

    if not a_domain or not b_domain:
        return 0.0

    return float(
        a_domain == b_domain
    )


def strong_identifier_count(
    row_a,
    row_b,
):

    count = 0

    if (
        clean(row_a["phone"])
        and clean(row_b["phone"])
        and normalize(row_a["phone"])
        == normalize(row_b["phone"])
    ):
        count += 1

    if (
        clean(row_a["email"])
        and clean(row_b["email"])
        and normalize(row_a["email"])
        == normalize(row_b["email"])
    ):
        count += 1

    if (
        clean(row_a["website"])
        and clean(row_b["website"])
        and domain_exact(
            row_a["website"],
            row_b["website"],
        )
    ):
        count += 1

    return count


def build_features():

    print("=" * 70)
    print("BUILDING IMPROVED ML FEATURE DATASET")
    print("=" * 70)

    dataset_a = pd.read_csv(
        A_PATH
    )

    dataset_b = pd.read_csv(
        B_PATH
    )

    # Ground-truth positives.
    positive_pairs = pd.read_csv(
        DATA_DIR / "ground_truth.csv"
    )

    # Hard negatives.
    negative_pairs = pd.read_csv(
        DATA_DIR / "negative_pairs.csv"
    )

    pairs = pd.concat(
        [
            positive_pairs,
            negative_pairs,
        ],
        ignore_index=True,
    )

    a_lookup = dataset_a.set_index(
        "record_id"
    )

    b_lookup = dataset_b.set_index(
        "record_id"
    )

    rows = []

    total = len(pairs)

    for i, pair in pairs.iterrows():

        record_a = a_lookup.loc[
            pair["record_id_a"]
        ]

        record_b = b_lookup.loc[
            pair["record_id_b"]
        ]

        name_basic = similarity(
            record_a["company_name"],
            record_b["company_name"],
        )

        name_fuzzy = jaro_winkler_similarity(
            record_a["company_name"],
            record_b["company_name"],
        )

        name_token = token_similarity(
            record_a["company_name"],
            record_b["company_name"],
        )

        name_sorted = token_sort_similarity(
            record_a["company_name"],
            record_b["company_name"],
        )

        address_basic = similarity(
            record_a["address"],
            record_b["address"],
        )

        address_token = token_similarity(
            record_a["address"],
            record_b["address"],
        )

        city_sim = similarity(
            record_a["city"],
            record_b["city"],
        )

        state_sim = similarity(
            record_a["state"],
            record_b["state"],
        )

        postal_sim = similarity(
            record_a["postal_code"],
            record_b["postal_code"],
        )

        phone_sim = phone_similarity(
            record_a["phone"],
            record_b["phone"],
        )

        email_sim = email_similarity(
            record_a["email"],
            record_b["email"],
        )

        website_sim = website_similarity(
            record_a["website"],
            record_b["website"],
        )

        website_domain = domain_exact(
            record_a["website"],
            record_b["website"],
        )

        missing_count = sum(
            [
                missing_both_or_one(
                    record_a["city"],
                    record_b["city"],
                ),
                missing_both_or_one(
                    record_a["state"],
                    record_b["state"],
                ),
                missing_both_or_one(
                    record_a["postal_code"],
                    record_b["postal_code"],
                ),
                missing_both_or_one(
                    record_a["phone"],
                    record_b["phone"],
                ),
                missing_both_or_one(
                    record_a["email"],
                    record_b["email"],
                ),
                missing_both_or_one(
                    record_a["website"],
                    record_b["website"],
                ),
            ]
        )

        strong_ids = strong_identifier_count(
            record_a,
            record_b,
        )


        # ---------------------------------------------------------
        # INTERACTION FEATURES
        # ---------------------------------------------------------

        # Strong identity evidence.
        identity_evidence_score = (
            0.30 * name_fuzzy
            + 0.20 * address_token
            + 0.20 * phone_sim
            + 0.15 * email_sim
            + 0.15 * website_sim
        )

        # Geographic evidence.
        geographic_evidence_score = (
            0.45 * city_sim
            + 0.30 * state_sim
            + 0.25 * postal_sim
        )

        # Detect the dangerous pattern:
        # strong geographic similarity but weak identity evidence.
        geographic_only_match = float(
            geographic_evidence_score >= 0.70
            and identity_evidence_score < 0.35
            and strong_ids == 0
        )

        # Agreement between name and contact information.
        name_contact_agreement = (
            name_fuzzy
            * (
                0.40 * phone_sim
                + 0.30 * email_sim
                + 0.30 * website_sim
            )
        )

        # Count strong conflicts.
        strong_conflict_count = 0

        if (
            postal_sim == 0
            and clean(record_a["postal_code"])
            and clean(record_b["postal_code"])
        ):
            strong_conflict_count += 1

        if (
            phone_sim < 0.30
            and clean(record_a["phone"])
            and clean(record_b["phone"])
        ):
            strong_conflict_count += 1

        if (
            email_sim < 0.30
            and clean(record_a["email"])
            and clean(record_b["email"])
        ):
            strong_conflict_count += 1

        if (
            website_sim < 0.30
            and clean(record_a["website"])
            and clean(record_b["website"])
        ):
            strong_conflict_count += 1

        rows.append(
            {
                "record_id_a":
                    pair["record_id_a"],

                "record_id_b":
                    pair["record_id_b"],

                # Name features
                "name_similarity":
                    name_basic,

                "name_fuzzy":
                    name_fuzzy,

                "name_token_similarity":
                    name_token,

                "name_token_sort":
                    name_sorted,

                # Address features
                "address_similarity":
                    address_basic,

                "address_token_similarity":
                    address_token,

                # Geographic features
                "city_similarity":
                    city_sim,

                "state_similarity":
                    state_sim,

                "postal_similarity":
                    postal_sim,

                # Contact features
                "phone_similarity":
                    phone_sim,

                "email_similarity":
                    email_sim,

                "website_similarity":
                    website_sim,

                "website_domain_exact":
                    website_domain,

                # Data-quality features
                "missing_field_count":
                    missing_count,

                "strong_identifier_count":
                    strong_ids,

                "identity_evidence_score":
                    identity_evidence_score,

                "geographic_evidence_score":
                    geographic_evidence_score,

                "geographic_only_match":
                    geographic_only_match,

                "name_contact_agreement":
                    name_contact_agreement,

                "strong_conflict_count":
                    strong_conflict_count,

                "label":
                    int(pair["same_entity"]),
            }
        )

        if (i + 1) % 1000 == 0:
            print(
                f"Processed "
                f"{i + 1:,}/{total:,}"
            )

    features = pd.DataFrame(rows)

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("IMPROVED FEATURE DATASET CREATED")
    print("=" * 70)

    print(
        f"\nRows: {len(features):,}"
    )

    print(
        f"Features: "
        f"{len(features.columns) - 3}"
    )

    print("\nFeature columns:")

    for column in features.columns:

        if column not in [
            "record_id_a",
            "record_id_b",
            "label",
        ]:
            print(
                f"  - {column}"
            )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    build_features()