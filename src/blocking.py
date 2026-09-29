import pandas as pd


def block_by_postal_code(df_a, df_b):
    """
    Generate candidate pairs where postal codes match.
    """

    candidates = []

    # Build lookup for Dataset B
    postal_lookup = {}

    for idx_b, row_b in df_b.iterrows():

        postal = row_b["normalized_postal_code"]

        if not postal:
            continue

        postal_lookup.setdefault(postal, []).append(idx_b)

    # Find matching postal codes
    for idx_a, row_a in df_a.iterrows():

        postal = row_a["normalized_postal_code"]

        if not postal:
            continue

        for idx_b in postal_lookup.get(postal, []):

            candidates.append(
                {
                    "index_a": idx_a,
                    "index_b": idx_b,
                    "blocking_method": "postal_code",
                }
            )

    return candidates


def block_by_city(df_a, df_b):
    """
    Generate candidate pairs where city matches.
    """

    candidates = []

    city_lookup = {}

    for idx_b, row_b in df_b.iterrows():

        city = row_b["normalized_city"]

        if not city:
            continue

        city_lookup.setdefault(city, []).append(idx_b)

    for idx_a, row_a in df_a.iterrows():

        city = row_a["normalized_city"]

        if not city:
            continue

        for idx_b in city_lookup.get(city, []):

            candidates.append(
                {
                    "index_a": idx_a,
                    "index_b": idx_b,
                    "blocking_method": "city",
                }
            )

    return candidates


def block_by_website(df_a, df_b):
    """
    Generate candidate pairs where website domains match.
    """

    candidates = []

    website_lookup = {}

    for idx_b, row_b in df_b.iterrows():

        website = row_b["normalized_website"]

        if not website:
            continue

        website_lookup.setdefault(
            website,
            [],
        ).append(idx_b)

    for idx_a, row_a in df_a.iterrows():

        website = row_a["normalized_website"]

        if not website:
            continue

        for idx_b in website_lookup.get(
            website,
            [],
        ):

            candidates.append(
                {
                    "index_a": idx_a,
                    "index_b": idx_b,
                    "blocking_method": "website",
                }
            )

    return candidates


def block_by_phone(df_a, df_b):
    """
    Generate candidate pairs where phone numbers match.
    """

    candidates = []

    phone_lookup = {}

    for idx_b, row_b in df_b.iterrows():

        phone = row_b["normalized_phone"]

        if not phone:
            continue

        phone_lookup.setdefault(
            phone,
            [],
        ).append(idx_b)

    for idx_a, row_a in df_a.iterrows():

        phone = row_a["normalized_phone"]

        if not phone:
            continue

        for idx_b in phone_lookup.get(
            phone,
            [],
        ):

            candidates.append(
                {
                    "index_a": idx_a,
                    "index_b": idx_b,
                    "blocking_method": "phone",
                }
            )

    return candidates


def block_by_name_prefix(df_a, df_b, prefix_length=4):
    """
    Generate candidate pairs using the first few characters
    of the normalized company name.
    """

    candidates = []

    name_lookup = {}

    for idx_b, row_b in df_b.iterrows():

        name = row_b["normalized_company_name"]

        if not name:
            continue

        key = name[:prefix_length]

        name_lookup.setdefault(
            key,
            [],
        ).append(idx_b)

    for idx_a, row_a in df_a.iterrows():

        name = row_a["normalized_company_name"]

        if not name:
            continue

        key = name[:prefix_length]

        for idx_b in name_lookup.get(
            key,
            [],
        ):

            candidates.append(
                {
                    "index_a": idx_a,
                    "index_b": idx_b,
                    "blocking_method": "name_prefix",
                }
            )

    return candidates


def generate_candidates(df_a, df_b):
    """
    Run multiple blocking strategies and combine
    their candidate pairs.

    A pair may be discovered by more than one strategy.
    """

    all_candidates = []

    all_candidates.extend(
        block_by_postal_code(
            df_a,
            df_b,
        )
    )

    all_candidates.extend(
        block_by_city(
            df_a,
            df_b,
        )
    )

    all_candidates.extend(
        block_by_website(
            df_a,
            df_b,
        )
    )

    all_candidates.extend(
        block_by_phone(
            df_a,
            df_b,
        )
    )

    all_candidates.extend(
        block_by_name_prefix(
            df_a,
            df_b,
        )
    )

    if not all_candidates:
        return pd.DataFrame(
            columns=[
                "index_a",
                "index_b",
                "blocking_methods",
            ]
        )

    candidates_df = pd.DataFrame(
        all_candidates
    )

    # Combine duplicate candidate pairs.
    grouped = (
        candidates_df
        .groupby(
            [
                "index_a",
                "index_b",
            ]
        )
        ["blocking_method"]
        .apply(
            lambda x: ", ".join(sorted(set(x)))
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "blocking_method": "blocking_methods"
        }
    )

    return grouped