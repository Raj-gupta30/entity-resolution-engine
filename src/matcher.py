import pandas as pd

from normalize import normalize_record
from similarity import calculate_features


def prepare_dataframe(df):
    normalized_records = []

    for _, row in df.iterrows():
        normalized = normalize_record(row)

        normalized_records.append(normalized)

    normalized_df = pd.DataFrame(normalized_records)

    normalized_df = normalized_df.rename(
        columns={
            "company_name": "normalized_company_name",
            "address": "normalized_address",
            "city": "normalized_city",
            "state": "normalized_state",
            "postal_code": "normalized_postal_code",
            "phone": "normalized_phone",
            "email": "normalized_email",
            "website": "normalized_website",
        }
    )

    return pd.concat(
        [
            df.reset_index(drop=True),
            normalized_df.reset_index(drop=True),
        ],
        axis=1,
    )


def calculate_match_score(features):
    weights = {
        "name_similarity": 0.25,
        "name_token_similarity": 0.15,
        "address_similarity": 0.20,
        "city_similarity": 0.05,
        "state_similarity": 0.05,
        "postal_match": 0.05,
        "phone_match": 0.10,
        "email_match": 0.05,
        "website_match": 0.10,
    }

    score = 0.0

    for feature, weight in weights.items():
        score += features[feature] * weight

    return score


def make_decision(score):
    if score >= 0.90:
        return "HIGH_CONFIDENCE_MATCH"

    if score >= 0.70:
        return "REVIEW_REQUIRED"

    return "NON_MATCH"


def match_records(record_a, record_b):
    normalized_a = normalize_record(record_a)
    normalized_b = normalize_record(record_b)

    features = calculate_features(normalized_a, normalized_b)

    score = calculate_match_score(features)

    decision = make_decision(score)

    return {
        "match_score": round(score, 4),
        "decision": decision,
        "features": features,
    }