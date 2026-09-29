from rapidfuzz import fuzz


def string_similarity(a, b):
    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def token_similarity(a, b):
    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b) / 100.0


def partial_similarity(a, b):
    if not a or not b:
        return 0.0

    return fuzz.partial_ratio(a, b) / 100.0


def exact_match(a, b):
    if not a or not b:
        return 0

    return int(a == b)


def calculate_features(record_a, record_b):
    return {
        "name_similarity": string_similarity(
            record_a["company_name"],
            record_b["company_name"],
        ),
        "name_token_similarity": token_similarity(
            record_a["company_name"],
            record_b["company_name"],
        ),
        "address_similarity": string_similarity(
            record_a["address"],
            record_b["address"],
        ),
        "city_similarity": string_similarity(
            record_a["city"],
            record_b["city"],
        ),
        "state_similarity": string_similarity(
            record_a["state"],
            record_b["state"],
        ),
        "postal_match": exact_match(
            record_a["postal_code"],
            record_b["postal_code"],
        ),
        "phone_match": exact_match(
            record_a["phone"],
            record_b["phone"],
        ),
        "email_match": exact_match(
            record_a["email"],
            record_b["email"],
        ),
        "website_match": exact_match(
            record_a["website"],
            record_b["website"],
        ),
    }