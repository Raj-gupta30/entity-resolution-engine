import re

from difflib import SequenceMatcher


def normalize_text(value):
    """
    Normalize text for comparison.
    """

    if value is None:
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def text_similarity(value_a, value_b):
    """
    Calculate character-level similarity
    between two text values.

    Returns a value between 0 and 1.
    """

    a = normalize_text(value_a)
    b = normalize_text(value_b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def exact_match(value_a, value_b):
    """
    Return 1 if two normalized values match exactly.
    """

    a = normalize_text(value_a)
    b = normalize_text(value_b)

    if not a or not b:
        return 0.0

    return float(a == b)


def phone_similarity(value_a, value_b):
    """
    Compare phone numbers after removing
    formatting characters.
    """

    a = re.sub(
        r"\D",
        "",
        str(value_a),
    )

    b = re.sub(
        r"\D",
        "",
        str(value_b),
    )

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    # Compare last 10 digits.
    if len(a) >= 10 and len(b) >= 10:
        if a[-10:] == b[-10:]:
            return 1.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def email_similarity(value_a, value_b):
    """
    Compare email addresses.
    """

    a = normalize_text(value_a)
    b = normalize_text(value_b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    # Compare username and domain separately.
    if "@" in a and "@" in b:

        user_a, domain_a = a.split(
            "@",
            1,
        )

        user_b, domain_b = b.split(
            "@",
            1,
        )

        user_score = text_similarity(
            user_a,
            user_b,
        )

        domain_score = text_similarity(
            domain_a,
            domain_b,
        )

        return (
            0.5 * user_score
            + 0.5 * domain_score
        )

    return text_similarity(
        a,
        b,
    )


def website_similarity(value_a, value_b):
    """
    Compare website domains.
    """

    a = normalize_text(value_a)
    b = normalize_text(value_b)

    a = re.sub(
        r"^https?://",
        "",
        a,
    )

    b = re.sub(
        r"^https?://",
        "",
        b,
    )

    a = a.removeprefix("www.")
    b = b.removeprefix("www.")

    a = a.rstrip("/")
    b = b.rstrip("/")

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return text_similarity(
        a,
        b,
    )


def generate_features(record_a, record_b):
    """
    Generate the complete feature vector
    for a pair of business records.
    """

    features = {

        # Company identity
        "name_similarity": text_similarity(
            record_a["company_name"],
            record_b["company_name"],
        ),

        # Address
        "address_similarity": text_similarity(
            record_a["address"],
            record_b["address"],
        ),

        # Geographic information
        "city_similarity": exact_match(
            record_a["city"],
            record_b["city"],
        ),

        "state_similarity": exact_match(
            record_a["state"],
            record_b["state"],
        ),

        "postal_similarity": exact_match(
            record_a["postal_code"],
            record_b["postal_code"],
        ),

        # Contact information
        "phone_similarity": phone_similarity(
            record_a["phone"],
            record_b["phone"],
        ),

        "email_similarity": email_similarity(
            record_a["email"],
            record_b["email"],
        ),

        "website_similarity": website_similarity(
            record_a["website"],
            record_b["website"],
        ),
    }

    return features