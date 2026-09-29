import random
import re
from pathlib import Path

import pandas as pd


random.seed(42)

N_ENTITIES = 1000
N_NEGATIVES = 5000


PREFIXES = [
    "Northstar", "Blue Horizon", "Evergreen", "Summit",
    "Pioneer", "Crestview", "Silverline", "Clearwater",
    "Redwood", "Brightway", "Oakridge", "Westfield",
    "Lakeside", "Grandview", "Riverside", "Stonebridge",
    "Highland", "Golden", "Maple", "Cedar",
    "Ironwood", "Greenfield", "Brookside", "Sunrise",
    "Horizon", "Liberty", "Prime", "United",
    "Metro", "Vertex", "Apex", "Atlas",
    "Nova", "Quantum", "Sterling", "Pacific",
    "Atlantic", "Central", "National", "Urban",
]

INDUSTRIES = [
    "Logistics", "Foods", "Healthcare", "Medical",
    "Technology", "Software", "Consulting", "Financial",
    "Automotive", "Manufacturing", "Retail",
    "Pharmaceuticals", "Energy", "Construction",
    "Engineering", "Marketing", "Security",
    "Telecom", "Insurance", "Education",
]

SUFFIXES = [
    "Group", "Solutions", "Services", "Systems",
    "Industries", "Holdings", "Partners", "Enterprises",
    "International", "Corporation", "Company", "Networks",
]

CITIES = [
    ("New York", "New York", "10001"),
    ("Los Angeles", "California", "90001"),
    ("Chicago", "Illinois", "60601"),
    ("Houston", "Texas", "77001"),
    ("Phoenix", "Arizona", "85001"),
    ("Philadelphia", "Pennsylvania", "19019"),
    ("San Antonio", "Texas", "78201"),
    ("San Diego", "California", "92101"),
    ("Dallas", "Texas", "75201"),
    ("San Jose", "California", "95101"),
    ("Austin", "Texas", "73301"),
    ("Jacksonville", "Florida", "32099"),
    ("Seattle", "Washington", "98101"),
    ("Denver", "Colorado", "80201"),
    ("Boston", "Massachusetts", "02108"),
    ("Atlanta", "Georgia", "30301"),
    ("Miami", "Florida", "33101"),
    ("Portland", "Oregon", "97201"),
    ("Detroit", "Michigan", "48201"),
    ("Minneapolis", "Minnesota", "55401"),
]

STREETS = [
    "Main Street",
    "Market Street",
    "Oak Avenue",
    "Maple Road",
    "Washington Street",
    "Lake Avenue",
    "Industrial Boulevard",
    "Park Road",
    "Center Street",
    "Commerce Drive",
]


def normalize(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def company_name(index):
    prefix = PREFIXES[index % len(PREFIXES)]

    industry = INDUSTRIES[
        (index // len(PREFIXES))
        % len(INDUSTRIES)
    ]

    suffix = SUFFIXES[
        (index // (
            len(PREFIXES) * len(INDUSTRIES)
        ))
        % len(SUFFIXES)
    ]

    return f"{prefix} {industry} {suffix}"


def address(index):
    number = 100 + (index * 37) % 9900
    street = STREETS[index % len(STREETS)]

    return f"{number} {street}"


def phone(index):
    number = 2000000000 + index * 7919

    digits = str(number)

    return (
        f"+1-{digits[:3]}-"
        f"{digits[3:6]}-"
        f"{digits[6:]}"
    )


def email(name):
    domain = normalize(name)

    return f"contact@{domain}.com"


def website(name):
    domain = normalize(name)

    return f"https://www.{domain}.com"


def build_canonical():

    rows = []


    for i in range(N_ENTITIES):

        city, state, base_postal = CITIES[
            i % len(CITIES)
        ]

        try:
            base_number = int(
                base_postal.replace("-", "")
            )
        except ValueError:
            base_number = 10000

        postal = str(
            base_number + (i * 17) % 900
        ).zfill(5)

        name = company_name(i)

        rows.append(
            {
                "entity_id": f"E{i:05d}",
                "company_name": name,
                "address": address(i),
                "city": city,
                "state": state,
                "postal_code": postal,
                "phone": phone(i),
                "email": email(name),
                "website": website(name),
            }
        )

    return pd.DataFrame(rows)


def typo(text):
    if not text:
        return text

    chars = list(text)

    operation = random.choice(
        ["delete", "swap", "replace", "none"]
    )

    if operation == "delete" and len(chars) > 5:
        idx = random.randrange(len(chars))
        chars.pop(idx)

    elif operation == "swap" and len(chars) > 5:
        idx = random.randrange(len(chars) - 1)
        chars[idx], chars[idx + 1] = (
            chars[idx + 1],
            chars[idx],
        )

    elif operation == "replace":
        idx = random.randrange(len(chars))
        chars[idx] = random.choice(
            "abcdefghijklmnopqrstuvwxyz"
        )

    return "".join(chars)


def corrupt_name(name):

    words = name.split()

    transformations = [
        lambda: name,
        lambda: name.upper(),
        lambda: name.lower(),
        lambda: name.replace("Group", "Grp."),
        lambda: name.replace("Company", "Co."),
        lambda: name.replace("Corporation", "Corp."),
        lambda: name.replace("International", "Intl."),
        lambda: name.replace("Services", "Svc."),
        lambda: name.replace("Solutions", "Sol."),
        lambda: name.replace("Industries", "Ind."),
        lambda: " ".join(reversed(words)),
        lambda: typo(name),
        lambda: name.replace(" ", ""),
    ]

    return random.choice(transformations)


def corrupt_address(value):

    transformations = [
        lambda: value,
        lambda: value.replace(
            "Street", "St."
        ),
        lambda: value.replace(
            "Avenue", "Ave."
        ),
        lambda: value.replace(
            "Road", "Rd."
        ),
        lambda: value.replace(
            "Boulevard", "Blvd."
        ),
        lambda: value.upper(),
        lambda: value.lower(),
        lambda: typo(value),
    ]

    return random.choice(transformations)


def corrupt_phone(value):

    digits = re.sub(
        r"\D",
        "",
        value,
    )

    transformations = [
        lambda: value,
        lambda: digits,
        lambda: f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}",
        lambda: f"{digits[1:4]}-{digits[4:7]}-{digits[7:]}",
        lambda: digits[:-1],
    ]

    return random.choice(transformations)


def corrupt_email(value):

    transformations = [
        lambda: value,
        lambda: value.upper(),
        lambda: value.replace(
            "contact",
            "info",
        ),
        lambda: value.replace(
            "contact",
            "sales",
        ),
        lambda: value[:-4],
    ]

    return random.choice(transformations)


def corrupt_website(value):

    domain = value.replace(
        "https://",
        "",
    )

    transformations = [
        lambda: value,
        lambda: domain,
        lambda: f"http://{domain}",
        lambda: f"www.{domain}",
        lambda: value.replace(
            "https://www.",
            "",
        ),
    ]

    return random.choice(transformations)


def maybe_missing(value, probability):

    if random.random() < probability:
        return ""

    return value


def create_dataset_a(canonical):

    rows = []

    for i, row in canonical.iterrows():

        rows.append(
            {
                "record_id": f"A{i:05d}",
                "company_name": row["company_name"],
                "address": row["address"],
                "city": row["city"],
                "state": row["state"],
                "postal_code": row["postal_code"],
                "phone": row["phone"],
                "email": row["email"],
                "website": row["website"],
            }
        )

    return pd.DataFrame(rows)


def create_dataset_b(canonical):

    rows = []

    for i, row in canonical.iterrows():

        rows.append(
            {
                "record_id": f"B{i:05d}",

                # Stronger name corruption
                "company_name": corrupt_name(
                    row["company_name"]
                ),

                "address": corrupt_address(
                    row["address"]
                ),

                # Geographic fields are sometimes missing
                "city": maybe_missing(
                    row["city"],
                    0.05,
                ),

                "state": maybe_missing(
                    row["state"],
                    0.05,
                ),

                "postal_code": maybe_missing(
                    row["postal_code"],
                    0.10,
                ),

                # Contact fields can be missing/corrupted
                "phone": maybe_missing(
                    corrupt_phone(
                        row["phone"]
                    ),
                    0.10,
                ),

                "email": maybe_missing(
                    corrupt_email(
                        row["email"]
                    ),
                    0.15,
                ),

                "website": maybe_missing(
                    corrupt_website(
                        row["website"]
                    ),
                    0.15,
                ),
            }
        )

    return pd.DataFrame(rows)


def create_positive_pairs(
    dataset_a,
    dataset_b,
):

    return pd.DataFrame(
        {
            "record_id_a":
                dataset_a["record_id"],

            "record_id_b":
                dataset_b["record_id"],

            "same_entity": 1,
        }
    )


def name_similarity_hint(name):

    """
    Used only to find hard negatives.
    This does NOT become a model feature.
    """

    a = normalize(name)

    return a

def clean_tokens(value):

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value,
    )

    return set(
        value.split()
    )


def create_hard_negative_pairs(
    dataset_a,
    dataset_b,
    positive_pairs,
):

    positive_set = set(
        zip(
            positive_pairs["record_id_a"],
            positive_pairs["record_id_b"],
        )
    )

    rows = []

    # Build pairs sharing city/state and
    # having similar company-name structure.
    for _ in range(N_NEGATIVES):

        while True:

            a_idx = random.randrange(
                len(dataset_a)
            )

            b_idx = random.randrange(
                len(dataset_b)
            )

            a = dataset_a.iloc[a_idx]
            b = dataset_b.iloc[b_idx]

            pair = (
                a["record_id"],
                b["record_id"],
            )

            if pair in positive_set:
                continue

            # Encourage hard negatives:
            # same city, same state, or similar
            # industry/prefix tokens.
            same_city = (
                a["city"] == b["city"]
            )

            same_state = (
                a["state"] == b["state"]
            )

            a_tokens = set(
                clean_tokens(
                    a["company_name"]
                )
            )

            b_tokens = set(
                clean_tokens(
                    b["company_name"]
                )
            )

            shared_tokens = len(
                a_tokens & b_tokens
            )

            if (
                shared_tokens >= 2
                or (
                    same_city
                    and same_state
                    and random.random() < 0.5
                )
                or random.random() < 0.05
            ):
                break

        rows.append(
            {
                "record_id_a":
                    a["record_id"],

                "record_id_b":
                    b["record_id"],

                "same_entity": 0,
            }
        )

    return pd.DataFrame(rows)


def main():

    data_dir = Path("data")

    data_dir.mkdir(
        exist_ok=True
    )

    print("=" * 60)
    print("GENERATING HARD ENTITY RESOLUTION DATASET")
    print("=" * 60)

    canonical = build_canonical()

    dataset_a = create_dataset_a(
        canonical
    )

    dataset_b = create_dataset_b(
        canonical
    )

    positive_pairs = create_positive_pairs(
        dataset_a,
        dataset_b,
    )

    negative_pairs = create_hard_negative_pairs(
        dataset_a,
        dataset_b,
        positive_pairs,
    )

    dataset_a.to_csv(
        data_dir / "dataset_a.csv",
        index=False,
    )

    dataset_b.to_csv(
        data_dir / "dataset_b.csv",
        index=False,
    )

    positive_pairs.to_csv(
        data_dir / "ground_truth.csv",
        index=False,
    )

    negative_pairs.to_csv(
        data_dir / "negative_pairs.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("DATASET CREATED")
    print("=" * 60)

    print(
        f"\nUnique entities : {N_ENTITIES:,}"
    )

    print(
        f"Positive pairs  : "
        f"{len(positive_pairs):,}"
    )

    print(
        f"Negative pairs  : "
        f"{len(negative_pairs):,}"
    )

    print("\nExample A:")
    print(
        dataset_a.head(5).to_string(
            index=False
        )
    )

    print("\nExample B:")
    print(
        dataset_b.head(5).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()