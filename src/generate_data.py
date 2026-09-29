import random
from pathlib import Path

import pandas as pd


random.seed(42)


PREFIXES = [
    "Northstar",
    "Blue Horizon",
    "Evergreen",
    "Summit",
    "Pioneer",
    "Crestview",
    "Silverline",
    "Clearwater",
    "Redwood",
    "Brightway",
    "Oakridge",
    "Westfield",
    "Lakeside",
    "Grandview",
    "Riverside",
    "Stonebridge",
    "Highland",
    "Golden",
    "Maple",
    "Cedar",
    "Ironwood",
    "Greenfield",
    "Brookside",
    "Sunrise",
    "Horizon",
    "Liberty",
    "Prime",
    "United",
    "Metro",
    "Vertex",
    "Apex",
    "Atlas",
    "Nova",
    "Quantum",
    "Sterling",
    "Pacific",
    "Atlantic",
    "Central",
    "National",
    "Urban",
]


INDUSTRIES = [
    "Logistics",
    "Foods",
    "Healthcare",
    "Medical",
    "Technology",
    "Software",
    "Consulting",
    "Financial",
    "Automotive",
    "Manufacturing",
    "Retail",
    "Pharmaceuticals",
    "Energy",
    "Construction",
    "Engineering",
    "Marketing",
    "Security",
    "Telecom",
    "Insurance",
    "Education",
]


SUFFIXES = [
    "Group",
    "Solutions",
    "Services",
    "Systems",
    "Industries",
    "Holdings",
    "Partners",
    "Enterprises",
    "International",
    "Corporation",
    "Company",
    "Networks",
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


def generate_company_name(index):
    """
    Generate a unique company name.
    """

    prefix = PREFIXES[
        index % len(PREFIXES)
    ]

    industry = INDUSTRIES[
        (index // len(PREFIXES))
        % len(INDUSTRIES)
    ]

    suffix = SUFFIXES[
        (index // (
            len(PREFIXES)
            * len(INDUSTRIES)
        ))
        % len(SUFFIXES)
    ]

    return (
        f"{prefix} "
        f"{industry} "
        f"{suffix}"
    )


def generate_phone(index):
    """
    Generate deterministic phone numbers.
    """

    number = (
        2000000000
        + index * 7919
    )

    return (
        f"+1-{str(number)[:3]}-"
        f"{str(number)[3:6]}-"
        f"{str(number)[6:]}"
    )


def generate_email(company_name):
    """
    Generate a company email.
    """

    domain = (
        company_name
        .lower()
        .replace(" ", "")
        .replace("&", "")
        .replace(".", "")
    )

    return (
        f"contact@{domain}.com"
    )


def generate_website(company_name):
    """
    Generate a company website.
    """

    domain = (
        company_name
        .lower()
        .replace(" ", "")
        .replace("&", "")
        .replace(".", "")
    )

    return (
        f"https://www.{domain}.com"
    )


def generate_address(index):
    """
    Generate a deterministic address.
    """

    number = (
        100
        + (index * 37) % 9900
    )

    street_names = [
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

    street = street_names[
        index % len(street_names)
    ]

    return (
        f"{number} {street}"
    )


def create_canonical_dataset(
    n=1000,
):

    records = []

    for i in range(n):

        city, state, postal = CITIES[
            i % len(CITIES)
        ]

        company_name = (
            generate_company_name(i)
        )

        record = {
            "entity_id":
                f"E{i:05d}",

            "company_name":
                company_name,

            "address":
                generate_address(i),

            "city":
                city,

            "state":
                state,

            "postal_code":
                postal,

            "phone":
                generate_phone(i),

            "email":
                generate_email(
                    company_name
                ),

            "website":
                generate_website(
                    company_name
                ),
        }

        records.append(record)

    return pd.DataFrame(records)


def corrupt_name(name):
    """
    Simulate real-world company-name variation.
    """

    replacements = [
        name,
        name.upper(),
        name.lower(),

        name.replace(
            "Corporation",
            "Corp."
        ),

        name.replace(
            "Company",
            "Co."
        ),

        name.replace(
            "International",
            "Intl."
        ),

        name.replace(
            "Group",
            "Grp."
        ),

        name.replace(
            "Services",
            "Svc."
        ),

        name.replace(
            "Solutions",
            "Solns."
        ),

        name.replace(
            "Industries",
            "Ind."
        ),

        name.replace(
            "Holdings",
            "Hldgs."
        ),

        name.replace(
            "Enterprises",
            "Ent."
        ),
    ]

    return random.choice(
        replacements
    ).strip()


def corrupt_address(address):

    replacements = [
        address,

        address.replace(
            "Street",
            "St."
        ),

        address.replace(
            "Road",
            "Rd."
        ),

        address.replace(
            "Avenue",
            "Ave."
        ),

        address.replace(
            "Boulevard",
            "Blvd."
        ),

        address.upper(),

        address.lower(),
    ]

    return random.choice(
        replacements
    )


def corrupt_phone(phone):

    digits = "".join(
        c
        for c in phone
        if c.isdigit()
    )

    styles = [
        phone,
        digits,

        f"({digits[1:4]}) "
        f"{digits[4:7]}-"
        f"{digits[7:]}",

        f"{digits[1:4]}-"
        f"{digits[4:7]}-"
        f"{digits[7:]}",
    ]

    return random.choice(styles)


def corrupt_email(email):

    return random.choice(
        [
            email,
            email.upper(),
            f" {email} ",
        ]
    ).strip()


def corrupt_website(website):

    domain = website.replace(
        "https://",
        "",
    )

    return random.choice(
        [
            website,
            domain,
            f"www.{domain}",
            f"http://{domain}",
            f"https://www.{domain}",
        ]
    )


def create_dataset_a(
    canonical,
):

    records = []

    for _, row in canonical.iterrows():

        records.append(
            {
                "record_id":
                    f"A{len(records):05d}",

                "company_name":
                    row["company_name"],

                "address":
                    row["address"],

                "city":
                    row["city"],

                "state":
                    row["state"],

                "postal_code":
                    row["postal_code"],

                "phone":
                    row["phone"],

                "email":
                    row["email"],

                "website":
                    row["website"],
            }
        )

    return pd.DataFrame(records)


def create_dataset_b(
    canonical,
):

    records = []

    for _, row in canonical.iterrows():

        records.append(
            {
                "record_id":
                    f"B{len(records):05d}",

                "company_name":
                    corrupt_name(
                        row["company_name"]
                    ),

                "address":
                    corrupt_address(
                        row["address"]
                    ),

                "city":
                    row["city"],

                "state":
                    row["state"],

                "postal_code":
                    row["postal_code"],

                "phone":
                    corrupt_phone(
                        row["phone"]
                    ),

                "email":
                    corrupt_email(
                        row["email"]
                    ),

                "website":
                    corrupt_website(
                        row["website"]
                    ),
            }
        )

    return pd.DataFrame(records)


def create_ground_truth(
    dataset_a,
    dataset_b,
):

    return pd.DataFrame(
        {
            "record_id_a":
                dataset_a["record_id"],

            "record_id_b":
                dataset_b["record_id"],

            "same_entity":
                1,
        }
    )


def create_negative_pairs(
    dataset_a,
    dataset_b,
    ground_truth,
    n=5000,
):

    positive_pairs = set(
        zip(
            ground_truth[
                "record_id_a"
            ],

            ground_truth[
                "record_id_b"
            ],
        )
    )

    records = []

    while len(records) < n:

        idx_a = random.randrange(
            len(dataset_a)
        )

        idx_b = random.randrange(
            len(dataset_b)
        )

        pair = (
            dataset_a.iloc[
                idx_a
            ]["record_id"],

            dataset_b.iloc[
                idx_b
            ]["record_id"],
        )

        if pair in positive_pairs:
            continue

        records.append(
            {
                "record_id_a":
                    pair[0],

                "record_id_b":
                    pair[1],

                "same_entity":
                    0,
            }
        )

    return pd.DataFrame(records)


def main():

    output_dir = Path("data")

    output_dir.mkdir(
        exist_ok=True
    )

    print(
        "Generating 1,000 unique businesses..."
    )

    canonical = (
        create_canonical_dataset(
            1000
        )
    )

    dataset_a = (
        create_dataset_a(
            canonical
        )
    )

    dataset_b = (
        create_dataset_b(
            canonical
        )
    )

    ground_truth = (
        create_ground_truth(
            dataset_a,
            dataset_b,
        )
    )

    negative_pairs = (
        create_negative_pairs(
            dataset_a,
            dataset_b,
            ground_truth,
            5000,
        )
    )

    dataset_a.to_csv(
        output_dir
        / "dataset_a.csv",
        index=False,
    )

    dataset_b.to_csv(
        output_dir
        / "dataset_b.csv",
        index=False,
    )

    ground_truth.to_csv(
        output_dir
        / "ground_truth.csv",
        index=False,
    )

    negative_pairs.to_csv(
        output_dir
        / "negative_pairs.csv",
        index=False,
    )

    print("\n" + "=" * 60)

    print(
        "DATA GENERATION COMPLETE"
    )

    print("=" * 60)

    print(
        f"\nUnique businesses: "
        f"{len(canonical):,}"
    )

    print(
        f"Positive pairs: "
        f"{len(ground_truth):,}"
    )

    print(
        f"Negative pairs: "
        f"{len(negative_pairs):,}"
    )

    print(
        "\nExample businesses:"
    )

    print(
        dataset_a[
            [
                "record_id",
                "company_name",
                "city",
            ]
        ].head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()