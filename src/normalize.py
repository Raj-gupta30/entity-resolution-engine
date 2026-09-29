import re
import unicodedata


COMPANY_SUFFIXES = [
    "incorporated",
    "corporation",
    "private limited",
    "pvt ltd",
    "limited",
    "company",
    "corp",
    "inc",
    "ltd",
    "llc",
]


def normalize_text(value):
    if value is None:
        return ""

    value = str(value)

    value = unicodedata.normalize("NFKD", value)

    value = value.lower()

    value = value.strip()

    value = re.sub(r"[^\w\s]", " ", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_company_name(value):
    value = normalize_text(value)

    for suffix in COMPANY_SUFFIXES:
        pattern = rf"\b{re.escape(suffix)}\b"
        value = re.sub(pattern, "", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_phone(value):
    if value is None:
        return ""

    digits = re.sub(r"\D", "", str(value))

    if len(digits) > 10:
        digits = digits[-10:]

    return digits


def normalize_email(value):
    if value is None:
        return ""

    return str(value).strip().lower()


def normalize_website(value):
    if value is None:
        return ""

    value = str(value).strip().lower()

    value = re.sub(r"^https?://", "", value)

    value = re.sub(r"^www\.", "", value)

    value = value.rstrip("/")

    return value


def normalize_address(value):
    value = normalize_text(value)

    replacements = {
        " street ": " st ",
        " road ": " rd ",
        " avenue ": " ave ",
        " boulevard ": " blvd ",
        " drive ": " dr ",
        " lane ": " ln ",
        " california ": " ca ",
        " washington ": " wa ",
    }

    value = f" {value} "

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_record(record):
    return {
        "company_name": normalize_company_name(record.get("company_name")),
        "address": normalize_address(record.get("address")),
        "city": normalize_text(record.get("city")),
        "state": normalize_text(record.get("state")),
        "postal_code": normalize_text(record.get("postal_code")),
        "phone": normalize_phone(record.get("phone")),
        "email": normalize_email(record.get("email")),
        "website": normalize_website(record.get("website")),
    }