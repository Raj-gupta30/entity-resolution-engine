import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# ENTITY RESOLUTION APP
# ============================================================

st.set_page_config(
    page_title="Entity Resolution Engine",
    page_icon="🔎",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join("models", "entity_matcher.pkl")

# Final threshold selected from threshold analysis
DEFAULT_THRESHOLD = 0.70

FEATURE_COLUMNS = [
    "name_similarity",
    "name_fuzzy",
    "name_token_similarity",
    "name_token_sort",
    "address_similarity",
    "address_token_similarity",
    "city_similarity",
    "state_similarity",
    "postal_similarity",
    "phone_similarity",
    "email_similarity",
    "website_similarity",
    "website_domain_exact",
    "missing_field_count",
    "strong_identifier_count",
    "identity_evidence_score",
    "geographic_evidence_score",
    "geographic_only_match",
    "name_contact_agreement",
    "strong_conflict_count",
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # Model was saved with joblib.dump()
    package = joblib.load(MODEL_PATH)

    if isinstance(package, dict):
        if "model" not in package:
            raise ValueError(
                "Model package does not contain a 'model' key."
            )

        model = package["model"]
        stored_features = package.get(
            "features",
            FEATURE_COLUMNS
        )

    else:
        model = package
        stored_features = FEATURE_COLUMNS

    return model, stored_features


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================

def sequence_similarity(a, b):
    """
    Character-level similarity using difflib.
    Returns value between 0 and 1.
    """
    from difflib import SequenceMatcher

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def token_similarity(a, b):
    """
    Token overlap similarity.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))

    if union == 0:
        return 0.0

    return intersection / union


def token_sort_similarity(a, b):
    """
    Sort tokens alphabetically before character comparison.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    a_sorted = " ".join(sorted(a.split()))
    b_sorted = " ".join(sorted(b.split()))

    return sequence_similarity(a_sorted, b_sorted)


def exact_similarity(a, b):
    """
    Exact normalized equality.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return 1.0 if a == b else 0.0


def domain_similarity(a, b):
    """
    Compare website domains.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    def extract_domain(value):
        value = value.replace("https://", "")
        value = value.replace("http://", "")
        value = value.split("/")[0]
        value = value.split(":")[0]
        return value.replace("www.", "")

    domain_a = extract_domain(a)
    domain_b = extract_domain(b)

    if not domain_a or not domain_b:
        return 0.0

    return 1.0 if domain_a == domain_b else 0.0


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(record_a, record_b):
    """
    Build the same 20 feature columns used by the final model.

    record_a and record_b should be dictionaries containing
    entity fields such as:

        name
        address
        city
        state
        postal
        phone
        email
        website
    """

    name_a = record_a.get("name", "")
    name_b = record_b.get("name", "")

    address_a = record_a.get("address", "")
    address_b = record_b.get("address", "")

    city_a = record_a.get("city", "")
    city_b = record_b.get("city", "")

    state_a = record_a.get("state", "")
    state_b = record_b.get("state", "")

    postal_a = record_a.get("postal", "")
    postal_b = record_b.get("postal", "")

    phone_a = record_a.get("phone", "")
    phone_b = record_b.get("phone", "")

    email_a = record_a.get("email", "")
    email_b = record_b.get("email", "")

    website_a = record_a.get("website", "")
    website_b = record_b.get("website", "")

    # --------------------------------------------------------
    # Basic similarities
    # --------------------------------------------------------

    name_similarity = sequence_similarity(name_a, name_b)
    name_fuzzy = sequence_similarity(name_a, name_b)
    name_token_similarity = token_similarity(name_a, name_b)
    name_token_sort = token_sort_similarity(name_a, name_b)

    address_similarity = sequence_similarity(
        address_a,
        address_b
    )

    address_token_similarity = token_similarity(
        address_a,
        address_b
    )

    city_similarity = exact_similarity(
        city_a,
        city_b
    )

    state_similarity = exact_similarity(
        state_a,
        state_b
    )

    postal_similarity = sequence_similarity(
        postal_a,
        postal_b
    )

    phone_similarity = sequence_similarity(
        phone_a,
        phone_b
    )

    email_similarity = sequence_similarity(
        email_a,
        email_b
    )

    website_similarity = sequence_similarity(
        website_a,
        website_b
    )

    website_domain_exact = domain_similarity(
        website_a,
        website_b
    )

    # --------------------------------------------------------
    # Missing fields
    # --------------------------------------------------------

    fields_a = [
        name_a,
        address_a,
        city_a,
        state_a,
        postal_a,
        phone_a,
        email_a,
        website_a,
    ]

    fields_b = [
        name_b,
        address_b,
        city_b,
        state_b,
        postal_b,
        phone_b,
        email_b,
        website_b,
    ]

    missing_field_count = sum(
        1
        for a, b in zip(fields_a, fields_b)
        if not normalize_text(a) or not normalize_text(b)
    )

    # --------------------------------------------------------
    # Strong identifiers
    # --------------------------------------------------------

    strong_identifier_count = 0

    if (
        normalize_text(phone_a)
        and normalize_text(phone_b)
        and normalize_text(phone_a) == normalize_text(phone_b)
    ):
        strong_identifier_count += 1

    if (
        normalize_text(email_a)
        and normalize_text(email_b)
        and normalize_text(email_a) == normalize_text(email_b)
    ):
        strong_identifier_count += 1

    if website_domain_exact == 1.0:
        strong_identifier_count += 1

    # --------------------------------------------------------
    # Identity evidence
    # --------------------------------------------------------

    identity_evidence_score = np.mean([
        name_similarity,
        name_fuzzy,
        name_token_similarity,
        name_token_sort,
        phone_similarity,
        email_similarity,
        website_similarity,
        website_domain_exact,
    ])

    # --------------------------------------------------------
    # Geographic evidence
    # --------------------------------------------------------

    geographic_evidence_score = np.mean([
        city_similarity,
        state_similarity,
        postal_similarity,
        address_similarity,
        address_token_similarity,
    ])

    # --------------------------------------------------------
    # Geographic-only match
    # --------------------------------------------------------

    geographic_only_match = 1.0 if (
        city_similarity >= 0.9
        and state_similarity >= 0.9
        and identity_evidence_score < 0.35
        and strong_identifier_count == 0
    ) else 0.0

    # --------------------------------------------------------
    # Name/contact agreement
    # --------------------------------------------------------

    name_contact_agreement = (
        name_fuzzy *
        np.mean([
            phone_similarity,
            email_similarity,
            website_similarity
        ])
    )

    # --------------------------------------------------------
    # Strong conflicts
    # --------------------------------------------------------

    strong_conflict_count = 0

    if (
        normalize_text(phone_a)
        and normalize_text(phone_b)
        and phone_similarity < 0.3
    ):
        strong_conflict_count += 1

    if (
        normalize_text(email_a)
        and normalize_text(email_b)
        and email_similarity < 0.3
    ):
        strong_conflict_count += 1

    if (
        normalize_text(website_a)
        and normalize_text(website_b)
        and website_similarity < 0.3
    ):
        strong_conflict_count += 1

    if (
        normalize_text(postal_a)
        and normalize_text(postal_b)
        and postal_similarity < 0.3
    ):
        strong_conflict_count += 1

    return {
        "name_similarity": name_similarity,
        "name_fuzzy": name_fuzzy,
        "name_token_similarity": name_token_similarity,
        "name_token_sort": name_token_sort,
        "address_similarity": address_similarity,
        "address_token_similarity": address_token_similarity,
        "city_similarity": city_similarity,
        "state_similarity": state_similarity,
        "postal_similarity": postal_similarity,
        "phone_similarity": phone_similarity,
        "email_similarity": email_similarity,
        "website_similarity": website_similarity,
        "website_domain_exact": website_domain_exact,
        "missing_field_count": missing_field_count,
        "strong_identifier_count": strong_identifier_count,
        "identity_evidence_score": identity_evidence_score,
        "geographic_evidence_score": geographic_evidence_score,
        "geographic_only_match": geographic_only_match,
        "name_contact_agreement": name_contact_agreement,
        "strong_conflict_count": strong_conflict_count,
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_match(record_a, record_b, model, feature_names):
    features = build_features(record_a, record_b)

    # Make sure feature order exactly matches training
    X = pd.DataFrame(
        [[features.get(col, 0.0) for col in feature_names]],
        columns=feature_names
    )

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    prediction = probability >= DEFAULT_THRESHOLD

    return probability, prediction, features


# ============================================================
# DISPLAY
# ============================================================

def show_result(probability, prediction, features):

    st.subheader("Matching Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Match Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Decision",
            "MATCH" if prediction else "NON-MATCH"
        )

    with col3:
        st.metric(
            "Threshold",
            f"{DEFAULT_THRESHOLD:.2f}"
        )

    if prediction:
        st.success(
            f"✓ MATCH — probability {probability:.2%} "
            f"is above the {DEFAULT_THRESHOLD:.2f} threshold."
        )
    else:
        st.error(
            f"✗ NON-MATCH — probability {probability:.2%} "
            f"is below the {DEFAULT_THRESHOLD:.2f} threshold."
        )

    st.subheader("Feature Analysis")

    feature_df = pd.DataFrame(
        {
            "Feature": list(features.keys()),
            "Value": list(features.values())
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.title("🔎 Entity Resolution Engine")

    st.markdown(
        """
        Compare two entity records and determine whether they
        represent the **same real-world entity**.

        The application uses the trained machine-learning model
        with the final validated decision threshold of **0.70**.
        """
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:
        model, feature_names = load_model()

    except Exception as e:
        st.error(f"Unable to load model: {e}")
        st.stop()

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    st.sidebar.header("Model Information")

    st.sidebar.write(
        "**Model:** Entity Matcher"
    )

    st.sidebar.write(
        "**Decision Threshold:** 0.70"
    )

    st.sidebar.write(
        "**Test Precision:** 72.87%"
    )

    st.sidebar.write(
        "**Test Recall:** 90.00%"
    )

    st.sidebar.write(
        "**Test F1:** 80.54%"
    )

    st.sidebar.write(
        "**ROC-AUC:** 97.35%"
    )

    # --------------------------------------------------------
    # Input records
    # --------------------------------------------------------

    st.header("Enter Entity Records")

    col_a, col_b = st.columns(2)

    with col_a:

        st.subheader("Record A")

        a_name = st.text_input(
            "Name",
            key="a_name"
        )

        a_address = st.text_input(
            "Address",
            key="a_address"
        )

        a_city = st.text_input(
            "City",
            key="a_city"
        )

        a_state = st.text_input(
            "State",
            key="a_state"
        )

        a_postal = st.text_input(
            "Postal Code",
            key="a_postal"
        )

        a_phone = st.text_input(
            "Phone",
            key="a_phone"
        )

        a_email = st.text_input(
            "Email",
            key="a_email"
        )

        a_website = st.text_input(
            "Website",
            key="a_website"
        )

    with col_b:

        st.subheader("Record B")

        b_name = st.text_input(
            "Name",
            key="b_name"
        )

        b_address = st.text_input(
            "Address",
            key="b_address"
        )

        b_city = st.text_input(
            "City",
            key="b_city"
        )

        b_state = st.text_input(
            "State",
            key="b_state"
        )

        b_postal = st.text_input(
            "Postal Code",
            key="b_postal"
        )

        b_phone = st.text_input(
            "Phone",
            key="b_phone"
        )

        b_email = st.text_input(
            "Email",
            key="b_email"
        )

        b_website = st.text_input(
            "Website",
            key="b_website"
        )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    if st.button(
        "🔍 Compare Records",
        type="primary",
        use_container_width=True
    ):

        record_a = {
            "name": a_name,
            "address": a_address,
            "city": a_city,
            "state": a_state,
            "postal": a_postal,
            "phone": a_phone,
            "email": a_email,
            "website": a_website,
        }

        record_b = {
            "name": b_name,
            "address": b_address,
            "city": b_city,
            "state": b_state,
            "postal": b_postal,
            "phone": b_phone,
            "email": b_email,
            "website": b_website,
        }

        try:

            probability, prediction, features = predict_match(
                record_a,
                record_b,
                model,
                feature_names
            )

            st.divider()

            show_result(
                probability,
                prediction,
                features
            )

        except Exception as e:
            st.error(
                f"Prediction failed: {e}"
            )

            st.info(
                "Check that the input fields and feature "
                "definitions match those used during training."
            )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "Entity Resolution Engine • Final ML model • "
        "Decision threshold: 0.70"
    )


if __name__ == "__main__":
    main()