# Entity Resolution Engine

## Project Overview

This project is an **Entity Resolution / Entity Matching system** that identifies whether two records belong to the same real-world entity.

The main idea is to compare records from two different datasets where the information may not be exactly the same. For example, the same person or business may have differences in name formatting, address, phone number, email, or website.

The project uses feature engineering and a machine learning model to calculate the probability that two records are a match.

---

## What This Project Does

The system takes two records and compares different fields such as:

* Name
* Address
* City
* State
* Postal code
* Phone
* Email
* Website

Instead of checking only exact values, the project calculates similarity scores between the fields.

These similarities are then given to a machine learning model, which predicts whether the records are:

* **Match**
* **Non-Match**

---

## Dataset

The dataset contains **6,000 record pairs**.

```text
Total pairs       : 6,000
Positive examples : 1,000
Negative examples : 5,000
Training data     : 4,800
Testing data      : 1,200
```

The dataset is imbalanced because there are more non-matching pairs than matching pairs.

---

## Features

The final feature dataset contains 20 features.

### Name Features

* `name_similarity`
* `name_fuzzy`
* `name_token_similarity`
* `name_token_sort`

### Address Features

* `address_similarity`
* `address_token_similarity`

### Location Features

* `city_similarity`
* `state_similarity`
* `postal_similarity`

### Contact Features

* `phone_similarity`
* `email_similarity`
* `website_similarity`
* `website_domain_exact`

### Additional Features

* `missing_field_count`
* `strong_identifier_count`
* `identity_evidence_score`
* `geographic_evidence_score`
* `geographic_only_match`
* `name_contact_agreement`
* `strong_conflict_count`

These additional features were added to make the model better at handling cases where some fields are missing or where geographic information matches but identity-related information does not.

---

## Machine Learning Model

The model is trained using the engineered features.

The trained model is saved here:

```text
models/entity_matcher.pkl
```

The model package contains the trained model and the feature information required by the application.

---

## Model Performance

The final model was tested on 1,200 test pairs.

The main results were:

```text
Precision : 0.6532
Recall    : 0.9700
F1 Score  : 0.7807
ROC-AUC   : 0.9735
```

Confusion matrix:

```text
[[897 103]
 [  6 194]]
```

The model has high recall, which means it is able to identify most of the actual matching records.

---

## Threshold Analysis

I also tested different probability thresholds instead of directly using the default 0.50 threshold.

The final threshold analysis gave:

```text
Threshold : 0.70
Precision : 0.7287
Recall    : 0.9000
F1 Score  : 0.8054
False Pos. : 67
```

Based on the F1 score, I use **0.70** as the decision threshold in the application.

The threshold analysis is saved in:

```text
data/threshold_results.csv
```

---

## Error Analysis

I performed separate error analysis to understand where the model was making incorrect predictions.

At the selected threshold:

```text
False positives : 67
False negatives : 20
```

The analysis showed that many false positives had:

* Very high city similarity
* High state similarity
* High geographic evidence
* Low name similarity
* Low contact similarity
* No strong identifiers

This showed that matching geographic information alone should not be enough to consider two records the same.

The detailed error analysis is saved in:

```text
data/error_analysis.csv
```

---

## Project Structure

```text
entity-resolution-engine/
│
├── app.py
│
├── data/
│   ├── dataset_a.csv
│   ├── dataset_b.csv
│   ├── ground_truth.csv
│   ├── negative_pairs.csv
│   ├── features.csv
│   ├── matching_results.csv
│   ├── evaluation_results.csv
│   ├── threshold_results.csv
│   └── error_analysis.csv
│
├── models/
│   └── entity_matcher.pkl
│
├── src/
│   ├── build_features.py
│   ├── train_model.py
│   ├── threshold_analysis.py
│   └── analyze_errors.py
│
└── README.md
```

---

## How to Run the Project

### 1. Create/activate the environment

I used Anaconda for this project.

```bash
conda activate base
```

### 2. Build the features

```bash
python src/build_features.py
```

This creates:

```text
data/features.csv
```

### 3. Train the model

```bash
python src/train_model.py
```

This creates:

```text
models/entity_matcher.pkl
```

### 4. Run threshold analysis

```bash
python src/threshold_analysis.py
```

This tests different thresholds and saves the results to:

```text
data/threshold_results.csv
```

### 5. Run error analysis

```bash
python src/analyze_errors.py
```

This generates detailed information about false positives and false negatives.

### 6. Run the application

```bash
streamlit run app.py
```

The Streamlit application provides the interface for testing entity pairs.

---

## Example

### Record A

```text
Name: John Smith
Address: 123 Main Street
City: New York
State: NY
Postal: 10001
Phone: 2125551234
Email: john.smith@gmail.com
Website: www.johnsmith.com
```

### Record B

```text
Name: John Smith
Address: 123 Main St.
City: New York
State: NY
Postal: 10001
Phone: (212) 555-1234
Email: john.smith@gmail.com
Website: johnsmith.com
```

Even though some values are formatted differently, the records represent the same entity.

The application calculates the feature similarities and gives a match probability.

---

## Important Implementation Detail

The trained model is saved using `joblib`.

Therefore, the application loads the model using:

```python
joblib.load()
```

and not:

```python
pickle.load()
```

This is important because the saved model contains a scikit-learn pipeline.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Fuzzy string matching
* Machine Learning
* Feature Engineering

---

## Main Learning From This Project

The main thing I learned from this project is that entity resolution is not just about finding similar names.

Different records can have:

* Different names but the same entity
* Similar names but different entities
* Missing information
* Different address formats
* Different phone formats
* Matching cities/states but different entities

Because of this, I used multiple features together instead of depending on one field.

I also used threshold analysis and error analysis to understand the model's behavior instead of only looking at accuracy.

---

## Final Result

The final system can:

1. Load two entity datasets.
2. Generate similarity features.
3. Train a machine learning model.
4. Predict the probability of a match.
5. Use a selected threshold for classification.
6. Analyze false positives and false negatives.
7. Provide an interactive Streamlit interface for testing.

The final model achieved a **ROC-AUC of 0.9735**, with an F1 score of **0.8054 at the selected 0.70 decision threshold**.
