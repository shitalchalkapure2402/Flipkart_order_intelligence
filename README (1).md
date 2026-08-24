# Flipkart Order Intelligence & Support Assistant

## Project Overview

This capstone project builds a machine-learning based **Flipkart Order Intelligence & Support Assistant**.

**Part 1** focuses on predicting whether an order is likely to be returned.

Target:

- `returned = 1` → Order returned
- `returned = 0` → Order not returned

---

## Part 1 — Workflow

1. Dataset generation and verification
2. Basic EDA
3. Missing-value analysis
4. Stratified train/test split
5. Preprocessing
6. DummyClassifier baseline
7. Logistic Regression
8. Logistic Regression threshold tuning
9. Random Forest with GridSearchCV
10. Feature importance
11. Permutation importance
12. Subgroup analysis
13. Save final Random Forest model

---

## Dataset

The dataset contains:

- **6000 rows**
- **13 columns**
- Target variable: `returned`

Main columns:

`order_id`, `product_category`, `price_inr`, `discount_pct`, `payment_method`, `customer_tenure_days`, `num_previous_orders`, `num_previous_returns`, `delivery_distance_km`, `delivery_days`, `rating_given`, `is_weekend_order`, `returned`

`order_id` is used only as an identifier and is removed before model training.

---

## Dataset Verification

### Overall results

- Total rows: **6000**
- Overall return rate: **22.75%**
- Missing `rating_given`: **13.05%**

### Return rate by product category

| Product Category | Return Rate |
|---|---:|
| Apparel | 26.43% |
| Footwear | 25.96% |
| Beauty | 20.03% |
| Home | 19.15% |
| Electronics | 18.69% |

### Return rate by payment method

| Payment Method | Return Rate |
|---|---:|
| COD | 30.75% |
| Prepaid_Card | 16.82% |
| Prepaid_UPI | 16.92% |
| Wallet | 17.85% |

---

## Missingness Analysis

`rating_given` has missing values.

The missingness pattern is classified as **MAR (Missing At Random)**.

Observed missingness:

- COD: **22.83%**
- Non-COD: **6.06%**
- Gap: **16.77 percentage points**

### Why MAR?

The missingness of `rating_given` depends on the observed `payment_method` column. Therefore it is not MCAR. The generator does not make missingness depend on the unobserved rating value itself, so it is treated as MAR.

---

## EDA

Basic EDA was performed before modelling:

- Dataset shape
- First rows
- Data types
- Missing values
- Target distribution
- Return rate by product category
- Return rate by payment method
- Average price by return status
- Average discount by return status
- Price distribution
- Missing rating percentage by payment method

The EDA provides an initial understanding of the dataset before preprocessing and model training.

---

# Preprocessing

The data is divided using a **stratified 80/20 train-test split** with `random_state=42`.

```text
Training data: 4800 rows
Testing data:  1200 rows
```

### Numerical features

- Median imputation for missing values
- StandardScaler for scaling

### Categorical features

- Most-frequent imputation
- One-hot encoding with `handle_unknown="ignore"`

`ColumnTransformer` and `Pipeline` are used so that preprocessing is fitted only on training data.

---

# Task 4 — DummyClassifier

The baseline uses:

```python
DummyClassifier(strategy="most_frequent")
```

### Results

| Metric | Result |
|---|---:|
| Accuracy | 77.25% |
| F1-score for returned=1 | 0.0000 |

The DummyClassifier predicts the majority class. Therefore, it can achieve high accuracy while detecting none of the returned orders.

### Key lesson

**High accuracy does not necessarily mean a useful classification model when the target classes are imbalanced.**

---

# Task 5 — Logistic Regression

Logistic Regression uses:

```python
class_weight="balanced"
```

### Threshold = 0.50

| Metric | Result |
|---|---:|
| Accuracy | 59.17% |
| F1 | 0.3921 |
| Recall | 57.88% |
| Precision | 29.64% |
| ROC-AUC | 0.6253 |

Although accuracy is lower than the DummyClassifier, Logistic Regression is more useful because it can identify returned orders.

## Threshold tuning

Thresholds from **0.10 to 0.90** were tested with a step of **0.01**.

Best threshold:

```text
0.44
```

Results:

```text
F1        = 0.4091
Recall    = 75.82%
Precision = 28.01%
```

Recall improved from **57.88% to 75.82%**, an improvement of **17.95 percentage points**.

### Business interpretation

Lowering the threshold makes the model more willing to flag an order as risky. This improves recall but can increase false positives, which is the trade-off between catching more returns and incorrectly flagging non-returned orders.

---

# Task 6 — Random Forest

Random Forest uses:

```python
class_weight="balanced"
random_state=42
```

GridSearchCV tests:

```text
n_estimators = [100, 200]
max_depth = [6, 10, None]
```

Five-fold `StratifiedKFold` cross-validation is used with ROC-AUC scoring.

### Best parameters

```text
n_estimators = 100
max_depth = 6
```

### Results

| Metric | Result |
|---|---:|
| Best CV ROC-AUC | 0.6178 |
| Test ROC-AUC | 0.6143 |
| CV/Test difference | 0.0036 |

The small CV/test difference indicates reasonably consistent performance on unseen data.

---

# Random Forest Threshold

The Random Forest uses its own `predict_proba()` output to find the F1-maximising threshold.

```text
t*_rf = 0.47
```

At this threshold:

| Metric | Result |
|---|---:|
| F1 | 0.4030 |
| Recall | 58.61% |
| Precision | 30.71% |

The Logistic Regression threshold is **not reused** for the Random Forest.

---

# Task 7 — Feature Importance

Top five impurity-based features:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | `payment_method_COD` | 0.1665 |
| 2 | `price_inr` | 0.1371 |
| 3 | `customer_tenure_days` | 0.1074 |
| 4 | `delivery_distance_km` | 0.0972 |
| 5 | `discount_pct` | 0.0890 |

### Interpretation

- **payment_method_COD:** Payment method helps distinguish different return-risk levels.
- **price_inr:** Order price contains information associated with return risk.
- **customer_tenure_days:** Customer tenure can help distinguish different risk levels.
- **delivery_distance_km:** It appears important by impurity importance, but permutation importance shows little independent predictive contribution.
- **discount_pct:** Discount level provides information useful for distinguishing return-risk levels.

## Impurity vs permutation importance

| Feature | Impurity | Permutation |
|---|---:|---:|
| payment_method | 0.1665 | 0.0975 |
| price_inr | 0.1371 | 0.0080 |
| customer_tenure_days | 0.1074 | -0.0052 |
| delivery_distance_km | 0.0972 | -0.0027 |
| discount_pct | 0.0890 | -0.0029 |

`customer_tenure_days` and `delivery_distance_km` show substantial drops under permutation importance.

### Required explanation

> Impurity-based importance can overrate a noisy continuous feature because it has many possible split points, giving decision trees more opportunities to find splits that appear useful even when the feature has weak real predictive signal.

---

# Task 8 — Subgroup Analysis

The final Random Forest threshold `t*_rf = 0.47` was used.

## Product category

| Category | Recall | Precision |
|---|---:|---:|
| Apparel | 58.00% | 30.85% |
| Beauty | 61.29% | 44.19% |
| Electronics | 46.15% | 27.91% |
| Footwear | 62.50% | 36.08% |
| Home | 70.59% | 22.43% |

Electronics has the lowest recall among product categories at **46.15%**.

## Payment method

| Payment Method | Recall | Precision |
|---|---:|---:|
| COD | 96.13% | 31.57% |
| Prepaid_Card | 6.12% | 25.00% |
| Prepaid_UPI | 12.50% | 31.58% |
| Wallet | 9.52% | 11.11% |

### Weakest subgroup

**Prepaid_Card** is the weakest subgroup, with only **6.12% recall** compared with **58.61% overall recall**.

### Specific intervention

A specific next step is to test a **lower decision threshold for Prepaid_Card orders** instead of applying the same global threshold of `0.47` to every payment method.

The subgroup threshold should be selected using validation data while monitoring recall, precision, F1 and false-positive rate.

The goal is to improve detection of actual Prepaid_Card returns while keeping false positives acceptable.

---

# Model Comparison

| Model | Accuracy | F1 | Recall | Precision | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| DummyClassifier | 77.25% | 0.0000 | 0.00% | 0.00% | — |
| Logistic Regression | 59.17% | 0.3921 | 57.88% | 29.64% | 0.6253 |
| Random Forest | — | 0.4030 | 58.61% | 30.71% | 0.6143 |

The DummyClassifier has higher accuracy but is not useful for detecting returns. Logistic Regression and Random Forest provide meaningful return-risk predictions.

---

# Project Structure

```text
Flipkart-Order-Intelligence/
│
├── generate_orders.py
├── orders_dataset.csv
├── README.md
├── requirements.txt
│
├── part1/
│   └── part1_model.py
│
├── models/
│   ├── return_risk_model.pkl
│   └── return_risk_threshold.json
│
├── part2/
└── part3/
```

---

# Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scriptsctivate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Part 1

From the project root:

```bash
python part1/part1_model.py
```

The script trains the models and saves:

```text
models/return_risk_model.pkl
models/return_risk_threshold.json
```

---

# Saved Model

The final Random Forest is saved as:

```text
models/return_risk_model.pkl
```

The saved pipeline contains:

```text
Preprocessing
     +
Random Forest
```

The selected Random Forest threshold is saved separately in:

```text
models/return_risk_threshold.json
```

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

---

# Learning Outcomes

This project demonstrates:

- Data loading and validation
- Basic EDA
- Missing-value analysis
- MAR missingness
- Train/test splitting
- Stratification
- Numerical imputation
- Categorical imputation
- One-hot encoding
- Feature scaling
- Pipelines
- ColumnTransformer
- DummyClassifier
- Logistic Regression
- Classification metrics
- Probability threshold tuning
- Random Forest
- GridSearchCV
- Stratified cross-validation
- ROC-AUC
- Impurity-based feature importance
- Permutation importance
- Subgroup analysis
- Model persistence

---

# Conclusion

The project demonstrates why a majority-class baseline can have high accuracy while being unable to detect returned orders. Logistic Regression provides meaningful return detection, and threshold tuning substantially improves recall. The tuned Random Forest provides the final persisted model required for the project.

The subgroup analysis identifies a major recall gap for Prepaid_Card orders. A future improvement is to test a payment-method-specific threshold for this subgroup using validation data.

This project uses a synthetic dataset for learning and capstone purposes. Results should not be interpreted as actual Flipkart production performance.
