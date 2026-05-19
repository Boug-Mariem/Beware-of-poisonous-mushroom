# Beware of Poisonous Mushrooms

A machine learning mini-project to predict whether a mushroom is **edible or poisonous** based on its physical characteristics.

---

## Overview

This project explores a mushroom dataset and builds classification models to distinguish between edible (`e`) and poisonous (`p`) mushrooms using two supervised learning algorithms:

- **K-Nearest Neighbors (KNN)**
- **Decision Tree**

---

## Dataset

- **Format:** CSV file
- **Shape:** 8124 rows × 23 columns
- **Target column:** `class` — `e` (edible) or `p` (poisonous)
- **Features:** Physical characteristics such as cap shape, cap color, odor, gill attachment, gill size, stalk shape, veil type, ring type, spore print color, habitat, and more.

---

## Project Pipeline

### 1. Data Loading
Raw data is loaded from a CSV file. Column names are assigned manually for clarity.

### 2. Handling Missing Values
Missing values were found only in the `stalk-root` column and replaced with the **most frequent value (mode)**.

### 3. Data Visualization
- **Count plots** for each feature to explore value distributions.
- **Mosaic plots** to visualize the relationship between each feature and the target class.

### 4. Categorical Encoding
All categorical variables (letter-coded) were converted to integers using `LabelEncoder` to make them compatible with ML algorithms.

### 5. Feature Selection (Chi-squared Test)
A Chi² test was used to measure the statistical dependency between each feature and the target. Features with low scores were dropped:
- Removed: `cap-color`, `veil-color`, `gill-attachment`, `veil-type`, `odor`

### 6. Train/Test Split
- **Training set:** 80%
- **Test set:** 20%

---

## Models

### K-Nearest Neighbors (KNN)

- Best `k`: **1** (minimizes false negatives — predicting a poisonous mushroom as edible)
- Best distance metric: **Manhattan** (cross-validation score = 1.0000)
- Cross-validation (5-fold): **1.0** on all folds
- Test accuracy: **1.0 (100%)**

| Metric | Class 0 (Poisonous) | Class 1 (Edible) |
|--------|---------------------|------------------|
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| F1-Score | 1.00 | 1.00 |

### Decision Tree

- Hyperparameter tuning via **GridSearchCV**
- Best parameters: `max_depth=10`, `min_samples_split=2`, `min_samples_leaf=1`
- Post-pruning via **cost-complexity pruning** (`ccp_alpha`): best value = `0.0`
- Test accuracy: **1.0 (100%)**
- AUC (ROC curve): **1.00**

---

## Results Comparison

| Model | Test Accuracy | Cross-Val Mean | F1-Score |
|-------|---------------|----------------|----------|
| KNN | 1.0 | 1.0 | 1.0 |
| Decision Tree | 1.0 | 1.0 | 1.0 |

Both models achieved **perfect classification** with no false positives or false negatives.

---

## Pipelines

Full preprocessing + training pipelines were built for both models using `sklearn.pipeline.Pipeline`:

1. Drop irrelevant columns
2. Impute missing values (most frequent strategy)
3. Encode categorical variables with `OrdinalEncoder`
4. Train KNN (k=1, Manhattan) or Decision Tree (max_depth=10, ccp_alpha=0.0)

---

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- statsmodels (mosaic plots)

---

## 👥 Authors
- **Mariem Boughizene**
- **Mohsen Allani**
