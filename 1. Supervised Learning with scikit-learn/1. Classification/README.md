# Supervised Learning with scikit-learn
## Segment 1 — Classification

*Comprehensive Revision Guide*

---

## Table of Contents

1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [Supervised Learning Fundamentals](#2-supervised-learning-fundamentals)
3. [Data Requirements and the scikit-learn Workflow](#3-data-requirements-and-the-scikit-learn-workflow)
4. [k-Nearest Neighbors (KNN)](#4-k-nearest-neighbors-knn)
5. [Measuring Model Performance](#5-measuring-model-performance)
6. [The Train/Test Split](#6-the-trainttest-split)
7. [Model Complexity, Overfitting, and Underfitting](#7-model-complexity-overfitting-and-underfitting)
8. [The Model Complexity Curve](#8-the-model-complexity-curve)
9. [Practical Code Patterns](#9-practical-code-patterns)
10. [Summary Cheat Sheet](#10-summary-cheat-sheet)

---

## 1. What is Machine Learning?

Machine learning is the process whereby computers **learn to make decisions from data without being explicitly programmed**. Rather than being given explicit rules, a machine learning model discovers patterns in data and uses those patterns to make predictions or decisions on new, unseen data.

### 1.1 Example Applications

Two representative examples introduced in the course:

- **Spam detection** — learning to predict whether an email is spam or not spam based on its content and sender.
- **Book clustering** — learning to group books into categories based on the words they contain, then assigning new books to one of the discovered clusters.

The first is an example of supervised learning; the second is an example of unsupervised learning.

### 1.2 Unsupervised Learning

Unsupervised learning uncovers hidden patterns and structures from **unlabeled** data — data where the categories or groupings are not known in advance. **Clustering** is the main branch of unsupervised learning. A business example: grouping customers into purchasing behavior segments without predefined categories.

> The course does *not* cover unsupervised learning — it is introduced only for contrast with supervised learning.

---

## 2. Supervised Learning Fundamentals

### 2.1 Definition

Supervised learning is a type of machine learning where the **values to be predicted are already known**, and a model is built with the aim of accurately predicting values of previously unseen data. It uses **features** (input variables) to predict the value of a **target variable** (the output).

Example: predicting a basketball player's position based on their points per game.

### 2.2 Naming Conventions

These terms are used interchangeably in the literature. Knowing both names is important for reading other materials.

| Course Term | Alternative Names |
|:---|:---|
| Feature | Predictor variable, independent variable |
| Target variable | Dependent variable, response variable |

### 2.3 The Two Types of Supervised Learning

**Classification** predicts the **label or category** of an observation. Example: predicting whether a bank transaction is fraudulent or not. When there are exactly two possible outcomes, this is called **binary classification**. The two outcomes are typically encoded as `1` (positive class, e.g., fraud) and `0` (negative class, e.g., not fraud).

**Regression** predicts **continuous values**. Example: predicting the price of a property given features such as number of bedrooms and size.

This segment focuses exclusively on **classification**.

### 2.4 Identifying the Target Variable — Exercise Example

The course uses the `churn_df` dataset to illustrate binary classification. The dataset columns include `account_length`, `total_day_charge`, `total_eve_charge`, `total_night_charge`, `total_intl_charge`, and `customer_service_calls`. The correct target variable for binary classification is `"churn"` — it represents whether a customer has left (churned) or stayed, which is a binary outcome. The other columns are continuous numeric values and serve as features.

---

## 3. Data Requirements and the scikit-learn Workflow

### 3.1 Prerequisites for Supervised Learning

Before running any scikit-learn model, the data must satisfy three requirements:

- **No missing values** — scikit-learn models will raise errors if `NaN` values are present.
- **Numeric format** — all features and the target must be expressed as numbers.
- **Correct data structure** — data must be stored as a **pandas DataFrame or Series**, or as a **NumPy array**.

This necessitates exploratory data analysis (EDA) using pandas methods for descriptive statistics and appropriate data visualizations before modelling begins.

### 3.2 The Universal scikit-learn Workflow

All supervised learning models in scikit-learn follow the **same four-step workflow**. This repeatability is one of scikit-learn's core design principles.

The correct order is tested directly in the drag-and-drop exercise:

1. **Import** the model class from the appropriate `sklearn` module.
2. **Instantiate** the model by creating an instance of the class.
3. **Fit** the model to the training data using `.fit(X, y)`.
4. **Predict** on new, unseen data using `.predict(X_new)`.

```python
from sklearn.module import Model    # Step 1: Import
model = Model()                     # Step 2: Instantiate
model.fit(X, y)                     # Step 3: Fit
model.predict(X_new)                # Step 4: Predict
```

### 3.3 Feature and Target Array Shapes

scikit-learn enforces specific shapes for `X` and `y`:

- **`X` (features)** must be a **2D array** — each column is a feature, each row is an observation.
- **`y` (target)** must be a **1D array** — one value per observation, same length as `X`.

Use the `.values` attribute on a pandas DataFrame or Series to convert to a NumPy array. Use `.shape` to verify dimensions before fitting.

---

## 4. k-Nearest Neighbors (KNN)

### 4.1 The Intuition

k-Nearest Neighbors (KNN) is the first classification algorithm covered in the course. The core idea is simple: to classify an unlabeled data point, find the `k` closest labeled data points in the feature space and let them **vote** on the label. The label chosen by the majority of those `k` neighbors is assigned to the new point. This is called **majority voting**.

**Example with k=3 vs. k=5:** Given a scatter plot with red and blue labeled points and one unlabeled black point:

- If `k=3`, two of the three closest neighbors are red → the point is classified as **red**.
- If `k=5`, three of the five closest neighbors are blue → the point is classified as **blue**.

The value of `k` directly controls the result, which is why choosing `k` carefully matters.

### 4.2 Decision Boundaries

KNN creates a **decision boundary** — a threshold in feature space that separates the regions where the model predicts one class from regions where it predicts another. In the telecom churn example, any customer in the gray-shaded region is predicted to churn; any customer in the red-shaded region is predicted not to churn. This boundary is used to classify all future unseen data points.

### 4.3 Fitting a KNN Classifier — Full Workflow

**Exercise example — k-Nearest Neighbors: Fit:**

```python
# Step 1: Import
from sklearn.neighbors import KNeighborsClassifier

# Step 2: Prepare feature and target arrays
y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Step 3: Instantiate with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Step 4: Fit to the data
knn.fit(X, y)
```

Key notes from the exercise:

- `y` is a **1D array** extracted from a single column using `.values`.
- `X` is a **2D array** extracted from multiple columns using double brackets `[[...]]` followed by `.values`.
- `account_length` is used as a feature because it indicates customer loyalty.
- `customer_service_calls` is used because frequent calls may signal dissatisfaction — both are meaningful predictors of churn.

### 4.4 Predicting on Unlabeled Data

**Exercise example — k-Nearest Neighbors: Predict:**

```python
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

# Use the fitted model to predict labels for new observations
y_pred = knn.predict(X_new)
print("Predictions: {}".format(y_pred))
```

- `X_new` must be a **2D NumPy array** with the same number of columns as the training `X`.
- `.predict()` returns one prediction per row in `X_new`.
- A prediction of `1` means the model predicts churn; `0` means no churn.
- Printing the shape of `X_new` first (`X_new.shape`) is good practice to verify the array is correctly formed.

---

## 5. Measuring Model Performance

### 5.1 What is Accuracy?

In classification, **accuracy** is the most commonly used performance metric. It is defined as:

```
Accuracy = Number of correct predictions / Total number of observations
```

A prediction is correct when the predicted label matches the true label.

### 5.2 Why Not Evaluate on Training Data?

It might seem natural to compute accuracy on the same data used to train the model. However, this gives an **overly optimistic estimate** — the model has already "seen" this data and may have memorized it rather than learning generalizable patterns. What we care about is how the model performs on **new, unseen data**. This is why a separate test set is required.

---

## 6. The Train/Test Split

### 6.1 The Concept

The standard approach is to split the available labeled data into two parts:

- **Training set** — used to fit the model (the model learns from this data).
- **Test set** — held out and used only to evaluate the model's performance after training.

Because the test set was never used during training, it gives an honest estimate of how well the model generalizes to new data.

### 6.2 Performing the Split with `train_test_split`

**Exercise example — Train/Test Split + Computing Accuracy:**

```python
from sklearn.model_selection import train_test_split

# Prepare feature and target arrays
X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing, 80% for training
    random_state=42,     # seed for reproducibility
    stratify=y           # preserve class proportions in both sets
)

# Instantiate and fit the model on training data only
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Compute and print accuracy on the test set
print(knn.score(X_test, y_test))
```

### 6.3 Key Parameters of `train_test_split`

| Parameter | Purpose | Notes |
|:---|:---|:---|
| `test_size` | Fraction of data used for the test set | Common values: `0.2` (20%) or `0.3` (30%) |
| `random_state` | Seed for the random number generator | Setting this ensures the same split every time the code is run — critical for reproducibility |
| `stratify` | Ensures class proportions are preserved in both splits | Set to `y`; if churn is 10% of all observations, both train and test sets will also contain approximately 10% churn |

The function returns **four arrays** that must be unpacked in this exact order: `X_train, X_test, y_train, y_test`.

### 6.4 The `.score()` Method

After fitting on `X_train` and `y_train`, use `.score(X_test, y_test)` to compute accuracy on the test set. The method internally calls `.predict()` and compares predictions to the true labels.

> **Important interpretation note:** An accuracy of 88% sounds high but must be evaluated in context. If the dataset has a 9-to-1 class imbalance (i.e., 90% of observations belong to one class), a model that always predicts the majority class would achieve 90% accuracy without learning anything useful. Always consider class balance when interpreting accuracy.

---

## 7. Model Complexity, Overfitting, and Underfitting

### 7.1 Interpreting k in KNN

The value of `k` (the number of neighbors) is a **hyperparameter** that directly controls model complexity:

- **Small k** → more complex model. The decision boundary is highly sensitive to individual data points. The model can capture fine-grained patterns but may also react to noise.
- **Large k** → simpler model. The decision boundary is smoother and less affected by any single observation. The model generalizes over a larger neighborhood.

### 7.2 Overfitting

**Overfitting** occurs when a model is too complex — it has learned the training data too well, including its noise and random fluctuations, rather than the underlying general pattern. An overfit model performs very well on training data but poorly on new, unseen test data.

### 7.3 Underfitting

**Underfitting** occurs when a model is too simple — it fails to detect the relationships that exist in the data. An underfit model performs poorly on both the training data and the test data. In KNN, very large values of `k` tend to produce underfitting because the model smooths over real patterns.

### 7.4 The Bias-Variance Trade-off in KNN Terms

| k Value | Model Type | Risk |
|:---|:---|:---|
| Very small (e.g., k=1) | High complexity | Overfitting — reacts to noise |
| Very large | Low complexity | Underfitting — misses real patterns |
| Optimal (e.g., k=13 in the course example) | Balanced | Best generalization to unseen data |

The goal is to find the `k` that achieves the best **test set accuracy** — not the best training set accuracy.

---

## 8. The Model Complexity Curve

### 8.1 Purpose

A model complexity curve plots the training and test accuracies across a range of `k` values. It is a diagnostic tool that helps you identify:

- The region of overfitting (small k, high train accuracy, low test accuracy).
- The region of underfitting (large k, both train and test accuracy plateau at a low level).
- The optimal `k` (where test accuracy peaks).

### 8.2 Building the Complexity Curve

**Exercise example — Overfitting and Underfitting:**

```python
# Create neighbors array
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies  = {}

for neighbor in neighbors:
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Fit the model
    knn.fit(X_train, y_train)

    # Compute accuracy on both sets and store with neighbor count as key
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor]  = knn.score(X_test,  y_test)

print(neighbors, '\n', train_accuracies, '\n', test_accuracies)
```

Key design choices: `np.arange(1, 13)` generates integers from 1 up to and **including** 12 (not 13). Both dictionaries use the neighbor count as the key and the accuracy score as the value.

### 8.3 Visualizing the Complexity Curve

**Exercise example — Visualizing Model Complexity:**

```python
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
```

Note: `.values()` is called on the dictionaries because `plt.plot()` requires an iterable of y-values, not a dictionary.

### 8.4 Reading the Complexity Curve

From the course example:

- As `k` increases beyond 15, both training and test accuracy **plateau** — this is the underfitting region.
- The **peak test accuracy** occurs at approximately **k=13 neighbors**, which is the optimal choice.
- At very small `k` values, the training accuracy is high but the test accuracy is lower — this is the overfitting region.

---

## 9. Practical Code Patterns

### Pattern 1: Full KNN Fit and Predict Workflow

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Prepare data
y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Instantiate, fit, and predict
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X, y)

X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

y_pred = knn.predict(X_new)
print("Predictions: {}".format(y_pred))
```

### Pattern 2: Train/Test Split and Accuracy Evaluation

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

### Pattern 3: Model Complexity Curve (Overfitting/Underfitting Diagnosis)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies  = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor]  = knn.score(X_test, y_test)

plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(),  label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
```

---

## 10. Summary Cheat Sheet

### 10.1 Core Concepts at a Glance

| Concept | Definition |
|:---|:---|
| Supervised learning | Learning from labeled data to predict values of unseen observations |
| Classification | Supervised learning for predicting discrete labels/categories |
| Regression | Supervised learning for predicting continuous numeric values |
| Binary classification | Classification with exactly two possible output labels (e.g., 0 and 1) |
| Feature (`X`) | Input variable(s) used to make predictions |
| Target variable (`y`) | Output variable the model is trained to predict |
| KNN | Algorithm that classifies by majority vote among the k nearest neighbors |
| Decision boundary | The threshold in feature space that separates predicted classes |
| Accuracy | Correct predictions / total observations |
| Train/test split | Dividing labeled data into a fitting set and an evaluation set |
| Overfitting | Model too complex; memorizes training noise; poor on test data |
| Underfitting | Model too simple; misses real patterns; poor on both sets |
| Model complexity curve | Plot of train and test accuracy vs. k to find the optimal hyperparameter |

### 10.2 The scikit-learn Workflow (in Order)

1. `from sklearn.module import Model`
2. `model = Model(hyperparameters)`
3. `model.fit(X_train, y_train)`
4. `model.predict(X_new)` or `model.score(X_test, y_test)`

### 10.3 `train_test_split` Parameter Reference

| Parameter | What It Does | Recommended Value |
|:---|:---|:---|
| `test_size` | Proportion of data held out for testing | `0.2` to `0.3` (20–30%) |
| `random_state` | Random seed for reproducibility | Any integer (e.g., `42`) |
| `stratify` | Preserves class proportions in both splits | Always set to `y` for classification |

### 10.4 KNN Hyperparameter Guidance

| k Value | Effect | Risk |
|:---|:---|:---|
| Very small (k=1 or 2) | Highly complex, sensitive to noise | Overfitting |
| Moderate (optimal range) | Balanced complexity | Best generalization |
| Very large | Overly simple, smooth boundary | Underfitting |

> Use the model complexity curve to identify the optimal `k` — it is the value where **test accuracy peaks**.

### 10.5 Common Mistakes to Avoid

| Mistake | Better Approach |
|:---|:---|
| Evaluating accuracy on training data | Always evaluate on the held-out test set using `.score(X_test, y_test)` |
| Forgetting `stratify=y` in the split | Always stratify for classification to preserve class proportions |
| Passing `X_new` as a 1D array to `.predict()` | `X_new` must always be a **2D array** with shape `(n_observations, n_features)` |
| Not setting `random_state` | Always set it for reproducibility — without it, the split changes every run |
| Interpreting high accuracy as always good | Check class balance — high accuracy can be misleading with imbalanced labels |
| Extracting a single feature column as 2D without `[[]]` | Use double brackets `df[["column"]].values` for a single feature to keep it 2D |
| Choosing `k` based on training accuracy | Choose `k` based on the **test accuracy** peak in the complexity curve |

---

*End of Revision Guide — Segment 1: Classification*
