# Logistic Regression using Scikit-learn

## Project Overview

This project demonstrates how to build a binary classification model using the Logistic Regression algorithm provided by Scikit-learn.

The model is trained on the Breast Cancer Wisconsin Dataset to predict whether a tumor is benign or malignant.

---

# Dataset

Dataset Name

Breast Cancer Wisconsin Dataset

Number of Samples

569

Number of Features

30

Target Classes

0 → Malignant

1 → Benign

---

# Libraries Used

- NumPy
- Scikit-learn

---

# Importing Libraries

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

### Explanation

| Library | Purpose |
|----------|---------|
| load_breast_cancer | Loads the dataset |
| LogisticRegression | Creates Logistic Regression model |
| train_test_split | Splits dataset |
| accuracy_score | Calculates model accuracy |

---

# Loading Dataset

```python
x, y = load_breast_cancer(return_X_y=True)
```

This function loads the Breast Cancer dataset.

```
x

↓

Input Features
```

```
y

↓

Target Labels
```

---

# Splitting Dataset

```python
x_train,x_test,y_train,y_test=train_test_split(
x,
y,
test_size=0.20,
random_state=23
)
```

80%

↓

Training Data

20%

↓

Testing Data

The model learns from the training dataset and is evaluated using the testing dataset.

---

# Creating the Model

```python
clf = LogisticRegression(
max_iter=10000,
random_state=0
)
```

## Parameters

### max_iter

Maximum number of iterations allowed for the optimization algorithm.

### random_state

Ensures reproducible results.

---

# Training the Model

```python
clf.fit(x_train,y_train)
```

The fit() method trains the Logistic Regression model using the training dataset.

Internally, Scikit-learn:

- Initializes weights
- Applies the Sigmoid Function
- Calculates loss
- Optimizes weights using numerical optimization
- Stops when convergence is reached

---

# Making Predictions

```python
clf.predict(x_test)
```

The trained model predicts whether each sample belongs to

```
Benign

or

Malignant
```

---

# Calculating Accuracy

```python
acc = accuracy_score(
y_test,
clf.predict(x_test)
)*100
```

Accuracy compares predicted labels with actual labels.

Formula

Accuracy = Correct Predictions / Total Predictions

---

# Displaying Output

```python
print(f"Logistic model accuracy is {acc:.2f}%")
```

Example Output

```
Logistic model accuracy is 97.36%
```

---

# Workflow

```
Load Dataset
      │
      ▼
Split Dataset
      │
      ▼
Create Logistic Regression Model
      │
      ▼
Train Model
      │
      ▼
Predict Test Data
      │
      ▼
Calculate Accuracy
      │
      ▼
Display Result
```

---

# Time Complexity

| Operation | Complexity |
|-----------|------------|
| Training | O(n × d × iterations) |
| Prediction | O(d) |

where

- n = Number of Samples
- d = Number of Features

---

# Advantages

- Easy to use
- Fast
- Good for binary classification
- Highly interpretable

---

# Limitations

- Assumes a linear decision boundary
- Less effective on complex non-linear datasets
- Sensitive to feature scaling

---

# Applications

- Disease Prediction
- Spam Detection
- Credit Risk Analysis
- Customer Churn Prediction
- Fraud Detection

---

# References

- Scikit-learn Documentation
- Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow

B.Tech CSE (AI & ML)

GLA University

GitHub: https://github.com/Clestialcosmos
