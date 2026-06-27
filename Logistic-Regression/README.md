# 📈 Logistic Regression

## Overview

Logistic Regression is a **supervised machine learning classification algorithm** used to predict the probability that an input belongs to a particular class. Unlike Linear Regression, Logistic Regression is designed for **classification tasks** and outputs values between **0 and 1** using the **Sigmoid Function**.

In this project, Logistic Regression is implemented using the **Breast Cancer Wisconsin Dataset** to classify tumors as **Malignant** or **Benign**.

---

# Table of Contents

1. Introduction
2. Problem Statement
3. Why Logistic Regression?
4. Mathematical Intuition
5. Sigmoid Function
6. Decision Boundary
7. Cost Function
8. Gradient Descent
9. Numerical Example
10. Algorithm Workflow
11. Python Implementation
12. Model Evaluation
13. Time & Space Complexity
14. Advantages
15. Limitations
16. Applications
17. References

---

# 1. Introduction

Logistic Regression is one of the most fundamental classification algorithms in Machine Learning.

Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts **probabilities**.

Example:

| Email | Spam |
|--------|------|
| Email A | Yes |
| Email B | No |

Instead of predicting a number like 53.2, Logistic Regression predicts

```
P(Spam)=0.92
```

meaning

```
92% probability of Spam
```

---

# 2. Problem Statement

Given various features describing a breast tumor, predict whether it is:

- Benign (0)
- Malignant (1)

This is a **Binary Classification Problem**.

---

# 3. Why Logistic Regression?

Suppose we use Linear Regression.

Output might become

```
1.7

or

-0.8
```

These values cannot represent probabilities.

We need outputs strictly between

```
0 and 1
```

Hence we use the Sigmoid Function.

---

# 4. Mathematical Intuition

The linear equation is

\[
z=w_1x_1+w_2x_2+\cdots+w_nx_n+b
\]

where

- x = input features
- w = weights
- b = bias

The value of **z** can range from

```
-∞ to +∞
```

To convert this into probability we apply the Sigmoid Function.

---

# 5. Sigmoid Function

The Sigmoid Function is

\[
\sigma(z)=\frac1{1+e^{-z}}
\]

Properties

- Output lies between 0 and 1
- Smooth and differentiable
- Converts linear output into probability

Examples

| z | Probability |
|---|-------------|
| -5 | 0.0067 |
| -2 | 0.119 |
| 0 | 0.50 |
| 2 | 0.881 |
| 5 | 0.993 |

Interpretation

```
Probability > 0.5

↓

Class = 1
```

```
Probability < 0.5

↓

Class = 0
```

---

# 6. Decision Boundary

The decision boundary separates two classes.

Rule

```
If Probability ≥ 0.5

Predict Class 1

Else

Predict Class 0
```

---

# 7. Cost Function

Mean Squared Error is not suitable for Logistic Regression.

Instead we use **Binary Cross Entropy (Log Loss)**.

The cost function is

\[
J(\theta)
=
-\frac1m
\sum
\left[
y\log(h)
+
(1-y)\log(1-h)
\right]
\]

where

- m = number of samples
- y = actual value
- h = predicted probability

Goal

Minimize the cost.

---

# 8. Gradient Descent

Gradient Descent updates model parameters to reduce the cost.

Weight Update

\[
w=w-\alpha\frac{\partial J}{\partial w}
\]

Bias Update

\[
b=b-\alpha\frac{\partial J}{\partial b}
\]

where

- α = Learning Rate

Training Process

```
Initialize Weights

↓

Calculate Prediction

↓

Compute Cost

↓

Compute Gradient

↓

Update Weights

↓

Repeat Until Convergence
```

---

# 9. Numerical Example

Suppose

Feature

```
Age = 30
```

Weight

```
w = 0.4
```

Bias

```
b = -8
```

### Step 1

Calculate z

```
z = wx+b

= (0.4×30)-8

=4
```

### Step 2

Apply Sigmoid

\[
\sigma(4)=\frac1{1+e^{-4}}
\]

```
Probability

=0.982
```

### Step 3

Prediction

```
0.982 > 0.5

Class = 1
```

---

# 10. Algorithm Workflow

```
Dataset

↓

Data Preprocessing

↓

Train-Test Split

↓

Train Logistic Regression

↓

Predict

↓

Evaluate Accuracy

↓

Confusion Matrix
```

---

# 11. Python Implementation

Main Steps

- Import Libraries
- Load Dataset
- Split Dataset
- Train Logistic Regression Model
- Predict Test Data
- Evaluate Accuracy

Example

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)
```

---

# 12. Model Evaluation

Common Evaluation Metrics

### Accuracy

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

### Precision

\[
Precision=\frac{TP}{TP+FP}
\]

### Recall

\[
Recall=\frac{TP}{TP+FN}
\]

### F1 Score

\[
F1=
2
\times
\frac{Precision\times Recall}
{Precision+Recall}
\]

### Confusion Matrix

| Actual | Predicted |
|---------|-----------|
| TP | FP |
| FN | TN |

---

# 13. Time & Space Complexity

| Operation | Complexity |
|-----------|------------|
| Training | O(n × d × iterations) |
| Prediction | O(d) |
| Space Complexity | O(d) |

where

- n = Number of Samples
- d = Number of Features

---

# 14. Advantages

- Easy to implement
- Fast training
- Probabilistic output
- Interpretable coefficients
- Works well for binary classification

---

# 15. Limitations

- Assumes linear decision boundary
- Sensitive to outliers
- Poor performance on highly non-linear data
- Requires feature scaling in many cases

---

# 16. Applications

- Medical Diagnosis
- Cancer Detection
- Credit Risk Prediction
- Email Spam Detection
- Customer Churn Prediction
- Fraud Detection
- Sentiment Analysis

---

# 17. Dataset

Dataset Used

**Breast Cancer Wisconsin Dataset**

Number of Features

```
30
```

Target Classes

```
0 → Benign

1 → Malignant
```

---

# Output

Example

```
Accuracy : 97.36%

Precision : 96.8%

Recall : 97.2%

F1 Score : 97%
```

---

# References

1. Scikit-learn Documentation
2. Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow
3. Pattern Recognition and Machine Learning – Christopher Bishop
4. Introduction to Statistical Learning
5. Machine Learning – Tom M. Mitchell

---

## Author

**Tanishq Tomar**

B.Tech CSE (AI & ML)

GLA University

GitHub: https://github.com/Clestialcosmos
