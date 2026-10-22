# Mini-ML-Project-telecom
End-to-end Telecom Customer Churn Prediction using Python, EDA, Logistic Regression, Random Forest, Model Evaluation, ROC-AUC Analysis, Threshold Optimization, and Cross-Validation.
This looks like a much more complete Telecom Customer Churn Prediction project. For GitHub, I'd recommend highlighting not only EDA but also machine learning, model comparison, and business impact.

GitHub Repository Description

End-to-end Telecom Customer Churn Prediction using Python, EDA, Logistic Regression, Random Forest, Model Evaluation, ROC-AUC Analysis, Threshold Optimization, and Cross-Validation.

README.md
# Telecom Customer Churn Prediction

## Project Overview

This project analyzes customer churn in a telecom company and builds machine learning models to predict whether a customer is likely to leave the service.

The project covers the complete data science workflow:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Correlation Analysis
- Churn Rate Analysis
- Feature Scaling
- Machine Learning Model Training
- Model Comparison
- Threshold Optimization
- ROC-AUC Evaluation
- Cross-Validation
- Business Value Calculation

---

## Business Problem

Customer acquisition is significantly more expensive than customer retention.

The goal of this project is to identify customers who are likely to churn so that proactive retention strategies can be applied.

---

## Dataset Features

Key features used in modeling:

- Tenure Months
- Monthly Charges
- Complaints
- Churn (Target Variable)

Target:

```python
churn
```

- 1 = Customer Churned
- 0 = Customer Retained

---

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Exploratory Data Analysis

### Data Quality Checks

- Missing Value Detection
- Missing Percentage Analysis
- Duplicate Detection
- Duplicate Removal
- Dataset Statistics
- Data Type Validation

### Churn Rate Analysis

```python
churn_rate = df['churn'].mean() * 100
```

Calculates the overall percentage of customers who churned.

### Churn Group Analysis

```python
df.groupby('churn').mean()
```

Compares average feature values between churned and retained customers.

---

## Correlation Analysis

A correlation matrix was generated to understand relationships between numerical variables.

### Correlation Heatmap

Features with strong positive or negative correlations can indicate important predictors of churn.

---

## Data Preparation

### Train-Test Split

```python
train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
```

- Training Set = 70%
- Test Set = 30%

### Feature Scaling

```python
StandardScaler()
```

Applied to ensure features are on the same scale before model training.

---

## Logistic Regression

The initial predictive model was built using Logistic Regression.

### Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

### Odds Ratio Interpretation

```python
odds_ratio = np.exp(logreg.coef_[0])
```

Odds ratios help quantify how each feature influences churn probability.

Example Interpretation:

- Odds Ratio > 1 → Increases churn likelihood
- Odds Ratio < 1 → Decreases churn likelihood

---

## ROC Curve and AUC

ROC curves were generated to evaluate classification performance across different thresholds.

Metrics:

- True Positive Rate (Recall)
- False Positive Rate
- ROC-AUC

A higher AUC indicates better model discrimination.

---

## Threshold Optimization

Different probability thresholds were tested:

```python
0.2
0.3
0.4
0.5
0.6
0.7
```

For each threshold:

- Precision
- Recall
- F1 Score
- True Positives
- False Positives
- Business Revenue Impact

were calculated.

### Business Value Formula

```python
Net Value =
4000 × TP
− 300 × (TP + FP)
```

This allows model selection based on business impact rather than accuracy alone.

---

## Machine Learning Models Compared

The following algorithms were evaluated:

### Logistic Regression

- Linear baseline model
- Interpretable coefficients

### K-Nearest Neighbors (KNN)

- Distance-based classification

### Naive Bayes

- Probabilistic classifier

### Decision Tree

- Rule-based model
- Easy to explain

### Random Forest

- Ensemble method
- Handles complex patterns

### Support Vector Machine (SVM)

- Margin-based classifier

---

## Model Comparison Metrics

Each model was evaluated using:

- Train Accuracy
- Test Accuracy
- Test F1 Score

The results were ranked by F1 Score to identify the best-performing model.

---

## Decision Tree Rules

Tree rules were exported using:

```python
export_text()
```

This provides a human-readable explanation of model decisions.

Example:

```text
if complaints > 2:
    churn = Yes
else:
    churn = No
```

---

## Cross Validation

Random Forest performance was further validated using:

```python
cross_val_score()
```

### Metrics Reported

- Fold-wise F1 Scores
- Mean F1 Score
- Standard Deviation

This helps evaluate model stability across different data splits.

---

## Visualizations

### Correlation Heatmap

- Feature relationships

### ROC Curve

- Classification performance

### Tenure Distribution

- Customer tenure analysis

### Monthly Charges Distribution

- Revenue distribution analysis

### Cross-Validation Boxplot

- Stability of model performance

---

## Key Learnings

This project demonstrates:

- End-to-End Machine Learning Workflow
- Customer Churn Prediction
- Feature Engineering Concepts
- Model Evaluation Techniques
- Threshold Tuning
- Business-Focused Analytics
- Cross-Validation Best Practices

---

## Future Enhancements

- Hyperparameter Tuning
- XGBoost
- LightGBM
- Feature Importance Analysis
- Customer Segmentation
- SHAP Explainability
- Deployment using Flask/FastAPI
- Interactive Dashboard using Power BI

---

## Author

Jeevitha S




This version looks strong enough for a Data Analyst, Business Analyst, or Junior Data Scientist portfolio project.
