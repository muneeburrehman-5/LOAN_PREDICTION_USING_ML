# Loan Prediction Using ML

## 1. Task objective
The objective of this project is to build a machine learning model that predicts whether a loan application will be approved based on applicant information.  
The notebook treats this as a binary classification problem using features such as:

- Age
- Income
- Credit Score
- Loan Amount
- Loan Term
- Employment Status

The target variable is **Loan_Approved**, where:

- `0` = Rejected
- `1` = Approved

---

## 2. My approach
I followed a complete end-to-end ML workflow:

### Data loading and inspection
- Loaded the dataset from `loan_prediction_dataset.csv`
- Checked for missing values
- Confirmed that the dataset contains no null values

### Feature engineering and encoding
- Used **LabelEncoder** to convert `Employment_Status` into numeric form
- Created a new feature:
  - `Employment_Status_Encoded`

### Exploratory data analysis
- Visualized:
  - target class distribution
  - loan amount vs income
  - income vs approval status
  - employment status distribution

### Model preparation
- Selected the following features:
  - `Age`
  - `Income`
  - `Credit_Score`
  - `Loan_Amount`
  - `Loan_Term`
  - `Employment_Status_Encoded`
- Split the dataset into training and testing sets using an **80/20 stratified split**
- Applied **StandardScaler** before training Logistic Regression

### Model training
I trained and compared two classification models:

1. **Logistic Regression**
   - Used `class_weight='balanced'` to handle class imbalance
2. **Decision Tree Classifier**
   - Used `max_depth=5`
   - Also used `class_weight='balanced'`

---

## 3. Results and insights

### Class imbalance
The dataset is imbalanced:

- Rejected: **1658**
- Approved: **342**

This means the model could easily bias toward predicting rejection unless class balancing is used.

### Model performance
#### Logistic Regression
- Accuracy: **88.00%**
- Strong at identifying approved loans
- Produced some false positives and false negatives

#### Decision Tree
- Accuracy: **100.00%**
- Perfect performance on the test set
- Best model in this notebook

### Key insights
- `class_weight='balanced'` was essential for improving minority-class prediction
- Decision Tree outperformed Logistic Regression on this dataset
- The dataset appears to have very strong separability between classes, which may explain the perfect score
- While the decision tree achieved 100% accuracy, such a result should still be validated carefully on unseen real-world data to rule out overfitting or data leakage

---

## Repository summary
This repository demonstrates a complete machine learning pipeline for loan approval prediction, including data preprocessing, encoding, visualization, model training, and evaluation.

If you want, I can also turn this into a **full polished GitHub README.md** with:
- project title
- installation
- usage
- dataset description
- model comparison
- conclusion
- license section
