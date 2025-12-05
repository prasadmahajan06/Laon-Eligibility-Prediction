# Loan-Eligibility-Prediction
Loan Eligibility Prediction Using Logistic Regression

📌 ***Loan Eligibility Prediction using Logistic Regression***
This project uses Logistic Regression to predict whether a loan applicant is eligible for a loan based on financial, demographic, and credit-related factors. The model analyzes historical loan data to identify patterns that influence loan approval decisions, helping financial institutions automate and optimize their screening process.

🚀 ***Project Overview***
The goal of this project is to build a binary classification model that predicts:
- ✔ Loan Approved (1)
- ✘ Loan Not Approved (0)

The project involves:
- Data cleaning & preprocessing
- Feature engineering
- Handling missing values
- Exploratory Data Analysis (EDA)
- Model training using Logistic Regression
- Model evaluation (accuracy, confusion matrix, ROC curve, etc.)
- Making predictions on new applicants

📂 ***Dataset Description***

Typical dataset fields include:
- Loan ID
- Gender
- Marital Status
- Dependents
- Education Level
- Applicant Income
- Co-Applicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status (Target Variable)

🗂️**Data Source**
- Primary Source: Loan Elibility Prediction Data <a href ="https://github.com/prasadmahajan06/Laon-Eligibility-Prediction/blob/main/loan-train.csv">Download Dataset Here</a>

🧰 ***Technologies & Libraries Used***

- Python 3.x
- Pandas – Data manipulation
- NumPy – Numeric calculations
- Matplotlib / Seaborn – Data visualization
- Scikit-learn – Logistic Regression model
- Jupyter Notebook – Analysis environment

🛠️ ***Steps Performed***

✔ 1. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Scaling numerical features (if needed)
- Converting target variable to binary format

✔ 2. Exploratory Data Analysis
- Income distribution
- Loan amount distribution
- Relationship between credit history & approval
- Correlation between variables

✔ 3. Model Training
- Train/Test split
- Logistic Regression training

✔ 4. Model Evaluation
- Accuracy score
- Confusion matrix
- Precision, Recall, F1-Score
- ROC curve and AUC score

⭐ ***Model Implementation (Code)***
### importing libraries
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

### Mounting drive
from google.colab import drive

drive.mount('/content/drive')

filepath = '/content/drive/MyDrive/loan-train.csv'

df = pd.read_csv(filepath)

df.describe()

df.info()

df.isna().sum()

### replacing the missing values in numerical columns using imputation
df['Credit_History'].fillna(df['Credit_History'].mode(), inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

### converting the categorical columns numerical values using encoding
df.Loan_Status = df.Loan_Status.replace({"Y": 1, "N" : 0})

df.Gender = df.Gender.replace({"Male": 1, "Female" : 0})

df.Married = df.Married.replace({"Yes": 1, "No" : 0})

df.Self_Employed = df.Self_Employed.replace({"Yes": 1, "No" : 0})

### Imputing missing values in newly converted categorical columns
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)

df['Married'].fillna(df['Married'].mode()[0], inplace=True)

df['Credit_History'].fillna(df['Credit_History'].mean(), inplace=True)

df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)

### Here Property_Area, Dependents and Education has multiple values so now we can use LabelEncoder from sklearn package
from sklearn.preprocessing import LabelEncoder

feature_col = ['Property_Area','Education', 'Dependents']

le = LabelEncoder()

for col in feature_col:

    df[col] = le.fit_transform(df[col])

df.isna().sum()

### Exploratory Data Analysis

### Line plot for all the numeric columns
df.plot(figsize=(18, 8))

plt.show()

### Histogram for visualizing Distriution of data
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)


df['ApplicantIncome'].hist(bins=10)

plt.title("Applicant Income Distribution")

plt.subplot(1, 2, 2)

plt.grid()

plt.hist(np.log(df['LoanAmount']))

plt.title("Log Loan Application Amount ")

plt.show()

### Relation between Applicant Income and loan amount
plt.figure(figsize=(18,6))

plt.scatter(df["ApplicantIncome"],df["LoanAmount"], marker='x')

plt.title("Relation between Applicant Income Vs Loan Amount")

plt.xlabel("Applicant Income")

plt.ylabel("Loan Amount")

plt.show()

### Corelation Heatmap for numeric columns
plt.figure(figsize=(8,4))

sns.heatmap(df.select_dtypes(include='number').corr(),
            cmap='coolwarm', annot=True, fmt='.1f', linewidths=.1)
            
plt.show()

### Choosing ML Model
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, RocCurveDisplay, roc_auc_score)

### Separating dependent and independent variable
X = df.drop(["Loan_ID","Loan_Status"], axis= 1)

y = df["Loan_Status"]

### split the data in train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Logistic regression model
log_clf = LogisticRegression()

log_clf.fit(X, y)

### Predicting y
y_predict = log_clf.predict(X_test)

### check the accuracy score
print(accuracy_score(y_test, y_predict))

📊 ***Results***
- Model Accuracy: 0.796
- Applicants with strong credit history have significantly higher approval rates
- Higher applicant income improves loan approval probability

🙋‍♂️ Author
- Created by: Prasad Mahajan
- GitHub: prasadmahajan06
