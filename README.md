# Loan Status Prediction

A machine learning project that predicts whether a loan application will be **approved (Y)** or **rejected (N)** based on the applicant's personal, financial, and property information. The notebook compares two classifiers: **Support Vector Machine (SVM)** and **Logistic Regression**.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Possible Improvements](#possible-improvements)

## Overview

Banks and financial institutions need to decide quickly and consistently whether to approve a loan. This project builds a binary classification model that learns from historical loan applications and predicts the outcome (`Loan_Status`) for new applicants.

## Dataset

The notebook uses the **Loan Prediction** training dataset (`train_u6lujuX_CVtuZ9i.csv`) with **614 rows and 13 columns**.

| Column | Description |
|---|---|
| `Loan_ID` | Unique loan identifier (dropped before training) |
| `Gender` | Male / Female |
| `Married` | Applicant marital status (Yes / No) |
| `Dependents` | Number of dependents (0, 1, 2, 3+) |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Self-employed (Yes / No) |
| `ApplicantIncome` | Applicant income |
| `CoapplicantIncome` | Co-applicant income |
| `LoanAmount` | Loan amount (in thousands) |
| `Loan_Amount_Term` | Loan term (in months) |
| `Credit_History` | Whether credit history meets guidelines (1 / 0) |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | **Target**: loan approved (Y) or not (N) |

> The dataset file is not included in this repository. Download it (e.g. from Kaggle / Analytics Vidhya "Loan Prediction") and update the path in the notebook.

## Project Workflow

1. **Load the data** with pandas and inspect it (`head`, `shape`, `describe`).
2. **Handle missing values**: the dataset contains missing values in `Gender`, `Married`, `Dependents`, `Self_Employed`, `LoanAmount`, `Loan_Amount_Term`, and `Credit_History`. Rows with missing values are dropped, leaving **480 rows**.
3. **Preprocessing / encoding**
   - `Loan_Status`: `N → 0`, `Y → 1`
   - `Dependents`: `3+ → 4`
   - `Married`, `Gender`, `Education`, `Self_Employed`: binary label encoding (0 / 1)
   - `Property_Area`: one-hot encoded (`Rural`, `Semiurban`, `Urban`)
   - `Loan_ID` is dropped.
4. **Exploratory data visualization** with seaborn count plots (`Education` vs `Loan_Status`, `Married` vs `Loan_Status`).
5. **Train/test split**: 80% / 20%, stratified on the target, `random_state=42` (384 training rows, 96 test rows).
6. **Model training and evaluation**
   - SVM with a linear kernel (`svm.SVC(kernel="linear")`)
   - Logistic Regression (`max_iter=1000`) with a classification report and confusion matrix

## Results

| Model | Training Accuracy | Test Accuracy |
|---|---|---|
| SVM (linear kernel) | 77.34% | 80.21% |
| Logistic Regression | 81.25% | **82.29%** |

**Logistic Regression: classification report (test set)**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (Rejected) | 0.81 | 0.57 | 0.67 | 30 |
| 1 (Approved) | 0.83 | 0.94 | 0.88 | 66 |

**Confusion matrix**

|  | Predicted Rejected | Predicted Approved |
|---|---|---|
| **Actual Rejected** | 17 | 13 |
| **Actual Approved** | 4 | 62 |

Logistic Regression performed slightly better than the linear SVM. The model is very good at identifying approved loans (recall 0.94) but misses a fair share of rejected ones (recall 0.57).

## Requirements

- Python 3.8+
- numpy
- pandas
- seaborn
- scikit-learn
- Jupyter Notebook / JupyterLab / Google Colab

Install the dependencies:

```bash
pip install numpy pandas seaborn scikit-learn jupyter
```

## How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Place the dataset CSV in the project folder.
3. Open the notebook and update the file path in the data-loading cell. The notebook currently uses a Google Colab path:
   ```python
   loan_dataset = pd.read_csv("/content/train_u6lujuX_CVtuZ9i (1).csv")
   ```
   Change it to your local path, for example:
   ```python
   loan_dataset = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
   ```
4. Launch Jupyter and run all cells:
   ```bash
   jupyter notebook Loan_Status_Prediction.ipynb
   ```

## Project Structure

```
.
├── Loan_Status_Prediction.ipynb   # Main notebook
├── train_u6lujuX_CVtuZ9i.csv      # Dataset (add manually)
└── README.md
```

## Possible Improvements

- Impute missing values (median / mode) instead of dropping rows, to keep more training data.
- Scale numeric features (`ApplicantIncome`, `LoanAmount`, ...) before training, especially for SVM.
- Handle class imbalance (e.g. `class_weight="balanced"` or SMOTE) to improve recall on rejected loans.
- Try other models (Random Forest, Gradient Boosting, XGBoost) and tune hyperparameters with cross-validation.
- Engineer new features such as total income (`ApplicantIncome + CoapplicantIncome`) or loan-to-income ratio.
- Add a prediction function or small app (Streamlit / Flask) to score new applicants.
