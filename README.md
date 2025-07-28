# 🏦 Loan Default Prediction & Expected Loss Estimator

This project builds a machine learning model that predicts a borrower's **Probability of Default (PD)** and calculates the **Expected Loss (EL)** using real-world financial features.

---

## 📁 Dataset

The dataset (`Loan Data.csv`) contains 10,000 customer records with the following features:

- `credit_lines_outstanding`
- `loan_amt_outstanding`
- `total_debt_outstanding`
- `income`
- `years_employed`
- `fico_score`
- `default` (target variable)

---

## 🔍 Objective

1. Train a **Logistic Regression** model to predict the probability of loan default.
2. Use this probability to calculate **Expected Loss (EL)** using the formula:

\[
\text{Expected Loss} = \text{PD} \times \text{Loan Amount} \times (1 - \text{Recovery Rate})
\]

- **Recovery Rate** assumed = `10%`
- **Loan Amount** = `loan_amt_outstanding`

---

## 🛠️ Technologies Used

- Python
- pandas, numpy
- scikit-learn (for model + scaling + metrics)
- matplotlib (optional for feature importance)

---

## 🧠 Model Pipeline

1. **Data Preprocessing**
   - Drop customer ID
   - Scale features using `StandardScaler`
2. **Train-Test Split**
   - 80/20 split using `train_test_split`
3. **Model Training**
   - Logistic Regression
4. **Evaluation**
   - Accuracy Score
   - ROC AUC Score
5. **Expected Loss Calculation**
   - Custom function using model predictions

---

## 🚀 Example Usage

python
new_borrower = {
    'credit_lines_outstanding': 2,
    'loan_amt_outstanding': 5000,
    'total_debt_outstanding': 8000,
    'income': 70000,
    'years_employed': 5,
    'fico_score': 640
}

loss = calculate_expected_loss(new_borrower, model, scaler)
print("Expected Loss:", loss)

---

## Output Example
Accuracy: 0.82
ROC AUC Score: 0.74
Expected Loss: 0.0081
