# Online Payments Fraud Detection using Machine Learning

## Overview

Online payment fraud is one of the most significant challenges faced by financial institutions and e-commerce platforms. This project builds a machine learning model to detect fraudulent online payment transactions, helping to protect customers and businesses from financial losses.

The model is trained on a labeled dataset of financial transactions and learns to distinguish between legitimate and fraudulent activity based on transaction features.

## Problem Statement

With the rapid growth of digital payments, fraudsters have developed increasingly sophisticated techniques to exploit vulnerabilities. Traditional rule-based fraud detection systems struggle to keep up with evolving fraud patterns. This project leverages machine learning to automatically identify suspicious transactions in real time.

## Dataset

The dataset used in this project contains simulated mobile money transactions and includes the following features:

| Feature | Description |
|---|---|
| `step` | Represents a unit of time (1 step = 1 hour) |
| `type` | Type of transaction (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER) |
| `amount` | Amount of the transaction |
| `nameOrig` | Customer initiating the transaction |
| `oldbalanceOrg` | Initial balance of the originating account |
| `newbalanceOrig` | New balance of the originating account after the transaction |
| `nameDest` | Recipient of the transaction |
| `oldbalanceDest` | Initial balance of the recipient account |
| `newbalanceDest` | New balance of the recipient account after the transaction |
| `isFraud` | Target variable — 1 if the transaction is fraudulent, 0 otherwise |
| `isFlaggedFraud` | Flag for transactions marked by business rules as potentially fraudulent |

## Technologies Used

- **Python 3**
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Matplotlib / Seaborn** — Data visualization
- **Scikit-learn** — Machine learning algorithms and evaluation metrics
- **Jupyter Notebook** — Interactive development environment

## Machine Learning Approach

### Steps

1. **Exploratory Data Analysis (EDA)** — Understanding data distributions, class imbalance, and correlations.
2. **Data Preprocessing** — Encoding categorical variables, handling missing values, and feature scaling.
3. **Feature Engineering** — Creating new features to improve model performance.
4. **Model Training** — Training classification models such as:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
5. **Model Evaluation** — Evaluating models using accuracy, precision, recall, F1-score, and ROC-AUC.

### Handling Class Imbalance

Fraudulent transactions are rare compared to legitimate ones, resulting in a highly imbalanced dataset. Techniques such as resampling or using class weights are applied to address this imbalance.

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Project

1. Clone the repository:

```bash
git clone https://github.com/Saket8538/Online-Payments-Fraud-Detection-using-Machine-Learning.git
cd Online-Payments-Fraud-Detection-using-Machine-Learning
```

2. Launch Jupyter Notebook:

```bash
jupyter notebook
```

3. Open the project notebook and run the cells sequentially.

## Results

The trained model is evaluated on a held-out test set. Key metrics include:

- **Accuracy** — Overall correctness of the model
- **Precision** — Fraction of predicted frauds that are actually fraudulent
- **Recall** — Fraction of actual frauds correctly identified
- **F1-Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Model's ability to distinguish between classes

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
