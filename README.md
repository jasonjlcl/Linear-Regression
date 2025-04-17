# 📈 Stock Market Direction Prediction with Logistic Regression

This project uses logistic regression to predict the daily direction (Up or Down) of the S&P 500 index using historical lagged return data from 2001 to 2005. It compares two models:
- One using **all available predictors**
- Another using **only Lag1 and Lag2**, the most statistically significant features

---

## 🧠 Objective
To explore the effectiveness of logistic regression in binary classification problems — specifically, stock market prediction — and demonstrate how model simplification can improve generalization.

---

## 📂 Dataset
The dataset used is `Smarket` from the [ISLR](https://www.statlearning.com/) package. It includes:
- Daily % returns for the S&P 500 from 2001 to 2005
- Lagged returns: `Lag1` to `Lag5`
- Trading `Volume`
- Market `Direction` (Up/Down)

---

## 🛠️ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/smarket-logistic-regression.git
   cd smarket-logistic-regression

pip install pandas numpy statsmodels scikit-learn
jupyter notebook
