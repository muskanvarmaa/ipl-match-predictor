## End to End Machine Learning Project

# IPL Match Outcome Predictor

A machine learning project that predicts IPL match winners using historical match data (2008–2017). Built as a hands-on exercise in the full ML pipeline — from raw data to model evaluation.

## Problem Statement

Given two teams, the venue, and the toss winner, predict whether **Team 1** will win the match.

## Dataset

IPL Complete Dataset (Kaggle) — `matches.csv` (756 matches, reduced to 752 after removing abandoned/no-result games).

## Approach

1. **Data Cleaning** — Dropped post-match columns to avoid data leakage.
2. **EDA** — Analyzed team win distribution, toss-decision trends, toss-winner vs match-winner correlation.
3. **Feature Engineering** — Reframed as binary classification (`team1_won`), One-Hot Encoded categorical features.
4. **Modeling** — Compared Logistic Regression, Random Forest, and XGBoost.
5. **Evaluation** — Confusion matrix and feature importance analysis.

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 55.6% |
| Random Forest | 53.6% |
| XGBoost | 55.0% |
| Toss-winner-only baseline | 52.3% |

## Key Finding

All models performed only marginally better than the toss-winner baseline. Feature importance revealed overfitting to defunct teams with few matches (e.g. Kochi Tuskers Kerala). This suggests static team/venue identity has limited predictive power — recent team form would likely be a stronger feature.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, matplotlib/seaborn, Google Colab