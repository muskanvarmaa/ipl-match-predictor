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
5. **Evaluation** — Confusion matrix, feature importance analysis, and bootstrap confidence intervals to validate whether model  improvements over baseline were statistically significant.

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 55.0% |
| Random Forest | 53.6% |
| XGBoost | 55.0% |
| Toss-winner-only baseline | 52.3% |

## Key Finding

All models performed only marginally better than the toss-winner baseline (XGBoost: +2.6 percentage points). A bootstrap confidence interval (1,000 resamples) on the accuracy difference between XGBoost and the baseline gave a 95% CI of (-6.6%, +11.9%) — meaning this improvement is not statistically significant at the current sample size (n=752). Feature importance also revealed overfitting to defunct teams with few matches (e.g. Kochi Tuskers Kerala). This suggests static team/venue identity has limited predictive power, and that both more data and recent team form would likely yield stronger, more reliable models.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, matplotlib/seaborn, Google Colab