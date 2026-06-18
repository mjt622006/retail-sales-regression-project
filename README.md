# Retail Store Sales Drivers — Multilinear Regression

A multilinear regression analysis identifying the key drivers of annual sales across 400 Dutch men's fashion stores (1990).

---

## Project Summary

**Goal:** Determine which store-level factors drive annual sales (`tsales`) using OLS regression.  
**Dataset:** `Clothing.csv` — 400 observations, Men's Fashion Stores, Netherlands (1990).  
**Approach:** Log-linear OLS regression with backward variable elimination and full diagnostic validation.

---

## Key Results

| Metric | Value |
|---|---|
| **R²** | 0.571 — model explains 57.1% of sales variance |
| **MAPE** | 45.1% average error on unseen stores |
| **Significant Predictors** | Store size, total hours worked, gross margin, part-time headcount |

**Biggest finding:** Part-time worker count carries the highest coefficient (+8.7% sales per additional worker), making flexible staffing the most actionable lever for store managers.

---

## Files

| File | Description |
|---|---|
| `retail_sales_regression.ipynb` | Main notebook — full analysis end to end |
| `Clothing.csv` | Dataset (400 stores, 13 variables) |
| `README.md` | This file |

---

## How to Run

**Option 1 — Colab (no setup required)**  
Click the "Open in Colab" badge above. Upload `Clothing.csv` when prompted.

**Option 2 — Local**  
```bash
pip install pandas numpy statsmodels seaborn matplotlib scikit-learn scipy
jupyter notebook retail_sales_regression.ipynb
```

---

## Methodology

1. Variable selection based on business logic and correlation analysis
2. Log-transformation of target (`tsales`) to fix right skew (2.39 → -0.69)
3. 80/20 train/test split
4. OLS regression with backward elimination of non-significant predictors
5. VIF check — all predictors below threshold of 5
6. Residual diagnostics (Q-Q plot + Residuals vs Fitted)
7. Accuracy assessment with back-conversion from log space to guilders

---

*Built as part of a Business Analytics portfolio project.*
