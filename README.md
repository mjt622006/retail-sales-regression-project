# 📊 Retail Store Sales Drivers — Multilinear Regression Analysis

**Identifying key sales drivers across 400 Dutch men's fashion stores using log-linear OLS regression.**

> A rigorous statistical analysis demonstrating data science fundamentals: exploratory analysis, model selection, diagnostic validation, and business-ready insights.

---

## 📈 Executive Summary

**Question:** Which store-level factors drive annual sales in a competitive retail market?

**Dataset:** 400 men's fashion stores in the Netherlands (1990). 13 variables spanning staffing, operations, and financial metrics.

**Method:** Log-linear OLS regression with backward variable elimination and full diagnostic validation.

**Key Finding:** Part-time worker count is the strongest sales driver (+8.7% sales per additional worker), followed by store size, total hours worked, and gross margin.

**Model Performance:** R² = 0.571 (explains 57.1% of variance); MAPE = 45.1% (average prediction error on unseen stores).

---

## 🎯 Business Insights

| Insight | Impact |
|---------|--------|
| **Part-time workforce is leverage** | Adding just 1 part-timer → ~+8.7% annual sales (largest coefficient) |
| **Scale matters, but efficiency too** | Store size matters, but hours-worked is more predictive than raw square footage |
| **Margin management drives profit** | Gross margin coefficient shows pricing/procurement discipline correlates with volume |
| **Model explains 57% of variance** | Leaves 43% to unmeasured factors (location, brand strength, local competition, season) |

---

## 📊 Files & Data

| File | Description |
|------|-------------|
| `retail_sales_regression.ipynb` | Complete Jupyter notebook — analysis start to finish |
| `Clothing.csv` | Dataset: 400 stores × 13 variables |
| `README.md` | This documentation |

### Dataset Variables

```
tsales:        Annual sales (guilders) — TARGET VARIABLE
size:          Store size (sq. meters)
hours:         Total hours worked (staff × hours)
parttime:      Part-time headcount
fulltime:      Full-time headcount
experience:    Avg years experience (staff)
margin:        Gross margin (%)
parking:       Parking spaces available
competition:   Number of nearby competitors
citycenter:    1 if in city center, 0 otherwise
age:           Years store has been open
```

---

## 🔬 Methodology

### 1. **Exploratory Data Analysis**
- Univariate distributions (histograms, summary stats)
- Correlation matrix → identify multicollinearity risk
- Target variable (`tsales`) shows right skew (skewness = 2.39) → log-transform

### 2. **Transformation & Preparation**
- Log-transform target: `log(tsales)` to normalize residuals
- Skewness after transform: -0.69 (much better)
- 80/20 train/test split for unbiased evaluation
- Standardize continuous predictors for interpretability

### 3. **Model Selection**
- Start with all 9 predictors
- Backward elimination: remove non-significant terms (α = 0.05)
- Final model: 5 significant predictors

### 4. **Diagnostic Validation**
- **VIF Check:** All predictors VIF < 5 → no multicollinearity
- **Residual Plots:** Q-Q plot shows near-normal residuals (slight heavy tails in extremes—acceptable)
- **Heteroscedasticity:** Residuals vs. Fitted plot shows no obvious fanning → homoscedastic

### 5. **Performance Assessment**
- Back-convert log predictions to guilders for business interpretation
- Metrics: R², Adjusted R², RMSE, MAE, MAPE
- Test set MAPE = 45.1% (reasonable given omitted variables)

---

## 📉 Key Results

### Regression Output

| Predictor | Coefficient | Std Error | t-stat | p-value | Interpretation |
|-----------|-------------|-----------|--------|---------|-----------------|
| **Intercept** | 8.21 | 0.15 | 54.7 | <0.001 | Baseline log(sales) |
| **log(size)** | 0.42 | 0.06 | 7.0 | <0.001 | +1% store size → +0.42% sales |
| **log(hours)** | 0.68 | 0.09 | 7.5 | <0.001 | +1% hours worked → +0.68% sales |
| **parttime** | 0.087 | 0.012 | 7.1 | <0.001 | **+1 part-timer → +8.7% sales** ⭐ |
| **margin** | 0.018 | 0.004 | 4.5 | <0.001 | +1% margin → +1.8% sales |

### Model Fit

```
R² (Train):                 0.571  →  57.1% of variance explained
Adjusted R²:                0.564  →  Penalizes complexity appropriately
F-statistic:                81.4   →  Highly significant (p < 0.001)
Residual Std Error:         0.248  →  Predictions typically off by ~25% in log space
```

### Prediction Accuracy

```
RMSE:        €42,500  (test set)
MAE:         €31,200  (test set)
MAPE:        45.1%    (test set)
```

*(Interpretation: On average, predictions miss by ~€31K or 45% of actual sales)*

---

## 💻 How to Run

### Option 1: Google Colab (Recommended — No Setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mjt622006/retail-sales-regression-project/blob/main/retail_sales_regression.ipynb)

1. Click the Colab badge above
2. When prompted, upload `Clothing.csv`
3. Click "Run All" or run cell-by-cell

### Option 2: Local Jupyter Notebook

```bash
# Install dependencies
pip install pandas numpy scipy statsmodels scikit-learn seaborn matplotlib jupyter

# Launch notebook
jupyter notebook retail_sales_regression.ipynb

# Then open http://localhost:8888 in browser
```

### Option 3: VS Code + Jupyter Extension

1. Open this repo in VS Code
2. Install "Jupyter" extension
3. Click "Run All" button in the notebook

---

## 🏗️ Notebook Structure

```
1. Load & Explore Data
   ├── Load Clothing.csv
   ├── Summary statistics
   ├── Univariate distributions
   └── Correlation matrix

2. Data Preparation
   ├── Check for missing values
   ├── Log-transform target (tsales)
   ├── Train/test split (80/20)
   └── Check skewness before/after

3. Model Building
   ├── Fit OLS with all predictors
   ├── Check p-values, t-stats
   ├── Backward elimination (remove α > 0.05)
   └── Final model: 5 predictors

4. Diagnostic Validation
   ├── VIF (variance inflation factor) → multicollinearity
   ├── Q-Q plot → residual normality
   ├── Residuals vs. Fitted → homoscedasticity
   ├── Durbin-Watson → autocorrelation

5. Performance & Interpretation
   ├── Back-convert predictions from log → guilders
   ├── Calculate RMSE, MAE, MAPE
   ├── Plot actual vs. predicted
   ├── Interpret coefficients (elasticities)
   └── Business insights & recommendations
```

---

## 🎓 What This Demonstrates

**For Recruiters:**

- ✅ **Statistical Modeling** — OLS regression, variable selection, diagnostics
- ✅ **Data Wrangling** — pandas for exploration, transformation, splitting
- ✅ **Hypothesis Testing** — p-values, t-tests, confidence intervals
- ✅ **Model Validation** — train/test split, diagnostic plots, accuracy metrics
- ✅ **Communication** — Clear interpretation of technical results for business audience
- ✅ **Problem-Solving** — Addresses a real business question with rigor

**Technical Skills:**
- Python (pandas, numpy, scipy, statsmodels, scikit-learn)
- Statistical analysis & inference
- Jupyter notebooks for reproducible research
- Data visualization (matplotlib, seaborn)

---

## 🔍 Key Assumptions & Limitations

### Model Assumptions (Checked ✓)
- ✓ **Linearity:** Log-linear form appropriate for sales data
- ✓ **No multicollinearity:** All VIF < 5
- ✓ **Homoscedastic residuals:** Variance roughly constant across fitted values
- ✓ **Normal errors:** Q-Q plot shows near-normality (minor heavy tails acceptable)

### Limitations (Be Honest)
- **43% unexplained variance** — Store location, brand reputation, local competition intensity, seasonality not captured
- **Time period (1990)** — Consumer behavior, retail landscape may differ today
- **Geographic scope** — Netherlands men's fashion; may not generalize to other retail
- **Omitted variables** — Could include: advertising spend, inventory turnover, foot traffic
- **Outliers:** 1-2 extreme cases could influence results; robust regression alternative available

---

## 📚 References & Methods

- **OLS Regression:** Greene, *Econometric Analysis* (7th ed.)
- **Diagnostics:** Fox, *Applied Regression Analysis and Generalized Linear Models* (3rd ed.)
- **Python Implementation:** statsmodels documentation
- **Business Interpretation:** Inspired by case studies in business analytics programs

---

## 🚀 Potential Extensions

- **Robust Regression** — Downweight outliers for more stable estimates
- **Interaction Terms** — Does margin×hours create synergy?
- **Categorical Predictors** — One-hot encode `citycenter` for more nuance
- **Time Series** — If you have multi-year data, use panel regression
- **Machine Learning** — Compare OLS to random forest / gradient boosting (may find non-linear patterns)
- **Cross-Validation** — k-fold CV for more robust error estimates

---

## 💡 Key Takeaway

**Part-time workforce is the most powerful sales lever.** A store manager wanting to boost sales should prioritize hiring flexible workers (even more impactful than expanding store size). This insight came from rigorous statistical analysis and survived multiple diagnostic checks—making it trustworthy for decision-making.

---

## 📄 License

MIT — Use for education, research, portfolio.

---

**Built as a portfolio project to demonstrate core data science competencies:**
- Hypothesis formulation & testing
- Statistical modeling & validation
- Clear communication of technical results

Happy analyzing! 📊
