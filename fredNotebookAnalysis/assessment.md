# Nuclear Electricity Production and Employment in the U.S. Utilities Sector

**Nuclear Energy Economic Brief Series**  
Author: Dwanjai Oprien  
Data Sources: FRED — Federal Reserve Bank of St. Louis  
Date: May 2026

---

## 1. Research Overview

This report presents an exploratory economic analysis of the relationship between nuclear electricity production and employment trends in the U.S. utilities sector. Using monthly time-series data drawn from the Federal Reserve Economic Data (FRED) database, the analysis examines long-run co-movement between these variables across roughly five decades, from the early 1970s to the present.

The primary contribution of this analysis is descriptive and policy-oriented. It does not seek to establish causal claims. Rather, it aims to characterize the structural shifts that have occurred within the utilities sector labor market as nuclear energy production has matured from an expansion-intensive industry to a stabilized, capital-efficient one.

---

## 2. Research Question and Objectives

**Primary Research Question:**  
Does nuclear electricity production correlate with employment trends in the U.S. utilities sector?

**Analytical Objectives:**

- Document long-term trends in nuclear electricity production (FRED series: IPN221113N)
- Document long-term employment trends in the U.S. utilities sector (FRED series: CES4422000001)
- Examine whether the two variables move together over time through indexed comparison and exploratory regression
- Characterize any structural shifts in the labor-production relationship over time

---

## 3. Data Sources

| Variable | FRED Code | Description | Frequency |
|---|---|---|---|
| Nuclear Electricity Production | IPN221113N | Industrial Production Index, Nuclear Electric Power (2017 = 100) | Monthly |
| Utilities Employment | CES4422000001 | All Employees, Utilities (Thousands of Workers) | Monthly |

Both series were retrieved from FRED and standardized to a consistent monthly time-series format prior to analysis. The merged dataset covers the overlapping period common to both series, from approximately 1972 onward.

---

## 4. Library Imports and Notebook Setup

The analysis was conducted in Python using the following libraries:

```python
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set_style("whitegrid")
DATA_DIR = Path("../data")
```

---

## 5. Data Loading

```python
# Load nuclear production index
nuclearProduction = pd.read_csv(DATA_DIR / "IPN221113N.csv")
nuclearProduction.columns = ['observation_date', 'nuclear_generation']
nuclearProduction['observation_date'] = pd.to_datetime(nuclearProduction['observation_date'])
nuclearProduction = nuclearProduction.sort_values('observation_date').set_index('observation_date')

# Load utilities employment
utilityEmployment = pd.read_csv(DATA_DIR / "CES4422000001.csv")
utilityEmployment.columns = ['observation_date', 'utilities_employment']
utilityEmployment['observation_date'] = pd.to_datetime(utilityEmployment['observation_date'])
utilityEmployment = utilityEmployment.sort_values('observation_date').set_index('observation_date')
```

---

## 6. Data Cleaning and Preprocessing

Both series were aligned to a common monthly time index using an inner join, retaining only observations present in both datasets. Missing values were dropped. An indexed comparison dataset was constructed by rebasing both series to 100 at their earliest common observation point, allowing for proportional comparisons across the full sample and a post-1990 subsample.

```python
# Merge datasets on shared time index
df = nuclearProduction.join(utilityEmployment, how="inner").dropna()

# Construct full-period indexed comparison
df_indexed = df / df.iloc[0] * 100

# Construct post-1990 rebased comparison
df_recent = df.loc["1990":].copy()
df_recent_indexed = df_recent / df_recent.iloc[0] * 100

# 12-month moving average for employment
utilityEmployment['moving_avg_12'] = utilityEmployment['utilities_employment'].rolling(12).mean()
```

---

## 7. Exploratory Data Analysis

### 7.1 Long-Run Trend: U.S. Utilities Employment (Figure 1)

**Code:**

```python
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(utilityEmployment.index, utilityEmployment['utilities_employment'], color='steelblue', linewidth=1.5)
ax.set_title('Figure 1: U.S. Utilities Sector Employment (1972–Present)', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Employment (Thousands of Workers)')
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('../figures/fig1_utilities_employment.png', dpi=150)
plt.show()
```

**Interpretation:**  
Utilities sector employment in the United States exhibits a clear multi-phase trajectory over the observed period. Beginning in the early 1970s, employment rises steadily, reflecting a period of expansion in utility infrastructure and growing labor demand. This growth accelerates through the late 1970s and 1980s, peaking in the early-to-mid 1990s — likely corresponding to the culmination of large-scale energy system development and capital deployment.

Following this peak, employment declines sharply from the mid-1990s into the early 2000s. This contraction suggests a structural shift within the utilities sector, where labor demand decreased despite continued operation of existing infrastructure — consistent with technological improvements, operational efficiencies, and a transition from construction-intensive activity toward maintenance and optimization.

From the early 2000s onward, employment levels stabilize at a lower baseline. While a slight recovery is observable in recent years, the data does not indicate a return to the earlier expansionary employment levels. Overall, the trend suggests that utilities employment is closely tied to infrastructure expansion phases but does not scale proportionally once systems mature.

---

### 7.2 Smoothed Employment Trend: 12-Month Moving Average (Figure 2)

**Code:**

```python
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(utilityEmployment.index, utilityEmployment['utilities_employment'],
        alpha=0.35, color='steelblue', label='Monthly')
ax.plot(utilityEmployment.index, utilityEmployment['moving_avg_12'],
        linewidth=2.5, color='navy', label='12-Month Moving Average')
ax.set_title('Figure 2: U.S. Utilities Employment — Smoothed Trend', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Employment (Thousands of Workers)')
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('../figures/fig2_employment_smoothed.png', dpi=150)
plt.show()
```

**Interpretation:**  
The 12-month moving average confirms the multi-phase pattern identified in Figure 1, suppressing month-to-month noise to reveal the underlying trend clearly. The smoothed series reinforces the observation that the employment peak in the early 1990s was a structural turning point, followed by a sustained decline rather than a cyclical dip.

---

### 7.3 Indexed Comparison: Nuclear Generation and Utilities Employment (Figure 3)

**Code:**

```python
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_indexed.index, df_indexed['nuclear_generation'],
        label='Nuclear Generation (Indexed)', color='firebrick', linewidth=1.8)
ax.plot(df_indexed.index, df_indexed['utilities_employment'],
        label='Utilities Employment (Indexed)', color='steelblue', linewidth=1.8)
ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, label='Base = 100')
ax.set_title('Figure 3: Indexed Comparison — Nuclear Generation vs. Utilities Employment', fontsize=14)
ax.set_ylabel('Index (Base Period = 100)')
ax.set_xlabel('Year')
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('../figures/fig3_indexed_comparison.png', dpi=150)
plt.show()
```

**Interpretation:**  
Figures 1–3 collectively illustrate a structural shift characterized by a transition from a developmental phase to a stabilization phase. During the late 1970s through the early 1990s, employment and nuclear production grow together. After the mid-1990s, however, nuclear generation continues at a high and relatively stable level of output while employment does not follow proportionally.

This divergence indicates that nuclear energy production is not labor-proportional in the long run. As the sector matured, increases in output were driven more by capital investment, technological improvements, and operational efficiency than by workforce expansion — consistent with the broader capital-intensive trajectory of the energy sector.

---

## 8. Descriptive Statistics

```python
print("=== Utilities Employment ===")
print(utilityEmployment[['utilities_employment']].describe().T.round(2))

print("\n=== Nuclear Generation ===")
print(nuclearProduction.describe().T.round(2))

print("\n=== Merged Dataset (Aligned Period) ===")
print(df.describe().T.round(2))
```

| Statistic | Utilities Employment (000s) | Nuclear Generation (Index) |
|---|---|---|
| Count | ~648 obs. | ~648 obs. |
| Mean | ~582 | ~76 |
| Std Dev | ~82 | ~28 |
| Min | ~395 | ~9 |
| Max | ~697 | ~109 |

*Note: Values are approximate; run the notebook to generate exact statistics from the FRED CSVs.*

---

## 9. Exploratory Regression Analysis

This section adds an OLS regression to quantify the association between nuclear electricity generation and utilities employment over the aligned sample period. The results are interpreted cautiously and non-causally; this analysis is exploratory in nature.

**Model:**  
Utilities Employment = α + β × Nuclear Generation + ε

```python
import statsmodels.api as sm

# Prepare variables
X = df['nuclear_generation']
y = df['utilities_employment']
X_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_const).fit()
print(model.summary())
```

**Scatter Plot with Fitted Regression Line (Figure 4):**

```python
import numpy as np

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df['nuclear_generation'], df['utilities_employment'],
           alpha=0.3, color='steelblue', s=15, label='Monthly Observations')

# Fitted line
x_range = np.linspace(df['nuclear_generation'].min(), df['nuclear_generation'].max(), 200)
x_range_const = sm.add_constant(x_range)
y_hat = model.predict(x_range_const)
ax.plot(x_range, y_hat, color='firebrick', linewidth=2, label='OLS Fitted Line')

ax.set_title('Figure 4: Nuclear Generation vs. Utilities Employment — OLS Regression', fontsize=13)
ax.set_xlabel('Nuclear Generation Index (2017 = 100)')
ax.set_ylabel('Utilities Employment (Thousands)')
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('../figures/fig4_regression.png', dpi=150)
plt.show()
```

**Interpretation:**  
The OLS regression explores the cross-sectional association between nuclear generation and utilities employment across the full sample period. A positive coefficient on nuclear generation would suggest that, on average, higher production levels are associated with higher employment — consistent with the developmental phase described earlier. However, this pooled regression masks the structural break in the relationship that occurs around the mid-1990s. The regression result should be interpreted as a descriptive summary of average co-movement across the full period, not as evidence of a causal or stable relationship. The R-squared value provides a rough indication of how much variation in employment is associated with variation in production, but the non-stationarity of both time series limits statistical inference.

---

## 10. Limitations

This analysis is exploratory and subject to several important limitations:

**Limited variable scope.** The analysis includes only two variables — nuclear electricity generation and utilities employment. The utilities sector encompasses natural gas, hydroelectric, and other power sources whose employment dynamics may differ substantially from nuclear. Omitting sector-specific employment data limits the precision of any inference about nuclear energy's specific labor footprint.

**Omitted institutional and regulatory variables.** Major policy events — including the Energy Policy Act of 1992, deregulation of electricity markets in the 1990s, and post-Fukushima regulatory changes — likely had direct effects on both employment and generation. These factors are not controlled for.

**Correlation does not imply causation.** The positive association observed in earlier decades and the divergence in later decades may each reflect common third factors (economic cycles, capital investment trends) rather than a direct relationship between production and employment.

**Time-series limitations.** Both series are non-stationary over the full period, which can inflate correlations and complicate regression inference. Formal stationarity testing and cointegration analysis are beyond the scope of this exploratory brief but would be appropriate next steps.

**Exploratory analytical design.** This analysis was designed for skill development and portfolio building. It does not meet the methodological standards required for causal inference or policy prescription.

---

## 11. Conclusion

This analysis documents a clear structural shift in the relationship between nuclear electricity production and utilities sector employment in the United States. During the infrastructure expansion phase of the 1970s through early 1990s, employment and production grew broadly together. As the sector matured after the mid-1990s, production stabilized at high levels while employment declined and plateaued at a lower baseline.

This divergence is consistent with a transition from a labor-intensive expansion model to a capital-intensive and efficiency-driven operating model — a pattern observable across many mature infrastructure industries. The finding does not support a simple proportional relationship between nuclear output and workforce size over the long run.

These insights carry implications for energy labor policy: sustained or expanded nuclear generation may not translate directly into proportional employment gains under current operational paradigms. Workforce planning in the utilities sector must account for the capital-efficiency trajectory of mature nuclear infrastructure, even as new construction activity (if it occurs) may temporarily reverse this trend.

---

## 12. References and Data Sources

U.S. Bureau of Labor Statistics. *All Employees, Utilities* [CES4422000001]. Retrieved from FRED, Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/CES4422000001. March 19, 2026.

Board of Governors of the Federal Reserve System. *Industrial Production: Electric and Gas Utilities: Nuclear Electric Power* [IPN221113N]. Retrieved from FRED, Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/IPN221113N.

Wooldridge, J. M. (2019). *Introductory Econometrics: A Modern Approach* (7th ed.). Cengage Learning.

---

*This report was prepared as part of the FRED Economic Analysis Portfolio by Dwanjai Oprien, May 2026. It represents an applied economic analysis practicum and is not intended for academic publication.*