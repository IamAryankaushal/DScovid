# 📊 COVID-19 Economic Impact Analysis (2020–2022)

This data science project analyzes how the COVID-19 pandemic impacted global economies — specifically GDP growth and unemployment trends — by integrating data from the **World Health Organization (WHO)** and **World Bank** across **100+ countries**.

The project walks through full-cycle data processing: from data cleaning and transformation, to SQL-based queries, insightful visualizations, and unsupervised machine learning (K-Means clustering + PCA).

---

## 🔍 Objectives

- Analyze COVID-19 case trends from 2020–2022 for selected countries.
- Examine the correlation between pandemic severity and GDP growth.
- Identify countries with major unemployment spikes using SQL.
- Build a comparative economic impact dashboard.
- Group countries into impact clusters using K-Means and PCA.

---

## 🗂 Datasets Used

| Dataset | Source | Description |
|--------|--------|-------------|
| `WHO-COVID-19-global-daily-data.csv` | [WHO](https://covid19.who.int) | Daily reported cases and deaths globally |
| `gdp_growth_annual.csv` | World Bank | Annual GDP growth rates per country |
| `global_unemployment.csv` | World Bank | Annual total unemployment rates per country |

---

## 🧰 Tools & Technologies

- **Python** – pandas, matplotlib, seaborn, scikit-learn, sqlite3
- **SQL** – In-memory analysis using SQLite
- **Machine Learning** – K-Means Clustering, Principal Component Analysis (PCA)
- **Visualization** – Line plots, scatter plots, heatmaps

---

## 🧼 Data Cleaning & Preparation

- Standardized country names across all datasets
- Removed unnecessary columns and handled missing data
- Transformed GDP and unemployment data from wide to long format
- Aggregated daily COVID data to yearly totals per country
- Merged all datasets on `Country` and `Year` to form a unified dataset

---

## 📈 Visualizations

### 1. **COVID-19 Case Trends**
- **Type**: Line Chart (7-day Rolling Avg)
- **Countries Selected**:
  - `India`, `United States`, `Brazil`, `Italy`

<img src="visualizations/covid_trend.png" width="600">

---

### 2. **GDP Growth vs COVID Severity**
- **Type**: Scatter Plot (log scale on cases)
- Visualizes how total COVID-19 case counts relate to GDP growth, year-wise.

<img src="visualizations/gdp_vs_cases.png" width="600">

---

### 3. **Comparative Heatmap Dashboard**
- **Type**: Heatmap (Top 20 countries by COVID deaths)
- Normalized comparison across:
  - Total cases
  - Total deaths
  - Avg GDP growth
  - Max unemployment

<img src="visualizations/heatmap_dashboard.png" width="600">

---

### 4. **Clustering (K-Means + PCA)**
- Grouped countries into 4 clusters based on pandemic + economic metrics:
  - Total cases
  - Total deaths
  - Avg GDP growth
  - Max unemployment
- Dimensionality reduction via PCA for visualization

<img src="visualizations/kmeans_pca.png" width="600">

---

## 🧠 Insights & Findings

- Some countries with high COVID cases (like the US) showed relatively resilient GDP trends due to strong economic buffers.
- Countries like **Brazil**, **India**, and **South Africa** saw both high mortality and economic contraction.
- Tourism and service-reliant economies faced unemployment spikes even if case counts were moderate.
- Clustering revealed 4 distinct global impact profiles:
  - High mortality + economic collapse
  - High cases but resilient economy
  - Low impact across all metrics
  - Moderate cases + sharp unemployment rise

---

