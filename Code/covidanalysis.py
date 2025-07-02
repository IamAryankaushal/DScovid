# ---Data Cleaning and Preparation ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Clean GDP Data ---
print("Cleaning GDP data...")
gdp_df = pd.read_csv('C:/covid/Code/gdp_growth_annual.csv')
gdp_df.rename(columns={
    '2019 [YR2019]': '2019',
    '2020 [YR2020]': '2020',
    '2021 [YR2021]': '2021',
    '2022 [YR2022]': '2022'
}, inplace=True)
gdp_df = gdp_df.drop(columns=['Series Name', 'Series Code', '2019'])
gdp_df = gdp_df.dropna()
print(gdp_df.info())
print("GDP cleaned")

# --- Clean Unemployment Data ---
print("Cleaning Unemployment Data...")
udf = pd.read_csv('C:/covid/Code/global_unemployment.csv')
udf.rename(columns={
    '2015 [YR2015]': '2015',
    '2016 [YR2016]': '2016',
    '2017 [YR2017]': '2017',
    '2018 [YR2018]': '2018',
    '2023 [YR2023]': '2023',
    '2024 [YR2024]': '2024',
    '2019 [YR2019]': '2019',
    '2020 [YR2020]': '2020',
    '2021 [YR2021]': '2021',
    '2022 [YR2022]': '2022'
}, inplace=True)
udf = udf.drop(columns=['Series Name', 'Series Code', '2015', '2016', '2017', '2018', '2019', '2023', '2024'])
udf = udf.dropna()
print(udf.info())
print("Unemployment data cleaned")

# --- Clean COVID Data ---
print("Cleaning COVID data...")
covid_df = pd.read_csv('C:/covid/Code/WHO-COVID-19-global-daily-data.csv')
covid_df['Date_reported'] = pd.to_datetime(covid_df['Date_reported'])
covid_df = covid_df[covid_df['Date_reported'].dt.year.isin([2020, 2021, 2022])]
covid_df['New_cases'] = covid_df['New_cases'].fillna(0)
covid_df['New_deaths'] = covid_df['New_deaths'].fillna(0)
covid_df['Year'] = covid_df['Date_reported'].dt.year

# Country name standardization
country_rename_map = {
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Côte d’Ivoire': "Cote d'Ivoire",
    'Democratic Republic of the Congo': 'Congo, Dem. Rep.',
    'Congo': 'Congo, Rep.',
    'Egypt': 'Egypt, Arab Rep.',
    'Gambia': 'Gambia, The',
    "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
    'Iran (Islamic Republic of)': 'Iran, Islamic Rep.',
    'Republic of Korea': 'Korea, Rep.',
    'Lao People\'s Democratic Republic': 'Lao PDR',
    'Russian Federation': 'Russia',
    'Syrian Arab Republic': 'Syria',
    'United Republic of Tanzania': 'Tanzania',
    'United States of America': 'United States',
    'Venezuela (Bolivarian Republic of)': 'Venezuela, RB',
    'Viet Nam': 'Vietnam',
    'Bahamas': 'Bahamas, The',
    'Brunei Darussalam': 'Brunei Darussalam',
    'Czechia': 'Czech Republic',
    'Hong Kong SAR': 'Hong Kong SAR, China',
    'Kyrgyzstan': 'Kyrgyz Republic',
    'Slovakia': 'Slovak Republic',
    'Türkiye': 'Turkey',
    'West Bank and Gaza': 'West Bank and Gaza',
    'Yemen': 'Yemen, Rep.'
}
covid_df['Country'] = covid_df['Country'].replace(country_rename_map)

# Aggregate COVID stats
covid_agg = covid_df.groupby(['Country', 'Year']).agg({
    'New_cases': 'sum',
    'New_deaths': 'sum'
}).reset_index()

print(covid_agg.info())
print("COVID data cleaned and aggregated")

# --- Transform and Merge Datasets ---
gdp_long = pd.melt(gdp_df, id_vars=['Country Name', 'Country Code'], value_vars=['2020', '2021', '2022'],
                   var_name='Year', value_name='GDP_Growth')
gdp_long['Year'] = gdp_long['Year'].astype(int)

unemp_long = pd.melt(udf, id_vars=['Country Name', 'Country Code'], value_vars=['2020', '2021', '2022'],
                     var_name='Year', value_name='Unemployment_Rate')
unemp_long['Year'] = unemp_long['Year'].astype(int)

eco_df = pd.merge(gdp_long, unemp_long, on=['Country Name', 'Country Code', 'Year'])
merged_df = pd.merge(covid_agg, eco_df, how='inner', left_on=['Country', 'Year'], right_on=['Country Name', 'Year'])
merged_df.drop(columns=['Country Name'], inplace=True)

print("Final merged dataset preview:")
print(merged_df.head())
#merged_df.to_csv("C:/covid/Code/Merged dataset.csv", index=False)

# SELECTING TEST COUNTRIES FOR THE PROJECT 
selected_countries = ['India', 'United States', 'Brazil', 'Italy']

# --- COVID Trend Line Plot ---
covid_trend = covid_df[covid_df['Country'].isin(selected_countries)].copy()
covid_trend['Cases_7DayAvg'] = covid_trend.groupby('Country')['New_cases'].transform(lambda x: x.rolling(7).mean())

plt.figure(figsize=(14, 6))
sns.lineplot(data=covid_trend, x='Date_reported', y='Cases_7DayAvg', hue='Country')
plt.title("COVID-19 Cases (7-day Rolling Avg) - Selected Countries - 2020 to 2022")
plt.xlabel("Date")
plt.ylabel("New Cases (7-day Avg)")
plt.xticks(rotation=45)
plt.tight_layout()
#plt.savefig("COVID-19 Cases (7-day Rolling Avg) - Selected Countries.png")
plt.show()


# --- GDP vs COVID Scatter Plot with Country Labels ---
selected_df = merged_df[merged_df['Country'].isin(selected_countries)].copy()

plt.figure(figsize=(14, 8))
sns.scatterplot(data=selected_df, x='New_cases', y='GDP_Growth', hue='Year', s=100, alpha=0.7)

for i, row in selected_df.iterrows():
    plt.text(row['New_cases'], row['GDP_Growth'], row['Country'], fontsize=8, alpha=0.7)

plt.xscale('log')
plt.xlabel("Total COVID-19 Cases (Log Scale)")
plt.ylabel("GDP Growth (%)")
plt.title("GDP Growth vs COVID-19 Cases (2020–2022)")
plt.grid(True)
plt.tight_layout()
plt.legend(title='Year')
#plt.savefig("GDP Growth vs COVID-19 Cases (2020–2022).png")
plt.show()

# --- Pearson Correlation Coefficient Calculation ---
for year in [2020, 2021, 2022]:
    sub = merged_df[merged_df['Year'] == year]
    corr = sub['New_cases'].corr(sub['GDP_Growth'])
    print(f"Pearson correlation (New Cases vs GDP Growth) in {year}: {corr:.3f}")

# --- Unemployment Spike SQL Analysis ---
unemp_pivot = unemp_long.pivot(index='Country Name', columns='Year', values='Unemployment_Rate').reset_index()
conn = sqlite3.connect(':memory:')
unemp_pivot.to_sql('unemployment', conn, index=False, if_exists='replace')

query = """
SELECT 
    [Country Name], 
    ROUND([2021] - [2020], 2) AS Spike_2020_2021,
    ROUND([2022] - [2021], 2) AS Spike_2021_2022,
    ROUND(MAX([2021] - [2020], [2022] - [2021]), 2) AS Max_Spike
FROM unemployment
ORDER BY Max_Spike DESC
LIMIT 10;
"""
sql_result = pd.read_sql(query, conn)
print("\nTop 10 Countries with Highest Unemployment Spike (via SQL):")
print(sql_result)

# --- Comparative Dashboard Summary ---
summary_df = merged_df.groupby('Country').agg({
    'New_cases': 'sum',
    'New_deaths': 'sum',
    'GDP_Growth': 'mean',
    'Unemployment_Rate': 'max'
}).reset_index()

summary_df.rename(columns={
    'New_cases': 'Total_Cases_2020_2022',
    'New_deaths': 'Total_Deaths_2020_2022',
    'GDP_Growth': 'Avg_GDP_Growth',
    'Unemployment_Rate': 'Max_Unemployment_Rate'
}, inplace=True)


print("\nComparative Dashboard Summary:")
print(summary_df.head())

# --- SQL Queries on Summary Table ---
summary_df.to_sql("summary", conn, index=False, if_exists='replace')

sql_top_deaths = pd.read_sql("""
SELECT Country, Total_Deaths_2020_2022
FROM summary
ORDER BY Total_Deaths_2020_2022 DESC
LIMIT 10;
""", conn)

sql_worst_gdp = pd.read_sql("""
SELECT Country, Avg_GDP_Growth
FROM summary
ORDER BY Avg_GDP_Growth ASC
LIMIT 10;
""", conn)

sql_peak_unemployment = pd.read_sql("""
SELECT Country, Max_Unemployment_Rate
FROM summary
ORDER BY Max_Unemployment_Rate DESC
LIMIT 10;
""", conn)

print("\nTop 10 Countries by COVID Deaths (SQL):")
print(sql_top_deaths)

print("\nTop 10 Countries with Lowest Avg GDP Growth (SQL):")
print(sql_worst_gdp)

print("\nTop 10 Countries by Peak Unemployment Rate (SQL):")
print(sql_peak_unemployment)

# --- Heatmap Visualization ---
heat_df = summary_df.copy()
cols_to_normalize = ['Total_Cases_2020_2022', 'Total_Deaths_2020_2022', 'Avg_GDP_Growth', 'Max_Unemployment_Rate']
heat_df['Avg_GDP_Growth'] = -heat_df['Avg_GDP_Growth']

scaler = MinMaxScaler()
heat_df_scaled = heat_df.copy()
heat_df_scaled[cols_to_normalize] = scaler.fit_transform(heat_df[cols_to_normalize])
heat_df_scaled.set_index('Country', inplace=True)

plt.figure(figsize=(14, 10))
sns.heatmap(heat_df_scaled.sort_values(by='Total_Deaths_2020_2022', ascending=False).head(20),
            cmap='Reds', linewidths=0.5, annot=True)
plt.title("COVID-19 Economic Impact Heatmap (Top 20 by Deaths)")
plt.tight_layout()
#plt.savefig("COVID-19 Economic Impact Heatmap (Top 20 by Deaths).png")
plt.show()

# --- K-Means Clustering & PCA Dimensionality Reduction ---

print("\nRunning K-Means clustering with PCA")

# Select & prepare features
cluster_df = summary_df.copy()
features = cluster_df[['Total_Cases_2020_2022', 'Total_Deaths_2020_2022',
                       'Avg_GDP_Growth', 'Max_Unemployment_Rate']].copy()
features['Avg_GDP_Growth'] = -features['Avg_GDP_Growth']  # Invert for consistency

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)
cluster_df['Cluster'] = cluster_labels

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)
cluster_df['PCA1'] = pca_components[:, 0]
cluster_df['PCA2'] = pca_components[:, 1]

# Plot Clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=cluster_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100)
for i, row in cluster_df.iterrows():
    plt.text(row['PCA1'], row['PCA2'], row['Country'], fontsize=8, alpha=0.7)

plt.title("K-Means Clustering of Countries by COVID-19 Impact (via PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
#plt.savefig("KMeans Clusters COVID Impact PCA.png")
plt.show()
