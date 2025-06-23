# --- Part 1: Data Cleaning and Preparation ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Clean GDP Data ---
print("Cleaning GDP data...")
gdp_df = pd.read_csv('C:/covid/data/gdp_growth_annual.csv')
gdp_df.rename(columns={
    '2019 [YR2019]': '2019',
    '2020 [YR2020]': '2020',
    '2021 [YR2021]': '2021',
    '2022 [YR2022]': '2022'
}, inplace=True)
gdp_df = gdp_df.drop(columns=['Series Name', 'Series Code', '2019'])
gdp_df = gdp_df.dropna()
print("GDP cleaned")

# --- Clean Unemployment Data ---
print("Cleaning Unemployment Data...")
udf = pd.read_csv('C:/covid/data/global_unemployment.csv')
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
print("Unemployment data cleaned")

# --- Clean COVID Data ---
print("Cleaning COVID data...")
covid_df = pd.read_csv('C:/covid/data/WHO-COVID-19-global-daily-data.csv')
covid_df['Date_reported'] = pd.to_datetime(covid_df['Date_reported'])
covid_df = covid_df[covid_df['Date_reported'].dt.year.isin([2020, 2021, 2022])]
covid_df['New_cases'] = covid_df['New_cases'].fillna(0)
covid_df['New_deaths'] = covid_df['New_deaths'].fillna(0)
covid_df['Year'] = covid_df['Date_reported'].dt.year

# Standardize country names for merging
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

# Aggregate yearly COVID stats per country
covid_agg = covid_df.groupby(['Country', 'Year']).agg({
    'New_cases': 'sum',
    'New_deaths': 'sum'
}).reset_index()


print("COVID data cleaned and aggregated")
print(covid_agg.head())
# Display preview of the cleaned datasets
#gdp_df.head(), udf.head(), covid_agg.head()

# --- Part 2: Transform and Merge Datasets ---
# Transform GDP data: wide to long format

gdp_long = pd.melt(gdp_df, 
                   id_vars=['Country Name', 'Country Code'], 
                   value_vars=['2020', '2021', '2022'],
                   var_name='Year', 
                   value_name='GDP_Growth')
gdp_long['Year'] = gdp_long['Year'].astype(int)

# Transform Unemployment data: wide to long format
unemp_long = pd.melt(udf, 
                     id_vars=['Country Name', 'Country Code'], 
                     value_vars=['2020', '2021', '2022'],
                     var_name='Year', 
                     value_name='Unemployment_Rate')
unemp_long['Year'] = unemp_long['Year'].astype(int)

# Merge GDP and Unemployment on Country Name and Year
eco_df = pd.merge(gdp_long, unemp_long, on=['Country Name', 'Country Code', 'Year'])

# Merge with COVID data (align Country name from COVID to GDP)
merged_df = pd.merge(covid_agg, 
                     eco_df, 
                     how='inner', 
                     left_on=['Country', 'Year'], 
                     right_on=['Country Name', 'Year'])

# Final clean-up
merged_df.drop(columns=['Country Name'], inplace=True)

# Preview final merged dataset
print("Final merged dataset preview:")
print(merged_df.head())
