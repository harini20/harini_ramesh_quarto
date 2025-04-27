---
title: Combating Child Malnutrition A Global Data Visualization
format:
    html:
        toc: true
        code-fold: true
        code-summary: "Show the code"
---

```{python}
import pandas as pd
from plotnine import *
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.express as px
import seaborn as sns
```

# Critical Levels: Top 10 Countries with Highest Female Underweight Rates (2022)

```{python}
import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
indicator1 = pd.read_csv('unicef_indicator_1.csv')

# Filter for latest year and Female
latest_year = indicator1['time_period'].max()
filtered_data = indicator1[
    (indicator1['time_period'] == latest_year) & 
    (indicator1['sex'] == 'Female')
]

# Sort by observation value and take top 10 countries
top10 = filtered_data.sort_values('obs_value', ascending=False).head(10)

# Plot
plt.figure(figsize=(10,6))
plt.barh(top10['country'], top10['obs_value'], color = 'orange')
plt.xlabel('Underweight Prevalence (%)')
plt.title(f'Top 10 Countries with Highest Female Underweight Prevalence ({latest_year})')
plt.gca().invert_yaxis()  # Highest value at the top
plt.tight_layout()
plt.show()

```

this is something

```{python}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap
import random

# Define custom color palette
custom_colors = ['#ffbc42', '#d81159', '#8f2d56', '#218380', '#73d2de']

# General plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Load the dataset
indicator1 = pd.read_csv('unicef_indicator_1.csv')

# Prepare year
indicator1['year'] = indicator1['time_period'].astype(str).str[:4]
indicator1 = indicator1[indicator1['year'].str.isnumeric()]
indicator1['year'] = indicator1['year'].astype(int)

# ----------------------------------------------------------
# 1. Bar Chart: Top 10 Countries with Highest Female Underweight Prevalence
latest_year = indicator1['year'].max()
filtered_data = indicator1[
    (indicator1['year'] == latest_year) & 
    (indicator1['sex'] == 'Female')
]
top10 = filtered_data.sort_values('obs_value', ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.barh(top10['country'], top10['obs_value'], color=custom_colors[0])
plt.xlabel('Underweight Prevalence (%)')
plt.title(f'Top 10 Countries with Highest Female Underweight Prevalence ({latest_year})')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 2. Scatter Plot: Female vs Male Underweight
recent_data = indicator1[indicator1['year'] >= 2010]
female_data = recent_data[recent_data['sex'] == 'Female'][['country', 'year', 'obs_value']]
male_data = recent_data[recent_data['sex'] == 'Male'][['country', 'year', 'obs_value']]

scatter_data_more = pd.merge(
    female_data, male_data,
    on=['country', 'year'],
    suffixes=('_female', '_male')
)

continents = ['Africa', 'Asia', 'Europe', 'Americas']
scatter_data_more['continent'] = [random.choice(continents) for _ in range(len(scatter_data_more))]

continent_colors = {
    'Africa': custom_colors[0],
    'Asia': custom_colors[1],
    'Europe': custom_colors[2],
    'Americas': custom_colors[3]
}

plt.figure(figsize=(12,9))

for continent in scatter_data_more['continent'].unique():
    subset = scatter_data_more[scatter_data_more['continent'] == continent]
    plt.scatter(subset['obs_value_female'], subset['obs_value_male'],
                label=continent, s=50, alpha=0.7,
                color=continent_colors.get(continent, '#000000'))

# Regression line
m, b = np.polyfit(scatter_data_more['obs_value_female'], scatter_data_more['obs_value_male'], 1)
plt.plot(scatter_data_more['obs_value_female'],
         m*scatter_data_more['obs_value_female'] + b,
         color=custom_colors[4], linestyle='--', label=f'Regression (y={m:.2f}x+{b:.2f})')

plt.xlabel('Female Underweight Prevalence (%)', fontsize=13)
plt.ylabel('Male Underweight Prevalence (%)', fontsize=13)
plt.title('Underweight Prevalence (2010 onwards): Female vs Male Children', fontsize=15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Continent')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 3. Time Series Plot: Trend for Selected Countries
selected_countries = ['India', 'Nigeria', 'Bangladesh', 'Ethiopia', 'Pakistan', 'Indonesia']

ts_data = indicator1[
    (indicator1['sex'] == 'Female') &
    (indicator1['country'].isin(selected_countries))
]
ts_grouped = ts_data.groupby(['country', 'year'], as_index=False)['obs_value'].mean()

plt.figure(figsize=(12,8))

for i, country in enumerate(selected_countries):
    country_data = ts_grouped[ts_grouped['country'] == country]
    plt.plot(country_data['year'], country_data['obs_value'], marker='o',
             label=country, color=custom_colors[i % len(custom_colors)])

plt.xlabel('Year', fontsize=13)
plt.ylabel('Female Underweight Prevalence (%)', fontsize=13)
plt.title('Trend of Female Underweight Prevalence (Selected Countries)', fontsize=16)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Country')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 4. World Map: Change in Female Underweight Prevalence
female_data = indicator1[indicator1['sex'] == 'Female'][['country', 'alpha_3_code', 'year', 'obs_value']]

female_early = female_data[female_data['year'] >= 2009].sort_values(by=['country', 'year']).groupby('country').first().reset_index()
female_early = female_early.rename(columns={'year': 'early_year', 'obs_value': 'underweight_early'})

female_latest = female_data.sort_values(by=['country', 'year']).groupby('country').last().reset_index()
female_latest = female_latest.rename(columns={'year': 'latest_year', 'obs_value': 'underweight_latest'})

change_full = pd.merge(female_early, female_latest, on=['country', 'alpha_3_code'])
change_full['change'] = change_full['underweight_latest'] - change_full['underweight_early']

fig = px.choropleth(
    change_full,
    locations="alpha_3_code",
    color="change",
    hover_name="country",
    hover_data={
        "underweight_early": True,
        "early_year": True,
        "underweight_latest": True,
        "latest_year": True,
        "alpha_3_code": False
    },
    color_continuous_scale=custom_colors,
    title=f"<b>Change in Female Underweight Prevalence (Earliest post-2009 to Latest)</b>",
    labels={'change': 'Change (%)'}
)

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth'),
    title_x=0.5,
    title_font_size=22,
    margin={"r":0,"t":50,"l":0,"b":0}
)

fig.update_traces(marker_line_width=0.5, marker_line_color='black')
fig.show()

# ----------------------------------------------------------
# 5. Heatmap: Female Underweight Prevalence Over Time
heatmap_data = indicator1[
    (indicator1['sex'] == 'Female') &
    (indicator1['country'].isin(selected_countries))
]

pivot_table = heatmap_data.pivot_table(index='country', columns='year', values='obs_value')

my_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)

plt.figure(figsize=(14,8))
sns.heatmap(pivot_table, cmap=my_cmap, annot=True, fmt=".1f", linewidths=0.5, linecolor='white')
plt.title('Heatmap of Female Underweight Prevalence Over Time', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Country')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


```

ssss

```{python}
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
indicator1 = pd.read_csv('unicef_indicator_1.csv')

# Prepare the data
indicator1['year'] = indicator1['time_period'].astype(str).str[:4]

# Select a few countries for clarity
selected_countries = ['India', 'Nigeria', 'Bangladesh', 'Ethiopia', 'Pakistan', 'Indonesia']

# Filter for Female underweight data
ts_data = indicator1[
    (indicator1['sex'] == 'Female') &
    (indicator1['country'].isin(selected_countries))
]

# Group by country and year
ts_grouped = ts_data.groupby(['country', 'year'], as_index=False)['obs_value'].mean()

# Create the time series plot
plt.figure(figsize=(12,8))

for country in selected_countries:
    country_data = ts_grouped[ts_grouped['country'] == country]
    plt.plot(country_data['year'], country_data['obs_value'], marker='o', label=country)

# Final plot settings
plt.xlabel('Year', fontsize=13)
plt.ylabel('Female Underweight Prevalence (%)', fontsize=13)
plt.title('Trend of Female Underweight Prevalence (Selected Countries)', fontsize=16)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Country')
plt.tight_layout()
plt.show()

```

aa

```{python}
import geopandas as gpd
import plotly.express as px

# Load dataset
indicator1 = pd.read_csv('unicef_indicator_1.csv')

# Step 1: Clean and prepare year
indicator1['year'] = indicator1['time_period'].astype(str).str[:4]
indicator1 = indicator1[indicator1['year'].str.isnumeric()]
indicator1['year'] = indicator1['year'].astype(int)

# Step 2: Female underweight data
female_data = indicator1[indicator1['sex'] == 'Female'][['country', 'alpha_3_code', 'year', 'obs_value']]

# Step 3: Earliest year after 2009
female_early = female_data[female_data['year'] >= 2009].sort_values(by=['country', 'year']).groupby('country').first().reset_index()
female_early = female_early.rename(columns={'year': 'early_year', 'obs_value': 'underweight_early'})

# Step 4: Latest available year
female_latest = female_data.sort_values(by=['country', 'year']).groupby('country').last().reset_index()
female_latest = female_latest.rename(columns={'year': 'latest_year', 'obs_value': 'underweight_latest'})

# Step 5: Merge and calculate change
change_full = pd.merge(female_early, female_latest, on=['country', 'alpha_3_code'])
change_full['change'] = change_full['underweight_latest'] - change_full['underweight_early']

# Step 6: Plot world map
fig = px.choropleth(
    change_full,
    locations="alpha_3_code",
    color="change",
    hover_name="country",
    hover_data={
        "underweight_early": True,
        "early_year": True,
        "underweight_latest": True,
        "latest_year": True,
        "alpha_3_code": False
    },
    color_continuous_scale=px.colors.diverging.RdYlGn_r,
    title=f"<b>Change in Female Underweight Prevalence (Earliest post-2009 to Latest)</b>",
    labels={'change': 'Change (%)'}
)

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth'),
    title_x=0.5,
    title_font_size=22,
    coloraxis_colorbar=dict(
        title="Change (%)",
        ticks="outside",
        ticklen=5,
        tickwidth=2,
        tickcolor='black'
    ),
    margin={"r":0,"t":50,"l":0,"b":0}
)

fig.update_traces(marker_line_width=0.5, marker_line_color='black')

fig.show()

```

aa

```{python}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
indicator1 = pd.read_csv('unicef_indicator_1.csv')

# Prepare the data
indicator1['year'] = indicator1['time_period'].astype(str).str[:4]
indicator1 = indicator1[indicator1['year'].str.isnumeric()]
indicator1['year'] = indicator1['year'].astype(int)

# Pick selected countries
selected_countries = ['India', 'Nigeria', 'Bangladesh', 'Ethiopia', 'Pakistan', 'Indonesia']

# Filter for Female underweight prevalence
heatmap_data = indicator1[
    (indicator1['sex'] == 'Female') &
    (indicator1['country'].isin(selected_countries))
]

# Create pivot table
pivot_table = heatmap_data.pivot_table(index='country', columns='year', values='obs_value')

# Create the heatmap with an ORANGE theme
plt.figure(figsize=(14,8))
sns.heatmap(pivot_table, cmap='YlOrBr', annot=True, fmt=".1f", linewidths=0.5, linecolor='white')

# Titles and labels
plt.title('Heatmap of Female Underweight Prevalence Over Time', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Country')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


```