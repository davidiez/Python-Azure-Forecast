# Python script to generate the components a forecasted plot (using Prophet model) of 60-day period. It will show 3 plots: trend, weekly and yearly forecast.
# To be used in any Power BI dashboard, getting as dataset an Amazon billing data in FOCUS format.

# Install Prophet if needed
# pip install prophet

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Power BI provides the dataset as 'dataset'
df = dataset.copy()

# Convert date and cost columns
df['ChargePeriodStart'] = pd.to_datetime(df['ChargePeriodStart'], errors='coerce')
df['ChargePeriodStart'] = df['ChargePeriodStart'].dt.tz_localize(None)  # Remove timezone
df['EffectiveCost'] = pd.to_numeric(df['EffectiveCost'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['ChargePeriodStart', 'EffectiveCost'])

# Remove top 5% outliers
df = df[df['EffectiveCost'] < df['EffectiveCost'].quantile(0.95)] 

# Aggregate cost by month
daily_cost = df.groupby(pd.Grouper(key='ChargePeriodStart', freq='MS'))['EffectiveCost'].sum().reset_index()
daily_cost.columns = ['ds', 'y']

# Initialize Prophet and add weekly and yearly seasonality
model = Prophet()
model.add_seasonality(name='weekly', period=7, fourier_order=5)
model.add_seasonality(name='yearly', period=365, fourier_order=15)

# Fit the model
model.fit(daily_cost)

# Forecast the next 60 days
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Plot forecast components (trend, seasonality)
fig2 = model.plot_components(forecast)

# Add titles and labels
plt.xlabel("Month")
plt.tight_layout()

# Print the plot
plt.show()

