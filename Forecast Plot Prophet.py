# Python script to generate a forecasted plot (using Prophet model) of 60-day period.
# To be used in any Power BI dashboard, getting as dataset an Amazon billing data in FOCUS format.

# Install Prophet if needed
# pip install prophet

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from prophet.plot import add_changepoints_to_plot

# Power BI provides the dataset as 'dataset'
df = dataset.copy()

# Definition of the function to do the transformation
def thousands(x, pos):
    return f'{x*1e-3:.0f}K'

# Convert date and cost columns
df['ChargePeriodStart'] = pd.to_datetime(df['ChargePeriodStart'], errors='coerce')
df['ChargePeriodStart'] = df['ChargePeriodStart'].dt.tz_localize(None)  # Remove timezone
df['EffectiveCost'] = pd.to_numeric(df['EffectiveCost'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['ChargePeriodStart', 'EffectiveCost'])

# Remove top 5% outliers
df = df[df['EffectiveCost'] < df['EffectiveCost'].quantile(0.95)] 

# Aggregate cost by day (as monthly seasonality is added manually)
daily_cost = df.groupby(pd.Grouper(key='ChargePeriodStart', freq='D'))['EffectiveCost'].sum().reset_index()
daily_cost.columns = ['ds', 'y']

# Initialize Prophet and add monthly seasonality
model = Prophet()
model.add_seasonality(name='monthly', period=90, fourier_order=5)

# Fit the model
model.fit(daily_cost)

# Forecast the next 60 days
future = model.make_future_dataframe(periods=60, freq='D')
forecast = model.predict(future)

# Plot the forecast
fig1 = model.plot(forecast)

# Apply formatter to Y-axis
ax = fig1.gca()
ax.yaxis.set_major_formatter(FuncFormatter(thousands))

# Add changepoint line
a = add_changepoints_to_plot(fig1.gca(), model, forecast)

# Add titles and labels
plt.title("60-day Forecast (Prophet model)", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Cost (NOK)")
plt.tight_layout()

# Print the plot
plt.show()

