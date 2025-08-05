# Python script to generate a forecasted plot (using 'statsmodels' SARIMAX model) of 60-day period.
# To be used in any Power BI dashboard, getting as dataset an Amazon billing data in FOCUS format.

# Install Prophet if needed
# pip install prophet

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore")

# Power BI provides the dataset as 'dataset'
df = dataset.copy()

# Format Y-axis labels
def thousands(x, pos):
    return f'{x*1e-3:.0f}K'

# Convert date and cost columns
df['ChargePeriodStart'] = pd.to_datetime(df['ChargePeriodStart'], errors='coerce')
df['ChargePeriodStart'] = df['ChargePeriodStart'].dt.tz_localize(None)
df['EffectiveCost'] = pd.to_numeric(df['EffectiveCost'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['ChargePeriodStart', 'EffectiveCost'])

# Remove top 5% outliers
df = df[df['EffectiveCost'] < df['EffectiveCost'].quantile(0.95)]

# Aggregate cost by day
daily_cost = df.groupby(pd.Grouper(key='ChargePeriodStart', freq='D'))['EffectiveCost'].sum().reset_index()
daily_cost.columns = ['ds', 'y']
daily_cost.set_index('ds', inplace=True)

# Fit SARIMA model
model = SARIMAX(daily_cost['y'], order=(1,1,1), seasonal_order=(1,1,1,12),
                enforce_stationarity=False, enforce_invertibility=False)
results = model.fit()

# Forecast the next 60 days
forecast_steps = 60
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=daily_cost.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot the forecast
plt.figure(figsize=(12,6))
plt.plot(daily_cost.index, daily_cost['y'], label='Observed')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='orange')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='orange', alpha=0.3)

# Format plot
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
plt.title("60-day Forecast (SARIMA model)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Cost (NOK)")
plt.legend()
plt.tight_layout()
plt.show()
