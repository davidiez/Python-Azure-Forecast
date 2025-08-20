# Python script to generate a forecasted plot (using 'statsmodels' SARIMAX model) of 60-day period.
# To be used in any Power BI dashboard, getting as dataset an Azure billing data in FOCUS format.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
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

# Add 7-day moving average
daily_cost['7d_avg'] = daily_cost['y'].rolling(window=7).mean()

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
plt.plot(daily_cost.index, daily_cost['y'], label='Observed', color='#1F77B4')
plt.plot(daily_cost.index, daily_cost['7d_avg'], label='7-day Avg', linestyle='--', color='gray')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='#FF7F0E')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='#FF7F0E', alpha=0.3)

# Format plot
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
plt.title("60-day Forecast (SARIMA model)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Cost (NOK)")
plt.grid(True, linestyle='--', alpha=0.5)

# Add confidence interval to legend
ci_patch = Patch(color='#FF7F0E', alpha=0.3, label='Confidence Interval')

# Create legend manually with all handles
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(ci_patch)
labels.append('Confidence Interval')
plt.legend(handles=handles, labels=labels)


plt.tight_layout()
plt.show()

# Optional: Export forecast data for tabular view
forecast_df = pd.DataFrame({
    'Date': forecast_index,
    'Forecast': forecast_mean,
    'Lower CI': forecast_ci.iloc[:, 0],
    'Upper CI': forecast_ci.iloc[:, 1]
})
