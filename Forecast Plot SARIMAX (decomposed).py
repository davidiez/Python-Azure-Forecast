# Python script to generate the decomposed view of a forecasted plot (using 'statsmodels' SARIMAX model) of 60-day period. It will show 4 plots: obeserved, trend, seasional and residual data.
# To be used in any Power BI dashboard, getting as dataset an Azure billing data in FOCUS format.

# Install Prophet if needed
# pip install prophet

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.seasonal import seasonal_decompose

# Format Y-axis labels
def thousands(x, pos):
    return f'{x*1e-3:.0f}K'

# Copy dataset from Power BI
df = dataset.copy()

# Preprocess
df['ChargePeriodStart'] = pd.to_datetime(df['ChargePeriodStart'], errors='coerce')
df['ChargePeriodStart'] = df['ChargePeriodStart'].dt.tz_localize(None)
df['EffectiveCost'] = pd.to_numeric(df['EffectiveCost'], errors='coerce')
df = df.dropna(subset=['ChargePeriodStart', 'EffectiveCost'])
df = df[df['EffectiveCost'] < df['EffectiveCost'].quantile(0.95)]

# Aggregate daily cost
daily_cost = df.groupby(pd.Grouper(key='ChargePeriodStart', freq='D'))['EffectiveCost'].sum().reset_index()
daily_cost.columns = ['ds', 'y']
daily_cost.set_index('ds', inplace=True)

# Decompose time series
decomposition = seasonal_decompose(daily_cost['y'], model='additive', period=30)

# Plot components
plt.figure(figsize=(12, 8))

#Observed
plt.subplot(411)
plt.plot(decomposition.observed)
plt.title('Observed')
plt.ylabel('Cost (NOK)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
plt.grid(True)

#Trend
plt.subplot(412)
plt.plot(decomposition.trend, color='green')
plt.title('Trend')
plt.ylabel('Cost (NOK)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
plt.grid(True)

#Seasonal
plt.subplot(413)
plt.plot(decomposition.seasonal, color='purple')
plt.title('Seasonal')
plt.ylabel('Cost (NOK)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
plt.grid(True)

#Residual
plt.subplot(414)
plt.plot(decomposition.resid, color='red')
plt.title('Residual')
plt.ylabel('Cost (NOK)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
plt.grid(True)

plt.tight_layout()
plt.show()
