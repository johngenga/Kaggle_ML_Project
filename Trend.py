
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
tunnel = pd.read_csv(
    "C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/tunnel.csv", parse_dates=["Day"]
)
tunnel = tunnel.set_index("Day")
tunnel.index = pd.to_datetime(tunnel.index, format='%d/%m/%Y')
tunnel = tunnel.to_period()

# Let's make a moving average plot to see what kind of trend this series has.
# Since this series has daily observations,
# let's choose a window of 365 days to smooth over any short-term changes within the year.

moving_average = tunnel.rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = tunnel.plot(style=".", color="0.5")
moving_average.plot(
    ax=ax, linewidth=3, title="Tunnel Traffic - 365-Day Moving Average", legend=False,
)

# We'll use a function from the statsmodels library called DeterministicProcess.
# The order argument refers to polynomial order: 1 for linear, 2 for quadratic, 3 for cubic, and so on.
# Non-random or completely determined, like the const and trend series are.
# Features derived from the time index will generally be deterministic.
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
X = dp.in_sample()
from sklearn.linear_model import LinearRegression
y = tunnel["NumVehicles"]  # the target

# The intercept is the same as the `const` feature from DeterministicProcess.
# LinearRegression behaves badly with duplicated features,
# We need to be sure to exclude it here.

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)

ax = tunnel.plot(style=".", color="0.5", title="Tunnel Traffic - Linear Trend")
_ = y_pred.plot(ax=ax, linewidth=3, label="Trend")

# To make a forecast, we apply our model to "out of sample (outside of the observation period) features.
# Here's how we could make a 30-day forecast:

X = dp.out_of_sample(steps=30)
y_fore = pd.Series(model.predict(X), index=X.index)
print(y_fore.head())

# plot a portion of the series to see the trend forecast for the next 30 days:
ax = tunnel["2005-05":].plot(title="Tunnel Traffic - Linear Trend Forecast", **plot_params)
ax = y_pred["2005-05":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()

plt.show()