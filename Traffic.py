
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
simplefilter("ignore")  # ignore warnings to clean up output cells
#'seaborn-v0_8-whitegrid'
# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
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
# Load Tunnel Traffic dataset
tunnel = pd.read_csv(
    "C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/tunnel.csv",
    parse_dates=["Day"])

# Create a time series in Pandas by setting the index to a date
# column. We parsed "Day" as a date type by using `parse_dates` when
# loading the data.

tunnel = tunnel.set_index("Day")

# By default, Pandas creates a `DatetimeIndex` with dtype `Timestamp`
# (equivalent to `np.datetime64`, representing a time series as a
# sequence of measurements taken at single moments. A `PeriodIndex`,
# on the other hand, represents a time series as a sequence of
# quantities accumulated over periods of time. Periods are often
# easier to work with, so that's what we'll use in this course.

tunnel.index = pd.to_datetime(tunnel.index, format='%d/%m/%Y')
tunnel = tunnel.to_period()

#    *** Time-step feature - we can create a time dummy by counting out the length of the series.***
df = tunnel.copy()
df['Time'] = np.arange(len(tunnel.index))

# The procedure for fitting a linear regression model follows the standard steps for scikit-learn.

from sklearn.linear_model import LinearRegression
# Training data
X = df.loc[:, ['Time']]  # features
y = df.loc[:, 'NumVehicles']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
y_pred = pd.Series(model.predict(X), index=X.index)

# print(y_pred)
# print(model.intercept_)

ax = y.plot(**plot_params)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Tunnel Traffic');

# Lag feature¶ - Pandas provides us a simple method to lag a series, the shift method
df['Lag_1'] = df['NumVehicles'].shift(1)

# print (df.head())
# plt.show()

# What to do with the MISSING VALUES produced. We'll just drop the missing values

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)  # drop missing values in the feature set
y = df.loc[:, 'NumVehicles']  # create the target
y, X = y.align(X, join='inner')  # drop corresponding values in target

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

# The relationship between the number of vehicles one day and the number the previous day.

fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of Tunnel Traffic')

plt.show()