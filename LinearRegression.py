import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Read and modify the data.
df = pd.read_csv(
    "C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

# The most basic time-step feature is the time dummy,
# which counts off time steps in the series from beginning to end.
df['Time'] = np.arange(len(df.index))

#The time dummy then lets us fit curves to time series in a time plot,
# where Time forms the x-axis.

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
# Plot the Time vs Hardcover
fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales')

# Lag features - Shift the observations of the target. E.g. 1 step lag.

df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])

# Linear regression with a lag feature produces the model:
# When you see a relationship like this, you know a lag feature will be useful.
fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales')

plt.show()




