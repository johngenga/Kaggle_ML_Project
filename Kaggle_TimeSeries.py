import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Read and modify the book_sales data
book_sales = pd.read_csv(
    "C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/book_sales.csv",
    index_col='Date',
    parse_dates=['Date']
).drop('Paperback', axis=1)

# Read and modify the store_sales data
dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}
store_sales = pd.read_csv(
    "C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/train.csv",
    dtype=dtype,
    parse_dates=['date'],
)
store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)
average_sales = store_sales.groupby('date').mean()['sales']

# Create a DataFrame for the average sales
df = average_sales.to_frame()

# Creating training data
df['time'] = np.arange(len(df.index))

# Features and target
X = df.loc[:, ['time']]
y = df.loc[:, 'sales']

# Train the model
model = LinearRegression()
model.fit(X, y)
# Predictions
y_pred = pd.Series(model.predict(X), index=X.index)

# Convert PeriodIndex to DatetimeIndex for plotting
df.index = df.index.to_timestamp()

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

# Plot the result
fig, ax = plt.subplots()
ax.plot(df.index, y, label='Actual Sales', color='0.75', linestyle='-', marker='.')
ax.plot(df.index, y_pred, label='Predicted Sales', color='0.25', linewidth=3)
ax.set_title('Time Plot of Total Store Sales')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()

# Fit a lag feature to Store Sales
df = average_sales.to_frame()
# Create a lag feature from the target 'sales'
lag_1 = df['sales'].shift(1)
df['lag_1'] = lag_1  # add to dataframe

X = df.loc[:, ['lag_1']].dropna()  # features
y = df.loc[:, 'sales']  # target
y, X = y.align(X, join='inner')  # drop corresponding values in target

# Create a LinearRegression instance and fit it to X and y.
model = LinearRegression()
model.fit(X, y)

# Create Store the fitted values as a time series with
# the same time index as the training data

y_pred = pd.Series(model.predict(X), index=X.index)

print(y_pred)
