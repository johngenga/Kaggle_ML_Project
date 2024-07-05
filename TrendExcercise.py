
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

retail_sales = pd.read_csv(
    "C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/us-retail-sales.csv",
    parse_dates=['Month'],
    index_col='Month',
).to_period('D')
food_sales = retail_sales.loc[:, 'FoodAndBeverage']
auto_sales = retail_sales.loc[:, 'Automobiles']

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

# ax = food_sales.plot(style=".", color="0.5")
# ax.set(title="US Food and Beverage Sales", ylabel="Millions of Dollars")

# Add methods to `food_sales` to compute a moving
# Average with appropriate parameters for trend estimation.
trend = food_sales.rolling(
    window=12,
    center=True,
    min_periods=6,
).mean()
# ax = food_sales.plot(style=".", color="0.5", alpha=0.5)
# ax = trend.plot(ax=ax, linewidth=3)
# plt.show()

from statsmodels.tsa.deterministic import DeterministicProcess
y = average_sales.copy()  # the target

dp = DeterministicProcess(
    index=y.index,
    constant=False,
    order=3,
    drop=True
)
X = dp.in_sample()
X_fore = dp.out_of_sample(steps=90)

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(style=".", color="0.5", alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend()

plt.show()