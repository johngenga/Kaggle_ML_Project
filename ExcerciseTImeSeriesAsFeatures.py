import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

store_sales = pd.read_csv("C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/train.csv",
                          usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
                          dtype={
                              'store_nbr': 'category',
                              'family': 'category',
                              'sales': 'float32',
                              'onpromotion': 'uint32',
                          },
                          parse_dates=['date'],
                          #infer_datetime_format=True,
                          )
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017', ['sales', 'onpromotion']]
)

# Sales of school and office supplies, however,
# show patterns of growth and decay
# To isolate any purely cyclic behavior,
# we'll start by deseasonalizing the series.

supply_sales = family_sales.loc(axis=1)[:, 'SCHOOL AND OFFICE SUPPLIES']
y = supply_sales.loc[:, 'sales'].squeeze()

fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    constant=True,
    index=y.index,
    order=1,
    seasonal=True,
    drop=True,
    additional_terms=[fourier],
)
X_time = dp.in_sample()
X_time['NewYearsDay'] = (X_time.index.dayofyear == 1)

model = LinearRegression(fit_intercept=False)
model.fit(X_time, y)
y_deseason = y - model.predict(X_time)
y_deseason.name = 'sales_deseasoned'

# Examine serial dependance.
onpromotion = supply_sales.loc[:, 'onpromotion'].squeeze().rename('onpromotion')

#  We can lag a time series in Pandas with the shift method. For this problem,
#  We'll fill in the missing values the lagging creates with 0.0.
def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)
def make_leads(ts, leads):
    return pd.concat(
        {
            f'y_lead_{i}': ts.shift(-i)
            for i in range(1, leads + 1)
        },
        axis=1)


X_lags = make_lags(y_deseason, lags=1)
X_lags = X_lags.fillna(0.0)
X_leads = make_leads(y_deseason, leads=1)
X_leads = X_leads.fillna(0.0)

X_promo = pd.concat([
    make_lags(onpromotion, lags=1),
    onpromotion,
    make_leads(onpromotion, leads=1),
], axis=1)

X = pd.concat([X_lags, X_promo], axis=1)
y, X = y.align(X, join='inner')

print(X)