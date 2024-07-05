import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Get_Data - - Training data
store_sales = pd.read_csv("C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/train.csv",
                          usecols = ['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
                          dtype={
                              'store_nbr': 'category',
                              'family': 'category',
                              'sales': 'float32',
                              'onpromotion': 'uint32',
                          },
                          parse_dates=['date'],
                          )
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'],observed=False)
    .mean()
    .unstack('family')
    .loc['2017']
)

# Get_Data - - Test data
test = pd.read_csv("C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/test.csv",
                   dtype={
                       'store_nbr': 'category',
                       'family': 'category',
                       'onpromotion': 'uint32',
                   },
                   parse_dates = ['date'],
                   )
test['date'] = test.date.dt.to_period('D')
test = test.set_index(['store_nbr', 'family', 'date']).sort_index()


