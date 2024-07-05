import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor

store_sales = pd.read_csv("C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/train.csv",
                          usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
                          dtype={
                              'store_nbr': 'category',
                              'family': 'category',
                              'sales': 'float32',
                          },
                          parse_dates=['date'],
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017']
)

##  Add fit and predict methods to this minimal class
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store column names from fit method

## 1) Define fit method for boosted hybrid
def fit(self, X_1, X_2, y):
    #fit self.model_1
    self.model_1.fit(X_1, y)
    # make predictions with self.model_1
    y_fit = pd.DataFrame(
        self.model_1.predict(X_1),
        index=X_1.index, columns=y.columns,
    )
    # compute residuals
    y_resid = y - y_fit
    y_resid = y_resid.stack().squeeze()  # wide to long

    # fit self.model_2 on residuals
    self.model_2.fit(X_2, y_resid)

    # Save column names for predict method
    self.y_columns = y.columns

## 2) Define predict method for boosted hybrid
def predict(self, X_1, X_2):
    y_pred = pd.DataFrame(
        # YOUR CODE HERE: predict with self.model_1
        self.model_1.predict(X_1),
        index=X_1.index, columns=self.y_columns,
    )
    y_pred = y_pred.stack().squeeze()  # wide to long

    # YOUR CODE HERE: add self.model_2 predictions to y_pred
    y_pred += self.model_2.predict(X_2)

    return y_pred.unstack()  # long to wide

# Add method to class
BoostedHybrid.predict = predict

##  2.1) use your new BoostedHybrid class to create a model for the Store Sales data.
#  Run the next cell to set up the data for training.

# Target series
y = family_sales.loc[:, 'sales']

# X_1: Features for Linear Regression
dp = DeterministicProcess(index=y.index, order=1)
X_1 = dp.in_sample()

# X_2: Features for XGBoost
X_2 = family_sales.drop('sales', axis=1).stack()  # onpromotion feature

# Label encoding for 'family'
le = LabelEncoder()  # from sklearn.preprocessing
X_2 = X_2.reset_index('family')
X_2['family'] = le.fit_transform(X_2['family'])

# Label encoding for seasonality
X_2["day"] = X_2.index.day  # values are day of the month

## 3) Train boosted hybrid
# Create the hybrid model by initializing a
# BoostedHybrid class with LinearRegression() and XGBRegressor() instances.
# Create LinearRegression + XGBRegressor hybrid with BoostedHybrid

# Fit and predict
model = BoostedHybrid(
    model_1=LinearRegression(),
    model_2=XGBRegressor(),
)


