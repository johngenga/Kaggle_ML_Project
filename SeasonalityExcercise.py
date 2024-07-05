
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

holidays_events = pd.read_csv("C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/holidays_events.csv",
                              dtype={'type': 'category',
                                     'locale': 'category',
                                     'locale_name': 'category',
                                     'description': 'category',
                                     'transferred': 'bool',
                                     },
                              parse_dates=[ 'date'],
                              # infer_datetime_format=True
                              )
holidays_events = holidays_events.set_index('date').to_period('D')

store_sales = pd.read_csv("C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/train.csv",
                          dtype={'store_nbr': 'category',
                                 'family': 'category',
                                 'sales': 'float32',
                                 },
                          parse_dates=['date'],
                          # infer_datetime_format=True,
                          )
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
average_sales = store_sales.groupby('date').mean().squeeze().loc['2017']

y = average_sales.copy()

# Create seasonal features. Use DeterministicProcess and CalendarFourier to create:
#      Indicators for weekly seasons and,
#      Fourier features of order 4 for monthly seasons.

fourier = CalendarFourier(freq="ME", order=4)
dp = DeterministicProcess(
    index=y.index,
    constant =True,
    order=1,
    seasonal= True,
    additional_terms= [fourier],
    drop=True
)
X = dp.in_sample()
y=y['sales']
model = LinearRegression().fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)
ax = y.plot(color='0.25', style='.', alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax.legend()

# Plot the prediction.
# plt.show()

# Deseasoning or Detrending = removing the seasons.
y_deseason = y - y_pred

# National and regional holidays in the training set
holidays = (
    holidays_events
    .query("locale in ['National', 'Regional']")
    .loc['2017':'2017-08-15', ['description']]
    .assign(description=lambda x: x.description.cat.remove_unused_categories())
)
# print(holidays)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

ax = y_deseason.plot(**plot_params)
plt.plot(holidays.index, y_deseason[holidays.index], color='C3')
ax.set_title('National and Regional Holidays')

# Create holiday features to help the model make use of this information about holiday features.

# Pandas solution :  Using Pandas makes it easier to join X_holidays to X2
#  since it returns a DataFrame retaining the date of each holiday.
X_holidays = pd.get_dummies(holidays)
# Join to training data
X2 = X.join(X_holidays, on='date').fillna(0.0)

df_test = pd.read_csv("C:/Users/User/PycharmProjects/LearningTimeSeries/store-sales-time-series-forecasting/test.csv",
                      dtype={
                          'store_nbr': 'category',
                          'family' : 'category',
                          'onpromotion' : 'uint32',
                      },
                      parse_dates = ['date'],
)
df_test['date'] = df_test.date.dt.to_period('D')
df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

# Create features for test set
X_test = dp.out_of_sample(steps=16)
X_test.index.name = 'date'
X_test['NewYear'] = (X_test.index.dayofyear == 1)

y_submit = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
y_submit = y_submit.stack(['store_nbr', 'family'])
y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])
y_submit.to_csv('submission.csv', index=False)



