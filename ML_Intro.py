import pandas as pd
from sklearn.tree import DecisionTreeRegressor

dataset_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(dataset_file_path)
#Read data to know what you are working with
print(home_data.describe())
#displays the top few rows of data
print(home_data.head())

y = home_data.SalePrice
 
features = [
    'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'
]

X = home_data[features]

#specify model
iowa_model = DecisionTreeRegressor()

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

#Training or Fitting model on Training data

iowa_model.fit(train_X, train_y)

val_prediction = iowa_model.predict(val_X)

print(val_prediction)

from sklearn.metrics import mean_absolute_error

# MAE calculation

mae_val = mean_absolute_error(val_y, val_prediction)
print(mae_val)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()
rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_val_predictions)

