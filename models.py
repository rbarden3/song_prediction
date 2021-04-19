# File Imports
from data import get_x_y

# Package Imports
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def random_forest(model, array_of_df):
    dict_x_y = get_x_y(array_of_df)
    X = dict_x_y['x']
    y = dict_x_y['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("y_pred -> ", y_pred)
    mean_error = mean_absolute_error(y_test, y_pred)
    print("\nmean_error -> ", mean_error)
    return model

def xgboost(array_of_df):
    dict_x_y = get_x_y(array_of_df)
    # X = pd.Series(dict_x_y['x'])
    # y = pd.Series(dict_x_y['y'])
    X = pd.DataFrame(dict_x_y['x']).iloc[:, :]
    y = pd.DataFrame(dict_x_y['y']).iloc[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    print("x_train -> ", X_train)
    # regressor = RandomForestRegressor()
    # regressor.fit(X_train, y_train)
    # Train
    # Instantiating a XGBoost classifier object
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    print("x_train", X_train)
    print("y_train", y_train)
    xgb_regressor = xgb.XGBRegressor(
        max_depth=6, learning_rate=0.1, objective='reg:linear', alpha=10, n_estimators=10)
    xgb_regressor.fit(X_train, y_train)

    # Predict
    pred_probs = xgb_regressor.predict_proba(X_test)
    pred_prob_again = xgb_regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_prob_again))

    print("pred_probs ->", pred_probs)

    # Results
    print("y_test -> ", y_test)
    # RMSE