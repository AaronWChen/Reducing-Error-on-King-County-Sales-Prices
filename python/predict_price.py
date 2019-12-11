import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
import sklearn.metrics as metrics
import joblib
import math


def load_data(rand_state=50):
  train_data = f"joblib/prepped_training_set_rand_state_{rand_state}.joblib"
  with open(train_data, "rb") as fo:
    trained_data = joblib.load(train_data)

  train_log_prices = f"joblib/log_train_prices_rand_state_{rand_state}.joblib"
  with open(train_log_prices, "rb") as fo:
    log_trained_prices = joblib.load(train_log_prices)
    with_log_train_prices = trained_data.join(log_trained_prices, how='inner')
    with_log_train_prices = with_log_train_prices['price']

  test_data = f"joblib/prepped_test_set_rand_state_{rand_state}.joblib"
  with open(test_data, "rb") as fo:
    test_dataset = joblib.load(test_data)

  test_log_prices = f"joblib/log_test_prices_rand_state_{rand_state}.joblib"
  with open(test_log_prices, "rb") as fo:
    log_test_priceset = joblib.load(test_log_prices)
    with_log_test_prices = test_dataset.join(log_test_priceset, how='inner')
    with_log_test_prices = with_log_test_prices['price']

  return trained_data, with_log_train_prices, test_dataset, with_log_test_prices


def calc_score_lr_rsme():
  lr = LinearRegression()
  
  trained_data, log_trained_prices, test_data, log_test_prices = load_data()
  non_log_test = np.exp(log_test_prices)
  
  lr.fit(trained_data, log_trained_prices)
  lr_score = lr.score(test_data, log_test_prices)
  lr_predictions = lr.predict(test_data)
  non_log_lr_predictions = np.exp(lr_predictions)
  lr_mse = metrics.mean_squared_error(non_log_test, non_log_lr_predictions)
  lr_rmse = math.sqrt(lr_mse)

  return lr_score, lr_rmse


def calc_score_lscv_rsme():
  lscv = LassoCV(max_iter=150000)

  trained_data, log_trained_prices, test_data, log_test_prices = load_data()
  non_log_test = np.exp(log_test_prices)
  
  lscv.fit(trained_data, log_trained_prices)
  lscv_score = lscv.score(test_data, log_test_prices)
  lscv_predictions = lscv.predict(test_data)
  non_log_lscv_predictions = np.exp(lscv_predictions)
  lscv_mse = metrics.mean_squared_error(non_log_test, non_log_lscv_predictions)
  lscv_rmse = math.sqrt(lscv_mse)
  lscv_a = lscv.alpha_

  return lscv_score, lscv_rmse, lscv_a


def display_results():
  lr_score, lr_rmse = calc_score_lr_rsme()
  lscv_score, lscv_rmse, lscv_a = calc_score_lscv_rsme()

  results = {'Linear Regression Score': lr_score,
              'Linear Regression RMSE ($)': lr_rmse,
              'Lasso CV Score': lscv_score,
              'Lasso CV RMSE ($)': lscv_rmse,
              'Lasso CV alpha coefficient': lscv_a}
  return results
