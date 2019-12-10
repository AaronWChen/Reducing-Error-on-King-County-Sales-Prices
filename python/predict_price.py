import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import joblib

lr = LinearRegression()

def load_data(rand_state=50):
  with open(f"joblib/prepped_training_set_rand_state_{rand_state}.joblib", 
            "wb") as fo:
    trained_data = joblib.load(fo)

  with open(f"joblib/log_train_prices_rand_state_{rand_state}.joblib",
            "wb") as fo:
    log_trained_prices = joblib.load(fo)

  with open(f"joblib/prepped_test_set_rand_state_{rand_state}.joblib", 
            "wb") as fo:
    test_data = joblib.load(fo)

  with open(f"joblib/log_test_prices_rand_state_{rand_state}.joblib",
            "wb") as fo:
    log_test_prices = joblib.load(fo)

  return trained_data, log_trained_prices, test_data, log_test_prices


def calc_score():
  trained_data, log_trained_prices, test_data, log_test_prices = load_data()

  lr.fit(trained_data, log_trained_prices)

  lr.score(test_data, log_test_prices)

