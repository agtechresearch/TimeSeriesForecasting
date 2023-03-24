#%%
import pandas as pd
from pandas import DataFrame
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optuna Libraries
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
# XGBRegressor
import xgboost as xgb
from xgboost import XGBRegressor

#import graphviz
import matplotlib.pyplot as plt
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.set_index('time', inplace=True)
    return df


def split_data(df, target_col, test_size=0.4, val_size=0.5, random_state=42):
    feature_columns = list(df.columns.difference([target_col]))
    X = df[feature_columns]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, X_val, y_val


def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100, seed=10):
    # optuna
    # random sampler
    sampler = TPESampler(seed=seed)

    # define function
    def objective(trial):
        xgb_param = {
        'max_depth': trial.suggest_int('max_depth',5, 15),
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.5),
        'gamma': trial.suggest_float("gamma", 0, 2),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 5),
        'subsample': trial.suggest_float("subsample", 0.3, 1),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1)
        }

        # Generate model
        model_xgb = xgb.XGBRegressor(**xgb_param)
        model_xgb = model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                           verbose=0, early_stopping_rounds=25)
                           
        # 평가 지표
                        
        MSE = mean_squared_error(y_val, model_xgb.predict(X_val))
        return MSE

    optuna_xgboost = optuna.create_study(direction='minimize', sampler=sampler)
    optuna_xgboost.optimize(objective, n_trials=n_trials)

    # best trial
    xgboost_trial = optuna_xgboost.best_trial
    xgboost_trial_params = xgboost_trial.params
    print('Best Trial: score {},\nparams {}'.format(xgboost_trial.value, xgboost_trial_params))
    return xgboost_trial_params


def train_model(X_train, y_train, hyperparameters):
    xgboost = XGBRegressor(**hyperparameters)
    xgboost.fit(X_train, y_train)
    return xgboost


def fit_xgb(X_train, y_train, X_test, y_test, xgboost_trial_params):
    # LGBM Regressor fit
    xgboost = XGBRegressor(**xgboost_trial_params)
    xgboost = xgboost.fit(X_train, y_train)

    # Predict the y_test
    pred = xgboost.predict(X_test)

    # results
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print("RMSE: %f" % (rmse))

    mae = mean_absolute_error(y_test, pred)
    print('mae: %f' %(mae))  

    r2 = r2_score(y_test, pred)
    print('R2: %f' %(r2))
# %%
