#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
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
        lgbm_param = {
            'objective': 'regression',
            'verbose': -1,
            'metric': 'mse', 
            'num_leaves': trial.suggest_int('num_leaves', 2, 1024, step=1, log=True), 
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_uniform('reg_lambda', 0.0, 10.0),
            'max_depth': trial.suggest_int('max_depth',3, 15),
            'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.5),
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_loguniform('subsample', 0.3, 1),
        }

        # Generate model
        model_lgbm = LGBMRegressor(**lgbm_param)
        model_lgbm = model_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0, early_stopping_rounds=25)

        # 평가 지표
        MSE = mean_squared_error(y_val, model_lgbm.predict(X_val))
        return MSE

    optuna_lgbm = optuna.create_study(direction='minimize', sampler=sampler)
    optuna_lgbm.optimize(objective, n_trials=n_trials)

    # best trial
    lgbm_trial = optuna_lgbm.best_trial
    lgbm_trial_params = lgbm_trial.params
    print('Best Trial: score {},\nparams {}'.format(lgbm_trial.value, lgbm_trial_params))
    return lgbm_trial_params


def train_model(X_train, y_train, hyperparameters):
    lgbm = LGBMRegressor(**hyperparameters)
    lgbm.fit(X_train, y_train)
    return lgbm


def fit_lgbm(X_train, y_train, X_test, y_test, lgbm_trial_params):
    # LGBM Regressor fit
    lgbm = LGBMRegressor(**lgbm_trial_params)
    lgbm = lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=25)  
    #early_stopping_rounds: 얼마나 성능이 개선되지 않았을 때 학습을 중단시킬 것인지

    # Predict the y_test
    pred = lgbm.predict(X_test)

    # results
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print("RMSE: %f" % (rmse))

    mae = mean_absolute_error(y_test, pred)
    print('mae: %f' %(mae))  

    r2 = r2_score(y_test, pred)
    print('R2: %f' %(r2))

'''
def train_and_evaluate(file_path, target_col, test_size=0.4, val_size=0.5, random_state=42, n_trials=100, seed=10):
    # Load data
    df = load_data(file_path)

    # Split data
    X_train, X_test, y_train, y_test, X_val, y_val = split_data(df, target_col, test_size, val_size, random_state)

    # Tune hyperparameters
    lgbm_trial_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials, seed)

    # Fit LGBM
    lgbm_model = fit_lgbm(X_train, y_train, X_test, y_test, lgbm_trial_params)

    # Return best iteration
    print("Best iteration:", lgbm_model.best_iteration_)
    return lgbm_model
'''
# %%
