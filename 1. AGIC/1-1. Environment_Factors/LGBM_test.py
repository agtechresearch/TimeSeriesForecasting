#%%
###############
##### CO2 #####
###############
from LGBM_def import load_data, split_data, tune_hyperparameters, train_model, fit_lgbm
file_path = "/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final_forEnv.csv"
target_col = "CO2air"


# 데이터 불러오기
df = load_data(file_path)

# train/test/validation 세트로 데이터 분리
X_train, X_test, y_train, y_test, X_val, y_val = split_data(df, target_col)


best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100, seed=10)
model = LGBMRegressor(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

model = train_model(X_train, y_train, best_params)


# predict test set
y_pred = model.predict(X_test)


# evaluate predictions
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print('CO2 MSE: {:.4f}'.format(mse))  #CO2 MSE: 0.0133
print('CO2 MAE: {:.4f}'.format(mae))   #CO2 MAE: 0.0805
print('CO2 RMSE: {:.4f}'.format(rmse))  #CO2 RMSE: 0.1152
print('CO2 R^2: {:.4f}'.format(r2))   #CO2 R^2: 0.5964

# %%
########################
########## RH ##########
########################
from LGBM_def import load_data, split_data, tune_hyperparameters, train_model, fit_lgbm

file_path = "/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final_forEnv.csv"
target_col = "Rhair"

# 데이터 불러오기
df = load_data(file_path)

# train/test/validation 세트로 데이터 분리
X_train, X_test, y_train, y_test, X_val, y_val = split_data(df, target_col)


best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100, seed=10)
model = LGBMRegressor(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

model = train_model(X_train, y_train, best_params)


# predict test set
y_pred = model.predict(X_test)

# evaluate predictions
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print('RH MSE: {:.4f}'.format(mse)) #
print('RH MAE: {:.4f}'.format(mae)) #
print('RH RMSE: {:.4f}'.format(rmse)) #
print('RH R^2: {:.4f}'.format(r2)) #


# %%
#######################
### stem elongation ###
#######################
from LGBM_def import load_data, split_data, tune_hyperparameters, train_model, fit_lgbm

file_path = "/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final.csv"
target_col = "Stem_elong"

# 데이터 불러오기
df = load_data(file_path)

# train/test/validation 세트로 데이터 분리
X_train, X_test, y_train, y_test, X_val, y_val = split_data(df, target_col)


best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100, seed=10)
model = LGBMRegressor(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

model = train_model(X_train, y_train, best_params)


# predict test set
y_pred = model.predict(X_test)

# evaluate predictions
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print('Stem_elong MSE: {:.4f}'.format(mse))  #0.0001
print('Stem_elong MAE: {:.4f}'.format(mae))  #0.0044
print('Stem_elong RMSE: {:.4f}'.format(rmse))  #0.0076
print('Stem_elong R^2: {:.4f}'.format(r2))  #0.9990


# %%
#########################
### number of trusses ###
#########################
from LGBM_def import load_data, split_data, tune_hyperparameters, train_model, fit_lgbm

file_path = "/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final.csv"
target_col = "Cum_trusses"

# 데이터 불러오기
df = load_data(file_path)

# train/test/validation 세트로 데이터 분리
X_train, X_test, y_train, y_test, X_val, y_val = split_data(df, target_col)


best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100, seed=10)
model = LGBMRegressor(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

model = train_model(X_train, y_train, best_params)


# predict test set
y_pred = model.predict(X_test)

# evaluate predictions
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print('Cum_trusses MSE: {:.4f}'.format(mse))  #0.0001
print('Cum_trusses MAE: {:.4f}'.format(mae))  #0.0039
print('Cum_trusses RMSE: {:.4f}'.format(rmse))  #0.0100
print('Cum_trusses R^2: {:.4f}'.format(r2))  #0.9987

#lgb.plot_importance(model, figsize=(10, 10), title='Feature importance - Cum_trusses(LGBM)')
#plt.show()

# %%
