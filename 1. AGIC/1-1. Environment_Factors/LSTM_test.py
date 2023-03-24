#%%
from LSTM_def import load_data, split_data, optimize_lstm, objective

file_path = "/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final_forEnv.csv"
target_col = "CO2air"

# 데이터 불러오기
df = load_data(file_path)
X_train, X_test, y_train, y_test = split_data(df, target_col)

# create a new study and run the optimization
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100)

# print the best set of hyperparameters
lstm_trial = study.best_trial
lstm_trial_params = lstm_trial.params
print('Best Trial: score {},\nparams {}'.format(lstm_trial.value, lstm_trial_params))

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
model = optimize_lstm(study.best_trial)
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# 예측값 계산
y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('CO2 MAE: {:.4f}'.format(mae)) 
print('CO2 RMSE: {:.4f}'.format(rmse))
# %%
from LSTM_def import load_data, split_data, optimize_lstm, objective

file_path = "/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final_forEnv.csv"
target_col = "CO2air"

# 데이터 불러오기
df = load_data(file_path)

# create a new study and run the optimization
def lstm_objective(trial):
    X_train, X_test, y_train, y_test = split_data(df, target_col)
    model = optimize_lstm(trial, X_train, y_train)
    # ModelCheckpoint 콜백 추가
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    # EarlyStopping 콜백 추가
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0, validation_data=(X_test, y_test),
              callbacks=[checkpoint, early_stop])
    # 저장된 가장 좋은 모델 불러오기
    model.load_weights('best_model.h5')
    score = model.evaluate(X_test, y_test, verbose=2)
    return score

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(lstm_objective, n_trials=100)

# print the best set of hyperparameters
lstm_trial = study.best_trial
lstm_trial_params = lstm_trial.params
print('Best Trial: score {},\nparams {}'.format(lstm_trial.value, lstm_trial_params))

# %%
