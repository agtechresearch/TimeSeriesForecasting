
#%%
import pandas as pd
from pandas import DataFrame
import datetime
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import joblib


'''
Load dataset

'''
df = pd.read_csv('/home/jy/TimeSeriesForecasting/df_AGIC_final.csv')
df.info()
df.describe()
df.columns # 'CO2air', 'EC_drain_PC', 'Tot_PAR', 'pH_drain_PC', 'water_sup','T_out', 'I_glob', 'RadSum', 'Windsp', 'Stem_elong', 'Cum_trusses']
df.set_index('time', inplace = True)


# train/test split
feature_columns = list(df.columns.difference(['Stem_elong', 'Stem_thick']))
X = df[feature_columns]
y = df['Stem_elong']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)

# reshape input data
X_train = np.array(X_train.values, dtype=np.float32)
#X_train = X_train.reshape(int(len(X_train)/3), 3, 10) #(samples, timestep, feature)

X_test = np.array(X_test.values, dtype=np.float32)
#X_test = X_test.reshape(int(len(X_test)/3), 3, 10)

y_train = np.array(y_train.values, dtype=np.float32)
y_test = np.array(y_test.values, dtype=np.float32)
#%%

'''
Model 1


# Model
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.3))
model.add(LSTM(units=64, recurrent_dropout=0))
model.add(Dropout(0.3))

model.add(Dense(1))

# optimizer : 모델 학습 최적화 방법, loss : 손실함수 설정
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') 
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# prediction
predictions = model.predict(X_test)

# calculate MAE
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)  #MAE: 0.031196348

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse)) #RMSE: 0.069858


# model save
#joblib.dump(model,'/home/jy/AGIC/agic_lstm_elong.pkl')

# 모델 불러옴
# file -> model load
#model_from_joblib = joblib.load('/home/jy/AGIC/agic_lstm_elong.pkl')
#model_from_joblib.evaluate(X_test, y_test)
'''


'''
Model 2


from keras.layers import Reshape #shape=(batch_size, time_steps, features)

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.3))
model.add(LSTM(units=64, recurrent_dropout=0))
model.add(Dropout(0.3))
model.add(Reshape((64, 1)))
model.add(LSTM(units = 16, recurrent_dropout=0))
model.add(Dense(1))

# optimizer : 모델 학습 최적화 방법, loss : 손실함수 설정
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') 
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# prediction
predictions = model.predict(X_test)

# calculate MAE
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)  #MAE: 0.017318934

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse)) #RMSE: 0.022941
'''

# %%
'''
Model 3 

from keras.layers import Reshape #shape=(batch_size, time_steps, features)

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.3))
model.add(LSTM(units=64, recurrent_dropout=0))
model.add(Dropout(0.3))
model.add(Reshape((64, 1)))
model.add(LSTM(units = 32, recurrent_dropout=0))
model.add(Dense(1))

# optimizer : 모델 학습 최적화 방법, loss : 손실함수 설정
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') 
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# prediction
predictions = model.predict(X_test)

# calculate MAE
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)  #MAE: 0.012230681

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse)) #RMSE: 0.019230
'''


'''
Model 4


# %%
from keras.layers import Reshape #shape=(batch_size, time_steps, features)

model = Sequential()
model.add(LSTM(units=256, return_sequences=True, recurrent_dropout=0, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.3))
model.add(LSTM(units=128, recurrent_dropout=0))
model.add(Dropout(0.3))
model.add(Reshape((128, 1)))
model.add(LSTM(units = 64, recurrent_dropout=0))
model.add(Dense(1))

# optimizer : 모델 학습 최적화 방법, loss : 손실함수 설정
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') 
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# prediction
predictions = model.predict(X_test)

# calculate MAE
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)  #MAE: 0.009475831

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse)) #RMSE: 0.019931
'''
# %%
'''
Model 5

from keras.layers import Reshape #shape=(batch_size, time_steps, features)

model = Sequential()
model.add(LSTM(units=256, return_sequences=True, recurrent_dropout=0, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.3))
model.add(LSTM(units=256, recurrent_dropout=0))
model.add(Dropout(0.3))
model.add(Reshape((256, 1)))
model.add(LSTM(units = 256, recurrent_dropout=0))
model.add(Dense(1))

# optimizer : 모델 학습 최적화 방법, loss : 손실함수 설정
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') 
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# prediction
predictions = model.predict(X_test)

# calculate MAE
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)  #MAE: 0.010202692

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse)) #RMSE: 0.015525
'''

# %%
'''
Model 6

from keras.layers import Reshape #shape=(batch_size, time_steps, features)

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.3))
model.add(LSTM(units=128, recurrent_dropout=0))
model.add(Dropout(0.3))
model.add(Reshape((128, 1)))
model.add(LSTM(units = 128, recurrent_dropout=0))
model.add(Dense(1))

# optimizer : 모델 학습 최적화 방법, loss : 손실함수 설정
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') 
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# prediction
predictions = model.predict(X_test)

# calculate MAE
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)  #MAE: 0.012215211

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse)) #RMSE: 0.020171
'''
# %%
'''
Model 7

from keras.layers import Reshape #shape=(batch_size, time_steps, features)

model = Sequential()
model.add(LSTM(units=64, return_sequences=True, recurrent_dropout=0, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.3))
model.add(LSTM(units=64, recurrent_dropout=0))
model.add(Dropout(0.3))
model.add(Reshape((64, 1)))
model.add(LSTM(units = 64, recurrent_dropout=0))
model.add(Dense(1))

# optimizer : 모델 학습 최적화 방법, loss : 손실함수 설정
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') 
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# prediction
predictions = model.predict(X_test)

# calculate MAE
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)  #MAE: 0.013567367

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse)) #RMSE: 0.026368
'''
# %%
'''
Model 8

from keras.layers import Reshape #shape=(batch_size, time_steps, features)

model = Sequential()
model.add(LSTM(units=512, return_sequences=True, recurrent_dropout=0, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.3))
model.add(LSTM(units=512, recurrent_dropout=0))
model.add(Dropout(0.3))
model.add(Reshape((512, 1)))
model.add(LSTM(units = 512, recurrent_dropout=0))
model.add(Dense(1))

# optimizer : 모델 학습 최적화 방법, loss : 손실함수 설정
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') 
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# prediction
predictions = model.predict(X_test)

# calculate MAE
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)  #MAE: 0.2693398

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse)) #RMSE: 0.315044
'''
# 각 레이어의 노드 수를 조절하며 성능을 비교해 보았음
# Model 4, 6, 7이 가장 좋은 성능을 보임. 세가지 모두 비슷한 성능을 보이므로 가장 간단한 model7이 선택되면 좋을듯.
# 추후 모델7에 다양한 하이퍼파라미터를 적용 예정