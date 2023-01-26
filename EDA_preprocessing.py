import pandas as pd
import datetime
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import numpy as np
import missingno as msno
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler


'''
Data load

'''
# Greenhouse Climate
gc = pd.read_csv('/home/jy/TimeSeries_Analysis/AGIC/GreenhouseClimate_automato_modified.csv')

# Outside Weather
w = pd.read_csv('/home/jy/TimeSeries_Analysis/AGIC/Weather_modified.csv')

# Crop Parameter
cp = pd.read_csv('/home/jy/TimeSeries_Analysis/AGIC/CropParameters_automato_modified__.csv')


'''
EDA and Preprocessing

'''
#  greenhouse climate에서 _sp, _vip로 끝나는 칼럼은 setpoint이므로 제거한다.
gc.info()  
gc = gc[['time', 'CO2air','EC_drain_PC', 'Rhair', 'Tair', 'Tot_PAR', 'pH_drain_PC','water_sup']]
gc.head()

w.info()  # P_output, T_sky는 전부 NaN이므로 제거한다.
w = w.drop(['P_out', 'T_sky', 'Winddir', 'Pyrgeo', 'u_wind'], axis = 1)
w

cp.info()
cp.head()
cp = cp.drop(['plant_dens', 'stem_dens '], axis = 1) # 타겟값을 제외하고 모두 제거한다.  
cp = cp.rename(columns = {'Time':'time'})

# datetime
gc['time'] = pd.to_datetime(gc['time'])
w['time'] = pd.to_datetime(w['time'])
cp['time'] = pd.to_datetime(cp['time'])

gc.describe()
w.describe()
cp.describe()

# merge (greenhouse climate + weather)
#gcw = pd.merge(gc, w, on='time')
#gcw.head()

'''
1. visualization

'''
gc.plot.box(subplots = True, figsize = (50, 20))
w.plot.box(subplots = True, figsize = (50, 20))
msno.matrix(gc)  #값이 있으면 검은색, 값이 없으면 흰색으로 나타남
msno.matrix(w)
plt.show()


'''
2. Missing values

'''
gc2 = gc.fillna(method = 'ffill')
gc3 = gc2.fillna(method = 'backfill')
gc3.isnull().sum()
gc4 = gc3


w2 = w.fillna(method = 'ffill')
w3 = w2.fillna(method = 'backfill')
w3.isnull().sum()
w4 = w3


# missing values -> cp -> 애매하게 채우는 것보단 삭제하는 것이 나을듯
cp.isnull().sum()
cp.head(23)
cp2 = cp.dropna(how = 'any')
cp2.isnull().sum()

'''
3. Outliers

'''
# IQR
# col = ['Tair', 'Rhair', 'CO2air', 'co2_dos', 'Tot_PAR', 'T_out', 'RH_out','I_glob', 'Windsp']
for c in gc3.columns:
    if gc3[c].dtype == float or gc3[c].dtype == int:
        q1 = gc3[c].quantile(0.25)
        q3 = gc3[c].quantile(0.75)
        IQR = q3 - q1
        gc3 = gc3[gc3[c].between(q1 - 1.5 * IQR, q3 + 1.5 * IQR, inclusive=True)]
        print("Column : " + c + "\'s outliers which out of IQR are removed.")
gc3.reset_index(drop=True, inplace=True)
#a = gc3['time'] # 정규화를 위해 저장
#gc3.set_index('time', inplace = True)


for c in w3.columns:
    if w3[c].dtype == float or w3[c].dtype == int:
        q1 = w3[c].quantile(0.25)
        q3 = w3[c].quantile(0.75)
        IQR = q3 - q1
        w3 = w3[w3[c].between(q1 - 1.5 * IQR, q3 + 1.5 * IQR, inclusive=True)]
        print("Column : " + c + "\'s outliers which out of IQR are removed.")
w3.reset_index(drop=True, inplace=True)
#b = w3['time'] # 정규화를 위해 저장
#w3.set_index('time', inplace = True)


'''
4. Merge

'''
gcw = pd.merge(gc3, w3, on = 'time', how = 'left')
gcw['date'] = gcw['time'].dt.date # 공통 column을 만들어준다.
cp2['date'] = cp2['time'].dt.date

dataset = pd.merge(gcw, cp2, on='date', how = 'outer')
dataset = dataset.drop(['date', 'time_y'], axis = 1) # 필요없음

# missing value -> crop_parameter -> 선형으로 비례하는 값들로 결측치를 채움
dataset['Stem_thick'].interpolate(method = 'linear', limit_direction = 'backward',inplace = True)
dataset['Cum_trusses'].interpolate(method = 'linear', limit_direction = 'backward',inplace = True)
dataset.isnull().sum()

dataset2 = dataset.fillna(method = 'ffill')
dataset3 = dataset2.fillna(method = 'backfill')

dataset3.isnull().sum()
a = dataset3['time_x'] # 정규화를 위해 저장


'''
5. Normalization

'''
dataset3.set_index('time_x', inplace = True)
scaler = MinMaxScaler()
scaler.fit(dataset3)
dataset3_scaled = scaler.transform(dataset3)
dataset3_scaled = pd.DataFrame(dataset3_scaled, columns = dataset3.columns)
dataset3_scaled.info()

dataset4 = pd.merge(a, dataset3_scaled, left_index = True, right_index = True)
dataset4.info()
dataset4 = dataset4.rename(columns = {'time_x':'time'})
dataset4.set_index('time', inplace = True)
dataset4.head()



'''
6. Variance Inflation Factor(VIF, 다중공선성)

'''
# visualization
dataset4.corr(method = 'pearson')
sns.heatmap(dataset4.corr(method = 'pearson'))

# VIF
a = dataset4.corr() >= 0.7
print('=== gc4의 correlation >= 0.7 : ', a.sum(True))

# VIF 출력을 위한 데이터 프레임 형성
vif = pd.DataFrame()

# VIF 값과 각 Feature 이름에 대해 설정
vif["VIF Factor"] = [variance_inflation_factor(dataset4.values, i) 
for i in range(dataset4.shape[1])]
vif["features"] = dataset4.columns 

# VIF 값이 높은 순으로 정렬
vif = vif.sort_values(by="VIF Factor", ascending=False)
vif = vif.reset_index().drop(columns='index')
vif

def vif(x):
    # vif 10 초과시 drop을 위한 임계값 설정
    thresh = 10
    # Filter method로 feature selection 진행 후 최종 도출 될 데이터 프레임 형성
    output = pd.DataFrame()
    # 데이터의 컬럼 개수 설정
    k = x.shape[1]
    # VIF 측정
    vif = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    for i in range(1,k):
        print(f'{i}번째 VIF 측정')
        # VIF 최대 값 선정
        a = np.argmax(vif)
        print(f'Max VIF feature & value : {x.columns[a]}, {vif[a]}')
        # VIF 최대 값이 임계치를 넘지 않는 경우 break
        if (vif[a] <= thresh):
            print('\n')
            for q in range(output.shape[1]):
                print(f'{output.columns[q]}의 vif는 {np.round(vif[q],2)}입니다.')
            break
        # VIF 최대 값이 임계치를 넘는 경우, + 1번째 시도인 경우 : if 문으로 해당 feature 제거 후 다시 vif 측정
        if (i == 1):
            output = x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        # VIF 최대 값이 임계치를 넘는 경우, + 1번째 이후 시도인 경우 : if 문으로 해당 feature 제거 후 다시 vif 측정
        elif (i > 1):
            output = output.drop(output.columns[a], axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
    return(output)
df = vif(dataset4)


'''
7. Save

'''
df.to_csv('df_AGIC_final.csv')