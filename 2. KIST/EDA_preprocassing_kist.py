#%%
import pandas as pd
import pandas_profiling
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

df = pd.read_csv('/home/jy/dataset/KIST/graphdata_kist210901~220531.csv')
df.head() # 1분간격 데이터셋
df.info()
df['시간'] = pd.to_datetime(df['시간'])

df.profile_report()

#%%
'''
1. visualization

'''
df.plot.box(subplots = True, figsize = (50, 20))
msno.matrix(df)  #값이 있으면 검은색, 값이 없으면 흰색으로 나타남
plt.show()


'''
2. Missing values

'''
# 결측치 앞과 뒤의 값으로 채운다.
# raw data = df
# 앞의 값으로 채움 = df2
# 뒤의 값으로 채움 = df3
df2 = df.fillna(method = 'ffill')  #method = 'ffill'은 앞의 값
df3 = df2.fillna(method = 'backfill')  #ethod = 'backfill'은 뒤의 값
df3.isnull().sum()
df3 = df3.drop(['온도'], axis = 1)

df4 = df3



'''
3. Outliers

'''
# IQR

for c in df3.columns:
    if df3[c].dtype == float or df3[c].dtype == int:
        q1 = df3[c].quantile(0.25)
        q3 = df3[c].quantile(0.75)
        IQR = q3 - q1
        df3 = df3[df3[c].between(q1 - 1.5 * IQR, q3 + 1.5 * IQR, inclusive=True)]
        print("Column : " + c + "\'s outliers which out of IQR are removed.")
df3.reset_index(drop=True, inplace=True)
a = df3['시간'] # 정규화를 위해 저장
#df3.set_index('time', inplace = True)


'''
4. Normalization

'''
df3.set_index('시간', inplace = True)
scaler = MinMaxScaler()
scaler.fit(df3)
df3_scaled = scaler.transform(df3)
df3_scaled = pd.DataFrame(df3_scaled, columns = df3.columns)
df3_scaled.info()

df4 = pd.merge(a, df3_scaled, left_index = True, right_index = True) # datetime이 사라져 저장해 뒀던 'time_x'를 붙임
df4.info()


df4 = df4.rename(columns = {'시간':'time'})
df4 = df4.rename(columns = {'내부온도':'temp'})
df4 = df4.rename(columns = {'내부습도':'humidity'})
#df4 = df4.rename(columns = {'온도':'temp_out'})
df4 = df4.rename(columns = {'엽온':'temp_leaf'})
df4 = df4.rename(columns = {'증발산량':'evaporation'}) 
df4 = df4.rename(columns = {'일사량계':'radiation'})
df4.set_index('time', inplace = True)
df4.head()   # VIF 분석시 엽온만 제거되는데, 엽온 예측을 위해 굳이 제거하지 않아도 되지 않을까



'''
5. Variance Inflation Factor(VIF, 다중공선성)


# visualization
df4.corr(method = 'pearson')
sns.heatmap(df4.corr(method = 'pearson'))

# VIF
a = df4.corr() >= 0.8
print('=== gc4의 correlation >= 0.8 : ', a.sum(True))

# VIF 출력을 위한 데이터 프레임 형성
vif = pd.DataFrame()

# VIF 값과 각 Feature 이름에 대해 설정
vif["VIF Factor"] = [variance_inflation_factor(df4.values, i) 
for i in range(df4.shape[1])]
vif["features"] = df4.columns 

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

df_final = vif(df4)  

'''

'''
6. Save

'''
#df_final.to_csv('df_KIST_final.csv')



'''
완성된 데이터에는 5가지의 환경 변수가 포함되어 있음
- temp : 온실 내부 온도
- humidity : 온실 내부 상대 습도
- CO2 : 온실 내부 CO2 농도
- evaporation : 온실 내부 증발산량
- radiation : 온실 내부 태양복사
+ temp_leaf : 엽온
'''
# %%
