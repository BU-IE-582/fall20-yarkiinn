```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
```


```python
import warnings
warnings.filterwarnings('ignore') #Added for unelegant shape of warnings
```


```python
df = pd.read_csv('elec_consump.csv',encoding="ISO-8859-1") #read data
```


```python
df.rename(columns={'Tarih': 'Date',"Saat":"Hour","Tüketim Miktarý (MWh)":"Consumption_Amount"}, inplace=True) #changed column names
```


```python
for i in range(0,len(df["Hour"])):  #deleted the first "0" of all hours to use them propely in further parts
    if df["Hour"].iloc[i].startswith("0"):
        df["Hour"].iloc[i]=df["Hour"].iloc[i][1:]  
```


```python
df["Consumption_Amount"]=df["Consumption_Amount"].str.replace(".","")   
```


```python
df["Consumption_Amount"]=df["Consumption_Amount"].str.replace(",",".")
```


```python
df["Consumption_Amount"] = df["Consumption_Amount"].apply(pd.to_numeric)    #transformed the format of consumption into float
```


```python
df["Date"]=pd.to_datetime(df['Date'])   #transformed the format of dates into datetime
```


```python
df["Consumption_Amount"]
```




    0        26277.24
    1        24991.82
    2        23532.61
    3        22464.78
    4        22002.91
               ...   
    43123    40720.16
    43124    39195.74
    43125    38310.79
    43126    37173.97
    43127    35725.46
    Name: Consumption_Amount, Length: 43128, dtype: float64




```python
df.insert(0,"index",range(0,len(df)))
```


```python
print(df)
```

           index       Date   Hour  Consumption_Amount
    0          0 2016-01-01   0:00            26277.24
    1          1 2016-01-01   1:00            24991.82
    2          2 2016-01-01   2:00            23532.61
    3          3 2016-01-01   3:00            22464.78
    4          4 2016-01-01   4:00            22002.91
    ...      ...        ...    ...                 ...
    43123  43123 2020-01-12  19:00            40720.16
    43124  43124 2020-01-12  20:00            39195.74
    43125  43125 2020-01-12  21:00            38310.79
    43126  43126 2020-01-12  22:00            37173.97
    43127  43127 2020-01-12  23:00            35725.46
    
    [43128 rows x 4 columns]
    


```python
df[df["Date"]=="2020-01-11"]["index"].head()   #Determined starting index of test data to split data into test and train
```




    42384    42384
    42385    42385
    42386    42386
    42387    42387
    42388    42388
    Name: index, dtype: int32




```python
df_test=df.iloc[42384:,:]
```


```python
print(df_test)  #stated test dataframe
```

           index       Date   Hour  Consumption_Amount
    42384  42384 2020-01-11   0:00            28701.62
    42385  42385 2020-01-11   1:00            27254.15
    42386  42386 2020-01-11   2:00            26078.57
    42387  42387 2020-01-11   3:00            25449.47
    42388  42388 2020-01-11   4:00            25324.01
    ...      ...        ...    ...                 ...
    43123  43123 2020-01-12  19:00            40720.16
    43124  43124 2020-01-12  20:00            39195.74
    43125  43125 2020-01-12  21:00            38310.79
    43126  43126 2020-01-12  22:00            37173.97
    43127  43127 2020-01-12  23:00            35725.46
    
    [744 rows x 4 columns]
    


```python
test_y=df_test["Consumption_Amount"]
```


```python
test_y
```




    42384    28701.62
    42385    27254.15
    42386    26078.57
    42387    25449.47
    42388    25324.01
               ...   
    43123    40720.16
    43124    39195.74
    43125    38310.79
    43126    37173.97
    43127    35725.46
    Name: Consumption_Amount, Length: 744, dtype: float64




```python
lag_168=df_test["index"].to_numpy()-168     #The consumption value of 168 hours ago was used for prediction
```


```python
lag_168_pred=df[df["index"].isin(lag_168)]["Consumption_Amount"].to_numpy()  #Lag 168 value calculated for the purpose stated above
```


```python
test_y_nump=test_y.to_numpy()
```


```python
err_168=np.abs((test_y_nump-lag_168_pred)/test_y_nump)*100      #mape formula was applied
```


```python
np.mean(err_168)    #mape value was calculated
```




    3.449188482612284




```python
lag_48=df_test["index"].to_numpy()-48      #The consumption value of 48 hours ago was used for prediction
```


```python
lag_48_pred=df[df["index"].isin(lag_48)]["Consumption_Amount"].to_numpy()    #Lag 48 value calculated for the purpose stated above
```


```python
err_48=np.abs((test_y_nump-lag_48_pred)/test_y_nump)*100
```


```python
np.mean(err_48)  #mape value was calculated
```




    8.060314509077507




```python
#Mape value of Lag 168 is better than Lag48. So, we can say that using Lag 168 has better forecasting performance.
```


```python
sns.boxplot(y=(err_168),palette="colorblind")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x60a152708>




![png](output_27_1.png)



```python
sns.boxplot(y=(err_48),palette="colorblind")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x60a58a708>




![png](output_28_1.png)



```python
#Part b
```


```python
lag_168_ind=df["index"].iloc[168:]    
```


```python
a=lag_168_ind.to_numpy()-168    #Calculated lag 168 values
```


```python
df[df["index"].isin(a)]["Consumption_Amount"]    
```




    0        26277.24
    1        24991.82
    2        23532.61
    3        22464.78
    4        22002.91
               ...   
    42955    40895.17
    42956    39240.65
    42957    38366.41
    42958    37446.73
    42959    36186.83
    Name: Consumption_Amount, Length: 42960, dtype: float64




```python
cols=["index","Date","Hour","Lag_48","Lag_168","Consumption_Amount"]  #Started to prepare for long format
```


```python
new_df=pd.DataFrame(columns=cols)
```


```python
new_df["Lag_168"]=df[df["index"].isin(a)]["Consumption_Amount"]
```


```python
new_df["index"]=range(168,len(df))
```


```python
lag_48_ind=df["index"].iloc[168:]          #Calculated lag 168 values
```


```python
b=lag_48_ind.to_numpy()-48       
```


```python
df[df["index"].isin(b)]["Consumption_Amount"]
```




    120      29189.27
    121      27614.02
    122      26578.97
    123      25719.19
    124      25864.63
               ...   
    43075    36310.77
    43076    35383.08
    43077    34491.45
    43078    33698.14
    43079    32295.74
    Name: Consumption_Amount, Length: 42960, dtype: float64




```python
lag_48_vals=df[df["index"].isin(b)]["Consumption_Amount"].to_frame()
```


```python
lag_48_vals.reset_index(inplace=True)
```


```python
lag_48_vals=lag_48_vals["Consumption_Amount"]
```


```python
new_df["Lag_48"]=lag_48_vals
```


```python
df[df["index"]>167]["Consumption_Amount"]
```




    168      28602.02
    169      27112.37
    170      25975.34
    171      25315.55
    172      25128.15
               ...   
    43123    40720.16
    43124    39195.74
    43125    38310.79
    43126    37173.97
    43127    35725.46
    Name: Consumption_Amount, Length: 42960, dtype: float64




```python
cons_vals=df[df["index"]>167]["Consumption_Amount"].to_frame()   #Consumption values were arranged based on lag values for long format
```


```python
cons_vals.reset_index(inplace=True)
```


```python
cons_vals=cons_vals["Consumption_Amount"]
```


```python
new_df["Consumption_Amount"]=cons_vals
```


```python
df[df["index"]>167]["Date"]
```




    168     2016-08-01
    169     2016-08-01
    170     2016-08-01
    171     2016-08-01
    172     2016-08-01
               ...    
    43123   2020-01-12
    43124   2020-01-12
    43125   2020-01-12
    43126   2020-01-12
    43127   2020-01-12
    Name: Date, Length: 42960, dtype: datetime64[ns]




```python
dates=df[df["index"]>167]["Date"].to_frame()    #Indexes were arranged based on lag values for long format
```


```python
dates.reset_index(inplace=True)
```


```python
dates=dates["Date"]
```


```python
new_df["Date"]=dates
```


```python
df[df["index"]>167]["Hour"]
```




    168       0:00
    169       1:00
    170       2:00
    171       3:00
    172       4:00
             ...  
    43123    19:00
    43124    20:00
    43125    21:00
    43126    22:00
    43127    23:00
    Name: Hour, Length: 42960, dtype: object




```python
hours=df[df["index"]>167]["Hour"].to_frame() #Hours were arranged based on lag values for long format
```


```python
hours.reset_index(inplace=True)
```


```python
hours=hours["Hour"]
```


```python
new_df["Hour"]=hours
```


```python
print(new_df)
```

           index       Date   Hour    Lag_48   Lag_168  Consumption_Amount
    0        168 2016-08-01   0:00  29189.27  26277.24            28602.02
    1        169 2016-08-01   1:00  27614.02  24991.82            27112.37
    2        170 2016-08-01   2:00  26578.97  23532.61            25975.34
    3        171 2016-08-01   3:00  25719.19  22464.78            25315.55
    4        172 2016-08-01   4:00  25864.63  22002.91            25128.15
    ...      ...        ...    ...       ...       ...                 ...
    42955  43123 2020-01-12  19:00  36310.77  40895.17            40720.16
    42956  43124 2020-01-12  20:00  35383.08  39240.65            39195.74
    42957  43125 2020-01-12  21:00  34491.45  38366.41            38310.79
    42958  43126 2020-01-12  22:00  33698.14  37446.73            37173.97
    42959  43127 2020-01-12  23:00  32295.74  36186.83            35725.46
    
    [42960 rows x 6 columns]
    


```python
new_df[new_df["Date"]=="2016-03-27"]["Hour"]
```




    1896     0:00
    1897     1:00
    1898     2:00
    1899     4:00
    1900     4:00
    1901     5:00
    1902     6:00
    1903     7:00
    1904     8:00
    1905     9:00
    1906    10:00
    1907    11:00
    1908    12:00
    1909    13:00
    1910    14:00
    1911    15:00
    1912    16:00
    1913    17:00
    1914    18:00
    1915    19:00
    1916    20:00
    1917    21:00
    1918    22:00
    1919    23:00
    Name: Hour, dtype: object




```python
junk1=new_df[new_df["Date"]=="2016-03-27"].index
junk2=new_df[new_df["Date"]=="2016-03-29"].index   
junk3=new_df[new_df["Date"]=="2016-04-03"].index
```


```python
new_df.drop(junk1,axis=0,inplace=True)
new_df.drop(junk2,axis=0,inplace=True)  # Three date point in data was discarded to cleanse inappropriate hours related to sunlight program in Turkey 
new_df.drop(junk3,axis=0,inplace=True)
```


```python
X_train=new_df.loc[:42216,["Lag_48","Lag_168"]]   #Lag data was splitted into test and train
X_test=new_df.loc[42216:,["Lag_48","Lag_168"]]
```


```python
y_train = new_df.loc[:42216,'Consumption_Amount']             #Consumption data was splitted into test and train
y_test = new_df.loc[42216:,'Consumption_Amount'].to_numpy()
```


```python
model=linear_model.LinearRegression() 
```


```python
model.fit(X_train,y_train)      #Prediction was made on Lag48 and Lag168 data by fitting linear model
predicted=model.predict(X_test)
```


```python
MAPE=np.mean(np.abs((y_test - predicted)/ y_test))*100   
```


```python
print(MAPE)    #MAPE value was calculated
```

    4.233438452971358
    


```python
model.coef_     #Coefficients were calculated
```




    array([0.3092438 , 0.64270557])




```python
model.intercept_  #Intercept was calculated
```




    1573.6541160669403




```python
APE=np.abs((y_test - predicted)/y_test)*100
plt.boxplot(APE)
plt.xlabel('MAPE')  
plt.show()
```


![png](output_71_0.png)



```python
#MAPE value of linear regression is higher than Lag168 MAPE value but better than Lag48 value which infers using Lag 168
# consumption values are better in terms of forecasting compared to linear regression for this data.
```


```python
#Part c
```


```python
features=['Lag_48','Lag_168']
final_df = pd.DataFrame()

for i in range(0,24):
    hour_data=new_df[new_df['Hour']==str(i)+':00']   #Seperated each hour nodes
    index_of_test=np.where(hour_data['Date']=="2020-01-11")
    train_data=hour_data.iloc[0:index_of_test[0][0],:]  #data splitted as train and test
    test_data=hour_data.iloc[index_of_test[0][0]:,:]
    train_X = train_data.loc[:,features]
    test_X = test_data.loc[:,features]

    train_y = train_data.loc[:,'Consumption_Amount']
    test_y = test_data.loc[:"Consumption_Amount"]

    model = linear_model.LinearRegression() 
    model.fit(train_X, train_y)
    predicted = model.predict(test_X)
    predicted_df = pd.DataFrame()
    predicted_df['Date']=test_data.loc[:,'Date']
    predicted_df['Hour']=test_data.loc[:,'Hour']  
    predicted_df['Consumption_Prediction']=predicted
    predicted_df['Actual_Consumption']=test_data.loc[:,'Consumption_Amount']
    final_df=pd.concat([final_df, predicted_df], axis=0, sort=False)
print(final_df)
```

                Date   Hour  Consumption_Prediction  Actual_Consumption
    42216 2020-01-11   0:00            29397.890757            28701.62
    42240 2020-02-11   0:00            29199.239620            27931.36
    42264 2020-03-11   0:00            29773.090504            31301.07
    42288 2020-04-11   0:00            29232.026265            31278.36
    42312 2020-05-11   0:00            30422.724214            31494.01
    ...          ...    ...                     ...                 ...
    42863 2020-11-27  23:00            35422.381959            36119.57
    42887 2020-11-28  23:00            34813.837657            34500.14
    42911 2020-11-29  23:00            34023.227558            32295.74
    42935 2020-11-30  23:00            34905.045490            35775.04
    42959 2020-01-12  23:00            34189.986217            35725.46
    
    [744 rows x 4 columns]
    


```python
#The predicted value and actual value of consumption was given for each hour.
```


```python
def MAPE(actual, prediction): 
    actual, prediction = np.array(actual), np.array(prediction)
    abs_MAPE = np.abs((actual - prediction) / actual)
    return np.mean(abs_MAPE) * 100
```


```python
act=final_df["Actual_Consumption"].to_numpy()
```


```python
pre=final_df["Consumption_Prediction"].to_numpy()
```


```python
abs_MAPE=np.abs((act - pre) / act)
```


```python
MAPE(act,pre)
```




    4.363534872320953




```python
plt.boxplot(abs_MAPE)
plt.xlabel('MAPE')  
plt.show()
```


![png](output_81_0.png)



```python
#Seasonality was checked for data to understand whether nights or evenings may differ in terms of consumption. MAPE value of hourly seperated
#data is close to MAPE value of linear regression.
```


```python
#Part d
```


```python
new_df[new_df["Date"]=="2020-01-11"]["index"]  #Determined starting index of test data
```




    42216    42384
    42217    42385
    42218    42386
    42219    42387
    42220    42388
    42221    42389
    42222    42390
    42223    42391
    42224    42392
    42225    42393
    42226    42394
    42227    42395
    42228    42396
    42229    42397
    42230    42398
    42231    42399
    42232    42400
    42233    42401
    42234    42402
    42235    42403
    42236    42404
    42237    42405
    42238    42406
    42239    42407
    Name: index, dtype: int32




```python
train=new_df.iloc[0:42216,:]
test=new_df.iloc[42216:,:]   #Splitted into trained and test
```


```python
train_=train.pivot(index='Date', columns='Hour', values=["Lag_48","Lag_168","Consumption_Amount"]) #48 features (72 with consumption amount actual values) was build such as 2 days ago and 7days ago
test_=test.pivot(index='Date', columns='Hour', values=["Lag_48","Lag_168","Consumption_Amount"]) #by using Lag-168 and Lag-48 values.
```


```python
print(train_)
```

                  Lag_48                                                    \
    Hour            0:00     10:00     11:00     12:00     13:00     14:00   
    Date                                                                     
    2016-01-02  30323.74  35450.49  36294.06  35204.25  34932.17  34697.06   
    2016-01-03  26608.30  25826.49  26509.59  26212.86  26348.07  26032.31   
    2016-01-04  29087.24  35071.25  35194.04  32717.89  32943.97  33377.70   
    2016-01-05  29066.79  35486.60  35810.48  33332.56  32932.17  34391.58   
    2016-01-06  25277.75  33473.30  34543.09  32547.88  33332.27  34119.17   
    ...              ...       ...       ...       ...       ...       ...   
    2020-12-06  31368.50  34352.81  35053.05  34043.71  34960.86  36159.30   
    2020-12-07  36573.49  38979.89  39817.24  38170.94  38234.85  40311.87   
    2020-12-08  34395.34  40157.01  40989.40  39978.07  40967.44  42329.40   
    2020-12-09  36451.72  39383.84  40156.17  39060.72  40345.43  42042.97   
    2020-12-10  31485.92  32694.83  33100.88  31980.88  32522.32  33102.26   
    
                                                        ... Consumption_Amount  \
    Hour           15:00     16:00     17:00     18:00  ...              22:00   
    Date                                                ...                      
    2016-01-02  34002.74  33457.63  34897.48  34596.05  ...           32777.70   
    2016-01-03  25786.34  25836.52  27066.61  29276.68  ...           30072.76   
    2016-01-04  33281.25  33056.87  32421.94  31943.77  ...           32119.82   
    2016-01-05  34048.34  33675.77  33208.01  32493.55  ...           29092.04   
    2016-01-06  33667.45  33942.88  33037.42  31468.56  ...           32421.19   
    ...              ...       ...       ...       ...  ...                ...   
    2020-12-06  36141.24  36305.28  35911.37  34888.35  ...           33305.15   
    2020-12-07  40699.23  41038.90  40752.17  39655.53  ...           36651.50   
    2020-12-08  42466.73  42787.16  42308.88  41089.48  ...           41641.76   
    2020-12-09  42356.33  42859.94  42854.02  42034.48  ...           37682.00   
    2020-12-10  33264.08  33698.06  34654.96  36092.63  ...           34871.41   
    
                                                                            \
    Hour           23:00      2:00      3:00      4:00      5:00      6:00   
    Date                                                                     
    2016-01-02  31448.40  24501.95  23973.46  23854.61  24195.66  25096.25   
    2016-01-03  28402.03  24743.83  24405.31  24226.34  24727.59  24855.56   
    2016-01-04  30783.43  26088.12  25487.10  25456.17  25863.30  26033.30   
    2016-01-05  27215.41  24527.90  23673.93  23353.26  22219.82  21375.04   
    2016-01-06  30710.30  25697.47  25062.99  24923.33  24205.53  24592.42   
    ...              ...       ...       ...       ...       ...       ...   
    2020-12-06  32186.24  28581.32  27786.17  27579.15  26714.32  26506.53   
    2020-12-07  35666.59  31845.94  31082.09  30239.42  29135.31  28177.21   
    2020-12-08  40112.36  34716.98  33805.81  33375.93  32915.32  32125.20   
    2020-12-09  36095.25  33560.72  32453.07  32195.08  32067.06  31334.34   
    2020-12-10  33037.24  26688.58  26223.44  26091.66  26343.24  26834.93   
    
                                              
    Hour            7:00      8:00      9:00  
    Date                                      
    2016-01-02  26488.24  32138.83  35430.82  
    2016-01-03  26277.09  30448.14  32151.77  
    2016-01-04  27445.54  31497.83  33735.57  
    2016-01-05  21885.64  23462.05  25288.95  
    2016-01-06  26508.98  30881.02  33605.72  
    ...              ...       ...       ...  
    2020-12-06  27439.33  31164.74  33151.62  
    2020-12-07  28039.97  28629.45  29578.29  
    2020-12-08  33457.86  37707.86  40260.15  
    2020-12-09  31625.48  34840.34  36792.22  
    2020-12-10  27820.86  31823.81  33636.65  
    
    [1759 rows x 72 columns]
    


```python
print(test_)
```

                  Lag_48                                                    \
    Hour            0:00     10:00     11:00     12:00     13:00     14:00   
    Date                                                                     
    2020-01-12  32332.68  28332.10  29003.50  28748.98  29081.29  29682.72   
    2020-04-11  27931.36  34530.73  34792.40  33256.31  33967.69  35239.46   
    2020-05-11  31301.07  35251.33  35451.90  34191.43  35387.31  36868.67   
    2020-06-11  31278.36  36953.41  37624.94  36237.72  37090.66  37947.51   
    2020-07-11  31494.01  37328.23  37995.61  36673.18  37474.83  38465.18   
    2020-08-11  32009.92  37023.74  37426.55  35504.94  35730.20  37473.42   
    2020-09-11  32079.29  33640.77  33982.07  32961.52  33436.74  33853.06   
    2020-10-11  30770.20  25872.33  26296.87  26247.39  26979.75  27311.96   
    2020-11-11  29431.56  34719.78  34762.37  32980.67  33549.23  35004.57   
    2020-11-13  32398.76  35411.64  35618.76  34000.28  34739.81  36053.33   
    2020-11-14  32363.50  35271.47  35403.36  33584.56  34366.97  35636.86   
    2020-11-15  32468.48  35447.62  35354.26  32980.12  33553.66  35534.50   
    2020-11-16  32450.53  34717.16  35131.00  34315.00  34813.13  35350.99   
    2020-11-17  31048.02  27488.11  27999.92  27963.55  28531.22  29082.54   
    2020-11-18  29921.97  36115.12  36215.12  34930.84  35562.96  36704.66   
    2020-11-19  33050.14  36641.78  36658.02  34863.45  35626.05  36604.82   
    2020-11-20  33367.10  36543.49  36760.46  35200.40  36006.68  37557.10   
    2020-11-21  33455.97  38233.15  38704.74  37415.32  38196.98  39412.98   
    2020-11-22  33651.81  39834.37  40475.03  38482.23  39003.06  40727.71   
    2020-11-23  33697.36  34969.54  35792.39  34480.23  35028.71  35312.45   
    2020-11-24  31685.68  27974.74  28525.59  28433.65  28962.66  29334.07   
    2020-11-25  30426.00  37012.71  36950.05  35296.28  35986.82  37089.41   
    2020-11-26  33520.81  37515.08  37261.41  35845.55  36583.20  37830.17   
    2020-11-27  33585.86  39507.26  39896.53  38446.53  39059.04  40148.58   
    2020-11-28  33902.71  39295.57  39312.61  37508.66  38124.78  39220.21   
    2020-11-29  33896.52  37808.82  37668.32  35381.81  35587.14  37205.46   
    2020-11-30  33935.56  35457.72  35822.08  34607.24  34962.65  35338.14   
    2020-12-11  32039.71  35239.96  35383.13  33645.52  34463.58  35696.60   
    
                                                        ... Consumption_Amount  \
    Hour           15:00     16:00     17:00     18:00  ...              22:00   
    Date                                                ...                      
    2020-01-12  30568.06  32342.53  35411.15  36984.04  ...           37173.97   
    2020-04-11  35526.07  36552.09  37959.05  38720.50  ...           35086.17   
    2020-05-11  36997.12  38021.90  39038.66  39314.53  ...           35449.89   
    2020-06-11  37981.78  38329.90  39271.36  39548.13  ...           35419.89   
    2020-07-11  38696.44  39111.30  40006.89  39891.23  ...           34199.98   
    2020-08-11  37614.03  38395.48  39456.23  39664.77  ...           32048.87   
    2020-09-11  34011.48  34934.08  36549.26  38006.32  ...           35314.45   
    2020-10-11  27914.76  29768.90  32450.62  34677.55  ...           35579.04   
    2020-11-11  35458.30  37024.05  38939.01  39482.19  ...           35748.63   
    2020-11-13  36466.69  37882.03  39718.50  40308.02  ...           35724.27   
    2020-11-14  36040.27  37712.17  38810.08  40355.11  ...           34725.68   
    2020-11-15  36199.20  37660.00  39696.79  39992.94  ...           32732.91   
    2020-11-16  35556.45  36387.39  37810.27  38873.11  ...           36467.46   
    2020-11-17  29875.90  31267.82  33723.79  35534.51  ...           36771.07   
    2020-11-18  37351.02  38792.51  40417.23  40819.19  ...           36816.58   
    2020-11-19  37306.60  38712.18  40754.69  41334.00  ...           37270.44   
    2020-11-20  38551.32  39933.51  41551.26  41501.39  ...           36786.90   
    2020-11-21  39853.82  40918.91  42208.11  41673.45  ...           35250.20   
    2020-11-22  40550.85  40976.32  41854.37  41564.30  ...           33602.91   
    2020-11-23  35408.44  36581.59  38630.06  39379.15  ...           36983.95   
    2020-11-24  30301.50  32108.31  34919.68  36647.32  ...           37446.73   
    2020-11-25  37586.08  39435.91  41569.99  41710.94  ...           37519.53   
    2020-11-26  38656.92  40396.69  42486.34  42255.24  ...           37607.20   
    2020-11-27  40330.14  41279.40  42837.79  42412.80  ...           37356.91   
    2020-11-28  39460.82  40532.72  42347.17  42312.06  ...           36016.67   
    2020-11-29  37760.66  39543.01  41895.73  42003.77  ...           33698.14   
    2020-11-30  35645.57  36905.20  39131.92  40203.21  ...           37225.65   
    2020-12-11  36271.37  37674.58  39588.11  39930.95  ...           35905.85   
    
                                                                            \
    Hour           23:00      2:00      3:00      4:00      5:00      6:00   
    Date                                                                     
    2020-01-12  35725.46  30884.17  30082.68  29844.18  30100.53  30743.44   
    2020-04-11  33533.38  28784.91  28117.57  27905.46  28226.50  28966.82   
    2020-05-11  34053.60  28798.70  28274.43  28252.16  28349.84  29122.89   
    2020-06-11  34157.70  29373.18  28514.08  28223.05  28561.94  29420.99   
    2020-07-11  32808.00  29333.87  28368.57  28365.36  28379.15  29135.26   
    2020-08-11  30885.94  27826.88  27090.40  26889.26  26760.23  26929.14   
    2020-09-11  33838.31  26904.55  26411.68  26359.61  26635.71  27660.58   
    2020-10-11  34176.40  29518.98  28949.57  28570.41  28835.19  29687.03   
    2020-11-11  34327.68  29728.25  28881.86  28766.83  28953.33  29860.46   
    2020-11-13  34360.70  29847.75  29145.47  28962.57  29296.21  30292.12   
    2020-11-14  32784.78  29924.72  29056.63  28771.48  29022.93  29557.49   
    2020-11-15  31167.42  28260.55  27501.22  27232.02  27252.27  27401.45   
    2020-11-16  35142.84  27332.25  26704.18  26573.26  27051.81  28037.82   
    2020-11-17  35374.13  30240.40  29579.22  29367.49  29740.95  30666.30   
    2020-11-18  35620.10  30472.93  29671.27  29514.25  29866.15  30633.98   
    2020-11-19  35731.83  30698.22  29846.76  29600.96  29899.31  30743.88   
    2020-11-20  35102.41  30832.30  29991.21  29795.53  29951.31  30758.76   
    2020-11-21  33817.76  30692.47  29828.03  29425.40  29748.05  30186.14   
    2020-11-22  32178.03  28703.45  28072.20  27651.60  27566.19  27737.26   
    2020-11-23  35594.03  28176.27  27599.20  27238.99  27754.59  28747.47   
    2020-11-24  36186.83  30513.78  29844.10  29650.85  29873.42  30934.51   
    2020-11-25  36160.62  31064.86  30213.43  29958.96  30234.26  31138.64   
    2020-11-26  36156.08  31023.35  30190.73  30055.35  30335.29  31201.79   
    2020-11-27  36119.57  30897.69  30191.20  29956.10  30199.97  31015.63   
    2020-11-28  34500.14  30439.27  30020.41  29825.63  29984.86  30444.97   
    2020-11-29  32295.74  29119.33  28484.61  28150.14  28065.44  28189.16   
    2020-11-30  35775.04  27689.33  27608.55  27566.71  27897.84  28873.20   
    2020-12-11  34497.57  29759.03  28965.76  28714.25  29072.24  30046.23   
    
                                              
    Hour            7:00      8:00      9:00  
    Date                                      
    2020-01-12  32279.80  36657.05  39815.64  
    2020-04-11  29817.81  34004.19  36566.70  
    2020-05-11  30291.01  33916.21  36927.15  
    2020-06-11  30594.99  34438.66  36638.12  
    2020-07-11  29387.25  32039.46  33274.45  
    2020-08-11  26111.23  25452.79  25483.71  
    2020-09-11  29448.99  33515.58  34884.78  
    2020-10-11  31025.24  34276.35  35415.76  
    2020-11-11  31165.52  34533.51  35809.71  
    2020-11-13  31490.94  34679.86  35732.52  
    2020-11-14  30030.71  32698.70  34328.19  
    2020-11-15  26907.92  26786.69  27095.15  
    2020-11-16  29930.18  34172.00  36130.29  
    2020-11-17  31968.39  35379.53  36886.53  
    2020-11-18  32002.03  35510.91  36754.20  
    2020-11-19  32074.70  35610.62  37711.29  
    2020-11-20  32370.10  36386.47  39029.49  
    2020-11-21  30564.89  32613.91  34058.84  
    2020-11-22  27439.05  26905.24  27247.25  
    2020-11-23  30989.04  35068.99  36907.37  
    2020-11-24  32239.30  35946.87  37504.70  
    2020-11-25  32838.39  36634.41  38845.22  
    2020-11-26  32775.21  36882.21  39059.09  
    2020-11-27  32518.79  36129.99  37851.53  
    2020-11-28  31159.67  33325.74  34868.28  
    2020-11-29  27883.71  27219.84  27521.40  
    2020-11-30  31137.18  35801.80  38418.47  
    2020-12-11  31394.88  34571.78  35745.74  
    
    [28 rows x 72 columns]
    


```python
X_train=train_[['Lag_48', 'Lag_168']]  #Splitted train and test data
X_test=test_[['Lag_48', 'Lag_168']]
```


```python
MAPEs={}
err_list = []
for i in range(0,24):
    y_train = train_.loc[:,('Consumption_Amount', str(i)+':00')].to_frame()
    y_test = test_.loc[:,('Consumption_Amount', str(i)+':00')].to_frame()
    alphas = np.logspace(-4, 1, 10)
    lassocv = linear_model.LassoCV(alphas=alphas,cv=10, random_state=0, max_iter = 2000) #10-fold cross validation was performed.
    lassocv.fit(X_train, y_train)
    lassocv_score_train = lassocv.score(X_train, y_train)
    lassocv_score_test = lassocv.score(X_test, y_test)
    lassocv_alphas = lassocv.alphas_
    lassocv_alpha = lassocv.alpha_
    best_lasso = linear_model.Lasso(alpha=lassocv_alpha)
    best_lasso.fit(X_train, y_train)
    predicted_y=best_lasso.predict(X_test)
    err=MAPE(y_test.to_numpy(),predicted_y)
    y_test = y_test.to_numpy().reshape(-1)
    err=MAPE(y_test,predicted_y)
    err_extended = np.divide(np.abs(y_test - predicted_y), y_test) * 100
    err_list.append(err_extended)
    MAPEs[i]=[]
    MAPEs[i].append(err)
MAPEs
```




    {0: [1.3536704422670456],
     1: [1.5160519570354045],
     2: [1.4525255671446071],
     3: [1.3652307976462048],
     4: [1.4420676130392673],
     5: [1.4073896026382995],
     6: [1.642707869571564],
     7: [1.6534942974546285],
     8: [2.216363815066442],
     9: [3.3026764368973054],
     10: [3.8355758583788537],
     11: [4.119821305307467],
     12: [4.485375978959793],
     13: [4.322648430217408],
     14: [4.039716642519007],
     15: [3.490473701876428],
     16: [2.510655502866077],
     17: [1.7786170291159846],
     18: [1.4300401002154561],
     19: [1.3815723162597244],
     20: [1.410609640341767],
     21: [1.4306426582173968],
     22: [1.3740100676891827],
     23: [1.5767204815078324]}




```python
plt.boxplot(err_list)
plt.xlabel('MAPES')  
plt.show()
```


![png](output_91_0.png)



```python
#With 10 fold cross validation best alpha value was tried to find by using L1 penalty.
```


```python
#For each hour MAPE values were calculated by using penalized regression. And we can say that it is better to predict consumption,from hours after 17:00 until night and from night
#till the early morning hours for penalized regression approach. Because MAPE values of these hours are relatviely smaller.
```


```python
hourly_regression_MAPE=4.363534872320953
```


```python
Lag_168_MAPE=3.449188482612284
```


```python
Lag_48_MAPE= 8.060314509077507
```


```python
regression_MAPE=4.233438452971358
```


```python
k=list(MAPEs.values())
```


```python
total=0
for i in range(len(k)):
    total+= k[i][0]
```


```python
print(total/24)   #average of MAPE values of penalized regression
```

    2.2724440880097143
    


```python
# I concluded that it is better to use penalized regression to predict electiricty consumption compared to other approaches. Lag_168 MAPE has
#also better values compared to other approaches. Hourly linear regression and linear regression MAPE values are close to each other. It is not
# that meaningful to use hourly regression based on MAPE values. Since it can be seen that regressin and hourly regression MAPE values are close
#to each other.
```
