```python
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
```


```python
#SkillCraft Data Set
```


```python
#Game action of players in terms of objects and map with keyboard,camera and mouse movements are collected from different games.
#Also player information such as age, playing hours per week and total hours of playing involves in data set.
# LeagueIndex of players are predicted.
```


```python
#Data set has 3395 instances with 4 integer, 1 ordinal, 15 continous variables. Ordinal variable "LeageueIndex" is the target variable.
```


```python
#1. GameID: Unique ID number for each game (integer)
#2. LeagueIndex: Bronze, Silver, Gold, Platinum, Diamond, Master, GrandMaster, and Professional leagues coded 1-8 (Ordinal)
#3. Age: Age of each player (integer)
#4. HoursPerWeek: Reported hours spent playing per week (integer)
#5. TotalHours: Reported total hours spent playing (integer)
#6. APM: Action per minute (continuous)
#7. SelectByHotkeys: Number of unit or building selections made using hotkeys per timestamp (continuous)
#8. AssignToHotkeys: Number of units or buildings assigned to hotkeys per timestamp (continuous)
#9. UniqueHotkeys: Number of unique hotkeys used per timestamp (continuous)
#10. MinimapAttacks: Number of attack actions on minimap per timestamp (continuous)
#11. MinimapRightClicks: number of right-clicks on minimap per timestamp (continuous)
#12. NumberOfPACs: Number of PACs per timestamp (continuous)
#13. GapBetweenPACs: Mean duration in milliseconds between PACs (continuous)
#14. ActionLatency: Mean latency from the onset of a PACs to their first action in milliseconds (continuous)
#15. ActionsInPAC: Mean number of actions within each PAC (continuous)
#16. TotalMapExplored: The number of 24x24 game coordinate grids viewed by the player per timestamp (continuous)
#17. WorkersMade: Number of SCVs, drones, and probes trained per timestamp (continuous)
#18. UniqueUnitsMade: Unique unites made per timestamp (continuous)
#19. ComplexUnitsMade: Number of ghosts, infestors, and high templars trained per timestamp (continuous)
#20. ComplexAbilitiesUsed: Abilities requiring specific targeting instructions used per timestamp (continuous)
```


```python
#The problem is multi-class classification problem (league index is between 1-8 ordinal variables)
```


```python
df=pd.read_csv("SkillCraft.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GameID</th>
      <th>LeagueIndex</th>
      <th>Age</th>
      <th>HoursPerWeek</th>
      <th>TotalHours</th>
      <th>APM</th>
      <th>SelectByHotkeys</th>
      <th>AssignToHotkeys</th>
      <th>UniqueHotkeys</th>
      <th>MinimapAttacks</th>
      <th>MinimapRightClicks</th>
      <th>NumberOfPACs</th>
      <th>GapBetweenPACs</th>
      <th>ActionLatency</th>
      <th>ActionsInPAC</th>
      <th>TotalMapExplored</th>
      <th>WorkersMade</th>
      <th>UniqueUnitsMade</th>
      <th>ComplexUnitsMade</th>
      <th>ComplexAbilitiesUsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>5</td>
      <td>27</td>
      <td>10</td>
      <td>3000</td>
      <td>143.7180</td>
      <td>0.003515</td>
      <td>0.000220</td>
      <td>7</td>
      <td>0.000110</td>
      <td>0.000392</td>
      <td>0.004849</td>
      <td>32.6677</td>
      <td>40.8673</td>
      <td>4.7508</td>
      <td>28</td>
      <td>0.001397</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>5</td>
      <td>23</td>
      <td>10</td>
      <td>5000</td>
      <td>129.2322</td>
      <td>0.003304</td>
      <td>0.000259</td>
      <td>4</td>
      <td>0.000294</td>
      <td>0.000432</td>
      <td>0.004307</td>
      <td>32.9194</td>
      <td>42.3454</td>
      <td>4.8434</td>
      <td>22</td>
      <td>0.001193</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.000208</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56</td>
      <td>4</td>
      <td>30</td>
      <td>10</td>
      <td>200</td>
      <td>69.9612</td>
      <td>0.001101</td>
      <td>0.000336</td>
      <td>4</td>
      <td>0.000294</td>
      <td>0.000461</td>
      <td>0.002926</td>
      <td>44.6475</td>
      <td>75.3548</td>
      <td>4.0430</td>
      <td>22</td>
      <td>0.000745</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.000189</td>
    </tr>
    <tr>
      <th>3</th>
      <td>57</td>
      <td>3</td>
      <td>19</td>
      <td>20</td>
      <td>400</td>
      <td>107.6016</td>
      <td>0.001034</td>
      <td>0.000213</td>
      <td>1</td>
      <td>0.000053</td>
      <td>0.000543</td>
      <td>0.003783</td>
      <td>29.2203</td>
      <td>53.7352</td>
      <td>4.9155</td>
      <td>19</td>
      <td>0.000426</td>
      <td>7</td>
      <td>0.0</td>
      <td>0.000384</td>
    </tr>
    <tr>
      <th>4</th>
      <td>58</td>
      <td>3</td>
      <td>32</td>
      <td>10</td>
      <td>500</td>
      <td>122.8908</td>
      <td>0.001136</td>
      <td>0.000327</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.001329</td>
      <td>0.002368</td>
      <td>22.6885</td>
      <td>62.0813</td>
      <td>9.3740</td>
      <td>15</td>
      <td>0.001174</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.000019</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape    
```




    (3395, 20)




```python
df=df[~df.isin(["?"]).any(axis=1)]    #missing values are discarded
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GameID</th>
      <th>LeagueIndex</th>
      <th>Age</th>
      <th>HoursPerWeek</th>
      <th>TotalHours</th>
      <th>APM</th>
      <th>SelectByHotkeys</th>
      <th>AssignToHotkeys</th>
      <th>UniqueHotkeys</th>
      <th>MinimapAttacks</th>
      <th>MinimapRightClicks</th>
      <th>NumberOfPACs</th>
      <th>GapBetweenPACs</th>
      <th>ActionLatency</th>
      <th>ActionsInPAC</th>
      <th>TotalMapExplored</th>
      <th>WorkersMade</th>
      <th>UniqueUnitsMade</th>
      <th>ComplexUnitsMade</th>
      <th>ComplexAbilitiesUsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>5</td>
      <td>27</td>
      <td>10</td>
      <td>3000</td>
      <td>143.7180</td>
      <td>0.003515</td>
      <td>0.000220</td>
      <td>7</td>
      <td>0.000110</td>
      <td>0.000392</td>
      <td>0.004849</td>
      <td>32.6677</td>
      <td>40.8673</td>
      <td>4.7508</td>
      <td>28</td>
      <td>0.001397</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>5</td>
      <td>23</td>
      <td>10</td>
      <td>5000</td>
      <td>129.2322</td>
      <td>0.003304</td>
      <td>0.000259</td>
      <td>4</td>
      <td>0.000294</td>
      <td>0.000432</td>
      <td>0.004307</td>
      <td>32.9194</td>
      <td>42.3454</td>
      <td>4.8434</td>
      <td>22</td>
      <td>0.001193</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.000208</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56</td>
      <td>4</td>
      <td>30</td>
      <td>10</td>
      <td>200</td>
      <td>69.9612</td>
      <td>0.001101</td>
      <td>0.000336</td>
      <td>4</td>
      <td>0.000294</td>
      <td>0.000461</td>
      <td>0.002926</td>
      <td>44.6475</td>
      <td>75.3548</td>
      <td>4.0430</td>
      <td>22</td>
      <td>0.000745</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.000189</td>
    </tr>
    <tr>
      <th>3</th>
      <td>57</td>
      <td>3</td>
      <td>19</td>
      <td>20</td>
      <td>400</td>
      <td>107.6016</td>
      <td>0.001034</td>
      <td>0.000213</td>
      <td>1</td>
      <td>0.000053</td>
      <td>0.000543</td>
      <td>0.003783</td>
      <td>29.2203</td>
      <td>53.7352</td>
      <td>4.9155</td>
      <td>19</td>
      <td>0.000426</td>
      <td>7</td>
      <td>0.0</td>
      <td>0.000384</td>
    </tr>
    <tr>
      <th>4</th>
      <td>58</td>
      <td>3</td>
      <td>32</td>
      <td>10</td>
      <td>500</td>
      <td>122.8908</td>
      <td>0.001136</td>
      <td>0.000327</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.001329</td>
      <td>0.002368</td>
      <td>22.6885</td>
      <td>62.0813</td>
      <td>9.3740</td>
      <td>15</td>
      <td>0.001174</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.000019</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (3338, 20)




```python
League_Indexes=[1,2,3,4,5,6,7,8]    
```


```python
df.LeagueIndex=df.LeagueIndex.astype(CategoricalDtype(categories=League_Indexes,ordered=True))  
#League index was transformed into ordinal variable
```


```python
df.LeagueIndex.head()
```




    0    5
    1    5
    2    4
    3    3
    4    3
    Name: LeagueIndex, dtype: category
    Categories (8, int64): [1 < 2 < 3 < 4 < 5 < 6 < 7 < 8]




```python
df_train=df.loc[:2225,:]  #Train data was splitted based on 2/3 portion of the data.
```


```python
df_test=df.loc[2226:,:]   #Test data was splitted based on 1/3 portion of the data.
```


```python
k=["GameID","LeagueIndex"]
```


```python
X_train=df_train.drop(k,axis=1)  #Target variable was discarded from train set
```


```python
X_test=df_test.drop(k,axis=1)
```


```python
y_train=df.loc[:2225,"LeagueIndex"]    #Train set of target variable 
```


```python
y_test=df.loc[2226:,"LeagueIndex"]     #Test set of target variable 
```


```python
#Penalized Linear Regression
```


```python
grid ={"alpha":[0.05,0.1,0.5,1,2]}  #lambda set
```


```python
lasso = linear_model.Lasso(random_state=0).fit(X_train,y_train)
```


```python
train=lasso.predict(X_train)
```


```python
y=lasso.predict(X_test)
```


```python
print(np.sqrt(mean_squared_error(y_train,train))) #training error in terms of RMSE
```

    1.037941354526766
    


```python
print(np.sqrt(mean_squared_error(y_test,y)))  #test error in terms of RMSE
```

    1.0360006013689902
    


```python
print(r2_score(y_train, train))   #r2 score of training
```

    0.4958381743359096
    


```python
print(r2_score(y_test,y))        #r2 score of test
```

    0.4660536019307584
    


```python
model=Lasso(random_state=0)
search = GridSearchCV(model, grid, scoring="neg_root_mean_squared_error",cv=5)
results=search.fit(X_train,y_train)
print("Negated_RMSE",results.best_score_)
print("lambda",results.best_params_)
```

    Negated_RMSE -2.966888517223163
    lambda {'alpha': 0.05}
    


```python
tuned_lasso = linear_model.Lasso(alpha=0.05,random_state=0).fit(X_train,y_train) #tuned model
```


```python
predicted_train=tuned_lasso.predict(X_train)
```


```python
predicted_y=tuned_lasso.predict(X_test)
```


```python
print(np.sqrt(mean_squared_error(y_train,predicted_train))) #training error in terms of RMSE
```

    1.0228144000054245
    


```python
print(np.sqrt(mean_squared_error(y_test,predicted_y)))  #test error in terms of RMSE
```

    1.0158246258745005
    


```python
print(r2_score(y_train, predicted_train))   #r2 score of training
```

    0.510426395853588
    


```python
print(r2_score(y_test,predicted_y))        #r2 score of test
```

    0.4866481636057216
    


```python
#CART
```


```python
cart_model=DecisionTreeClassifier(random_state=42).fit(X_train,y_train)
```


```python
cart_train=cart_model.predict(X_train)
```


```python
cart_test=cart_model.predict(X_test)
```


```python
print(np.sqrt(mean_squared_error(y_train,cart_train)))
```

    1.068474000922569
    


```python
print(np.sqrt(mean_squared_error(y_test,cart_test))) 
```

    1.3318363828969004
    


```python
accuracy_score(cart_test,y_test)
```




    0.3132854578096948




```python
cart_params= {"max_depth":[2,3,4,5,8,10],
              "min_samples_leaf":[1,2,5,30,50,100]}     
```


```python
cart_cv_model=GridSearchCV(cart_model,cart_params,cv=10).fit(X_train,y_train)  
```


```python
cart_cv_model.best_params_
```




    {'max_depth': 5, 'min_samples_leaf': 50}




```python
tuned_cart=DecisionTreeClassifier(random_state=42,max_depth=5,min_samples_leaf=50).fit(X_train,y_train)  #tuned model
```


```python
prediction_train_cart=tuned_cart.predict(X_train)
```


```python
prediction_cart=tuned_cart.predict(X_test)
```


```python
print(np.sqrt(mean_squared_error(y_train,prediction_train_cart)))
```

    1.068474000922569
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_cart))) 
```

    1.1841260062019305
    


```python
accuracy_score(prediction_train_cart,y_train)   #f1 score of training
```




    0.4545863309352518




```python
accuracy_score(prediction_cart,y_test)  #f1 score of test
```




    0.3608617594254937




```python
#Random Forest
```


```python
rf_model=RandomForestClassifier(random_state=0).fit(X_train,y_train)
```


```python
train_rf=rf_model.predict(X_train)
```


```python
test_rf=rf_model.predict(X_test)
```


```python
print(np.sqrt(mean_squared_error(y_train,train_rf)))
```

    0.0
    


```python
print(np.sqrt(mean_squared_error(y_test,test_rf))) 
```

    1.0550383521920188
    


```python
accuracy_score(y_test,test_rf)  #f1 score
```




    0.39497307001795334




```python
rf_params={"min_samples_split":[2,5,10,80,100]}
```


```python
rf_cv_model=GridSearchCV(rf_model,rf_params,cv=10).fit(X_train,y_train)
```


```python
rf_cv_model.best_params_
```




    {'min_samples_split': 2}




```python
tuned_rf=RandomForestClassifier(n_estimators=500,min_samples_leaf=5,min_samples_split=2,random_state=0).fit(X_train,y_train)  #tuned model
```


```python
prediction_train_rf=tuned_rf.predict(X_train)
```


```python
prediction_rf=tuned_rf.predict(X_test)
```


```python
print(np.sqrt(mean_squared_error(y_train,prediction_train_rf)))
```

    0.602378379617177
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_rf))) 
```

    1.0370168512279345
    


```python
accuracy_score(prediction_train_rf,y_train)   #f1 score of training
```




    0.8943345323741008




```python
accuracy_score(y_test,prediction_rf)    #f1 score of test
```




    0.40933572710951527




```python
#Stochastic Gradient Boosting
```


```python
sgb_model=GradientBoostingClassifier(random_state=41).fit(X_train,y_train)
```


```python
train_sgb=sgb_model.predict(X_train)
```


```python
test_sgb=sgb_model.predict(X_test)
```


```python
print(np.sqrt(mean_squared_error(y_train,train_sgb)))
```

    0.5959998551494199
    


```python
print(np.sqrt(mean_squared_error(y_test,test_sgb))) 
```

    1.0942972790832375
    


```python
accuracy_score(y_test,test_sgb)  #f1_score test
```




    0.3752244165170557




```python
sgb_params={"learning_rate":[0.001,0.1,0.01],
           "max_depth":[3,5,8],
           "n_estimators":[100,200,500],}
```


```python
sgb_cv_model=GridSearchCV(sgb_model,sgb_params,cv=5).fit(X_train,y_train)
```


```python
sgb_cv_model.best_params_
```




    {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}




```python
tuned_sgb=GradientBoostingClassifier(min_samples_leaf=10,learning_rate=0.01,max_depth=3,n_estimators=500,random_state=41).fit(X_train,y_train)
```


```python
prediction_train_sgb=tuned_sgb.predict(X_train)
```


```python
prediction_sgb=tuned_sgb.predict(X_test)
```


```python
print(np.sqrt(mean_squared_error(y_train,prediction_train_sgb)))
```

    0.7931246828777478
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_sgb))) 
```

    1.0839948723602943
    


```python
accuracy_score(prediction_train_sgb,y_train)   #f1_score training
```




    0.7252697841726619




```python
accuracy_score(y_test,prediction_sgb)  #f1_score test
```




    0.37971274685816875




```python
#    test_error(RMSE)      cross_validation_error(RMSE)       Performance              Performance(CV)
#PNL   1.0360006013689902     1.0158246258745005        0.4660536019307584           0.4866481636057216   r2_score
#CART   1.3318363828969004     1.1841260062019305        0.3132854578096948           0.3608617594254937   f1_score
#RF    1.0550383521920188     1.0370168512279345        0.39497307001795334          0.40933572710951527  f1_score
#SGB    1.0942972790832375     1.1684819595383606        0.3752244165170557           0.37971274685816875  f1_score

#According to the results, suprisingly the best performance measure is penalized regression although it is a multiclass classification problem.
#For all methods, cross validation error is less than test error but there are slight differences.In terms of error best option is SGB.
#However, model performances are poor and it might be because of training set has large numbers of features.
#It seems overfitting is the case due to the fact that large f1 scores (training performances) for models except penalized regression 
#but poor test performances.
#In penalized regression both training and test performance metrics (r2 score) are poor. Therefore underfitting might be the case.

```


```python
#Credit Card Client Data Set
```


```python
#Demographic data of credit cart clients, their marital status, education are collected. Also payment history, bill statement and credit amount
#involve in data set.
```


```python
#Data set is mixed with numeric, ordinal and categorical features
```


```python
#This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
#X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
#X2: Gender (1 = male; 2 = female).
#X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
#X4: Marital status (1 = married; 2 = single; 3 = others).
#X5: Age (year).
#X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
#X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
#X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.


```


```python
#Target variable is default of payment (Yes=1, No=0)
```


```python
#The problem is binary classification problem (class imbalance  0    23364
#                                                               1     6636)
```


```python
df2=pd.read_excel("Credit_Card_Clients.xls")
```


```python
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>...</th>
      <th>X15</th>
      <th>X16</th>
      <th>X17</th>
      <th>X18</th>
      <th>X19</th>
      <th>X20</th>
      <th>X21</th>
      <th>X22</th>
      <th>X23</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID</td>
      <td>LIMIT_BAL</td>
      <td>SEX</td>
      <td>EDUCATION</td>
      <td>MARRIAGE</td>
      <td>AGE</td>
      <td>PAY_0</td>
      <td>PAY_2</td>
      <td>PAY_3</td>
      <td>PAY_4</td>
      <td>...</td>
      <td>BILL_AMT4</td>
      <td>BILL_AMT5</td>
      <td>BILL_AMT6</td>
      <td>PAY_AMT1</td>
      <td>PAY_AMT2</td>
      <td>PAY_AMT3</td>
      <td>PAY_AMT4</td>
      <td>PAY_AMT5</td>
      <td>PAY_AMT6</td>
      <td>default payment next month</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
df2.isnull().sum().sum()  #Checking if any NA value exists
```




    0




```python
df2.shape
```




    (30001, 25)




```python
df2=df2.loc[1:,:]    #Removed unnamed row
```


```python
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>...</th>
      <th>X15</th>
      <th>X16</th>
      <th>X17</th>
      <th>X18</th>
      <th>X19</th>
      <th>X20</th>
      <th>X21</th>
      <th>X22</th>
      <th>X23</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>50000</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
df2["Y"].value_counts()  #class imbalance
```




    0    23364
    1     6636
    Name: Y, dtype: int64




```python
df2.shape
```




    (30000, 25)




```python
df2 = df2.apply(pd.to_numeric)  #All data converted to numeric form
```


```python
df2_train=df2.loc[:20000,:]
```


```python
df2_test=df2.loc[20000:,:]
```


```python
X_train=df2_train.drop("Y",axis=1)
```


```python
X_test=df2_test.drop("Y",axis=1)
```


```python
y_train=df2.loc[:20000,"Y"]
```


```python
y_test=df2.loc[20000:,"Y"]
```


```python
#Penalized Regression
```


```python
lasso= linear_model.Lasso(random_state=0).fit(X_train,y_train)
```


```python
y=lasso.predict(X_test)
```


```python
train=lasso.predict(X_train)
```


```python
lasso.score(X_train,y_train)  #training score 
```




    0.030594157804576527




```python
lasso.score(X_test,y_test)    #test score 
```




    0.025297721408646945




```python
print(np.sqrt(mean_squared_error(y_train,train))) #RMSE of train
```

    0.41301112981128707
    


```python
print(np.sqrt(mean_squared_error(y_test,y)))   #RMSE of test
```

    0.400553160633912
    


```python
grid ={"alpha":[0.05,0.1,0.5,1,2]}
```


```python
model=Lasso(random_state=0)
search = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=10)
results=search.fit(X_train,y_train)
print("neg_RMSE",results.best_score_)
print("lambda",results.best_params_)   #found best lambda
```

    neg_RMSE -0.3980460558533604
    lambda {'alpha': 0.05}
    


```python
tuned_lasso = linear_model.Lasso(alpha=0.05,random_state=0).fit(X_train,y_train)
```


```python
predicted_y=tuned_lasso.predict(X_test)
```


```python
predicted_train=tuned_lasso.predict(X_train)
```


```python
tuned_lasso.score(X_train,y_train)  #training score 
```




    0.1020831204759104




```python
tuned_lasso.score(X_test,y_test)    #test score 
```




    0.10383968487050232




```python
print(np.sqrt(mean_squared_error(y_train,predicted_train)))
```

    0.39749073172703947
    


```python
print(np.sqrt(mean_squared_error(y_test,predicted_y))) 
```

    0.38407587295080414
    


```python
# CART
```


```python
cart_model=DecisionTreeClassifier(random_state=42).fit(X_train,y_train)
```


```python
y_cart=cart_model.predict(X_test)
```


```python
train_cart=cart_model.predict(X_train)
```


```python
accuracy_score(y_train,train_cart)
```




    1.0




```python
accuracy_score(y_test,y_cart)
```




    0.6095390460953904




```python
print(np.sqrt(mean_squared_error(y_train,train_cart)))
```

    0.0
    


```python
print(np.sqrt(mean_squared_error(y_test,y_cart))) 
```

    0.6248687493422995
    


```python
cart_params= {"max_depth":[2,3,4,5],
              "min_samples_leaf":[1,2,5,50,100]}
```


```python
cart_model=DecisionTreeClassifier(random_state=42).fit
```


```python
cart_cv_model=GridSearchCV(cart_model,cart_params,cv=10).fit(X_train,y_train)
```


```python
cart_cv_model.best_params_
```




    {'max_depth': 2, 'min_samples_leaf': 100}




```python
tuned_cart=DecisionTreeClassifier(random_state=42,max_depth=2,min_samples_leaf=100).fit(X_train,y_train)
```


```python
prediction_cart=tuned_cart.predict(X_test)
```


```python
prediction_train_cart=tuned_cart.predict(X_train)
```


```python
accuracy_score(y_train,prediction_train_cart)
```




    0.8142




```python
accuracy_score(y_test,prediction_cart)
```




    0.832016798320168




```python
print(np.sqrt(mean_squared_error(y_train,prediction_train_cart)))
```

    0.4310452412450461
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_cart))) 
```

    0.4098575382737666
    


```python
#Random Forest
```


```python
rf_model=RandomForestClassifier(random_state=0).fit(X_train,y_train)
```


```python
y_rf=rf_model.predict(X_test)
```


```python
rf_train=rf_model.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,rf_train)))
```

    0.007071067811865475
    


```python
print(np.sqrt(mean_squared_error(y_test,y_rf))) 
```

    0.41506959864922377
    


```python
accuracy_score(y_train,rf_train)
```




    0.99995




```python
accuracy_score(y_test,y_rf)
```




    0.8277172282771723




```python
rf_params={"min_samples_split":[2,5,10,80,100]}
```


```python
rf_cv_model=GridSearchCV(rf_model,rf_params,cv=10).fit(X_train,y_train)
```


```python
rf_cv_model.best_params_
```




    {'min_samples_split': 100}




```python
tuned_rf=RandomForestClassifier(n_estimators=500,min_samples_leaf=5,min_samples_split=100,random_state=0).fit(X_train,y_train)
```


```python
prediction_rf=tuned_rf.predict(X_test)
```


```python
prediction_rf_train=tuned_rf.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,prediction_rf_train)))
```

    0.415571895103603
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_rf))) 
```

    0.4058121876519178
    


```python
accuracy_score(y_train,prediction_rf_train)
```




    0.8273




```python
accuracy_score(y_test,prediction_rf)
```




    0.8353164683531646




```python
#Sthocastic Gradient Boosting
```


```python
sgb_model=GradientBoostingClassifier(random_state=30).fit(X_train,y_train)
```


```python
y_sgb=sgb_model.predict(X_test)
```


```python
sgb_train=sgb_model.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,sgb_train)))
```

    0.4235563716909474
    


```python
print(np.sqrt(mean_squared_error(y_test,y_sgb))) 
```

    0.4083911380868018
    


```python
accuracy_score(y_train,sgb_train)
```




    0.8206




```python
accuracy_score(y_test,y_sgb)
```




    0.8332166783321668




```python
sgb_params={"learning_rate":[0.001,0.1,0.01],
           "max_depth":[3,5,8],
           "n_estimators":[100,200,500],}
```


```python
sgb_cv_model=GridSearchCV(sgb_model,sgb_params,cv=10,random_state=30).fit(X_train,y_train)
```


```python
sgb_cv_model.best_params_
```




    {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500}




```python
tuned_sgb=GradientBoostingClassifier(min_samples_leaf=10,learning_rate=0.01,max_depth=5,n_estimators=500).fit(X_train,y_train)
```


```python
prediction_sgb=tuned_sgb.predict(X_test)
```


```python
prediction_sgb_train=tuned_sgb.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,prediction_sgb_train)))
```

    0.41309805131469696
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_sgb))) 
```

    0.40912499517119777
    


```python
accuracy_score(y_train,prediction_sgb_train)
```




    0.82935




```python
accuracy_score(y_test,prediction_sgb)
```




    0.8326167383261673




```python
#     test_error(RMSE)      cross_validation_error(RMSE)      Performance              Performance(CV)
#PNL   0.400553160633912     0.38407587295080414        0.025297721408646945          0.10383968487050232   r2_score
#CART  0.6248687493422995     0.4098575382737666        0.6095390460953904           0.832016798320168   accuracy
#RF    0.41506959864922377    0.4058121876519178        0.8277172282771723          0.8353164683531646  accuracy
#SGB    0.4083911380868018     0.4083911380868018        0.8332166783321668         0.8326167383261673  accuracy

#Penalized regression has very poor performance on binary classification problem which is quite expected.Training performance is %100 in CART
# before cross validation that indicates overfitting problem with 60% test performance. But afrer cross validation in training and 
#test measure in terms of accuracy is close which indicates that overfitting problem is handled after cross validation.Also in RF, training performance is 
#99.9% which indicates overfitting is the case with 82% test performance. However, after cross validation, it is handled and difference between
#training and test error is slight.
#Except penalized regression,
# all algorithms works well on that problem with close performance measures in terms of accuracy.In terms of error rates, best algorithm is SGB
#but all algorithms have close error rates except CART. In SGB in terms of training and test error difference, it seems no overfitting/underfitting case
# observed. Also, it should be added that with cross validation test errors of algorithm decrease in terms of RMSE.
```


```python
#Bias Correction Data Set
```


```python
#Meterologial weather data consists of air temperature, cloud cover, solar radiation,humidity, latitude, longtitude etc. are collected.
```


```python
# Data set includes categorical and continous variables.
```


```python
#Next day maximum temperature is target variable
```


```python
For more information, read [Cho et al, 2020].
#1. station - used weather station number: 1 to 25
#2. Date - Present day: yyyy-mm-dd ('2013-06-30' to '2017-08-30')
#3. Present_Tmax - Maximum air temperature between 0 and 21 h on the present day (Â°C): 20 to 37.6
#4. Present_Tmin - Minimum air temperature between 0 and 21 h on the present day (Â°C): 11.3 to 29.9
#5. LDAPS_RHmin - LDAPS model forecast of next-day minimum relative humidity (%): 19.8 to 98.5
#6. LDAPS_RHmax - LDAPS model forecast of next-day maximum relative humidity (%): 58.9 to 100
#7. LDAPS_Tmax_lapse - LDAPS model forecast of next-day maximum air temperature applied lapse rate (Â°C): 17.6 to 38.5
#8. LDAPS_Tmin_lapse - LDAPS model forecast of next-day minimum air temperature applied lapse rate (Â°C): 14.3 to 29.6
#9. LDAPS_WS - LDAPS model forecast of next-day average wind speed (m/s): 2.9 to 21.9
#10. LDAPS_LH - LDAPS model forecast of next-day average latent heat flux (W/m2): -13.6 to 213.4
#11. LDAPS_CC1 - LDAPS model forecast of next-day 1st 6-hour split average cloud cover (0-5 h) (%): 0 to 0.97
#12. LDAPS_CC2 - LDAPS model forecast of next-day 2nd 6-hour split average cloud cover (6-11 h) (%): 0 to 0.97
#13. LDAPS_CC3 - LDAPS model forecast of next-day 3rd 6-hour split average cloud cover (12-17 h) (%): 0 to 0.98
#14. LDAPS_CC4 - LDAPS model forecast of next-day 4th 6-hour split average cloud cover (18-23 h) (%): 0 to 0.97
#15. LDAPS_PPT1 - LDAPS model forecast of next-day 1st 6-hour split average precipitation (0-5 h) (%): 0 to 23.7
#16. LDAPS_PPT2 - LDAPS model forecast of next-day 2nd 6-hour split average precipitation (6-11 h) (%): 0 to 21.6
#17. LDAPS_PPT3 - LDAPS model forecast of next-day 3rd 6-hour split average precipitation (12-17 h) (%): 0 to 15.8
#18. LDAPS_PPT4 - LDAPS model forecast of next-day 4th 6-hour split average precipitation (18-23 h) (%): 0 to 16.7
#19. lat - Latitude (Â°): 37.456 to 37.645
#20. lon - Longitude (Â°): 126.826 to 127.135
#21. DEM - Elevation (m): 12.4 to 212.3
#22. Slope - Slope (Â°): 0.1 to 5.2
#23. Solar radiation - Daily incoming solar radiation (wh/m2): 4329.5 to 5992.9
#24. Next_Tmax - The next-day maximum air temperature (Â°C): 17.4 to 38.9
#25. Next_Tmin - The next-day minimum air temperature (Â°C): 11.3 to 29.8


```


```python
#This problem is a regression problem
```


```python
df3=pd.read_csv("Bias_correction.csv")
```


```python
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>station</th>
      <th>Date</th>
      <th>Present_Tmax</th>
      <th>Present_Tmin</th>
      <th>LDAPS_RHmin</th>
      <th>LDAPS_RHmax</th>
      <th>LDAPS_Tmax_lapse</th>
      <th>LDAPS_Tmin_lapse</th>
      <th>LDAPS_WS</th>
      <th>LDAPS_LH</th>
      <th>...</th>
      <th>LDAPS_PPT2</th>
      <th>LDAPS_PPT3</th>
      <th>LDAPS_PPT4</th>
      <th>lat</th>
      <th>lon</th>
      <th>DEM</th>
      <th>Slope</th>
      <th>Solar radiation</th>
      <th>Next_Tmax</th>
      <th>Next_Tmin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2013-06-30</td>
      <td>28.7</td>
      <td>21.4</td>
      <td>58.255688</td>
      <td>91.116364</td>
      <td>28.074101</td>
      <td>23.006936</td>
      <td>6.818887</td>
      <td>69.451805</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.6046</td>
      <td>126.991</td>
      <td>212.3350</td>
      <td>2.7850</td>
      <td>5992.895996</td>
      <td>29.1</td>
      <td>21.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2013-06-30</td>
      <td>31.9</td>
      <td>21.6</td>
      <td>52.263397</td>
      <td>90.604721</td>
      <td>29.850689</td>
      <td>24.035009</td>
      <td>5.691890</td>
      <td>51.937448</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.6046</td>
      <td>127.032</td>
      <td>44.7624</td>
      <td>0.5141</td>
      <td>5869.312500</td>
      <td>30.5</td>
      <td>22.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>2013-06-30</td>
      <td>31.6</td>
      <td>23.3</td>
      <td>48.690479</td>
      <td>83.973587</td>
      <td>30.091292</td>
      <td>24.565633</td>
      <td>6.138224</td>
      <td>20.573050</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.5776</td>
      <td>127.058</td>
      <td>33.3068</td>
      <td>0.2661</td>
      <td>5863.555664</td>
      <td>31.1</td>
      <td>23.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>2013-06-30</td>
      <td>32.0</td>
      <td>23.4</td>
      <td>58.239788</td>
      <td>96.483688</td>
      <td>29.704629</td>
      <td>23.326177</td>
      <td>5.650050</td>
      <td>65.727144</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.6450</td>
      <td>127.022</td>
      <td>45.7160</td>
      <td>2.5348</td>
      <td>5856.964844</td>
      <td>31.7</td>
      <td>24.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>2013-06-30</td>
      <td>31.4</td>
      <td>21.9</td>
      <td>56.174095</td>
      <td>90.155128</td>
      <td>29.113934</td>
      <td>23.486480</td>
      <td>5.735004</td>
      <td>107.965535</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.5507</td>
      <td>127.135</td>
      <td>35.0380</td>
      <td>0.5055</td>
      <td>5859.552246</td>
      <td>31.2</td>
      <td>22.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
df3=df3.dropna()
```


```python
df3.shape
```




    (7588, 25)




```python
df3_train=df3.loc[:5059,:]
```


```python
df3_test=df3.loc[5060:,:]
```


```python
k=["Date","Next_Tmax"]
```


```python
X_train=df3_train.drop(k,axis=1)
```


```python
X_test=df3_test.drop(k,axis=1)
```


```python
y_train=df3.loc[:5059,"Next_Tmax"]
```


```python
y_test=df3.loc[5060:,"Next_Tmax"]
```


```python
#Penalized Regression
```


```python
lasso=linear_model.Lasso(random_state=0).fit(X_train,y_train)
```


```python
lasso_y=lasso.predict(X_test)
```


```python
lasso_train=lasso.predict(X_train)
```


```python
lasso.score(X_train,y_train)
```




    0.6699317099876898




```python
lasso.score(X_test,y_test)
```




    0.7374328801128973




```python
print(np.sqrt(mean_squared_error(y_train,lasso_train)))
```

    1.6258429015218507
    


```python
print(np.sqrt(mean_squared_error(y_test,lasso_y))) 
```

    1.805276724985845
    


```python
grid ={"alpha":[0.05,0.1,0.5,1,2]}
```


```python
model=Lasso(random_state=0)
search = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=10)
results=search.fit(X_train,y_train)
print("neg_RMSE",results.best_score_)
print("lambda",results.best_params_)
```

    neg_RMSE -1.5015184906852317
    lambda {'alpha': 0.05}
    


```python
tuned_lasso = linear_model.Lasso(alpha=0.05,random_state=0).fit(X_train,y_train)
```


```python
predicted_y=tuned_lasso.predict(X_test)
```


```python
predicted_train=tuned_lasso.predict(X_train)
```


```python
tuned_lasso.score(X_train,y_train)
```




    0.7531220453884154




```python
tuned_lasso.score(X_test,y_test)
```




    0.8122101221471366




```python
print(np.sqrt(mean_squared_error(y_train,predicted_train)))
```

    1.4061052600711186
    


```python
print(np.sqrt(mean_squared_error(y_test,predicted_y))) 
```

    1.5267210812837684
    


```python
#CART
```


```python
cart_model=DecisionTreeRegressor(random_state=42).fit(X_train,y_train)
```


```python
y_cart=cart_model.predict(X_test)
```


```python
cart_training=cart_model.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,cart_training)))
```

    5.707073882996834e-16
    


```python
print(np.sqrt(mean_squared_error(y_test,y_cart))) 
```

    2.1111962198154752
    


```python
cart_model.score(X_train,y_train)
```




    1.0




```python
cart_model.score(X_test,y_test)
```




    0.6409044909819488




```python
cart_params= {"max_depth":[2,3,4,5],
              "min_samples_leaf":[1,2,5,50,100]}
```


```python
cart_cv_model=GridSearchCV(cart_model,cart_params,cv=10).fit(X_train,y_train)
```


```python
cart_cv_model.best_params_
```




    {'max_depth': 5, 'min_samples_leaf': 100}




```python
tuned_cart=DecisionTreeRegressor(random_state=42,max_depth=5,min_samples_leaf=100).fit(X_train,y_train)
```


```python
prediction_cart=tuned_cart.predict(X_test)
```


```python
prediction_cart_training=tuned_cart.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,prediction_cart_training)))
```

    1.4386111636474053
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_cart))) 
```

    1.6922219885566547
    


```python
tuned_cart.score(X_train,y_train)
```




    0.7415756116474015




```python
tuned_cart.score(X_test,y_test)
```




    0.7279155548449676




```python
#Random Forest
```


```python
rf_model=RandomForestRegressor(random_state=0).fit(X_train,y_train)
```


```python
y_rf=rf_model.predict(X_test)
```


```python
rf_training=rf_model.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,rf_training)))
```

    0.3180828387990583
    


```python
print(np.sqrt(mean_squared_error(y_test,y_rf))) 
```

    1.6637075297927426
    


```python
rf_model.score(X_train,y_train)
```




    0.9873664000033533




```python
rf_model.score(X_test,y_test)
```




    0.7769990367818713




```python
rf_params={"min_samples_split":[2,5,10,80,100]}
```


```python
rf_cv_model=GridSearchCV(rf_model,rf_params,cv=10).fit(X_train,y_train)
```


```python
rf_cv_model.best_params_
```




    {'min_samples_split': 2}




```python
tuned_rf=RandomForestRegressor(n_estimators=500,min_samples_leaf=5,min_samples_split=2,random_state=0).fit(X_train,y_train)
```


```python
prediction_rf=tuned_rf.predict(X_test)
```


```python
predicion_rf_training=tuned_rf.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,predicion_rf_training)))
```

    0.5406330653181552
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_rf))) 
```

    1.6574234832679255
    


```python
tuned_rf.score(X_train,y_train)
```




    0.9635034466695156




```python
tuned_rf.score(X_test,y_test)
```




    0.7786804643914644




```python
#Sthocastic Gradient Boosting
```


```python
sgb_model=GradientBoostingRegressor(random_state=42).fit(X_train,y_train)
```


```python
y_sgb=sgb_model.predict(X_test)
```


```python
sgb_training=sgb_model.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,sgb_training)))
```

    0.9304871913348146
    


```python
print(np.sqrt(mean_squared_error(y_test,y_sgb))) 
```

    1.568927507873894
    


```python
sgb_model.score(X_train,y_train)
```




    0.8918896077093496




```python
sgb_model.score(X_test,y_test)
```




    0.801683645934918




```python
sgb_params={"learning_rate":[0.001,0.1,0.01],
           "max_depth":[3,5,8],
           "n_estimators":[100,200,500],}
```


```python
sgb_cv_model=GridSearchCV(sgb_model,sgb_params,cv=5).fit(X_train,y_train)
```


```python
sgb_cv_model.best_params_
```


```python
tuned_sgb=GradientBoostingRegressor(min_samples_leaf=10,learning_rate=0.01,max_depth=5,n_estimators=500,random_state=42).fit(X_train,y_train)
```


```python
prediction_sgb=tuned_sgb.predict(X_test)
```


```python
prediction_sgb_training=tuned_sgb.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,prediction_sgb_training)))
```

    0.7743819387422018
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_sgb))) 
```

    1.5888836414701704
    


```python
tuned_sgb.score(X_train,y_train)
```




    0.9251215104593288




```python
tuned_sgb.score(X_test,y_test)
```




    0.7966065504533887




```python
#     test_error(RMSE)      cross_validation_error(RMSE)      Performance              Performance(CV)
#PNL   1.805276724985845     1.5267210812837684        0.7374328801128973          0.8122101221471366  r2_score
#CART  2.1111962198154752     1.6922219885566547        0.640904490981948           0.7279155548449676  r2_score
#RF    1.6637075297927426    1.6574234832679255        0.7769990367818713          0.7786804643914644  r2_score
#SGB    1.568927507873894     1.5888836414701704       0.801683645934918          0.7966065504533887  r2_score

```


```python
#In terms of perfoormance measure best algorithms is SGB before cross validation. After cross validation, best performance measure is
# penalized regresion which is meaningful due to the fact that the problem is regression problem. In terms of error best method is SB before cross validation.
#However, after CV penalized regression is the best in terms of RMSE value. In CART, before CV training performance is 100% and which and test performance is 64%
#which indicates overfitting clearly. After CV, training performance and test performance are 74% and 72%, respectively. In RF, before and after CV
# very high training performance are observed. It seems that it couldn't be handled with CV and this situation may be derived from choice of parameter set.
# Providing better parameters, tuning could achieved with close performance measures in terms of training and test. In SGB, cross validation error is higher
# than test error which is unusual. But it may be derived from choice of parameter set as well. Except CART, test error and cross validation error are consistent.

```


```python
#Spambase Data Set
```


```python
# Collection of spam and non-spam emails. The last column is target variable whether the e-mail was considered spam (1) or not (0).
```


```python
#48 continuous real [0,100] attributes of type word_freq_WORD
#percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. 
#A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.

#6 continuous real [0,100] attributes of type char_freq_CHAR]
#=percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail

#1 continuous real [1,...] attribute of type capital_run_length_average
#= average length of uninterrupted sequences of capital letters

#1 continuous integer [1,...] attribute of type capital_run_length_longest
#= length of longest uninterrupted sequence of capital letters

#1 continuous integer [1,...] attribute of type capital_run_length_total
#= sum of length of uninterrupted sequences of capital letters
#= total number of capital letters in the e-mail

#1 nominal {0,1} class attribute of type spam
#= denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail.

```


```python
# Problem is a binary classification problem with 4601 instances.Class imbalance   0    2788
#                                                                                  1    1813 
```


```python
df4=pd.read_csv("spambase.csv",header=None)
```


```python
df4 = shuffle(df4)  #shuffled the data
```


```python
df4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.96</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.92</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.462</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.312</td>
      <td>6</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1400</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.695</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.315</td>
      <td>12</td>
      <td>44</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2354</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.94</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.766</td>
      <td>4</td>
      <td>53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3415</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.166</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2647</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.333</td>
      <td>3</td>
      <td>12</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
df4=df4.reset_index()
```


```python
df4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.96</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.462</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.312</td>
      <td>6</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1400</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.695</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.315</td>
      <td>12</td>
      <td>44</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2354</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.94</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.766</td>
      <td>4</td>
      <td>53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3415</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.166</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2647</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.333</td>
      <td>3</td>
      <td>12</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>




```python
df4.drop("index",axis=1,inplace=True)
```


```python
df4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.96</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.92</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.462</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.312</td>
      <td>6</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.695</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.315</td>
      <td>12</td>
      <td>44</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.94</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.766</td>
      <td>4</td>
      <td>53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.166</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.333</td>
      <td>3</td>
      <td>12</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
df4.isnull().sum().sum()  #Checking if any NA value exists
```




    0




```python
df4[57].value_counts()   #class imbalance can be shown
```




    0    2788
    1    1813
    Name: 57, dtype: int64




```python
df4_train=df4.loc[:3068,:]       
```


```python
df4_test=df4.loc[3069:,:]
```


```python
X_train=df4_train.drop(57,axis=1)     #Prepared train and test splits based on 2/3 train and 1/3 test rule
```


```python
X_test=df4_test.drop(57,axis=1)
```


```python
y_train=df4.loc[:3068,57]
```


```python
y_test=df4.loc[3069:,57]
```


```python
#Penalized Regression
```


```python
lasso= linear_model.Lasso(random_state=0).fit(X_train,y_train)
```


```python
predicted_y=lasso.predict(X_test)
```


```python
predicted_train=lasso.predict(X_train)
```


```python
lasso.score(X_train,y_train)   #tuned training score in terms of r2
```




    0.06490393427499241




```python
lasso.score(X_test,y_test)     #tuned test score in terms of r2
```




    0.08471411685043628




```python
print(np.sqrt(mean_squared_error(y_train,predicted_train)))    # RMSE value of training
```

    0.4712321374761744
    


```python
print(np.sqrt(mean_squared_error(y_test,predicted_y)))      # RMSE value of test
```

    0.4698243323374636
    


```python
grid ={"alpha":[0.05,0.1,0.5,1,2]}    #lambda set for trial
```


```python
model=Lasso(random_state=0)
search = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=10)
results=search.fit(X_train,y_train)
print("neg_RMSE",results.best_score_)   #best error score in terms of negative RMSE
print("lambda",results.best_params_)   #best lambda after cross valiadation
```

    neg_RMSE -0.4269502665559724
    lambda {'alpha': 0.05}
    


```python
tuned_lasso = linear_model.Lasso(alpha=0.05,random_state=0).fit(X_train,y_train)
```


```python
predicted_y_tuned=tuned_lasso.predict(X_test)
```


```python
predicted_train_tuned=tuned_lasso.predict(X_train)
```


```python
tuned_lasso.score(X_train,y_train)   #tuned training score in terms of r2
```




    0.31155576819539643




```python
tuned_lasso.score(X_test,y_test)     #tuned test score in terms of r2
```




    0.3138104697220183




```python
print(np.sqrt(mean_squared_error(y_train,predicted_train_tuned)))    # RMSE value of training
```

    0.4043348442580087
    


```python
print(np.sqrt(mean_squared_error(y_test,predicted_y_tuned)))      # RMSE value of test
```

    0.40679833518520586
    


```python
#CART
```


```python
cart_model=DecisionTreeClassifier(random_state=42).fit(X_train,y_train)
```


```python
prediction_cart=cart_model.predict(X_test)
```


```python
prediction_cart_training=cart_model.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,prediction_cart_training)))
```

    0.01805101203579608
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_cart))) 
```

    0.2935334391437564
    


```python
cart_model.score(X_train,y_train)
```




    0.9996741609644836




```python
cart_model.score(X_test,y_test)
```




    0.9138381201044387




```python
cart_params= {"max_depth":[2,3,4,5],
              "min_samples_leaf":[1,2,5,50,100]}
```


```python
cart_cv_model=GridSearchCV(cart_model,cart_params,cv=10).fit(X_train,y_train)
```


```python
cart_cv_model.best_params_
```




    {'max_depth': 5, 'min_samples_leaf': 2}




```python
tuned_cart=DecisionTreeClassifier(random_state=42,max_depth=5,min_samples_leaf=2).fit(X_train,y_train)
```


```python
prediction_cart_tuned=tuned_cart.predict(X_test)
```


```python
cart_training_tuned=tuned_cart.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,cart_training_tuned)))
```

    0.2634458475000221
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_cart_tuned))) 
```

    0.30444916659298726
    


```python
tuned_cart.score(X_train,y_train)
```




    0.9305962854349951




```python
tuned_cart.score(X_test,y_test)
```




    0.9073107049608355




```python
#Random Forest
```


```python
rf_model=RandomForestClassifier(random_state=0).fit(X_train,y_train)
```


```python
prediction_rf=rf_model.predict(X_test)
```


```python
predicion_rf_training=rf_model.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,predicion_rf_training)))
```

    0.01805101203579608
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_rf))) 
```

    0.2299392586384182
    


```python
rf_model.score(X_train,y_train)
```




    0.9996741609644836




```python
rf_model.score(X_test,y_test)
```




    0.9471279373368147




```python
rf_params={"min_samples_split":[2,5,10,80,100]}
```


```python
rf_cv_model=GridSearchCV(rf_model,rf_params,cv=10).fit(X_train,y_train)
```


```python
rf_cv_model.best_params_
```




    {'min_samples_split': 2}




```python
tuned_rf=RandomForestClassifier(n_estimators=500,min_samples_leaf=5,min_samples_split=2,random_state=0).fit(X_train,y_train)
```


```python
prediction_rf_tuned=tuned_rf.predict(X_test)
```


```python
tuned_rf_training=tuned_rf.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,tuned_rf_training)))
```

    0.1823062851979558
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_rf_tuned))) 
```

    0.2542074151587066
    


```python
tuned_rf.score(X_train,y_train)
```




    0.9667644183773216




```python
tuned_rf.score(X_test,y_test)
```




    0.935378590078329




```python
#Sthocastic Gradient Boosting
```


```python
sgb_model=GradientBoostingClassifier(random_state=0).fit(X_train,y_train)
```


```python
prediction_sgb=sgb_model.predict(X_test)
```


```python
prediction_sgb_training=sgb_model.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,prediction_sgb_training)))
```

    0.17501105490381674
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_sgb))) 
```

    0.2383034027229726
    


```python
sgb_model.score(X_train,y_train)
```




    0.9693711306614532




```python
sgb_model.score(X_test,y_test)
```




    0.9432114882506527




```python
sgb_params={"learning_rate":[0.001,0.1,0.01],
           "max_depth":[3,5,8],
           "n_estimators":[100,200,500]}
```


```python
sgb_cv_model=GridSearchCV(sgb_model,sgb_params,cv=5).fit(X_train,y_train)
```


```python
sgb_cv_model.best_params_
```




    {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500}




```python
tuned_sgb=GradientBoostingClassifier(random_state=0,min_samples_leaf=10,learning_rate=0.1,max_depth=3,n_estimators=500).fit(X_train,y_train)
```


```python
prediction_sgb_tuned=tuned_sgb.predict(X_test)
```


```python
tuned_sgb_training=tuned_sgb.predict(X_train)
```


```python
print(np.sqrt(mean_squared_error(y_train,prediction_sgb_training)))
```

    0.17501105490381674
    


```python
print(np.sqrt(mean_squared_error(y_test,prediction_sgb))) 
```

    0.2383034027229726
    


```python
tuned_sgb.score(X_train,y_train)
```




    0.9977191267513849




```python
tuned_sgb.score(X_test,y_test)
```




    0.956266318537859




```python
#     test_error(RMSE)      cross_validation_error(RMSE)      Performance              Performance(CV)
#PNL   0.4698243323374636    0.40679833518520586      0.08471411685043628          0.3138104697220183  r2_score
#CART  0.2935334391437564     0.2634458475000221        0.9138381201044387           0.9471279373368147  accuracy
#RF    0.2299392586384182    0.2542074151587066       0.9471279373368147         0.935378590078329 accuracy
#SGB    0.2383034027229726     0.2383034027229726      0.9432114882506527          0.956266318537859  accuracy

```


```python
#In terms of performance, random forest is the best algorithm before CV. However, penalized regression is the worst which is quite expected
#due to the fact that the problem is binary classification problem.
#After cross validation SGB becomes the best method in terms of accuracy. Test and cross validation errors are consistent for all methods and test errors
#are slightly higher than cross validation errors. Training and test performance are close for all methods which indicates no over/underfitting.
#In RF, cross validation performance is slightly less than performance measure without CV.It may be derived from choice of parameters set.
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
