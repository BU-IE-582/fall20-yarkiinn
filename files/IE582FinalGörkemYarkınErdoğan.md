```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
```


```python
# Two approach for bag representation and metric value as AUS
from mil.bag_representation import MILESMapping, MeanMinMaxBagRepresentation
from mil.metrics import AUC
```


```python
# importing mil models
from mil.models import APR, AttentionDeepPoolingMil, MILES

# importing sklearn models
from mil.models import RandomForestClassifier, SVC
```


```python
# standarize bags of lists
from mil.preprocessing import StandarizerBagsList
```


```python
#mil validators for 10 CV and trainer
from mil.validators import KFold
from mil.trainer import Trainer
```


```python
df=pd.read_csv("Musk1.csv",header=None)
```


```python
df.shape
```




    (476, 168)




```python
df.rename(columns = {0:'Class'}, inplace = True)  #Renamed 0 column as "Class"
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
      <th>Class</th>
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
      <th>158</th>
      <th>159</th>
      <th>160</th>
      <th>161</th>
      <th>162</th>
      <th>163</th>
      <th>164</th>
      <th>165</th>
      <th>166</th>
      <th>167</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-198</td>
      <td>-109</td>
      <td>-75</td>
      <td>-117</td>
      <td>11</td>
      <td>23</td>
      <td>-88</td>
      <td>...</td>
      <td>-238</td>
      <td>-74</td>
      <td>-129</td>
      <td>-120</td>
      <td>-38</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>6</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-191</td>
      <td>-142</td>
      <td>-65</td>
      <td>-117</td>
      <td>55</td>
      <td>49</td>
      <td>-170</td>
      <td>...</td>
      <td>-238</td>
      <td>-302</td>
      <td>60</td>
      <td>-120</td>
      <td>-39</td>
      <td>31</td>
      <td>48</td>
      <td>-37</td>
      <td>5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-191</td>
      <td>-142</td>
      <td>-75</td>
      <td>-117</td>
      <td>11</td>
      <td>49</td>
      <td>-161</td>
      <td>...</td>
      <td>-238</td>
      <td>-73</td>
      <td>-127</td>
      <td>-120</td>
      <td>-38</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>5</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-198</td>
      <td>-110</td>
      <td>-65</td>
      <td>-117</td>
      <td>55</td>
      <td>23</td>
      <td>-95</td>
      <td>...</td>
      <td>-238</td>
      <td>-302</td>
      <td>60</td>
      <td>-120</td>
      <td>-39</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>6</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>42</td>
      <td>-198</td>
      <td>-102</td>
      <td>-75</td>
      <td>-117</td>
      <td>10</td>
      <td>24</td>
      <td>-87</td>
      <td>...</td>
      <td>-238</td>
      <td>-73</td>
      <td>-127</td>
      <td>51</td>
      <td>128</td>
      <td>144</td>
      <td>43</td>
      <td>-30</td>
      <td>14</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 168 columns</p>
</div>




```python
df["Class"].value_counts()  #269 0 and 207 1 class
```




    0    269
    1    207
    Name: Class, dtype: int64




```python
df[1].unique()  # Observed 92 unique bags
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
           52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
           69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
           86, 87, 88, 89, 90, 91, 92], dtype=int64)




```python
bag_id=df[1].unique()  
```


```python
nofins=df[1].value_counts(sort=False).values  #Observed how many instance each bag has
```


```python
nofins
```




    array([ 4,  4,  2,  3,  4,  2,  2,  2,  5,  6,  5,  8,  4,  2,  4,  3,  5,
            4,  8,  4,  4,  4,  2,  8,  4,  4,  2,  8,  4,  4,  4,  8,  2,  4,
            4,  2,  8,  8,  2,  4,  3,  5,  4,  5,  5,  8,  6,  4,  4,  2,  4,
            2,  9,  2,  2,  2,  4,  2,  9, 32,  4,  2,  2,  4,  4,  2,  2,  2,
            2,  4,  2,  4,  2,  2,  2,  2,  8,  4,  8,  2, 40, 40,  2,  2,  2,
            2,  4, 16,  4,  4,  3,  8], dtype=int64)




```python
df[3].value_counts(sort=False).values
```




    array([ 4,  1,  1,  1,  1,  1,  5,  2,  2,  6,  2,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  2,  3,  1,  1,  1,  2,  1,  1,  1,  1,  1,
            3,  1,  5,  1,  1,  1,  2,  2,  1,  1,  2,  5,  3,  2,  3,  4,  1,
            1,  2,  3,  1,  2,  2,  1,  1,  6, 11,  5,  7,  4,  2,  3,  1,  1,
            5,  3,  3,  5,  5,  4,  1,  4,  5,  4,  6,  3,  3,  7,  3,  2,  1,
            4,  2,  3,  1,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,
            1,  6,  1,  1,  1,  2,  1,  1,  2,  1,  2,  3,  5,  2,  5,  5,  1,
            1,  1,  1,  3,  4,  4,  1,  5, 19, 10,  9,  8, 20,  5,  6, 27, 35,
           20, 13,  1,  1,  1,  2,  1,  1,  1,  1,  2], dtype=int64)




```python
df[166].value_counts(sort=False).values
```




    array([ 5,  4,  8, 10, 14, 20,  6,  5,  8,  6, 11,  9,  8, 14, 11,  5,  3,
            5, 11, 10, 12,  7,  7,  5,  7,  9, 10,  1,  2,  3, 10,  5,  8,  8,
            2,  5,  5,  1,  3,  1,  1,  1,  1,  2,  1,  7,  7,  1,  1,  1,  1,
            3,  1,  1,  2,  4,  1,  1,  8,  1,  6,  1,  1,  1,  5,  4,  1,  3,
            1,  1,  1,  1,  1,  1,  7, 12, 11,  5,  3,  1,  1,  1,  1,  2,  1,
            5,  1,  1,  7,  2,  1,  1,  1,  3,  1,  5,  1,  1,  5, 13,  3,  2,
            1,  1,  1,  4,  1, 11,  1], dtype=int64)




```python
df[2].max() #max of 2nd column
```




    130




```python
df[2].min()  #max of 2nd column
```




    -9




```python
df[160].max() #max of 160th column
```




    173




```python
df[160].min()   #min of 160th column
```




    -217




```python
df[165].max()  #max of 165th column
```




    24




```python
df[165].min()  #min of 165th column
```




    -132




```python
label=df.groupby([1])["Class"].first().values
```


```python
label
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0], dtype=int64)




```python
len(label)
```




    92




```python
k=df.loc[df.iloc[:,1]==1] #class 1
```


```python
k  #class 1
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
      <th>Class</th>
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
      <th>158</th>
      <th>159</th>
      <th>160</th>
      <th>161</th>
      <th>162</th>
      <th>163</th>
      <th>164</th>
      <th>165</th>
      <th>166</th>
      <th>167</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-198</td>
      <td>-109</td>
      <td>-75</td>
      <td>-117</td>
      <td>11</td>
      <td>23</td>
      <td>-88</td>
      <td>...</td>
      <td>-238</td>
      <td>-74</td>
      <td>-129</td>
      <td>-120</td>
      <td>-38</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>6</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-191</td>
      <td>-142</td>
      <td>-65</td>
      <td>-117</td>
      <td>55</td>
      <td>49</td>
      <td>-170</td>
      <td>...</td>
      <td>-238</td>
      <td>-302</td>
      <td>60</td>
      <td>-120</td>
      <td>-39</td>
      <td>31</td>
      <td>48</td>
      <td>-37</td>
      <td>5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-191</td>
      <td>-142</td>
      <td>-75</td>
      <td>-117</td>
      <td>11</td>
      <td>49</td>
      <td>-161</td>
      <td>...</td>
      <td>-238</td>
      <td>-73</td>
      <td>-127</td>
      <td>-120</td>
      <td>-38</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>5</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-198</td>
      <td>-110</td>
      <td>-65</td>
      <td>-117</td>
      <td>55</td>
      <td>23</td>
      <td>-95</td>
      <td>...</td>
      <td>-238</td>
      <td>-302</td>
      <td>60</td>
      <td>-120</td>
      <td>-39</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>6</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 168 columns</p>
</div>




```python
X_train=df.loc[:317,2:]
```


```python
X_test=df.loc[318:"Class"]
```


```python
for i in range(1,93):
    X_test=df.loc[df.iloc[:,1]==i]     #train and test data
    X_train=df.loc[df.iloc[:,1]==i]
    X_train_np=X_train.to_numpy()
    X_test_np=X_test.to_numpy()
   
```


```python
bags_train=X_train_np.tolist()
bags_test=X_test_np.tolist()
```


```python
#Support Vector Classifier
```


```python
# instantiate trainer
trainer = Trainer()

# preparing trainer
metrics = ['acc', AUC]
model = SVC(kernel='linear', class_weight='balanced')
pipeline = [('scale', StandarizerBagsList()), ('disc_mapping', MILESMapping())]   #MILESMapping approach
trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)

# fitting trainer
valid=KFold(n_splits=10)
trained = trainer.fit(bags_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)

# printing validation results for each fold
print(trained['metrics_val'])

# predicting metrics for the test set
trainer.predict_metrics(bags_test, y_test)
```

    10/10 [==============================] - 2s 183ms/step - train_accuracy: 1.0000 - train_auc: 1.0000 - val_accuracy: 1.0000 - val_auc: 1.0000
    [{'accuracy': 0.875, 'auc': 0.875}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.875, 'auc': 0.8333334}, {'accuracy': 0.5714286, 'auc': 0.7}, {'accuracy': 0.71428573, 'auc': 0.8}, {'accuracy': 0.85714287, 'auc': 0.8333334}, {'accuracy': 0.85714287, 'auc': 0.8333334}, {'accuracy': 0.85714287, 'auc': 0.875}, {'accuracy': 0.42857143, 'auc': 0.5}, {'accuracy': 1.0, 'auc': 1.0}]
    




    {'accuracy': 0.68421054, 'auc': 0.72727275}




```python

# instantiate trainer
trainer = Trainer()

# preparing trainer
metrics = ['acc', AUC]
model = SVC(kernel='linear', C=1, class_weight='balanced')
pipeline = [('scale', StandarizerBagsList()), ('disc_mapping', MeanMinMaxBagRepresentation())]   #MeanMinMaxBagRepresentation
trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)

# fitting trainer
valid=KFold(n_splits=10)
trained = trainer.fit(bags_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)

# printing validation results for each fold
print(trained['metrics_val'])

# predicting metrics for the test set
trainer.predict_metrics(bags_test, y_test)
```

    10/10 [==============================] - 1s 65ms/step - train_accuracy: 1.0000 - train_auc: 1.0000 - val_accuracy: 1.0000 - val_auc: 1.0000
    [{'accuracy': 0.875, 'auc': 0.875}, {'accuracy': 0.875, 'auc': 0.90000004}, {'accuracy': 0.625, 'auc': 0.6333334}, {'accuracy': 0.85714287, 'auc': 0.75}, {'accuracy': 0.71428573, 'auc': 0.5}, {'accuracy': 0.5714286, 'auc': 0.5833334}, {'accuracy': 0.5714286, 'auc': 0.5416667}, {'accuracy': 0.85714287, 'auc': 0.8333333}, {'accuracy': 0.71428573, 'auc': 0.7083333}, {'accuracy': 1.0, 'auc': 1.0}]
    




    {'accuracy': 0.8947368, 'auc': 0.875}




```python
#Random Forest Classifier
```


```python
# instantiate trainer
trainer = Trainer()

# preparing trainer
metrics = ['acc', AUC]

model = RandomForestClassifier(random_state=0,n_estimators=200,max_depth=2)  #trying with max_depth 2
pipeline = [('scale', StandarizerBagsList()), ('disc_mapping',  MILESMapping())] #MILESMapping approach
trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)

# fitting trainer
valid=KFold(n_splits=10)  #10-cross validation
trained = trainer.fit(bags_train, y_train, validation_strategy=valid, verbose=1)

# printing validation results for each fold
print(trained['metrics_val'])

# predicting metrics for the test set
trainer.predict_metrics(bags_test, y_test)
```

    10/10 [==============================] - 5s 545ms/step - train_accuracy: 0.9848 - train_auc: 0.9844 - val_accuracy: 1.0000 - val_auc: 1.0000
    [{'accuracy': 0.875, 'auc': 0.875}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.75, 'auc': 0.73333335}, {'accuracy': 0.85714287, 'auc': 0.9}, {'accuracy': 0.85714287, 'auc': 0.75}, {'accuracy': 0.71428573, 'auc': 0.75}, {'accuracy': 0.71428573, 'auc': 0.6666667}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.5714286, 'auc': 0.625}, {'accuracy': 1.0, 'auc': 1.0}]
    




    {'accuracy': 0.84210527, 'auc': 0.8295455}




```python
# instantiate trainer
trainer = Trainer()

# preparing trainer
metrics = ['acc', AUC]

model = RandomForestClassifier(random_state=0,n_estimators=200,max_depth=5) #trying with max_depth 5
pipeline = [('scale', StandarizerBagsList()), ('disc_mapping',  MILESMapping())] #MILESMapping approach
trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)

# fitting trainer
valid=KFold(n_splits=10)
trained = trainer.fit(bags_train, y_train, validation_strategy=valid, verbose=1)

# printing validation results for each fold
print(trained['metrics_val'])

# predicting metrics for the test set
trainer.predict_metrics(bags_test, y_test)
```

    10/10 [==============================] - 6s 646ms/step - train_accuracy: 1.0000 - train_auc: 1.0000 - val_accuracy: 1.0000 - val_auc: 1.0000
    [{'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.625, 'auc': 0.6333334}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.71428573, 'auc': 0.65}, {'accuracy': 0.71428573, 'auc': 0.75}, {'accuracy': 0.71428573, 'auc': 0.7083333}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.85714287, 'auc': 0.875}, {'accuracy': 1.0, 'auc': 1.0}]
    




    {'accuracy': 0.94736844, 'auc': 0.9375}




```python
# instantiate trainer
trainer = Trainer()

# preparing trainer
metrics = ['acc', AUC]

model = RandomForestClassifier(random_state=0,n_estimators=200,max_depth=8)  #trying with max_depth 8
pipeline = [('scale', StandarizerBagsList()), ('disc_mapping',  MILESMapping())] #MILESMapping approach
trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)

# fitting trainer
valid=KFold(n_splits=10)
trained = trainer.fit(bags_train, y_train, validation_strategy=valid, verbose=1)

# printing validation results for each fold
print(trained['metrics_val'])

# predicting metrics for the test set
trainer.predict_metrics(bags_test, y_test)
```

    10/10 [==============================] - 6s 584ms/step - train_accuracy: 1.0000 - train_auc: 1.0000 - val_accuracy: 1.0000 - val_auc: 1.0000
    [{'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.625, 'auc': 0.6333334}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.71428573, 'auc': 0.65}, {'accuracy': 0.71428573, 'auc': 0.75}, {'accuracy': 0.71428573, 'auc': 0.7083333}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.85714287, 'auc': 0.875}, {'accuracy': 1.0, 'auc': 1.0}]
    




    {'accuracy': 0.94736844, 'auc': 0.9375}




```python
# instantiate trainer
trainer = Trainer()

# preparing trainer
metrics = ['acc', AUC]
model = RandomForestClassifier(random_state=0,n_estimators=200,max_depth=2) #trying with max_depth 2
pipeline = [('scale', StandarizerBagsList()), ('disc_mapping',  MeanMinMaxBagRepresentation())]  #MeanMinMaxBagRepresentation
trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)


# fitting trainer
valid=KFold(n_splits=10)
trained = trainer.fit(bags_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)

# printing validation results for each fold
print(trained['metrics_val'])

# predicting metrics for the test set
trainer.predict_metrics(bags_test, y_test)
```

    10/10 [==============================] - 4s 405ms/step - train_accuracy: 0.9545 - train_auc: 0.9540 - val_accuracy: 0.8571 - val_auc: 0.7500
    [{'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.75, 'auc': 0.8}, {'accuracy': 0.625, 'auc': 0.6333334}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.5714286, 'auc': 0.625}, {'accuracy': 0.5714286, 'auc': 0.5416667}, {'accuracy': 0.71428573, 'auc': 0.6666666}, {'accuracy': 0.85714287, 'auc': 0.8333333}, {'accuracy': 0.85714287, 'auc': 0.75}]
    




    {'accuracy': 0.7894737, 'auc': 0.7670455}




```python

# instantiate trainer
trainer = Trainer()

# preparing trainer
metrics = ['acc', AUC]
model = RandomForestClassifier(random_state=0,n_estimators=200,max_depth=5) #trying with max_depth 5
pipeline = [('scale', StandarizerBagsList()), ('disc_mapping',  MeanMinMaxBagRepresentation())]  #MeanMinMaxBagRepresentation
trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)


# fitting trainer
valid=KFold(n_splits=10)
trained = trainer.fit(bags_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)

# printing validation results for each fold
print(trained['metrics_val'])

# predicting metrics for the test set
trainer.predict_metrics(bags_test, y_test)
```

    10/10 [==============================] - 4s 423ms/step - train_accuracy: 1.0000 - train_auc: 1.0000 - val_accuracy: 1.0000 - val_auc: 1.0000
    [{'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.875, 'auc': 0.90000004}, {'accuracy': 0.625, 'auc': 0.6333334}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.85714287, 'auc': 0.75}, {'accuracy': 0.5714286, 'auc': 0.625}, {'accuracy': 0.71428573, 'auc': 0.6666667}, {'accuracy': 0.85714287, 'auc': 0.8333333}, {'accuracy': 0.85714287, 'auc': 0.8333333}, {'accuracy': 1.0, 'auc': 1.0}]
    




    {'accuracy': 0.8947368, 'auc': 0.875}




```python

# instantiate trainer
trainer = Trainer()

# preparing trainer
metrics = ['acc', AUC]
model = RandomForestClassifier(random_state=0,n_estimators=200,max_depth=8) #trying with max_depth 8
pipeline = [('scale', StandarizerBagsList()), ('disc_mapping',  MeanMinMaxBagRepresentation())]  #MeanMinMaxBagRepresentation
trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)


# fitting trainer
valid=KFold(n_splits=10)
trained = trainer.fit(bags_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)

# printing validation results for each fold
print(trained['metrics_val'])

# predicting metrics for the test set
trainer.predict_metrics(bags_test, y_test)
```

    10/10 [==============================] - 5s 454ms/step - train_accuracy: 1.0000 - train_auc: 1.0000 - val_accuracy: 1.0000 - val_auc: 1.0000
    [{'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.875, 'auc': 0.90000004}, {'accuracy': 0.625, 'auc': 0.6333334}, {'accuracy': 1.0, 'auc': 1.0}, {'accuracy': 0.85714287, 'auc': 0.75}, {'accuracy': 0.5714286, 'auc': 0.625}, {'accuracy': 0.71428573, 'auc': 0.6666667}, {'accuracy': 0.85714287, 'auc': 0.8333333}, {'accuracy': 0.85714287, 'auc': 0.8333333}, {'accuracy': 1.0, 'auc': 1.0}]
    




    {'accuracy': 0.8947368, 'auc': 0.875}




INTRODUCTION
I observed the shape, uniqueness and minimum and maximum of some columns. It can be seen that dataset has 167 features which can be considered as high dimensional and also observed columns has minimum values frequently are negative integers whereas maximums are positive integers. In addition to this, range of these values are significantly large. Also when one observes uniqueness, value counts of columns have significant amount values with one repetition. Therefore one of the alternative bag level representations that I suggest is bag- level MILES with SVC with linear kernel and Random Forest classifiers with max depth 2, 5 and 8 and number of tree used as 200 constantly. Multiple-Instance Learning via Embedded Instance Selection (MILES) is an approach to MI learning based on the diverse density framework. In contrast to standard diverse density algorithms, it embeds bags into a single-instance feature space. Earlier diverse density-based methods have used the standard MI assumption mentioned above and further assume the existence of a single target point. Instead, MILES uses a symmetric assumption, where multiple target points are allowed, each of which may be related to either positive or negative bags (Foulds & Frank, 2008). Another alternative approach that I suggest is MeanMinMax bag representation. As a heuristic approach, I tried this one since data has a large range between negative and positive values so that I can generalize one instance in each bag as average of the minimum and maximum. Also this approach was applied with SVC with linear kernel and Random Forest classifier.







ANALYSIS, DISCUSSION and CONCLUSION

                    	AUC
MeanMinMaxBagRepresentation MILESMapping
SVC	                0.875	0.727
Random Forest (2)	0.767	0.829
Random Forest (5)	0.875	0.937
Random Forest (8)	0.875	0.937
Table 1. AUC values for different approaches

As it can be seen in from the Table 1 best area under curve value was found on MILESMapping approach with random forest classifier with maximum depth 5 and 8. Also the poorest AUC value was observed in MILESMapping with SVC classifier.  While maximum depth of trees increased to some level (5 for this example), both approach provided better AUC values. But maximum depth with 8 has no increment or decrement for this specific case. Also we can observe that random forest provided better value on MILESMapping for each random forest trials whereas MeanMinMaxBagRepresetation provided better AUC value on SVC.





References
Foulds, J., & Frank, E. (2008). Revisiting Multiple-Instance Learning via Embedded Instance Selection. Australasian Joint Conference on Artificial Intelligence, 300-310.



