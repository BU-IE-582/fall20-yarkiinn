```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS
import seaborn as sns
```


```python
#Question A
X = pd.read_csv("X_TRAIN.csv",header=None,delim_whitespace=True )  #I ordered the whitspaces to read data as dataframe
Y = pd.read_csv('Y_TRAIN.csv',header=None,delim_whitespace=True ) 
Z = pd.read_csv('Z_TRAIN.csv',header=None,delim_whitespace=True )
```


```python
X.rename(columns = {0:'Class'}, inplace = True) 
Y.rename(columns = {0:'Class'}, inplace = True)   #Changed the 0(class name) as "Class"
Z.rename(columns = {0:'Class'}, inplace = True)
```


```python
class_info=X.iloc[:,0]   #taking class column from first(X) data
```


```python
cum_sum_x=np.cumsum(X.iloc[:,1:],axis=1)   #I produced the cumulative sums of accleration over time and found the velocity vector.
cum_sum_y=np.cumsum(Y.iloc[:,1:],axis=1)   # Excluded class and started from index 1.
cum_sum_z=np.cumsum(Z.iloc[:,1:],axis=1)
```


```python
X_= pd.concat([class_info, cum_sum_x], axis=1, sort=False)
Y_= pd.concat([class_info, cum_sum_y], axis=1, sort=False) #Added class information
Z_= pd.concat([class_info, cum_sum_z], axis=1, sort=False)
```


```python
X_.insert(loc=0, column='Index', value=range(0,896))
Y_.insert(loc=0, column='Index', value=range(0,896)) #Creating an index column
Z_.insert(loc=0, column='Index', value=range(0,896))
```


```python
X_
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
      <th>Index</th>
      <th>Class</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>306</th>
      <th>307</th>
      <th>308</th>
      <th>309</th>
      <th>310</th>
      <th>311</th>
      <th>312</th>
      <th>313</th>
      <th>314</th>
      <th>315</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6.0</td>
      <td>-0.304243</td>
      <td>-0.608486</td>
      <td>-0.912730</td>
      <td>-1.216973</td>
      <td>-1.521216</td>
      <td>-1.825459</td>
      <td>-2.129702</td>
      <td>-2.433946</td>
      <td>...</td>
      <td>5.970108</td>
      <td>5.228380</td>
      <td>4.533681</td>
      <td>3.886012</td>
      <td>3.238344</td>
      <td>2.590675</td>
      <td>1.943006</td>
      <td>1.295337</td>
      <td>0.647669</td>
      <td>-1.989999e-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5.0</td>
      <td>1.627311</td>
      <td>3.254622</td>
      <td>4.881933</td>
      <td>6.509244</td>
      <td>8.136556</td>
      <td>9.763867</td>
      <td>11.391178</td>
      <td>13.018489</td>
      <td>...</td>
      <td>2.138701</td>
      <td>1.901068</td>
      <td>1.663434</td>
      <td>1.425800</td>
      <td>1.188166</td>
      <td>0.950533</td>
      <td>0.712899</td>
      <td>0.475265</td>
      <td>0.237632</td>
      <td>-2.102000e-06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5.0</td>
      <td>0.661277</td>
      <td>1.322553</td>
      <td>1.983830</td>
      <td>2.645106</td>
      <td>3.306383</td>
      <td>3.967659</td>
      <td>4.628936</td>
      <td>5.290212</td>
      <td>...</td>
      <td>0.961456</td>
      <td>0.724527</td>
      <td>0.532679</td>
      <td>0.385913</td>
      <td>0.284228</td>
      <td>0.223025</td>
      <td>0.166387</td>
      <td>0.110925</td>
      <td>0.055463</td>
      <td>5.146001e-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3.0</td>
      <td>0.005185</td>
      <td>0.010370</td>
      <td>0.015554</td>
      <td>0.020739</td>
      <td>0.025924</td>
      <td>0.031109</td>
      <td>0.036293</td>
      <td>0.041478</td>
      <td>...</td>
      <td>-8.792173</td>
      <td>-7.644351</td>
      <td>-6.550008</td>
      <td>-5.489646</td>
      <td>-4.436487</td>
      <td>-3.392020</td>
      <td>-2.373040</td>
      <td>-1.463090</td>
      <td>-0.672077</td>
      <td>-6.300000e-08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4.0</td>
      <td>1.286198</td>
      <td>2.572396</td>
      <td>3.858593</td>
      <td>5.144791</td>
      <td>6.430989</td>
      <td>7.717187</td>
      <td>9.003385</td>
      <td>10.289582</td>
      <td>...</td>
      <td>13.093926</td>
      <td>11.656943</td>
      <td>10.219960</td>
      <td>8.782978</td>
      <td>7.345995</td>
      <td>5.905653</td>
      <td>4.453140</td>
      <td>2.984694</td>
      <td>1.500315</td>
      <td>2.645900e-06</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>891</th>
      <td>891</td>
      <td>3.0</td>
      <td>0.117811</td>
      <td>0.235622</td>
      <td>0.353433</td>
      <td>0.471244</td>
      <td>0.589055</td>
      <td>0.706866</td>
      <td>0.824677</td>
      <td>0.942488</td>
      <td>...</td>
      <td>-2.543936</td>
      <td>-2.361904</td>
      <td>-2.148004</td>
      <td>-1.909160</td>
      <td>-1.653556</td>
      <td>-1.381190</td>
      <td>-1.083863</td>
      <td>-0.756098</td>
      <td>-0.394810</td>
      <td>2.185001e-07</td>
    </tr>
    <tr>
      <th>892</th>
      <td>892</td>
      <td>1.0</td>
      <td>-1.232590</td>
      <td>-2.465181</td>
      <td>-3.697771</td>
      <td>-4.930362</td>
      <td>-6.162952</td>
      <td>-7.395542</td>
      <td>-8.628133</td>
      <td>-9.860723</td>
      <td>...</td>
      <td>-0.852997</td>
      <td>-0.823414</td>
      <td>-0.774643</td>
      <td>-0.706685</td>
      <td>-0.622468</td>
      <td>-0.523887</td>
      <td>-0.412102</td>
      <td>-0.287526</td>
      <td>-0.150158</td>
      <td>1.647300e-06</td>
    </tr>
    <tr>
      <th>893</th>
      <td>893</td>
      <td>2.0</td>
      <td>0.282877</td>
      <td>0.565755</td>
      <td>0.848632</td>
      <td>1.131510</td>
      <td>1.414387</td>
      <td>1.697264</td>
      <td>1.980142</td>
      <td>2.263019</td>
      <td>...</td>
      <td>19.966182</td>
      <td>17.684324</td>
      <td>15.402467</td>
      <td>13.120609</td>
      <td>10.838751</td>
      <td>8.560205</td>
      <td>6.310830</td>
      <td>4.118563</td>
      <td>2.007065</td>
      <td>3.336000e-07</td>
    </tr>
    <tr>
      <th>894</th>
      <td>894</td>
      <td>7.0</td>
      <td>1.248704</td>
      <td>2.497409</td>
      <td>3.746113</td>
      <td>4.994818</td>
      <td>6.243522</td>
      <td>7.492226</td>
      <td>8.740931</td>
      <td>9.989635</td>
      <td>...</td>
      <td>0.030701</td>
      <td>-0.217793</td>
      <td>-0.431280</td>
      <td>-0.549265</td>
      <td>-0.568286</td>
      <td>-0.498443</td>
      <td>-0.372638</td>
      <td>-0.233362</td>
      <td>-0.102691</td>
      <td>-8.426000e-07</td>
    </tr>
    <tr>
      <th>895</th>
      <td>895</td>
      <td>5.0</td>
      <td>-0.168518</td>
      <td>-0.337036</td>
      <td>-0.505553</td>
      <td>-0.674071</td>
      <td>-0.842589</td>
      <td>-1.011107</td>
      <td>-1.179625</td>
      <td>-1.348142</td>
      <td>...</td>
      <td>-15.934478</td>
      <td>-14.472657</td>
      <td>-12.998853</td>
      <td>-11.429036</td>
      <td>-9.659839</td>
      <td>-7.691264</td>
      <td>-5.662732</td>
      <td>-3.675465</td>
      <td>-1.787888</td>
      <td>6.081000e-07</td>
    </tr>
  </tbody>
</table>
<p>896 rows × 317 columns</p>
</div>




```python
first_gesture=X_.loc[X_.iloc[:,1]==1]
```


```python
first_gesture
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
      <th>Index</th>
      <th>Class</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>306</th>
      <th>307</th>
      <th>308</th>
      <th>309</th>
      <th>310</th>
      <th>311</th>
      <th>312</th>
      <th>313</th>
      <th>314</th>
      <th>315</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>1.0</td>
      <td>-0.791447</td>
      <td>-1.582894</td>
      <td>-2.378767</td>
      <td>-3.188832</td>
      <td>-4.038062</td>
      <td>-4.941527</td>
      <td>-5.872673</td>
      <td>-6.823277</td>
      <td>...</td>
      <td>-0.755218</td>
      <td>-0.671305</td>
      <td>-0.587391</td>
      <td>-0.503478</td>
      <td>-0.419565</td>
      <td>-0.335652</td>
      <td>-0.251739</td>
      <td>-0.167826</td>
      <td>-0.083913</td>
      <td>2.499990e-09</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>1.0</td>
      <td>-0.449602</td>
      <td>-0.899205</td>
      <td>-1.348807</td>
      <td>-1.798410</td>
      <td>-2.248012</td>
      <td>-2.697615</td>
      <td>-3.147217</td>
      <td>-3.596820</td>
      <td>...</td>
      <td>4.582439</td>
      <td>3.940151</td>
      <td>3.343462</td>
      <td>2.798052</td>
      <td>2.287614</td>
      <td>1.806075</td>
      <td>1.350176</td>
      <td>0.899205</td>
      <td>0.449603</td>
      <td>3.950000e-07</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>1.0</td>
      <td>-0.187469</td>
      <td>-0.374938</td>
      <td>-0.562407</td>
      <td>-0.749876</td>
      <td>-0.937344</td>
      <td>-1.124813</td>
      <td>-1.312282</td>
      <td>-1.499751</td>
      <td>...</td>
      <td>-4.826368</td>
      <td>-4.290105</td>
      <td>-3.753842</td>
      <td>-3.217579</td>
      <td>-2.681316</td>
      <td>-2.145053</td>
      <td>-1.608790</td>
      <td>-1.072527</td>
      <td>-0.536264</td>
      <td>-6.107000e-07</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>1.0</td>
      <td>0.301482</td>
      <td>0.602963</td>
      <td>0.904445</td>
      <td>1.205927</td>
      <td>1.507408</td>
      <td>1.808890</td>
      <td>2.110372</td>
      <td>2.411854</td>
      <td>...</td>
      <td>-1.805298</td>
      <td>-1.529600</td>
      <td>-1.272680</td>
      <td>-1.034536</td>
      <td>-0.815170</td>
      <td>-0.614581</td>
      <td>-0.432770</td>
      <td>-0.269736</td>
      <td>-0.125479</td>
      <td>1.970000e-07</td>
    </tr>
    <tr>
      <th>37</th>
      <td>37</td>
      <td>1.0</td>
      <td>0.163202</td>
      <td>0.326404</td>
      <td>0.489607</td>
      <td>0.652809</td>
      <td>0.816011</td>
      <td>0.979213</td>
      <td>1.142415</td>
      <td>1.305617</td>
      <td>...</td>
      <td>1.557704</td>
      <td>1.384626</td>
      <td>1.211547</td>
      <td>1.038469</td>
      <td>0.865391</td>
      <td>0.692313</td>
      <td>0.519235</td>
      <td>0.346157</td>
      <td>0.173078</td>
      <td>1.600000e-07</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>865</th>
      <td>865</td>
      <td>1.0</td>
      <td>0.948343</td>
      <td>1.896686</td>
      <td>2.845028</td>
      <td>3.793371</td>
      <td>4.741714</td>
      <td>5.690057</td>
      <td>6.638399</td>
      <td>7.586742</td>
      <td>...</td>
      <td>-2.295508</td>
      <td>-2.088826</td>
      <td>-1.873569</td>
      <td>-1.649738</td>
      <td>-1.416498</td>
      <td>-1.166779</td>
      <td>-0.900582</td>
      <td>-0.617536</td>
      <td>-0.317342</td>
      <td>3.209950e-09</td>
    </tr>
    <tr>
      <th>878</th>
      <td>878</td>
      <td>1.0</td>
      <td>0.423145</td>
      <td>0.846290</td>
      <td>1.269435</td>
      <td>1.692580</td>
      <td>2.115726</td>
      <td>2.538871</td>
      <td>2.962016</td>
      <td>3.385161</td>
      <td>...</td>
      <td>11.712916</td>
      <td>10.569039</td>
      <td>9.385773</td>
      <td>8.163117</td>
      <td>6.901071</td>
      <td>5.599636</td>
      <td>4.258812</td>
      <td>2.878597</td>
      <td>1.458993</td>
      <td>-2.379997e-08</td>
    </tr>
    <tr>
      <th>883</th>
      <td>883</td>
      <td>1.0</td>
      <td>-1.143164</td>
      <td>-2.286329</td>
      <td>-3.429493</td>
      <td>-4.572657</td>
      <td>-5.715822</td>
      <td>-6.858986</td>
      <td>-8.002150</td>
      <td>-9.145314</td>
      <td>...</td>
      <td>-2.818904</td>
      <td>-2.600923</td>
      <td>-2.370183</td>
      <td>-2.096840</td>
      <td>-1.787341</td>
      <td>-1.467403</td>
      <td>-1.145531</td>
      <td>-0.806289</td>
      <td>-0.424446</td>
      <td>-4.629000e-07</td>
    </tr>
    <tr>
      <th>885</th>
      <td>885</td>
      <td>1.0</td>
      <td>-0.289544</td>
      <td>-0.579087</td>
      <td>-0.868631</td>
      <td>-1.158174</td>
      <td>-1.447718</td>
      <td>-1.737262</td>
      <td>-2.036735</td>
      <td>-2.349168</td>
      <td>...</td>
      <td>6.399303</td>
      <td>5.687297</td>
      <td>4.996575</td>
      <td>4.310490</td>
      <td>3.622981</td>
      <td>2.914187</td>
      <td>2.190986</td>
      <td>1.461760</td>
      <td>0.730880</td>
      <td>-3.239997e-08</td>
    </tr>
    <tr>
      <th>892</th>
      <td>892</td>
      <td>1.0</td>
      <td>-1.232590</td>
      <td>-2.465181</td>
      <td>-3.697771</td>
      <td>-4.930362</td>
      <td>-6.162952</td>
      <td>-7.395542</td>
      <td>-8.628133</td>
      <td>-9.860723</td>
      <td>...</td>
      <td>-0.852997</td>
      <td>-0.823414</td>
      <td>-0.774643</td>
      <td>-0.706685</td>
      <td>-0.622468</td>
      <td>-0.523887</td>
      <td>-0.412102</td>
      <td>-0.287526</td>
      <td>-0.150158</td>
      <td>1.647300e-06</td>
    </tr>
  </tbody>
</table>
<p>122 rows × 317 columns</p>
</div>




```python
x=X_.loc[first_gesture.iloc[0,0]][2:]
y=Y_.loc[first_gesture.iloc[0,0]][2:]
z=Z_.loc[first_gesture.iloc[0,0]][2:]
```


```python
x
```




    1     -7.914472e-01
    2     -1.582894e+00
    3     -2.378767e+00
    4     -3.188832e+00
    5     -4.038062e+00
               ...     
    311   -3.356523e-01
    312   -2.517392e-01
    313   -1.678261e-01
    314   -8.391307e-02
    315    2.499990e-09
    Name: 10, Length: 315, dtype: float64




```python
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)   
pyplot.show()         
```


![png](output_12_0.png)



```python
second_gesture=X_.loc[X_.iloc[:,1]==2]
```


```python
x=X_.loc[second_gesture.iloc[0,0]][2:]
y=Y_.loc[second_gesture.iloc[0,0]][2:]
z=Z_.loc[second_gesture.iloc[0,0]][2:]
```


```python
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
pyplot.show()
```


![png](output_15_0.png)



```python
third_gesture=X_.loc[X_.iloc[:,1]==3]
```


```python
x=X_.loc[third_gesture.iloc[0,0]][2:]
y=Y_.loc[third_gesture.iloc[0,0]][2:]
z=Z_.loc[third_gesture.iloc[0,0]][2:]
```


```python
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
pyplot.show()
```


![png](output_18_0.png)



```python

```


```python
fourth_gesture=X_.loc[X_.iloc[:,1]==4]
```


```python
x=X_.loc[fourth_gesture.iloc[0,0]][2:]
y=Y_.loc[fourth_gesture.iloc[0,0]][2:]
z=Z_.loc[fourth_gesture.iloc[0,0]][2:]
```


```python
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
pyplot.show()
```


![png](output_22_0.png)



```python

```


```python
fifth_gesture=X_.loc[X_.iloc[:,1]==5]
```


```python
x=X_.loc[fifth_gesture.iloc[0,0]][2:]
y=Y_.loc[fifth_gesture.iloc[0,0]][2:]
z=Z_.loc[fifth_gesture.iloc[0,0]][2:]
```


```python
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
pyplot.show()
```


![png](output_26_0.png)



```python

```


```python
sixth_gesture=X_.loc[X_.iloc[:,1]==6]
```


```python
x=X_.loc[sixth_gesture.iloc[0,0]][2:]
y=Y_.loc[sixth_gesture.iloc[0,0]][2:]
z=Z_.loc[sixth_gesture.iloc[0,0]][2:]
```


```python
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
pyplot.show()
```


![png](output_30_0.png)



```python

```


```python
seventh_gesture=X_.loc[X_.iloc[:,1]==7]
```


```python
x=X_.loc[seventh_gesture.iloc[0,0]][2:]
y=Y_.loc[seventh_gesture.iloc[0,0]][2:]
z=Z_.loc[seventh_gesture.iloc[0,0]][2:]
```


```python
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
pyplot.show()
```


![png](output_34_0.png)



```python

```


```python
eighth_gesture=X_.loc[X_.iloc[:,1]==8]
```


```python
x=X_.loc[eighth_gesture.iloc[0,0]][2:]
y=Y_.loc[eighth_gesture.iloc[0,0]][2:]
z=Z_.loc[eighth_gesture.iloc[0,0]][2:]
```


```python
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
pyplot.show()
```


![png](output_38_0.png)



```python
#I plotted 3D scatter plot of velocity vector for each classes and try to find relation between actual gestures.
#As it is seen in the scatterplots, there are similar shapes and figures.Therefore we can conclude
#that accelaration data is meaningful.
```


```python
#Question B
```


```python
X["Time_Id"] = X.index +1
Y["Time_Id"] = Y.index +1  #Made the time ids for long format by adding 1 on each indexes.
Z["Time_Id"] = Z.index +1
```


```python
X_MELT = pd.melt(X,id_vars=["Class","Time_Id"],var_name="Time_Index", value_name="X") 
Y_MELT = pd.melt(Y,id_vars=["Class","Time_Id"],var_name="Time_Index", value_name="Y") #By using melt function transformed data wide to long format
Z_MELT = pd.melt(Z,id_vars=["Class","Time_Id"],var_name="Time_Index", value_name="Z")
```


```python
long_format=pd.merge(pd.merge(X_MELT,Y_MELT),Z_MELT)   #Merge all components based on class and time indexes
```


```python
long_format
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
      <th>Time_Id</th>
      <th>Time_Index</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>1</td>
      <td>1</td>
      <td>-0.304243</td>
      <td>-2.119396</td>
      <td>-1.528965</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>2</td>
      <td>1</td>
      <td>1.627311</td>
      <td>0.666624</td>
      <td>1.786869</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>3</td>
      <td>1</td>
      <td>0.661277</td>
      <td>-0.189730</td>
      <td>0.521249</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>4</td>
      <td>1</td>
      <td>0.005185</td>
      <td>0.374067</td>
      <td>0.309455</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>5</td>
      <td>1</td>
      <td>1.286198</td>
      <td>-0.397437</td>
      <td>-0.466022</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>282235</th>
      <td>3.0</td>
      <td>892</td>
      <td>315</td>
      <td>0.394810</td>
      <td>-0.511710</td>
      <td>0.214065</td>
    </tr>
    <tr>
      <th>282236</th>
      <td>1.0</td>
      <td>893</td>
      <td>315</td>
      <td>0.150160</td>
      <td>-0.437178</td>
      <td>0.332617</td>
    </tr>
    <tr>
      <th>282237</th>
      <td>2.0</td>
      <td>894</td>
      <td>315</td>
      <td>-2.007065</td>
      <td>0.316739</td>
      <td>0.267828</td>
    </tr>
    <tr>
      <th>282238</th>
      <td>7.0</td>
      <td>895</td>
      <td>315</td>
      <td>0.102690</td>
      <td>-0.269599</td>
      <td>-0.433044</td>
    </tr>
    <tr>
      <th>282239</th>
      <td>5.0</td>
      <td>896</td>
      <td>315</td>
      <td>1.787888</td>
      <td>-0.622273</td>
      <td>-0.736781</td>
    </tr>
  </tbody>
</table>
<p>282240 rows × 6 columns</p>
</div>




```python
dimensions=["X","Y","Z"]
pca_data=long_format.loc[:,dimensions].values     #Prepared data for PCA
```


```python
pca= PCA(n_components=1)
pca_fit=pca.fit_transform(pca_data)   #Reduced data by PCA to 1D
```


```python
pca_fit
```




    array([[-2.49257458],
           [ 2.15017535],
           [ 0.42976291],
           ...,
           [-0.48215096],
           [-0.38684878],
           [-0.08759219]])




```python
print("Explained variance",pca.explained_variance_)
print("Explained variance ratio: ",pca.explained_variance_ratio_)  #Found explained _variance and explained variance ratio
```

    Explained variance [1.46711782]
    Explained variance ratio:  [0.49059498]
    


```python
df_pca=long_format[["Class","Time_Id"]]   
```


```python
df_pca["Principal Component"]=pca_fit   #Merge reduced with class and time id
```

    C:\Users\lenovo\anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    


```python
df_pca
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
      <th>Time_Id</th>
      <th>Principal Component</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>1</td>
      <td>-2.492575</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>2</td>
      <td>2.150175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>3</td>
      <td>0.429763</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>4</td>
      <td>0.440825</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>5</td>
      <td>0.008132</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>282235</th>
      <td>3.0</td>
      <td>892</td>
      <td>-0.083781</td>
    </tr>
    <tr>
      <th>282236</th>
      <td>1.0</td>
      <td>893</td>
      <td>-0.069781</td>
    </tr>
    <tr>
      <th>282237</th>
      <td>2.0</td>
      <td>894</td>
      <td>-0.482151</td>
    </tr>
    <tr>
      <th>282238</th>
      <td>7.0</td>
      <td>895</td>
      <td>-0.386849</td>
    </tr>
    <tr>
      <th>282239</th>
      <td>5.0</td>
      <td>896</td>
      <td>-0.087592</td>
    </tr>
  </tbody>
</table>
<p>282240 rows × 3 columns</p>
</div>




```python
s1=random.choice(list(df_pca[df_pca["Class"] == 1]["Time_Id"])) #Produced two random sample from class one's Time id's
s2=random.choice(list(df_pca[df_pca["Class"] == 1]["Time_Id"])) 
```


```python
print(s1)
print(s2)
```

    754
    866
    


```python
sample_1=df_pca[(df_pca["Class"] == 1) & (df_pca["Time_Id"] ==754)].iloc[:,2] #Use these two random sample and 
sample_2=df_pca[(df_pca["Class"] == 1) & (df_pca["Time_Id"] ==866)].iloc[:,2] #created two time series
```


```python
plt.plot(sample_1,color = "g")
plt.plot(sample_2,color = "b")
plt.legend(["S1", "S2"])
plt.show()
```


![png](output_55_0.png)



```python

```


```python
s1=random.choice(list(df_pca[df_pca["Class"] == 2]["Time_Id"]))
s2=random.choice(list(df_pca[df_pca["Class"] == 2]["Time_Id"]))
```


```python
print(s1)
print(s2)
```

    780
    65
    


```python
sample_1=df_pca[(df_pca["Class"] == 2) & (df_pca["Time_Id"] ==780)].iloc[:,2]
sample_2=df_pca[(df_pca["Class"] == 2) & (df_pca["Time_Id"] ==65)].iloc[:,2]
```


```python
plt.plot(sample_1,color = "g")
plt.plot(sample_2,color = "b")
plt.legend(["S1", "S2"])
plt.show()
```


![png](output_60_0.png)



```python

```


```python
s1=random.choice(list(df_pca[df_pca["Class"] == 3]["Time_Id"]))
s2=random.choice(list(df_pca[df_pca["Class"] == 3]["Time_Id"]))
```


```python
sample_1=df_pca[(df_pca["Class"] == 3) & (df_pca["Time_Id"] ==s1)].iloc[:,2]
sample_2=df_pca[(df_pca["Class"] == 3) & (df_pca["Time_Id"] ==s2)].iloc[:,2]
```


```python
plt.plot(sample_1,color = "g")
plt.plot(sample_2,color = "b")
plt.legend(["S1", "S2"])
plt.show()
```


![png](output_64_0.png)



```python

```


```python
s1=random.choice(list(df_pca[df_pca["Class"] == 4]["Time_Id"]))
s2=random.choice(list(df_pca[df_pca["Class"] == 4]["Time_Id"]))
```


```python
sample_1=df_pca[(df_pca["Class"] == 4) & (df_pca["Time_Id"] ==s1)].iloc[:,2]
sample_2=df_pca[(df_pca["Class"] == 4) & (df_pca["Time_Id"] ==s2)].iloc[:,2]
```


```python
plt.plot(sample_1,color = "g")
plt.plot(sample_2,color = "b")
plt.legend(["S1", "S2"])
plt.show()
```


![png](output_68_0.png)



```python

```


```python
s1=random.choice(list(df_pca[df_pca["Class"] == 5]["Time_Id"]))
s2=random.choice(list(df_pca[df_pca["Class"] == 5]["Time_Id"]))
```


```python
sample_1=df_pca[(df_pca["Class"] == 5) & (df_pca["Time_Id"] ==s1)].iloc[:,2]
sample_2=df_pca[(df_pca["Class"] == 5) & (df_pca["Time_Id"] ==s2)].iloc[:,2]
```


```python
plt.plot(sample_1,color = "g")
plt.plot(sample_2,color = "b")
plt.legend(["S1", "S2"])
plt.show()
```


![png](output_72_0.png)



```python

```


```python
s1=random.choice(list(df_pca[df_pca["Class"] == 6]["Time_Id"]))
s2=random.choice(list(df_pca[df_pca["Class"] == 6]["Time_Id"]))
```


```python
sample_1=df_pca[(df_pca["Class"] == 6) & (df_pca["Time_Id"] ==s1)].iloc[:,2]
sample_2=df_pca[(df_pca["Class"] == 6) & (df_pca["Time_Id"] ==s2)].iloc[:,2]
```


```python
plt.plot(sample_1,color = "g")
plt.plot(sample_2,color = "b")
plt.legend(["S1", "S2"])
plt.show()
```


![png](output_76_0.png)



```python

```


```python
s1=random.choice(list(df_pca[df_pca["Class"] == 7]["Time_Id"]))
s2=random.choice(list(df_pca[df_pca["Class"] == 7]["Time_Id"]))
```


```python
sample_1=df_pca[(df_pca["Class"] == 7) & (df_pca["Time_Id"] ==s1)].iloc[:,2]
sample_2=df_pca[(df_pca["Class"] == 7) & (df_pca["Time_Id"] ==s2)].iloc[:,2]
```


```python
plt.plot(sample_1,color = "g")
plt.plot(sample_2,color = "b")
plt.legend(["S1", "S2"])
plt.show()
```


![png](output_80_0.png)



```python

```


```python
s1=random.choice(list(df_pca[df_pca["Class"] == 8]["Time_Id"]))
s2=random.choice(list(df_pca[df_pca["Class"] == 8]["Time_Id"]))
```


```python
sample_1=df_pca[(df_pca["Class"] == 8) & (df_pca["Time_Id"] ==s1)].iloc[:,2]
sample_2=df_pca[(df_pca["Class"] == 8) & (df_pca["Time_Id"] ==s2)].iloc[:,2]
```


```python
plt.plot(sample_1,color = "g")
plt.plot(sample_2,color = "b")
plt.legend(["S1", "S2"])
plt.show()
```


![png](output_84_0.png)



```python
# I repeated producing time series from random two sample for each classes (8 times)
#We can say that time series of first components are similar for classes 1,2,3,5,8.
```


```python
#Question C
```


```python
dimensions=["X","Y","Z"]
pca_data=long_format.loc[:,dimensions].values
```


```python
pca= PCA()
pca_fit=pca.fit_transform(pca_data) 
```


```python
explained_variance = pca.explained_variance_ratio_ 
cum_explained_variance=np.cumsum(explained_variance)
print(cum_explained_variance)
plt.plot(cum_explained_variance)  
plt.grid() 
plt.xticks((0,1,2))         # From the graph we can understand rhat with two components we can explain over 80% of variance.
plt.xlabel('n_component') 
plt.ylabel('Cumulative_Variance_explained')  
plt.show()
```

    [0.49059498 0.83729864 1.        ]
    


![png](output_89_1.png)



```python
pca_values_list = []

for i in range(1,long_format["Class"].nunique()+1):
    
    k= long_format[long_format["Class"] == i]   #I applied PCA for each class and calculated their pca components and 
                                                # explained variance ratio.
    pca = PCA(n_components=1)
    pca.fit(k[["X","Y","Z"]])
    print("\ PCA for the class {} :".format(i))
    print("First component: ",pca.components_)
    print("Explained variance ratio: ",pca.explained_variance_ratio_)
    
    pca_values_list.append(pca.components_)
```

    \ PCA for the class 1 :
    First component:  [[0.35679434 0.6911427  0.62850582]]
    Explained variance ratio:  [0.46253306]
    \ PCA for the class 2 :
    First component:  [[0.45541761 0.68455467 0.56919215]]
    Explained variance ratio:  [0.51254787]
    \ PCA for the class 3 :
    First component:  [[ 0.67472328  0.53068805 -0.51295097]]
    Explained variance ratio:  [0.54074403]
    \ PCA for the class 4 :
    First component:  [[ 0.68063228  0.63420651 -0.36677214]]
    Explained variance ratio:  [0.55012869]
    \ PCA for the class 5 :
    First component:  [[0.39887719 0.64259    0.65419804]]
    Explained variance ratio:  [0.6472234]
    \ PCA for the class 6 :
    First component:  [[-0.2060269   0.67972303  0.70393858]]
    Explained variance ratio:  [0.57182387]
    \ PCA for the class 7 :
    First component:  [[0.22902999 0.71502583 0.66051746]]
    Explained variance ratio:  [0.5192033]
    \ PCA for the class 8 :
    First component:  [[0.57443435 0.69279232 0.43596328]]
    Explained variance ratio:  [0.61342685]
    


```python
# I can say that value of components of 1 and 5 are close to each other. 3 and 4 is also close to each other 
#in the way of first component value. The first component values of 7 and 8 is close to each other as well.
#We can consider that pattern of these class are similar. Also class 5 and 8 eight is responsible for relatively higher explained variance
# which means they are relatively more meaningful than other classes but difference is not significant with other classes.
```


```python
#Question D
```


```python
df_merged = pd.concat([X,Y.iloc[:,1:],Z.iloc[:,1:]],axis=1) #Merged class and components to calculate distance matrix.
```


```python
df_merged
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
      <th>306</th>
      <th>307</th>
      <th>308</th>
      <th>309</th>
      <th>310</th>
      <th>311</th>
      <th>312</th>
      <th>313</th>
      <th>314</th>
      <th>315</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>-0.304243</td>
      <td>-0.304243</td>
      <td>-0.304243</td>
      <td>-0.304243</td>
      <td>-0.304243</td>
      <td>-0.304243</td>
      <td>-0.304243</td>
      <td>-0.304243</td>
      <td>-0.304243</td>
      <td>...</td>
      <td>0.523217</td>
      <td>0.513994</td>
      <td>0.503481</td>
      <td>0.492967</td>
      <td>0.474522</td>
      <td>0.456077</td>
      <td>0.437632</td>
      <td>0.419187</td>
      <td>0.400743</td>
      <td>0.382298</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>1.627311</td>
      <td>1.627311</td>
      <td>1.627311</td>
      <td>1.627311</td>
      <td>1.627311</td>
      <td>1.627311</td>
      <td>1.627311</td>
      <td>1.627311</td>
      <td>1.627311</td>
      <td>...</td>
      <td>-0.427010</td>
      <td>-0.427010</td>
      <td>-0.427010</td>
      <td>-0.427172</td>
      <td>-0.428773</td>
      <td>-0.440720</td>
      <td>-0.452667</td>
      <td>-0.464613</td>
      <td>-0.476560</td>
      <td>-0.488507</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>0.661277</td>
      <td>0.661277</td>
      <td>0.661277</td>
      <td>0.661277</td>
      <td>0.661277</td>
      <td>0.661277</td>
      <td>0.661277</td>
      <td>0.661277</td>
      <td>0.661277</td>
      <td>...</td>
      <td>-0.862717</td>
      <td>-0.862717</td>
      <td>-0.862717</td>
      <td>-0.862717</td>
      <td>-0.862717</td>
      <td>-0.862717</td>
      <td>-0.862717</td>
      <td>-0.862717</td>
      <td>-0.862717</td>
      <td>-0.862717</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0.005185</td>
      <td>0.005185</td>
      <td>0.005185</td>
      <td>0.005185</td>
      <td>0.005185</td>
      <td>0.005185</td>
      <td>0.005185</td>
      <td>0.005185</td>
      <td>0.005185</td>
      <td>...</td>
      <td>-0.187384</td>
      <td>-0.123549</td>
      <td>-0.055870</td>
      <td>0.011808</td>
      <td>0.079487</td>
      <td>0.157056</td>
      <td>0.253740</td>
      <td>0.445503</td>
      <td>0.648538</td>
      <td>0.851573</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>1.286198</td>
      <td>1.286198</td>
      <td>1.286198</td>
      <td>1.286198</td>
      <td>1.286198</td>
      <td>1.286198</td>
      <td>1.286198</td>
      <td>1.286198</td>
      <td>1.286198</td>
      <td>...</td>
      <td>1.867472</td>
      <td>1.834319</td>
      <td>1.756000</td>
      <td>1.638582</td>
      <td>1.521164</td>
      <td>1.453266</td>
      <td>1.515219</td>
      <td>1.632637</td>
      <td>1.750054</td>
      <td>1.867472</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>891</th>
      <td>3.0</td>
      <td>0.117811</td>
      <td>0.117811</td>
      <td>0.117811</td>
      <td>0.117811</td>
      <td>0.117811</td>
      <td>0.117811</td>
      <td>0.117811</td>
      <td>0.117811</td>
      <td>0.117811</td>
      <td>...</td>
      <td>-0.212484</td>
      <td>-0.137582</td>
      <td>-0.066376</td>
      <td>-0.010642</td>
      <td>0.026809</td>
      <td>0.064260</td>
      <td>0.101712</td>
      <td>0.139163</td>
      <td>0.176614</td>
      <td>0.214065</td>
    </tr>
    <tr>
      <th>892</th>
      <td>1.0</td>
      <td>-1.232590</td>
      <td>-1.232590</td>
      <td>-1.232590</td>
      <td>-1.232590</td>
      <td>-1.232590</td>
      <td>-1.232590</td>
      <td>-1.232590</td>
      <td>-1.232590</td>
      <td>-1.232590</td>
      <td>...</td>
      <td>0.222775</td>
      <td>0.234980</td>
      <td>0.247184</td>
      <td>0.259389</td>
      <td>0.271593</td>
      <td>0.283798</td>
      <td>0.296003</td>
      <td>0.308207</td>
      <td>0.320412</td>
      <td>0.332617</td>
    </tr>
    <tr>
      <th>893</th>
      <td>2.0</td>
      <td>0.282877</td>
      <td>0.282877</td>
      <td>0.282877</td>
      <td>0.282877</td>
      <td>0.282877</td>
      <td>0.282877</td>
      <td>0.282877</td>
      <td>0.282877</td>
      <td>0.282877</td>
      <td>...</td>
      <td>-0.044573</td>
      <td>-0.043387</td>
      <td>-0.038164</td>
      <td>0.000695</td>
      <td>0.027133</td>
      <td>0.039175</td>
      <td>0.088921</td>
      <td>0.158608</td>
      <td>0.223306</td>
      <td>0.267828</td>
    </tr>
    <tr>
      <th>894</th>
      <td>7.0</td>
      <td>1.248704</td>
      <td>1.248704</td>
      <td>1.248704</td>
      <td>1.248704</td>
      <td>1.248704</td>
      <td>1.248704</td>
      <td>1.248704</td>
      <td>1.248704</td>
      <td>1.248704</td>
      <td>...</td>
      <td>-0.160172</td>
      <td>-0.162960</td>
      <td>-0.190327</td>
      <td>-0.296460</td>
      <td>-0.407102</td>
      <td>-0.504592</td>
      <td>-0.559242</td>
      <td>-0.549343</td>
      <td>-0.505912</td>
      <td>-0.433044</td>
    </tr>
    <tr>
      <th>895</th>
      <td>5.0</td>
      <td>-0.168518</td>
      <td>-0.168518</td>
      <td>-0.168518</td>
      <td>-0.168518</td>
      <td>-0.168518</td>
      <td>-0.168518</td>
      <td>-0.168518</td>
      <td>-0.168518</td>
      <td>-0.168518</td>
      <td>...</td>
      <td>-0.875563</td>
      <td>-0.875563</td>
      <td>-0.874925</td>
      <td>-0.869817</td>
      <td>-0.859210</td>
      <td>-0.848602</td>
      <td>-0.828104</td>
      <td>-0.800426</td>
      <td>-0.768603</td>
      <td>-0.736781</td>
    </tr>
  </tbody>
</table>
<p>896 rows × 946 columns</p>
</div>




```python
scaled=StandardScaler().fit_transform(df_merged.iloc[:,1:])  #Applied multi-dimensional scaling 
```


```python
distance=distance_matrix(scaled,scaled) #Obtained symmetric distance matrix
```


```python
 MDS_features= MDS(n_components=2,).fit_transform(distance)  #Scaling was applied in 2D space.
```

    C:\Users\lenovo\anaconda3\lib\site-packages\sklearn\manifold\_mds.py:419: UserWarning: The MDS API has changed. ``fit`` now constructs an dissimilarity matrix from data. To use a custom dissimilarity matrix, set ``dissimilarity='precomputed'``.
      warnings.warn("The MDS API has changed. ``fit`` now constructs an"
    


```python
 MDS_features_df=pd.DataFrame(MDS_features)
```


```python
MDS_features_df.insert(0,"Class",df_merged.iloc[:,0])  #class information was joined
```


```python
MDS_features_df.columns=["Class","0","1"]
```


```python
sns.lmplot( x='0', y='1', data=MDS_features_df, fit_reg=False, hue='Class', legend='full');
```


![png](output_101_0.png)



```python
# I can conclude on this that it seems very hard to distinguish class 1 and 6. It is very good visualization to see
# the behaviours of classes and determine their relations. Class 2 and 7 is also very close classes in terms of seperability.
#But class 3 and 4 , 5 and 6 and 7 and 8 are distunguished in terms of seperability, respectively.
#Overall we can state that relation of classes in the visualization also valid for real gestures.
```
