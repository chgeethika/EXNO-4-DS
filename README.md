# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![o 1](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/7fec46a6-ccad-49b5-ad57-4469be3ee5d7)
![o 2](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/7e00759a-91ad-4da4-a1a8-1ce376081591)

```
df.dropna()
```
![o 3](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/27f99045-cb94-4bc5-b273-67e61e1b7331)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![o 4](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/09e44516-ab72-49d2-b095-e5516c6b585e)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![o 5](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/8a42bb20-731a-4318-9b00-897f65c1a39b)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![o 6](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/673f3e87-6236-4a80-8749-78ef59affe06)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![o 7](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/c1ea80d1-cae6-497e-8e5e-e8cf3dd85cd2)

```
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
df1
```
![o 8](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/84d1293b-ff99-4f30-bbe4-9ade58ff9995)

```
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2.head()
```
![o 9](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/901d2619-ef1e-46bb-86c4-36ffb88643f2)


```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![o 10](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/f77e9573-50ee-4640-9b7e-d3341488313c)



```
data.isnull().sum()
```
![o 11](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/3228889c-09f7-474b-87e4-58259192a413)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![12](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/82b67abd-0ef7-48c5-a8f4-ba44380d9dc0)

```
data2 = data.dropna(axis=0)
data2
```
![13](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/bca8e2ea-7da2-49ab-9ab7-4b6f189699a4)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![14](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/df026888-8a42-4b91-93c6-121668b876b2)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![15](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/abefff60-09e3-4d5f-879e-499ba56b5ff4)

```
data2
```
![16](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/02f0af7a-7405-45e3-9866-c9785e9401a9)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![17](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/29ce027d-a91a-4500-8264-95883e9fdcb5)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![18](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/867a9e38-42aa-4512-918b-a492af2586c8)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![19](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/e6eabe48-b831-4516-a1dc-e7a2fdbe235c)

```
y=new_data['SalStat'].values
print(y)
```
![20](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/729a84b6-6dba-4a3b-8550-426237358e97)

```
x = new_data[features].values
print(x)
```
![21](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/1219e14d-17a6-405c-aa20-3acdb114f365)

```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![22](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/bbdc10b4-11d2-4c73-998c-442c0f350aaf)

```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![23](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/5269a749-8f7f-4a50-8463-afb5b56a9c1c)

```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
![24](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/fdfac49a-b648-40d4-ba63-eaba6fc883ba)

```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
![25](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/978455cf-7630-4773-88e9-5884a90afac5)

```
data.shape
```
![26](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/5f9a5548-37d4-4ce8-83db-6b865b2f677f)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![27](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/e5cca7f3-0999-4509-8497-593589bad1d6)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![28](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/b5a8286a-eed7-4fa7-81db-e8ce72a5e7b7)

```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![29](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/4d67bf80-859a-43a9-be68-71903f1a93d6)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![30](https://github.com/chgeethika/EXNO-4-DS/assets/142209368/d26a4b7f-da08-4be4-99ab-963dbbe3a0bf)



















# RESULT:
       # INCLUDE YOUR RESULT HERE
