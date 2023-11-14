# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necssary Libraries.
2. Read the File using Pandas.
3. Implement Basic Operatoins(df.head(),df.info())
4. Implement Label encoding.
5. Implement Decision Tree Regressor.
6. Evaluate Accuracy.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: B VIJAY KUMAR
RegisterNumber:  212222230173
*/
```
##### READ THE FILE:
```
import pandas as pd
df = pd.read_csv('Salary.csv')
```
##### INFO AND HEAD
```
df.head()
df.info()
```
##### NULL DETECTION
```
df.isnull().sum()
```
##### IMPLEMENT LABEL ENCODING
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Position'] = le.fit_transform(df['Position'])
```
##### SPLIT THE DATA INTO TRAINING AND TESTING SETS
```
x = df[['Position','Level']]
x.head()
y = df[['Salary']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
##### IMPLEMENT DECISION TREE REGRESSOR
```
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
```
##### ERROR DETECTION
```
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
print(mse)
r2 = r2_score(y_test, y_pred)
print(f'r2 score = {r2}')
```
##### PREDICT WITH NEW DATA
```
dt.predict([[5,6]])
```
##### 
## Output:
##### READ THE FILE:
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119657657/90d9b3d4-e77a-4b22-b8e6-107ab51ea5d3)


##### HEAD 

![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119657657/7edd9576-4e4d-4b46-a2dd-c5d5ca9c9cf4)


##### INFO
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119657657/bc4f1d27-3c9d-4271-bc14-a765c997da1c)

##### NULL DETECTION
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119657657/62a26173-d57f-4687-a2f8-9486298522ed)

##### IMPLEMENT LABEL ENCODING
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119657657/98904c2c-d949-4f37-938c-36469269a613)

##### ERROR DETECTION
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119657657/881b5dff-86c0-4e04-ae20-50302d67b6f2)

##### PREDICT WITH NEW DATA

![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119657657/e9f3f14c-4d68-4ac6-bdad-ab8d89e2da92)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
