import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

#getting data and storing in dataset variable df using pandas
df =pd.read_csv('data.csv')
#print(df.info())
#print(df.columns)

#getting all the feature colums in x
x =df.drop(["id","diagnosis"],axis=1)

#replacing missing values with mean

imputer = Imputer(missing_values = 'NaN',strategy= 'mean' , axis = 0)
imputer = imputer.fit(x)
x = imputer.transform(x)
#gettng the result column in y
y =df["diagnosis"]

#converting categorical data into indicator/dummy variable
y = pd.get_dummies(df["diagnosis"],drop_first=True)
#print(df.head())


x1,x2,y1,y2 = train_test_split(x,y,test_size = 0.3)

# =============================================================================
model = KNeighborsClassifier(n_neighbors=10)
model.fit(x1,y1)
pre = model.predict(x2)
print (model.score(x2,y2))
print (classification_report(y2,pre))
print (confusion_matrix(y2,pre))