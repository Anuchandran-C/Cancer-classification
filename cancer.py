import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
df =pd.read_csv("C:\\Users\\HP\\Desktop\\My Python\\Project\\breast-cancer-wisconsin-data\\data.csv")
#print(df.info())
print(df.columns)
x =df.drop(["id","diagnosis"],axis=1)
y =df["diagnosis"]
df["diagnosis"] = pd.get_dummies(df["diagnosis"],drop_first=True)
print(df.head())