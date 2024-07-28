import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#データ読み込み
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")
gender_submission = pd.read_csv("dataset/gender_submission.csv")

#特徴量エンジニアリング
#train
t_train = train["Survived"]
train["FamilySize"] = train["SibSp"] + train["Parch"] 
x_train = train[["Pclass","Sex","Age","FamilySize","Fare","Embarked"]]
x_train["Sex"].replace(["male","female"],[0,1],inplace=True)
# Embarkedをonehot変換
Embarked = pd.get_dummies(x_train["Embarked"])
Embarked.replace([False,True],[0,1],inplace=True)
x_train = x_train.drop(columns=['Embarked'])
x_train = pd.concat([x_train,Embarked],axis = 1)
x_train.fillna(x_train.mean(), inplace=True)
#正規化
x_train = (x_train - x_train.mean()) / x_train.std()
x_train
#test
test["FamilySize"] = test["SibSp"] + test["Parch"] 
x_test = test[["Pclass","Sex","Age","FamilySize","Fare","Embarked"]]
x_test["Sex"].replace(["male","female"],[0,1],inplace=True)
# Embarkedをonehot変換
Embarked = pd.get_dummies(x_test["Embarked"])
Embarked.replace([False,True],[0,1],inplace=True)
x_test = x_test.drop(columns=['Embarked'])
x_test = pd.concat([x_test,Embarked],axis = 1)
x_test.fillna(x_test.mean(), inplace=True)
#正規化
x_test = (x_test - x_test.mean()) / x_test.std()
x_test