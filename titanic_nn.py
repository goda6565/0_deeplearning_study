import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

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

#NNを作成する
Model = Sequential()
Model.add(Dense(100, input_dim=8))
Model.add(Activation("relu"))
Model.add(Dropout(0.15))
Model.add(Dense(100))
Model.add(Activation("relu"))
Model.add(Dropout(0.5))
Model.add(Dense(100))
Model.add(Activation("relu"))
Model.add(Dropout(0.5))
Model.add(Dense(100))
Model.add(Activation("relu"))
Model.add(Dropout(0.5))
Model.add(Dense(2))
Model.add(Activation("softmax"))
Model.compile(optimizer = "adam", loss = "categorical_crossentropy")
t_train = to_categorical(t_train, num_classes=2)
Model.fit(x_train,t_train,epochs=50)
t_test = Model.predict(x_test)
t_test = np.argmax(t_test, axis = 1)
x_test["Survived"] = t_test
t_output = x_test["Survived"]
t_output.index = t_output.index + 892
t_output.index.rename('PassengerId', inplace=True)
# DataFrameをCSVに出力
t_output.to_csv("output.csv")