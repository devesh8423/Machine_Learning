import simple_LR_oops as LR
import pandas as pd
import numpy as np
from sklearn .model_selection import train_test_split
df=pd.read_excel("placement.xlsx")
# x=df["cgpa"]
x=df.iloc[:,0].values
y=df.iloc[:,1].values
# y=df["package"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LR.Meralr()
lr.fit(x_train,y_train)
y_pred_train=lr.prediction(x_train)
y_pred_test=lr.prediction(x_test)
# print(y_pred_train[:3])
# print(y_pred_test[:3])
lr.r2(x_test,y_pred_test,y_test)
lr.r2(x_train,y_pred_train,y_train)