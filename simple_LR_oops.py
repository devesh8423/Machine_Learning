# Install the actual library:
# (Optional) Replace “sklearn” with “scikit-learn” in your requirements files if needed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Meralr:
    def __init__(self):
        self.m=None
        self.b=None

    def fit(self,x_train,y_train):
        num=0
        den=0
        for i in range(x_train.shape[0]):
            num=num+(x_train[i]-x_train.mean())*(y_train[i]-y_train.mean())
            den=den+(x_train[i]-x_train.mean())*+(x_train[i]-x_train.mean())
        self.m=num/den
        self.b=y_train.mean()-(self.m*x_train.mean())
        print("Coef:",self.m)
        print("Intercept:",self.b)


        
    def prediction(self,x_test):
        z=self.m*x_test+self.b
        return z
    
    def r2(self,x_test,y_test,y_pred_test):
        l=[]
        residual=y_test-y_pred_test
        residual_sq=residual**2
        rss=residual_sq.sum()
        # print(rss)
        y_mean=y_test.mean()
        for i in y_test:
            l.append((i-y_mean)**2)
        tss=sum(l)
        # print(tss)
        r2_score=1-(rss/tss)
        print("r2_score:",r2_score)
if __name__=="__main__":
    df=pd.read_excel("placement.xlsx")
    x=df.iloc[:,0].values
    y=df.iloc[:,1].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    lr=Meralr()
    lr.fit(x_train,y_train)
    y_pred_test=lr.prediction(x_test)
    lr.r2(x_test,y_test,y_pred_test)