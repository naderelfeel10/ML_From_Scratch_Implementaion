import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error


def gradient_descent(x,y,initial_start=[0,0],lr=0.01,max_iter=1000):
    n = x.shape[0]
    for i in range(max_iter):
        y_pred = initial_start[1]*x + initial_start[0]
        cost = 1/n * np.sum((y - y_pred)**2)
        md = -2/n * np.sum((y - y_pred)*x)
        bd = -2/n * np.sum(y-y_pred)
        initial_start[1] = initial_start[1] - lr*md
        initial_start[0] = initial_start[0] - lr*bd
        print(initial_start,cost)

#gradient_descent(np.array([1,2,3,4,5]),np.array([5,7,9,11,13]))


class MyLinearModel:
    def __init__(self,lr=0.01,max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
    def fit(self,X,y,initial_start):
        self.X = pd.concat([pd.DataFrame(np.ones((X.shape[0],1))),pd.DataFrame(X)],axis=1) # i added first col as ones instead of bias (in dot product)
        self.y = pd.DataFrame(y)
        self.weights = initial_start
        #print(self.X)
        for i in range(self.max_iter):
            y_pred = np.dot(self.X,self.weights)
            gradients = -1/self.X.shape[0] * np.dot(self.X.T,(self.y - y_pred))
            self.weights = self.weights - self.lr*gradients

    def predict(self,X):
        X = pd.concat([pd.DataFrame(np.ones(X.shape[0])),pd.DataFrame(X)],axis=1)  # i added first col as ones instead of bias (in dot product)
        return np.dot(X,self.weights)
    
    def MSE(self, y_true, y_predict):
        y_true = np.array(y_true).flatten()
        y_predict = np.array(y_predict).flatten()
        n = len(y_predict)
        return np.sum((y_true - y_predict) ** 2) / n



if __name__ == '__main__':

    x = pd.DataFrame(np.linspace(0,10,50))
    #print(x)
    y = pd.DataFrame(2 * x + 3 + np.random.normal(0,3,size=x.shape))
    #print(y)
    mymodel = MyLinearModel()
    mymodel.fit(x,y,initial_start=np.zeros((x.shape[1]+1,1)))
    y_predict = mymodel.predict(x)
    print(y_predict)
    plt.scatter(x,y)
    #print(y_predict)
    plt.plot(x, y_predict, color='red')
    print("MSE : ",mymodel.MSE(y,y_predict))
    plt.show()
