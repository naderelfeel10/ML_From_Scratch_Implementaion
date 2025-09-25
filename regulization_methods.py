import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from Linear_model import MyLinearModel
from sklearn.preprocessing import PolynomialFeatures,StandardScaler



class MyRegularizationRegressor:
    def __init__(self,method='ridge',lr=0.01,lamda=0.1,max_iter=1000):
        self.lr = lr
        self.lamda = lamda
        self.max_iter = max_iter
        self.method = method
    def fit(self,X,y,initial_start):
        self.X = pd.concat([pd.DataFrame(np.ones(X.shape[0])),X],axis=1)  # i added first col as ones instead of bias (in dot product)
        self.y = y
        self.weights = initial_start
        #print(self.X)
        for i in range(self.max_iter):
            y_pred = np.dot(self.X,self.weights)
            if self.method == 'ridge':
                penalty = 2*self.lamda *  self.weights
            elif self.method == 'lasso':
                penalty = 2*self.lamda *  np.sign(self.weights)
            else :
                penalty = 0

            ridge_gradients = -1/self.X.shape[0] * np.dot(self.X.T,(self.y - y_pred)) + penalty
            self.weights = self.weights - self.lr*ridge_gradients

    def predict(self,X):
        X = pd.concat([pd.DataFrame(np.ones(X.shape[0])),X],axis=1)  
        return np.dot(X,self.weights)
    
    def MSE(self, y_true, y_predict):
        y_true = np.array(y_true).flatten()
        y_predict = np.array(y_predict).flatten()
        n = len(y_predict)
        return np.sum((y_true - y_predict) ** 2) / n


if __name__ == '__main__':
    
    x = pd.DataFrame(np.linspace(0,10,50))
    #print(x)
    y = pd.DataFrame(x**2 + 2 * x + 3 + np.random.normal(0, 6, size=(50,1)))
    #print(y)
    mymodel = MyRegularizationRegressor(method='ridge')
    mymodel.fit(x,y,initial_start=np.zeros((x.shape[1]+1,1)))
    y_ridge = mymodel.predict(x)
    plt.scatter(x,y)
    #print(y_predict)
    plt.plot(x, y_ridge, color='red')
    print("MSE : ",mymodel.MSE(y,y_ridge))
    print("variance of ridge : ",np.mean(y_ridge.var(axis=0)))
    
    
    degrees = list(range(1, 6))  
    
    mse_linear, var_linear = [], []
    mse_ridge, var_ridge = [], []
    mse_lasso, var_lasso = [], []
    
    for i in degrees:
        poly = PolynomialFeatures(degree=i, include_bias=False)
        x_poly = pd.DataFrame(poly.fit_transform(x))
        scaler = StandardScaler()
        #scaler = MinMaxScaler()
        x_poly = pd.DataFrame(scaler.fit_transform(x_poly))
    
        model = MyLinearModel()
        model.fit(x_poly, y, initial_start=np.zeros((x_poly.shape[1]+1,1)))
        y_pred = model.predict(x_poly)
        mse_linear.append(model.MSE(y, y_pred))
    
        var_linear.append(np.mean(y_pred.var(axis=0)))
    
    
        model = MyRegularizationRegressor(method='ridge')
        model.fit(x_poly, y, initial_start=np.zeros((x_poly.shape[1]+1,1)))
        y_pred = model.predict(x_poly)
        mse_ridge.append(model.MSE(y, y_pred))
    
        var_ridge.append(np.mean(y_pred.var(axis=0)))
    
    
    
        model = MyRegularizationRegressor(method='lasso')
        model.fit(x_poly, y, initial_start=np.zeros((x_poly.shape[1]+1,1)))
        y_pred = model.predict(x_poly)
        mse_lasso.append(model.MSE(y, y_pred))
        var_lasso.append(np.mean(y_pred.var(axis=0)))
    
    
    
    
    
    plt.figure(figsize=(12,8))
    
    
    # MSE 
    plt.subplot(2,1,1)
    plt.plot(degrees, mse_linear, marker='o', color='blue', label="Linear MSE")
    plt.plot(degrees, mse_ridge, marker='s', color='red', label="Ridge MSE")
    
    
    plt.plot(degrees, mse_lasso, marker='^', color='green', label="Lasso MSE")
    plt.title("MSE Comparison: Linear vs Ridge vs Lasso")
    plt.xlabel("Polynomial Degree")
    
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    
    
    
    
    # variances 
    plt.subplot(2,1,2)
    
    plt.plot(degrees, var_linear, marker='o', color='blue', label="Linear Variance")
    plt.plot(degrees, var_ridge, marker='s', color='red', label="Ridge Variance")
    
    plt.plot(degrees, var_lasso, marker='^', color='green', label="Lasso Variance")
    plt.title("Variance Comparison: Linear vs Ridge vs Lasso")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Variance")
    plt.legend()
    
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()