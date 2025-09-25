import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score,precision_score,recall_score
from sklearn.metrics import log_loss, accuracy_score
from Linear_model import MyLinearModel
from sklearn.preprocessing import StandardScaler

def MylogLoss(y,y_predict):
        n = len(y)
        eps = 1e-15
        y_predict = np.clip(y_predict, eps, 1 - eps)
        y_predict = y_predict.flatten()
        return -(1/n) * np.sum(
            y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)
        )

def my_accuracy_score(y_true,y_predict):
    #print(y_true.shape)
    #print(len(y_true))
    #print(y_predict.shape)
    y_predict = y_predict.flatten()
    return  1/len(y_true) * np.sum(np.logical_not(np.bitwise_xor(y_true,y_predict)).astype(int)) 
     

def confusion_elements(y_true, y_pred):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, FP, TN, FN

def precession(y_true,y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP/(TP+FP)

def recall(y_true,y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP/(TP+FN)

def F1_score(y_true,y_pred):
    precession_ = precession(y_true,y_pred)
    recall_ = recall(y_true,y_pred)
    return 2/ (1/recall_ + 1/precession_)


def sigmoid(predictions):
    return 1 / (1 + np.exp(-predictions))


class MyLogisticRegression:
    def __init__(self,lr=0.01,max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter

    def fit(self,X,y,initial_start):
        self.X = pd.concat([pd.DataFrame(np.ones((X.shape[0],1))),pd.DataFrame(X)],axis=1) # i added first col as ones instead of bias (in dot product)
        self.y = pd.DataFrame(y)
        self.weights = initial_start
        for i in range(self.max_iter):
            y_pred = np.dot(self.X,self.weights)
            y_pred = sigmoid(y_pred)
            gradients = 1/self.X.shape[0] * np.dot(self.X.T,(y_pred - self.y))
            self.weights = self.weights - self.lr*gradients

    def predict_probabiliteis(self,X):
        #predictions = self.lr.predict(X)
        #return 1 / (1 + np.exp(-predictions))
        y_pred = np.dot(pd.concat([pd.DataFrame(np.ones((X.shape[0],1))),pd.DataFrame(X)],axis=1),self.weights)
        y_pred = sigmoid(y_pred)
        return y_pred
    
    def predict(self,X,threshold=0.5):
        return (self.predict_probabiliteis(X) >= threshold).astype(int)
    



if __name__ == '__main__':


    X, y = make_classification(n_samples=200, n_features=2, 
                               n_redundant=0, n_informative=2,
                               random_state=42, n_clusters_per_class=1)


    lg = MyLogisticRegression()
    lg.fit(X,y,initial_start=np.zeros((X.shape[1]+1,1)))
    probs = lg.predict_probabiliteis(X)
    #print(probs)
    y_predict = lg.predict(X)
    print(y_predict.shape)
    logLoss = MylogLoss(y, probs)
    print("My log loss:", logLoss)
    print("My accuracy score : ", my_accuracy_score(y,y_predict))

    TP, FP, TN, FN = confusion_elements(y, y_predict)
    print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
    print("Precision:", precession(y, y_predict))
    print("Recall:", recall(y, y_predict))
    print("f1-score :", F1_score(y, y_predict))


    # 5. Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid = np.c_[xx.ravel(), yy.ravel()]  
    Z = lg.predict(grid)                
    Z = Z.reshape(xx.shape)    

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title("My Logistic Regression ")


    model = LogisticRegression()
    model.fit(X, y)
    y_probs = model.predict_proba(X)[:, 1]   
    y_preds = model.predict(X)               


    print("Sklearn log loss:", log_loss(y, y_probs))
    print("sklearn accuracy score : ", accuracy_score(y,y_preds))
    TP, FP, TN, FN = confusion_elements(y, y_preds)
    print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
    print("Precision:", precision_score(y, y_preds))
    print("Recall :", recall_score(y, y_preds))
    print("f1 score :", f1_score(y, y_preds))


    # 5. Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title("sklearn Logistic Regression ")
    plt.show()
