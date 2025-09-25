import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Linear_model import MyLinearModel
from sklearn.preprocessing import PolynomialFeatures,StandardScaler


#x = pd.DataFrame(np.linspace(0,10,50))
#print(x)
#y = pd.DataFrame(2 * x + 3 + np.random.normal(0,3,size=x.shape))
np.random.seed(0)

x = pd.DataFrame(np.linspace(-3, 3, 50).reshape(-1, 1))

# nonlinear relationship: quadratic
#y = pd.DataFrame(3 * x**4+ 0.5 * x**2 + 2 * x + 3 + np.random.normal(0, 3, size=(100,1)))
y = pd.DataFrame(x**2 + 2 * x + 3 + np.random.normal(0, 2, size=(50,1)))

#print(y)
mymodel = MyLinearModel()
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x))
mymodel.fit(x,y,initial_start=np.zeros((x.shape[1]+1,1)))
y_predict = mymodel.predict(x)
plt.scatter(x,y)
#print(y_predict)
plt.plot(x, y_predict, color='red')
print("MSE of linear model : ",mymodel.MSE(y,y_predict))
print("variance of linear model : ",np.var(y_predict))
#plt.show()
print("*"*100)
mse = []
var = []
degrees = list(range(1, 7))  # 1st to 9th degree
for i in range(1,7):
    poly = PolynomialFeatures(degree=i,include_bias=False)
    x_poly = pd.DataFrame(poly.fit_transform(x))
    #scaler = StandardScaler()
    #x_poly = pd.DataFrame(scaler.fit_transform(x_poly))
    mymodel = MyLinearModel()
    mymodel.fit(x_poly,y,initial_start=np.zeros((x_poly.shape[1]+1,1)))
    y_poly_predict = mymodel.predict(x_poly)
    print(f"MSE of {i}th degree model : ",mymodel.MSE(y,y_poly_predict))
    print(f"variance of {i}th degree model : ",np.mean(y_poly_predict.var(axis=0)))
    mse.append(mymodel.MSE(y,y_poly_predict))
    var.append(np.mean(y_poly_predict.var(axis=0)))



plt.figure(figsize=(8,5))
plt.plot(degrees, mse, marker='o', label="MSE", color="blue")
plt.plot(degrees, var, marker='s', label="Variance", color="red")

plt.xlabel("Polynomial Degree")
plt.ylabel("Error")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.grid(True)
plt.show()