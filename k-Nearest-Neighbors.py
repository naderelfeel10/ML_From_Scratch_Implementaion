import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification



class MyKNN:
    def __init__(self,k):
        self.k = k

    def get_k_nearest(self,point,data,y):
        y = np.array(y)
        distances = []
        for p in data:
            distances.append(np.sum((point - p)**2))
        min_k_points = np.argsort(distances)[:self.k]
        return y[min_k_points]
    
    
    def predict(self,points,data,y):
        predictions = []
        for p in points:
            nearest_points = self.get_k_nearest(p,data,y)
            counter = Counter(nearest_points)

            most_common = counter.most_common(1)[0][0]
            predictions.append(most_common)


        return np.array(predictions)




if __name__ == '__main__':
    data = np.array([[1,2],[2,3],[3,4],[6,7],[7,8]])
    y = np.array([1,0,1,0,1])
    point = np.array([[2,2],[2,3]])


    model = MyKNN(k=3)
    knn_points = model.get_k_nearest(point, data,y)

    print("Nearest points:", knn_points)

    predictions  = model.predict(point,data,y)
    print(predictions)


    X, y = make_classification(
        n_samples=200,
        n_features=2,    
        n_informative=2,   
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    knn = MyKNN(k=5) 

    y_pred = knn.predict(X_test,X_train,y_train)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()], X_train, y_train)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title("KNN Classification (k=5)")
    plt.show()