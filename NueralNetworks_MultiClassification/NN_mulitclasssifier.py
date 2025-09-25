import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from NN_functions import soft_max_batch,dtanh,cross_entropy_batch,feed_forward,backprobagation


mnist = fetch_openml("mnist_784")
X, y = mnist["data"].to_numpy(), mnist["target"].astype(int).to_numpy()

random_indices = np.random.choice(X.shape[0], size=5000, replace=True)
X = X[random_indices]
y = y[random_indices]
np.save('X.npy', X)
np.save('y.npy', y)

np.random.seed(0)


def load_data():
    X = np.load('X.npy')
    y = np.load('y.npy')

    # Normalize data
    X = X / 255.0

    return X, y



class NeuralNetworkMultiClassifier:
    def __init__(self, input_dim, hidden_dims,output_dim):
        self.W = []
        self.B = []
        self.W.append(np.random.randn(input_dim, hidden_dims[0]))
        self.B.append(np.zeros((1, hidden_dims[0])))
        #hidden_dims = hidden_dims[1:]

        for i in range(1,len(hidden_dims)):
            print("###",i)
            self.W.append(np.random.randn(hidden_dims[i-1], hidden_dims[i]))
            self.B.append(np.zeros((1, hidden_dims[i])))

        self.W.append(np.random.randn(hidden_dims[-1], output_dim))
        self.B.append(np.zeros((1, output_dim)))
        self.B = self.B[::-1]
        self.B = self.B[::-1]


    def updata_weights(self,dW,dB,lr):
        for i in range(0,len(dW)):
            self.W[i] -= lr*dW[i]
            self.B[i] -= lr*dB[i] 


    def train(self, X_train, y_train, X_test, y_test, learning_rate = 1e-2, n_epochs = 20, batch_size = 32):
        #print((self.W[0].shape))
        #print((self.W[1].shape))
        #print((self.W[2].shape))

        #n_samples = X_train.shape[0]
        #n_batches = n_samples // batch_size

        for epoch in range(n_epochs):
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]


                nets,outs = feed_forward(X_batch,self.W,self.B)
                #print(nets,outs)
                dE_dnets,dE_douts,dWs,dBs = backprobagation(X_batch,y_batch,self.W,outs)
                #print(dE_dnets,dE_douts,dWs,dBs)
                self.updata_weights(dWs,dBs,learning_rate)
                loss = cross_entropy_batch(y_batch, outs[-1])
                acc = self.test(X_test, y_test)
                
            print(f"Epoch {epoch+1}, Last Loss: {loss}, Current Accuracy: {acc}")  




    def test(self,X_test,y_test):
        y_pred = feed_forward(X_test,self.W,self.B)[-1][-1]
        y_pred = np.argmax(y_pred,axis=1)
        y_true = np.argmax(y_test,axis=1)
        return accuracy_score(y_true,y_pred)



if __name__ == '__main__':
    X, y = load_data()
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    hidden_layers_dims = [20,15]
    output_layer_dim = 10
    nn = NeuralNetworkMultiClassifier(X_train.shape[1],hidden_layers_dims,output_layer_dim)

    nn.train(X_train, y_train, X_test, y_test)
    print(X_train.shape)