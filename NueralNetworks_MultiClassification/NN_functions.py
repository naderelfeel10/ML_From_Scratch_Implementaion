import numpy as np

def soft_max_batch(X):
    max_vector = np.max(X,axis=1,keepdims=True)
    #print(max_vector)
    X = X-max_vector
    #print(X)
    X_sum = (np.sum(np.exp(X),axis=1,keepdims=True))
    #print((X_sum))
    res = np.exp(X)/X_sum
    #print(res)
    return res


def cross_entropy_batch(y_true,y_pred):
    y_pred = np.log(y_pred)*-1
    #print(y_pred)
    #print(y_true)
    res = np.sum(y_true*y_pred,axis=1,keepdims=True)
    return np.sum(res) / y_true.shape[0]





def feed_forward(X_batch,Ws,Bs):
    n1 = np.dot(X_batch,Ws[0]) + Bs[0]
    o1 = np.tanh(n1) #[-1,1]
    nets = [n1]
    outs = [o1]
    idx = 1
    for w,b in zip(Ws[1:],Bs[1:]):
        nets.append(np.dot(outs[idx-1],w) + b)
        outs.append(np.tanh(nets[idx]))   # other nodes act func is tanh
        idx+=1
    outs[-1] = soft_max_batch(np.arctanh(outs[-1]))
    return nets,outs



def dtanh(y):   # tanh derivative
    return 1 - y ** 2




def backprobagation(X_batch,y_batch,Ws,outs):
    dE_dnets =[]
    dE_dnets.append(outs[-1] - y_batch)
    dE_douts =[]
    #for i in range(0,len(Ws)):
        #dE_douts.append(np.dot(dE_dnets[i],Ws[len(Ws)-i-1].T))
        #dE_dnets.append(dE_douts[i] * dtanh(outs[len(Ws)-i-1]))

    Ws_rev = Ws.copy()[::-1]
    outs_rev = outs.copy()[::-1]
    outs_rev = outs_rev[1:]
    #print(len(outs_rev))
    #print(len(outs))
    for w,net,o in zip(Ws_rev,dE_dnets,outs_rev):
        #print(w.shape,'  ',net.shape,'      ',o.shape)
        dE_douts.append(np.dot(net,w.T))
        #print(dE_douts[-1].shape)
        dE_dnets.append(dE_douts[-1]* dtanh(o))
            

    dE_douts.append(X_batch)
    #dE_douts = dE_douts[::-1]
    #dE_dnets = dE_dnets[::-1]
    dWs = []
    dBs = []
    dE_dnets = dE_dnets[::-1]   # reverse to match forward order
    # Step 3: compute gradients
    dWs, dBs = [], []
    layer_inputs = [X_batch] + outs[:-1]   # input to each layer
    for a, dnet in zip(layer_inputs, dE_dnets):
        dWs.append(np.dot(a.T, dnet))
        dBs.append(np.sum(dnet, axis=0, keepdims=True))

    dE_douts = dE_douts[::-1]

    return dE_dnets,dE_douts,dWs,dBs   


    

def check(your_answer, right_answer, name):
    if your_answer.shape != right_answer.shape:
        print(f"\nSomething wrong for {name}")
        print("your answer shape")
        print(your_answer.shape)
        print("Optimal answer shape")
        print(right_answer.shape)
        exit()


    if np.allclose(your_answer, right_answer, atol=1e-6):
        print("Good job ",name)
    else:
        print(f"\nSomething wrong for {name}")
        print("your answer")
        print(your_answer.shape, your_answer)
        print("Optimal answer")
        print(right_answer.shape, right_answer)
        exit()

if __name__ == '__main__':

    ## softmax
    input = np.array([
        [1,2,3,4],
        [1,3,7,8],
        [2,6,10,5]
    ])
    SM_result = soft_max_batch(input)
    print(SM_result)
    print("*"*100)

    ## cross entropy   : -segma(GT*log(predictions))

    y_true = np.array([
        [1,0],
        [0,1],
        [0.8,0.2]
    ])

    y_pred = np.array([
        [0.9,0.1],
        [0.2,0.8],
        [0.7,0.3]
    ])
    CE_result = cross_entropy_batch(y_true,y_pred)
    print(CE_result)

    # feed forward :
    W1 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\W1.npy')
    W2 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\W2.npy')
    W3 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\W3.npy')
    Ws = [W1,W2,W3]
    #print(Ws) 
    B1 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\b1.npy')
    B2 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\b2.npy')
    B3 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\b3.npy')
    Bs = [B1,B2,B3]
    #print(Bs) 
    X_batch = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\X_batch.npy')
    FF_res = feed_forward(X_batch,Ws,Bs)
    #print(FF_res[1][0])
    net_1 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\out3.npy')
    print("*"*100)
    #print(net_1)
    if np.allclose(FF_res[1][2],net_1,atol=1e-6):
        print("Greatest Ever")

    y_batch = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\y_batch.npy')
    back_prob_res = backprobagation(X_batch,y_batch,Ws[1:],outs=FF_res[1])
    print(Ws[2].shape)
    print(FF_res[1][1].shape)
    print("*"*20)
    X_batch = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\X_batch.npy')
    y_batch = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\y_batch.npy')
    W2 =   np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\W2.npy')
    W3 =   np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\W3.npy')
    out1 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\out1.npy')
    out2 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\out2.npy')
    out3 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\out3.npy')


    opt_dW1 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\dW1.npy')
    opt_db1 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\db1.npy')
    opt_dW2 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\dW2.npy')
    opt_db2 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\db2.npy')
    opt_dW3 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\dW3.npy')
    opt_db3 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\db3.npy')


    opt_dE_dnet3 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\dE_dnet3.npy')
    opt_dE_dout2 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\dE_dout2.npy')
    opt_dE_dnet2 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\dE_dnet2.npy')
    opt_dE_dnet1 = np.load(r'D:\DeepLearning\ML\ML_From_Scratch_Implementation\NueralNetworks_MultiClassification\results\dE_dnet1.npy')

    print("*"*100)
    check(back_prob_res[2][0], opt_dW1, 'dW1')
    check(back_prob_res[3][0], opt_db1, 'db1')
    check(back_prob_res[2][1], opt_dW2, 'dW2')
    check(back_prob_res[3][1], opt_db2, 'db2')
    check(back_prob_res[2][2], opt_dW3, 'dW3')
    check(back_prob_res[3][2], opt_db3, 'db3')
    check(back_prob_res[0][2], opt_dE_dnet3, 'dE_dnet3')
    check(back_prob_res[0][1], opt_dE_dnet2, 'dE_dnet2')
    check(back_prob_res[0][0], opt_dE_dnet1, 'dE_dnet1')
    check(back_prob_res[1][2], opt_dE_dout2, 'dE_dout2')
