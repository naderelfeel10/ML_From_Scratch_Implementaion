import numpy as np

def softmax_iterative(arr):
    res = []
    arr -= np.max(arr)
    sum = np.sum(np.exp(arr))
    for i in arr:
        tmp = (np.exp(i)/sum)
        res.append(tmp)
    return res    

def softmax_grad_iterative(arr):
    res = np.zeros((len(arr),len(arr)) ,dtype=float)
    for idx,i in enumerate(arr):
        for jdx,j in enumerate(arr):
            if(idx == jdx):
                tmp = i * (1-i)
                res[idx][jdx] = tmp
            else:
                tmp = - i * j
                res[idx][jdx] = tmp
                

    return res


def softmax_vecotized(arr):
    max = np.max(arr)
    arr -= max
    arr = np.exp(arr)
    tmp1 = np.ones(len(arr),dtype=int).reshape(-1,1)
    sum = np.dot(arr,tmp1)
    res = arr / sum
    return res


if __name__ == '__main__':
    
    probs = softmax_iterative(np.array([5,7,8]))
    print(probs)
    dervs = softmax_grad_iterative(np.array(probs))
    print(dervs)
    probs_v = softmax_vecotized(np.array([5,7,8]))
    print(probs_v)
    

