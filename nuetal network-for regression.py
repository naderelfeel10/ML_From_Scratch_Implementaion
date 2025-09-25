import random
import math
import numpy as np


class NeuralNetwork:
    def __init__(self, hidden_layers_weights_list,output_layer_weights, activation_type = 'poly'):
        self.lr = 0.5  # learning rate

        #self.hidden_layer = NeuronLayer(hidden_layer_weights, activation_type)
        self.output_layer = NeuronLayer(output_layer_weights, activation_type)
        self.hidden_layers = []
        for i in range(len(hidden_layers_weights_list)):
            h = NeuronLayer(hidden_layers_weights_list[i], activation_type)
            self.hidden_layers.append(h)

        self.hidden_layers_weights_list = hidden_layers_weights_list

        #self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

    def feed_forward(self, input):
        self.h_net = []
        self.h_out = []
        temp = input
        for i in range(len(self.hidden_layers)):
            h1_net,h1_out = self.hidden_layers[i].feed_forward(temp)
            self.h_net.append(h1_net)
            self.h_out.append(h1_out)
            temp = h1_out

        #h1_net,h1_out = self.hidden_layer.feed_forward(input)
        self.o_net,self.o_out = self.output_layer.feed_forward(h1_out)

        return self.h_net,self.h_out,self.o_net,self.o_out

    def my_compute_delta(self, target):
        # Base case rule for output layer
        print(self.h_net)
        self.dE_dO_net = []
        #dE_dO_out = layers_data[-1] - target
        dE_dO_out = np.array(self.o_out) - np.array(target)
        dO_out_do_net = []
        print(self.output_layer.neurons)
        for n in self.output_layer.neurons:
            #d1 = n.activation_derv(self.o_net)
            #dO_out_do_net = d1
            dO_out_do_net = [n.activation_derv([net])[0] for n, net in zip(self.output_layer.neurons, self.o_net)]
            #dO_out_do_net.extend(d1)

        #self.dE_dO_net = [dE_dO_out[0]*dO_out_do_net[0],dE_dO_out[1]*dO_out_do_net[1]]
        for i in range(len(dO_out_do_net)):
            o1 =  dE_dO_out[i]*dO_out_do_net[i]
            self.dE_dO_net.append(o1)

        # Sum on neighbours formula for hidden neuron
        '''
        self.dE_dH_net = []
        tmp =0
        dE_dH_out = []
        all_layer_weights = self.hidden_layers_weights_list + [self.output_layer_weights]
        h_layer_weights = all_layer_weights[::-1]
        h_layer_weights.pop(-1)
        print("h_fasdf  :   ",len(h_layer_weights))
        print("all_layer_weights : ",all_layer_weights)
        print("w : ",self.hidden_layers_weights_list)
        print("w2 : ",h_layer_weights)

        net_tmp = self.dE_dO_net.copy()
        print(net_tmp)
        for idx,n in enumerate(h_layer_weights):
            print("gggggggggggggggggggggg")
            print(n)
            for h in range(n.shape[1]):
                for j in range(len(self.dE_dO_net)):
                    #dE_dH_out = [self.dE_dO_net[0] * self.output_layer_weights[0][0] + self.dE_dO_net[1] * self.output_layer_weights[1][0]
                            #, self.dE_dO_net[0] * self.output_layer_weights[0][1] + self.dE_dO_net[1] * self.output_layer_weights[1][1] ]
                    print(net_tmp[j])
                    print(n[j][idx])
                    print(self.dE_dO_net[j])
                    tmp += net_tmp[j]*n[j][h]
                print("tmp : ",tmp)    
                dE_dH_out.append(tmp)
                tmp=0
            #print("dE_dH_out : ",dE_dH_out)
            dH_out_dH_net = []
            #print(idx)
            #print(self.hidden_layers[0].neurons)
            #print("hnet: ",self.h_net)
            #print(len(self.hidden_layers))
            #print("idx : ",idx)

            for n in self.hidden_layers[idx-1].neurons:
                print("h_net : ",self.h_net)
                d1 = n.activation_derv(self.h_net[len(self.hidden_layers)-idx-1])
                dH_out_dH_net = d1
                print(dH_out_dH_net)
            net_tmp =  dH_out_dH_net
            print("dH_out_dH_net : ",dH_out_dH_net)
            #print("net_tmp : ",tmp*net_tmp)
            
            for i in range(len(dH_out_dH_net)):
                #print(i)
                #print(dE_dH_out,dH_out_dH_net)
                o1 = dE_dH_out[i]*dH_out_dH_net[i]
                self.dE_dH_net.append(o1)
                print("dE_dH_net : ",self.dE_dH_net)
        '''  
        self.dE_dH_net = []
        net_tmp = self.dE_dO_net.copy()

        for layer_idx in reversed(range(len(self.hidden_layers))):  # loop from last hidden layer backward
            weight_matrix = self.output_layer_weights if layer_idx == len(self.hidden_layers)-1 else self.hidden_layers_weights_list[layer_idx+1]
            hidden_deltas = []

            for h in range(len(self.hidden_layers[layer_idx].neurons)):
                # sum of deltas from next layer * weights
                sum_delta = sum(net_tmp[k] * weight_matrix[k][h] for k in range(weight_matrix.shape[0]))
                # derivative of this hidden neuron
                deriv = self.hidden_layers[layer_idx].neurons[h].activation_derv([self.h_net[layer_idx][h]])[0]
                hidden_deltas.append(sum_delta * deriv)

            net_tmp = hidden_deltas
            self.dE_dH_net = hidden_deltas + self.dE_dH_net  # prepend to maintain correct order


            #self.dE_dH_net.append(tmp*)
               


        #self.dE_dH_net = [dE_dH_out[0]*dH_out_dH_net[0],dE_dH_out[1]*dH_out_dH_net[1]]
        print("dE_dH_net : ",self.dE_dH_net)    
        #print(self.dE_dO_net)
        #print(self.dE_dH_net)


    def compute_delta(self, target):
        dE_dO_out = np.array(self.o_out) - np.array(target)
        self.dE_dO_net = [
            delta * n.activation_derv([net])[0]
            for delta, n, net in zip(dE_dO_out, self.output_layer.neurons, self.o_net)
        ]

        self.dE_dH_net = []
        net_tmp = self.dE_dO_net.copy()
        all_layer_weights = self.hidden_layers_weights_list + [self.output_layer_weights]

        for layer_idx in reversed(range(len(self.hidden_layers))):
            weight_matrix = all_layer_weights[layer_idx + 1]  
            hidden_deltas = []
            for h in range(weight_matrix.shape[1]):  
                sum_delta = sum(net_tmp[k] * weight_matrix[k][h] for k in range(weight_matrix.shape[0]))
                deriv = self.hidden_layers[layer_idx].neurons[h].activation_derv([self.h_net[layer_idx][h]])[0]
                hidden_deltas.append(sum_delta * deriv)
            net_tmp = hidden_deltas
            self.dE_dH_net = hidden_deltas + self.dE_dH_net  

        print("dE_dO_net:", self.dE_dO_net)
        print("dE_dH_net:", self.dE_dH_net)

    def my_update_weights(self,input):
        # Update output layer
        #self.h_out.append([1,2])
        print("h_out : ",self.h_out[::-1])
        dE_dw = []
        all_layers = [input] + self.h_out + [self.o_net]
        self.h_net 
        print(all_layers[::-1])
        self.h_out = self.h_out[::-1]
        for idx,i in enumerate(all_layers):
            for j in range(len(i)):
                for h in range(len(all_layers[idx+1])):
                    print(h)
                    tmp = all_layers[idx+1][j]*i[h]
                    dE_dw.append(tmp)
                    print("dE_dw",dE_dw)

        dE_dw = [self.dE_dO_net[0]*layers_data[1][0],self.dE_dO_net[0]*layers_data[1][1],
                 self.dE_dO_net[1]*layers_data[1][0],self.dE_dO_net[1]*layers_data[1][1]]


        print(dE_dw)
        o_weights = self.output_layer_weights.copy()
        #print(o_weights)
        wdx=0
        for i in o_weights:
            for w in i:
                w = w - self.lr * dE_dw[wdx]
                self.output_layer_weights[wdx//2][wdx%2] = w
                print("o",w)
                wdx +=1

        # Update hidden layer
        dE_dw = [self.dE_dH_net[0]*input[0],self.dE_dH_net[0]*input[1],
                 self.dE_dH_net[1]*input[0],self.dE_dH_net[1]*input[1]]
        #print(dE_dw)
        
        h_weights = self.hidden_layer_weights.copy()
        wdx=0
        for i in h_weights:
            for w in i:
                w = w - self.lr * dE_dw[wdx]
                self.hidden_layer_weights[wdx//2][wdx%2] = w
                print("h: ",w)
                wdx +=1


    def update_weights(self, input):
        # Update output layer weights
        print("h_out : ", self.h_out[::-1])
        dE_dw = []

        # Gradients for output layer weights
        print("deOnet : ",self.dE_dO_net)
        for i, delta in enumerate(self.dE_dO_net):  # loop over each output neuron
            for h in range(len(self.h_out[-1])):  # inputs to output layer = last hidden layer outputs
                grad = delta * self.h_out[-1][h]
                dE_dw.append(grad)
                # Update weight
                self.output_layer_weights[i][h] -= self.lr * grad

        print("Updated output layer weights:", self.output_layer_weights)

        # Update hidden layer weights (only works for single hidden layer now)
        dE_dw = []
        for i, delta in enumerate(self.dE_dH_net):  # loop over each hidden neuron
            for h in range(len(input)):  # inputs to hidden = original input
                grad = delta * input[h]
                dE_dw.append(grad)
                # Update weight
                self.hidden_layers_weights_list[0][i][h] -= self.lr * grad

        print("Updated hidden layer weights:", self.hidden_layers_weights_list[0])

    def train_step(self, input, target):
        h_net,h_out,o_net,o_out = self.feed_forward(input)
        layers_data = [h_net,h_out,o_net,o_out]
        print('network output:', o_out)
        print(layers_data)
        flat = [item for sublist in layers_data for item in (sublist if isinstance(sublist[0], (list, np.ndarray)) else [sublist])]
        #layers_data = np.array(flat)
        #layers_data = np.hstack(flat)   # if you want to flatten all into one vector
        layers_data = [np.array(f) for f in flat]   # keep as structured lis
        print(layers_data)
        #self.compute_delta(target,layers_data)
        self.my_compute_delta(target)
        self.update_weights(input)


class NeuronLayer:
    def __init__(self, weights, activation_type):
        # Consist of list of Neuron
        self.neurons = [Neuron(w, activation_type) for w in weights]
        layer = []
        for i in range(weights.shape[0]):
            n = Neuron(weights[i,:],activation_type)
            layer.append(n)

        self.layer = layer

    def feed_forward(self, inputs):
        outputs = []
        # TODO
        h1_net = []
        h1_out = []
        for i in range(len(self.layer)):
            n_net  = self.layer[i].calc_net_out(inputs)
            n_out = self.layer[i].activation(n_net)
            h1_net.append(n_net)
            h1_out.append(n_out)

        return h1_net,h1_out


class Neuron:
    def __init__(self, weights, activation_type):
        self.weights = weights
        self.activation_type = activation_type

    def calc_net_out(self, input):
        n_net = np.dot(input,self.weights.T)
        return n_net

    def activation(self, net):
        if self.activation_type == 'poly':
            return net**2
        elif self.activation_type == 'sigmoid':
            net = np.array(net, dtype=float)   # ensure array
            return 1 / (1 + np.exp(-net))

    def activation_derv(self,net):
        dervs = []
        if self.activation_type == 'poly':
            for i in range(len(net)):
                dervs.append(2*net[i])
               
        elif self.activation_type == 'sigmoid':
            for i in range(len(net)):
                dervs.append(self.activation(net)*(1 -(self.activation(net))))

        return dervs





def poly():     # 2 x 2 x 2
    hidden_layer_weights = np.array([[1, 1],
                                     [2, 1]])
    output_layer_weights = np.array([[2, 1],
                                     [1, 0]])
    hidden_layer_weights_list = [hidden_layer_weights]
    nn = NeuralNetwork(hidden_layer_weights_list, output_layer_weights, 'poly')

    
    nn.train_step([1, 1],[290, 14])
    print(nn.output_layer_weights)


    '''
    network output: [289, 16]
    Delta o[0]: -34.0
    Delta o[1]: 16.0
    Delta h[0]: -208.0
    Delta h[1]: -204.0
    node o: 0 - w_ho: 0: Delata -136.0 => new w = 70.0
    node o: 0 - w_ho: 1: Delata -306.0 => new w = 154.0
    node o: 1 - w_ho: 0: Delata 64.0 => new w = -31.0
    node o: 1 - w_ho: 1: Delata 144.0 => new w = -72.0
    node h: 0 - w_ih: 0: Delata -208.0 => new w = 105.0
    node h: 0 - w_ih: 1: Delata -208.0 => new w = 105.0
    node h: 1 - w_ih: 0: Delata -204.0 => new w = 104.0
    node h: 1 - w_ih: 1: Delata -204.0 => new w = 103.0
    '''


def sigm():     # 2 4 3
    hidden_layer_weights = np.array([[0.1, 0.1],      # 4x2 NOT 2x4
                                     [0.2, 0.1],
                                     [0.1, 0.3],
                                     [0.5, 0.01]])
    hidden_layer_weights_list = [hidden_layer_weights]

    output_layer_weights = np.array([[0.1, 0.2, 0.1, 0.2],
                                     [0.1, 0.1, 0.1, 0.5],
                                     [0.1, 0.4, 0.3, 0.2]])

    nn = NeuralNetwork(hidden_layer_weights_list, output_layer_weights, 'sigmoid')

    nn.train_step([1, 2], [0.4, 0.7, 0.6])


    '''
    network output: [0.5913212667539777, 0.6219200057374265, 0.6508562785102494]
    Delta o[0]: 0.04623477887224621
    Delta o[1]: -0.01835937944358026
    Delta o[2]: 0.011556701931083076
    Delta h[0]: 0.000963950492482261
    Delta h[1]: 0.0028912254002713203
    Delta h[2]: 0.001386714367431997
    Delta h[3]: 0.000556197739142091
    node o: 0 - w_ho: 0: Delata 0.026559222739603632 => new w = 0.0867203886301982
    node o: 0 - w_ho: 1: Delata 0.027680191578841717 => new w = 0.18615990421057915
    node o: 0 - w_ho: 2: Delata 0.030893513891333994 => new w = 0.08455324305433301
    node o: 0 - w_ho: 3: Delata 0.028996038295713737 => new w = 0.18550198085214314
    node o: 1 - w_ho: 0: Delata -0.010546408134670482 => new w = 0.10527320406733524
    node o: 1 - w_ho: 1: Delata -0.010991533920193718 => new w = 0.10549576696009687
    node o: 1 - w_ho: 2: Delata -0.01226751284879592 => new w = 0.10613375642439797
    node o: 1 - w_ho: 3: Delata -0.01151404380893776 => new w = 0.5057570219044689
    node o: 2 - w_ho: 0: Delata 0.006638660943333523 => new w = 0.09668066952833325
    node o: 2 - w_ho: 1: Delata 0.006918854837737182 => new w = 0.39654057258113146
    node o: 2 - w_ho: 2: Delata 0.007722046916941944 => new w = 0.29613897654152904
    node o: 2 - w_ho: 3: Delata 0.007247759802026145 => new w = 0.19637612009898694
    node h: 0 - w_ih: 0: Delata 0.000963950492482261 => new w = 0.09951802475375887
    node h: 0 - w_ih: 1: Delata 0.001927900984964522 => new w = 0.09903604950751775
    node h: 1 - w_ih: 0: Delata 0.0028912254002713203 => new w = 0.19855438729986435
    node h: 1 - w_ih: 1: Delata 0.005782450800542641 => new w = 0.09710877459972869
    node h: 2 - w_ih: 0: Delata 0.001386714367431997 => new w = 0.09930664281628401
    node h: 2 - w_ih: 1: Delata 0.002773428734863994 => new w = 0.298613285632568
    node h: 3 - w_ih: 0: Delata 0.000556197739142091 => new w = 0.49972190113042897
    node h: 3 - w_ih: 1: Delata 0.001112395478284182 => new w = 0.00944380226085791
    '''


if __name__ == '__main__':
    #poly()
    sigm()

