import random

import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):


    def __init__(self,sizes):
        """
        @sizes : A list of number of neuorons in each layer
        Select random biases and weights for the start of the network
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weights = [np.random.randn(x,y)
                        for x,y in zip(sizes[1:],sizes[:-1])]

    def feedforward(self,a):
        """
        @a : Take a as input layer
        return: the outpur of the neural network values for the particular example 
                for the fixed weights and biases
        """
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,a) + b
            a = sigmoid(z)
        return a

    def SGD(self,training_data, epochs ,mini_batch_size,alpha, test_data=None):
        if test_data: test_len = len(test_data)
        n = len(training_data)
        for i in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k+mini_batch_size]
                             for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,alpha)

            if(test_data):
                print "Epoch {0}:{1}/{2}".format(
                    i+1,self.evaluate(test_data),test_len)
            else:
                print "Epoch {0} completed".format(i+1)

    def update_mini_batch(self,mini_batch,alpha):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            changeb,changew = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, changeb)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, changew)]
        self.weights = [w-(alpha/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(alpha/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    
    def backprop(self,x,y):
        """
        @x : A particular example from training set
        @y : output of that particular example
        return : Two list one of which is change in cost with respect to weight
                and  other for biases for each layer. 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,activation)
            zs.append[z]
            activation = sigmoid(z)
            activations.append(activation)
        #The above for loop give output of network for x
        #and store actiavtion and z for each layer
        #Now the below code is for backward pass
        delta = self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,actiavtions[-2].transpose())

        for l in xrange(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1],actiavtions[-l-1].transpose())
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)
            
        

    def evaluate(self,test_data):
        test_results = [ (np.argmax(self.feedforward(x)),y)
                         for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)



    def cost_derivative(self,output_actiavtions, y):
        return (output_activations - y)







            
        
