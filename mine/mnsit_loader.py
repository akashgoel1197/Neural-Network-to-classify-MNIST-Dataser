""" mnist_loader
    ~~~~~~~~~~~~~~~~~~
    A python code to load the MNIST imagae data.
"""

####Libraries
# Standard library
import cPickle
import gzip
#Third-party Library
import numpy as np

def load_data():
    """
    This function load The MNIST data in training_data , testing_data, validation_data
    as a tuple.
    The 'training_data consist of tuple with first entry as a numpy array with 50,000
    entries.Each entry is, in turn , a numpy ndarray with 784 values,representing 28*28=
    784 pixel in mnsit image.
    The testing and cross validation has same format as above but with Number of entries
    being equal to 10,000.

    """
    f = gzip.open("../data/mnist.pkl.gz",'rb')
    
    training_data,validation_data,test_data = cPickle.load(f)
    f.close()
    return (training_data,validation_data,test_data)


def load_data_wrapper():
    """
    same as load_data
    just change the format of training_data from 2-tuple(x,y) to a list conating 50,000 2-tuple(x,y)
    such that x is of ndarrya of 784 dimension representing image  and  y is a numpy array of 784*10 dimension
    and only changing validation_data and testing data as a list conatining 10000 2-tuple(x,y)
    """
    tr_d , va_d,te_d = load_data()
    training_inputs = [ np.reshape(x,(784,1))   for x in tr_d[0]]
    training_labels = [ vectorized_result(y)    for y in tr_d[1]]
    training_data = zip(training_inputs,training_labels)
    validation_value = [ np.reshape(x,(784,1)) for x in va_d[0]]
##    validation_label = [y for y in va_d[1]]
    validation_data = zip(validation_value,va_d[1])
    test_value = [np.reshape(x,(784,1)) for x in te_d[0]]
##    test_label = [ y for x in te_d[1]]
    test_data = zip(test_value,te_d[1])
                  
    return (training_data,validation_data,test_data)
    
def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e
