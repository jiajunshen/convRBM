import theano
from conv import conv
import numpy as np
from convExpend import convExpend
from convRBM import convRBM
from convTheano import convTheano
from sklearn.utils import check_random_state
import amitgroup.io.mnist as mn
import time
import os

def testConvOperation():
    a = np.array((1,0,0,1,0,0,1,0,0))
    a = np.array((a,a,a))
    a = np.array((a,a,a))
    a = a.reshape((9,9))
    b = np.array((2,2,2))
    b = np.array((b,b,b))
    b = b.reshape((3,3))
    print a
    print b
    print conv(a.flatten(),b.flatten())
    print convExpend(a.flatten(),b.flatten())

def testConvTheano(border = 'valid'):
    a = np.array((1,0,1,0,1,0))
    b = np.array((0,1,0,1,0,1))
    A = np.array((a,b,a,b,a,b))
    B = np.array((b,a,b,a,b,a))
    c = np.array((1,1,1))
    C = np.array((c,c,c))
    D = np.zeros((3,3))
    D[1,1] = 1
    Z = np.array((A,B))
    #print Z.shape
    Y = np.array((C,D))
    #print Y.shape
    return convTheano(Z.reshape(2,-1),Y.reshape(2,-1))


def testConvSpeed():
    a = np.ones(9)
    a = np.array((a, 2*a, 3*a, 4*a, 5*a, 6*a, 7*a, 8*a, 9*a))
    b = np.ones(3)
    b = np.array((b,b,b))
    a = a.reshape(9,9)
    b = b.reshape(3,3)
    current = time.time()
    
    for i in range(10000): 
        result = conv(a.flatten(),b.flatten())
    print time.time() - current
    current = time.time()
    testConvTheano()
    print time.time()-current
    print result

def testTotal():
    r = testInit()
    visibleNodes = np.ones((50,28,28))
    for i in xrange(50):
        visibleNodes[i,14,:] = 1
    visibleNodes = visibleNodes.reshape(50,-1)
    r.fit(visibleNodes)
    return r

def testInit(useTheano = False):
    n_groups = 16
    n_components = 24 * 24
    window_size = 5
    learning_rate = 0.1
    batch_size = 10
    n_iter = 1000
    verbose = False
    r = convRBM(n_groups = n_groups, n_components = n_components, window_size = window_size, learning_rate = learning_rate, batch_size = batch_size, n_iter = n_iter, verbose = verbose, use_theano = useTheano)
    return r

def testMeanHidden():
    r = testInit()
    rng = check_random_state(r.random_state)
    visibleSamples = 20
    r.components_ = np.asarray(rng.normal(0,0.01,(r.n_groups,r.window_size * r.window_size)),order = 'fortran')
    r.intercept_hidden_=np.zeros((r.n_groups,r.n_components))
    r.intercept_visible_=np.zeros(28 * 28)
    visibleNodes = np.ones((20,28*28))
    hiddenMean = r._mean_hiddens(visibleNodes,1)
    return r, hiddenMean

def testMeanHiddenTheano():
    r = testInit(useTheano = True)
    rng = check_random_state(r.random_state)
    visibleSamples = 20
    r.components_ = np.asarray(rng.normal(0,0.01,(r.n_groups,r.window_size * r.window_size)),order = 'fortran')
    r.intercept_hidden_=np.zeros((r.n_groups))
    r.intercept_visible_=0
    visibleNodes = np.ones((20,28*28))
    hiddenMean = r._mean_hiddens_theano(visibleNodes)
    return r,hiddenMean

def testMeanVisible():
    r,hiddenMean = testMeanHidden()
    rng = check_random_state(r.random_state)
    sample_H = []
    for i in range(r.n_groups):
        sample_H_k = r._bernoulliSample(hiddenMean,rng)
        sample_H.append(sample_H_k)
    sample_H = np.array(sample_H)
    sample_H = np.swapaxes(sample_H, 0, 1)
    return r._mean_visibles(sample_H)

def testMeanVisibleTheano():
    r,hiddenMean = testMeanHiddenTheano()
    visibleNodes = np.ones((20,28*28))
    rng = check_random_state(r.random_state)
    sample_H = r._bernoulliSample(hiddenMean,rng)
    
    result = r._mean_visibles_theano(sample_H,visibleNodes)
    return result

def testGradience():
    r,hiddenMean = testMeanHidden()
    visibleSamples = 20
    visibleNodes = np.ones((20,28*28))
    probability_H = r._mean_hiddens(visibleNodes,1)
    gradience_Positive = r._gradience(visibleNodes, probability_H)
    return r,visibleNodes, probability_H,gradience_Positive
    
def testGradienceTheano():
    r,hiddenMean = testMeanHiddenTheano()
    visibleSamples = 20
    visibleNodes = np.ones((20,28*28))
    probability_H = hiddenMean
    gradience_Positive = r._gradience_theano(visibleNodes, probability_H)
    return gradience_Positive



 
def testRun():
    r = testInit()
    visibleSamples = 20
    visibleNodes = np.zeros((20,28,28))
    visibleNodes[:,14:16,:] = 1
    visibleNodes = visibleNodes.reshape(20,28*28)
    r.fit(visibleNodes)
    return r

def testRunTheano():
    r = testInit(useTheano = True)
    visibleSamples = 20
    visibleNodes = np.zeros((20,28,28))
    visibleNodes[:,14:16,:] = 1
    visibleNodes = visibleNodes.reshape(20,28*28)
    r.fit(visibleNodes)
    return r


def testRunMnist():
    n_groups = 16
    n_components = 24*24
    window_size = 5
    learning_rate = 0.1
    batch_size = 50
    n_iter = 20
    r = convRBM(n_groups = n_groups, n_components = n_components, window_size = window_size, learning_rate = learning_rate, batch_size = batch_size, n_iter = n_iter, verbose = False)
    digits = [0,1,2,3,4,5,6,7,8,9]
    images,labels = mn.load_mnist('training',digits,'/Users/jiajunshen/Dropbox/Research/data/',False,slice(0,6000,5),True,False)
    return r

def testRunMnistTheano():
    n_groups = 16
    n_components = 24 * 24
    window_size = 5
    learning_rate = 0.1
    batch_size = 50
    n_iter = 200
    r = convRBM(n_groups = n_groups, n_components = n_components, window_size = window_size, learning_rate = learning_rate, batch_size = batch_size, n_iter = n_iter, verbose = False,use_theano = True)
    digits = [0,1,2,3,4,5,6,7,8,9]
    images,labels = mn.load_mnist('training',digits,'/home/jiajun/mnist',False,slice(0,6000,5),True,False)
    r.fit(images.reshape(1200,28*28))
    return r
