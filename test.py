from conv import conv
import numpy as np
from convExpend import convExpend
from convRBM import convRBM
from sklearn.utils import check_random_state
import amitgroup.io.mnist as mn
import time

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

def testConvSpeed():
    a = np.ones(9)
    a = np.array((a, 2*a, 3*a, 4*a, 5*a, 6*a, 7*a, 8*a, 9*a))
    b = np.ones(3)
    b = np.array((b,b,b))
    a = a.reshape(9,9)
    b = b.reshape(3,3)
    current = time.time()
    result = conv(a.flatten(),b.flatten())
    for i in range(10000):
        conv(a.flatten(),b.flatten())    
    print time.time() - current
    print result

def testTotal():
    r = testInit()
    visibleNodes = np.ones((50,28,28))
    for i in xrange(50):
        visibleNodes[i,14,:] = 1
    visibleNodes = visibleNodes.reshape(50,-1)
    r.fit(visibleNodes)
    return r

def testInit():
    n_groups = 16
    n_components = 24 * 24
    window_size = 5
    learning_rate = 0.1
    batch_size = 10
    n_iter = 20
    verbose = False
    r = convRBM(n_groups = n_groups, n_components = n_components, window_size = window_size, learning_rate = learning_rate, batch_size = batch_size, n_iter = n_iter, verbose = verbose)
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


def testGradience():
    r,hiddenMean = testMeanHidden()
    visibleSamples = 20
    visibleNodes = np.ones((20,28*28))
    probability_H = r._mean_hiddens(visibleNodes,1)
    gradience_Positive = r._gradience(visibleNodes, probability_H)
    return r,visibleNodes, probability_H,gradience_Positive
    
    
def testRun():
    r = testInit()
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
    r.fit(images.reshape(1200,28*28))
    return r
