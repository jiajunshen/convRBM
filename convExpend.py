import numpy as np
from math import sqrt
def convExpend(Z,Y):
# Take Z = array(n_z * n_z) input hiddenVariable and Y = array(n_y *  n_y) 
# weight filter and do anti-convolution operation. The result should
# be n_z + n_y -1; For eg, the input layer consists of an Nh*Nh
# array of binary units. Each group of hidens is associated with a 
# Nw * Nw, where Nw = Nv - Nh + 1; Also Nv = Nh + Nw - 1.
    hiddenNodeSize = int(sqrt(Z.shape[0]))
    Z = Z.reshape((hiddenNodeSize,hiddenNodeSize))
    windowSize = int(sqrt(Y.shape[0]))
    Y = Y.reshape((windowSize,windowSize))
    convedSize = hiddenNodeSize + windowSize - 1
    convedData = np.zeros((convedSize, convedSize))
    for i in range(hiddenNodeSize):
        for j in range(hiddenNodeSize):
            convedData[i:i + windowSize, j:j + windowSize] += Z[i,j] * Y
    return convedData

def convExpendGroup(Z,Y):
# Here Z is k group of hidden variable and y is k groups of weight filters
# Z = array(k,n_z * n_z) and y = array(k, n_y * n_y).
# We want to get for sum_i^k(convExpand(Z_i,Y_i))
    k = Z.shape[0]
    if(k!=Y.shape[0]):
        print "Group Size not correct"
        return
    else:
        convList = np.array([convExpend(Z[i,:],Y[i,:]) for i in xrange(k)])
    return convList.sum(axis = 0)
