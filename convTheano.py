import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
from math import sqrt

def convTheano(Z,Y,flip=False,boarder='valid'):
    """
    Take Z = array(batch_size, n_z * n_z) input imageand Y = array(nb_filters, n_y * n_y) 
    convolution filter and do convolution operation. The result should
    be n_z - n_y + 1; For eg, the input layer consists of an Nv*Nv
    array of binary units. Each group of hidens is associated with a 
    Nw * Nw, where Nw = Nv - Nh + 1; Also Nh = Nv - Nw + 1.
    """
    labelZ = False 
    if (np.array(Z.shape).shape[0] != 2):
        Z = Z.reshape(1,Z.shape[0])
        labelZ = True
    labelY = False
    if(np.array(Y.shape).shape[0] != 2):
        Y = Y.reshape(1,Y.shape[0])
        labelY = True

    windowSize = int(sqrt(Y.shape[1]))
    visualSize = int(sqrt(Z.shape[1]))
    if(boarder=='valid'):
        convedSize = visualSize - windowSize + 1
    else:
        convedSize = visualSize + windowSize - 1

    batch_size = Z.shape[0]
    nb_filters = Y.shape[0]

    x = T.tensor4(name = 'x')
    y = T.tensor4(name = 'y')
    output = conv.conv2d(x,y,boarder_mode = boarder)
    f = theano.function([x,y],output)
    
    Z = Z.reshape(batch_size,1,visualSize,visualSize)
    Y = Y.reshape(nb_filters,1,windowSize,windowSize)
    
    if(flip):
        Y = np.array([np.flipud(np.fliplr(Y[i,0,:,:])) for i in range(nb_filters)]).reshape(nb_filters,1,windowSize,windowSize)

    result = f(Z,Y)
    result = np.array(result).reshape(batch_size,nb_filters,convedSize * convedSize)
    if(labelZ):
        return result.reshape(nb_filters,convedSize * convedSize)
    else if(labelY):
        return result.reshape(batch_size,convedSize * convedSize)
    else:
        return result
