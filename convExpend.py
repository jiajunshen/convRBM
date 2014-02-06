import numpy as np

def convExpend(Z,Y):
# Take Z = array(n_z, n_z) input hiddenVariable and Y = array(n_y, n_y) 
# weight filter and do anti-convolution operation. The result should
# be n_z + n_y -1; For eg, the input layer consists of an Nh*Nh
# array of binary units. Each group of hidens is associated with a 
# Nw * Nw, where Nw = Nv - Nh + 1; Also Nv = Nh + Nw - 1.
    windowSize = Y.shape[0]
    hiddenNodeSize = Z.shape[0]
    convedSize = hiddenNodeSize + windowSize - 1
    convedData = np.zeros((convedSize, convedSize))
    for i in range(hiddenNodeSize):
        for j in range(hiddenNodeSize):
            convedData[i:i + windowSize, j:j + windowSize] += Z[i,j] * Y
    return convedData
