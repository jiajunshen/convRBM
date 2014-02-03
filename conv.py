import numpy as np

def conv(Z,Y):
# Take Z = array(n_z, n_z) input imageand Y = array(n_y, n_y) 
# convolution filter and do convolution operation. The result should
# be n_z - n_y + 1; For eg, the input layer consists of an Nv*Nv
# array of binary units. Each group of hidens is associated with a 
# Nw * Nw, where Nw = Nv - Nh + 1; Also Nh = Nv - Nw + 1.
    windowSize = Y.shape[0]
    visualSize = Z.shape[0]
    convedSize = visualSize - windowSize + 1
    convedData = np.array([Z[i:i+windowSize,j:j+windowSize] for i in range(convedSize) for j in range(convedSize)])
    convedResult = np.array([(convedData[i,:,:] * Y).sum() for i in range(pow(convedSize,2))])
    convedResult = convedResult.reshape((convedSize,convedSize))
    return convedResult
