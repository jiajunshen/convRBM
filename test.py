from conv import conv
import numpy as np
from convExpend import convExpend

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
