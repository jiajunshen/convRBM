import numpy as np 
from conv import conv
a = 8 * np.ones((8,8))
b = np.ones((3,3))
print conv(a,b)
print conv(a,b).shape
