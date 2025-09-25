import numpy as np
import matplotlib.pyplot as plt
#kernel for this problem
def kernel(x):
        return np.exp(-x**2)
#
X = np.linspace(-5,5,100).reshape(-1,1)
K = kernel(X)

plt.plot (X,K)
plt.show()

#covariance decerases as distance increases!!!!