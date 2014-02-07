"""convolutional Restricted Boltzmann Machine
"""

# Main author: Jiajun Shen<jiajun@cs.uchicago.edu>

import time
import numpy as np
import heapq

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_arrays
from sklearn.utils import check_random_state
from sklearn.utils import issparse
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import logistic_sigmoid
from conv import conv
from conv import convExpend
import amitgroup.plot as gr


class convRBM(BaseEstimator, TransformerMixin):
    """convolutional Restricted Boltzmann Machine (RBM).

    A convolutional Restricted Boltzmann Machine with binary visible units and
    binary hiddens. Parameters are estimated using Stochastic descent
    for learning binary CRBM's weights

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Parameters
    ----------
    n_components : int, optional
        Number of binary hidden units.

    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, optional
        Number of examples per minibatch.

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, optional
        The verbosity level. Enabling it (with a non-zero value) will compute 
        the log-likelihood of each mini-batch and hence cause a runtime overhead
        in the order of 10%.

    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    `components_` : array-like, shape (n_components, n_features), optional
        Weight matrix, where n_features in the number of visible
        units and n_components is the number of hidden units.

    `intercept_hidden_` : array-like, shape (n_components,), optional
        Biases of the hidden units.

    `intercept_visible_` : array-like, shape (n_features,), optional
        Biases of the visible units.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import BernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BernoulliRBM(n_components=2)
    >>> model.fit(X)
    BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10,
           random_state=None, verbose=False)

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008
    [3]

    [4]

    """
    def __init__(self, n_groups = 16,n_components, window_size = 5, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=False, random_state=None, getNum = 1):
        self.getNum = getNum
        self.n_groups = n_group
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.window_size = window_size
        self.verbose = verbose
        self.random_state = random_state
        
    
    def gen_even_slices(self,n, n_packs, n_samples=None):
        """Generator to create n_packs slices going up to n.

        Pass n_samples when the slices are to be used for sparse matrix indexing;
        slicing off-the-end raises an exception, while it works for NumPy arrays.
    
        Examples
        --------
        >>> from sklearn.utils import gen_even_slices
        >>> list(gen_even_slices(10, 1))
        [slice(0, 10, None)]
        >>> list(gen_even_slices(10, 10))                     #doctest: +ELLIPSIS
        [slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
        >>> list(gen_even_slices(10, 5))                      #doctest: +ELLIPSIS
        [slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
        >>> list(gen_even_slices(10, 3))
        [slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]
        """
        start = 0
        for pack_num in range(n_packs):
            this_n = n // n_packs
            if pack_num < n % n_packs:
                this_n += 1
            if this_n > 0:
                end = start + this_n
                if n_samples is not None:
                    end = min(n_samples, end)
                yield slice(start, end, None)
                start = end



    def transform(self, X):
        """Compute the hidden layer activation probabilities, P(h=1|v=X).

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        h : array, shape (n_samples, n_components)
            Latent representations of the data.
        """
        X, = check_arrays(X, sparse_format='csr', dtype=np.float)
        return self._mean_hiddens(X)

    def _mean_hiddens(self, v,k):
        """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        n_samples = v.shape[0]
        activations = np.array([conv(v[i,:],self.components_[k]) + self.intercept_hidden_[i] for i in range(n_samples)])
        return logistic_sigmoid(activations)

    
    def _mean_visible(self, h):
        """
        Computes the probabilities P(v=1|h).
        
        Parameters
        ----------
        h : array-like, shape (n_samples, n_groups, n_components)
            values of the hidden layer.
        
        Returns
        -------
        v: array-like,shape (n_samples, n_features)
        """
        n_samples = h.shape[0]
        activations = np.array([convExpendGroup(h[i],self.components_) + self.intercept_visible_ for i in range(n_samples)])
        return logistic_sigmoid(activations) 


    def _gradience(v,mean_h):
        """Computer the gradience given the v and h.
        This is for getting the Grad0k./ If it is, we need to focus on the Ph0k
        Parameters
        ----------
        v: array-like, shape (n_samples, n_features)
            values of the visible layer.
        h: array-like, shape (n_samples, n_components)
        
        Returns
        --------
        Grad: array-like,shape (weight_windowSize * weight_windowSize)     
        
        """
        visibleData = v.reshape(n_samples, n_visible_windowSize,n_visible_windowSize)
        weights =  np.array([conv(visibleData[i,:],mean_h[i,:]) for i in range(v.shape[0])]).sum(axis = 0) 
        return np.ravel(weights)

    def _bernoulliSample(self,p,rng):
        p[rng.uniform(size = p.shape) < p] = 1.
        return np.floor(p,p)



    def _sample_hiddens(self, v, k, rng):
        """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to sample from.

        rng : RandomState
            Random number generator to use.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer.
        """
        p = self._mean_hiddens(v,k)
        p[rng.uniform(size=p.shape) < p] = 1.
        return np.floor(p, p)
    

    def _sample_visibles(self, h, rng):
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng : RandomState
            Random number generator to use.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = logistic_sigmoid(np.dot(h, self.components_)
                             + self.intercept_visible_)
        p[rng.uniform(size=p.shape) < p] = 1.
        return np.floor(p, p)

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
	#print safe_sparse_dot(v, self.components_.T)
        #print  self.intercept_hidden_
        return (- safe_sparse_dot(v, self.intercept_visible_)
                - np.log(1. + np.exp(safe_sparse_dot(v, self.components_.T)
                            + self.intercept_hidden_)).sum(axis=1))

    def fit(self, X):
        """Fit the model to the data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training Data
        i
        Returns
        -------
        self: convolutionRBM
              The fitted Model.
        """
        X, = check_arrays(X, sparse_format = 'csr', dtype = np.float)
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)
        
        self.components_ = np.asarray(
            rng.normal(0, 0.001, (self.n_groups,self.window_size * self.window_size)),order='fortran')
        self.intercept_hidden_ = np.zeros(self.n_groups, self.n_components)          

        self.intercept_visible_ = np.zeros(X.shape[1],)
                        

        lr = float(self.learning_rate) / v.shape[0]
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples)/ self.batch_size))
        batch_slices = list(self.gen_even_slices(n_batches * self.batch_size, n_batches, n_samples))
       
        verbose = self.verbose
        
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng)

            if verbose:
                end = time.time()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(), end - begin))
                begin = end

        return self



    def _fit(self, v_pos,rng):
        """Inner fit for one mini-batch
        
        Adjust the parameters to maximize the likelihood of v using 
        Stochastic CD gradient descent for learning binary CRBM's 
        Weights

        Parameters
        -------------
        v_pos: array-like, shape(n_samples, n_features)
               the data to use for training.

        rng: RandomState
             Random number generator to use for sampling
        """
        sample_H = []
        gradience_Positive = []
        gradience_Negtive = []
        for i in range(self.n_group):
            probability_H = self._mean_hiddens(v_pos,i)
            gradience_Positive += self._gradience(v_pos,probability_H)
            sample_H_k = self._bernoulliSample(probability_H)
            sample_H += sample_H_k
        
        v_reconstruct = _mean_visibles(sample_H)

        for j in range(self.n_group):
            probability_H = self._mean_hidden(v_reconstruct,j)
            gradience_Negtive += self._gradience(v_reconstruct, probability_H)
            self.components_[j] += lr * (gradience_Positive - gradience_Negtive[j]) 

        return



    def score_samples(self, v):
        """Compute the pseudo-likelihood of v.

        Parameters
        ----------
        v : {array-like, sparse matrix} shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        pseudo_likelihood : array-like, shape (n_samples,)
            Value of the pseudo-likelihood (proxy to likelihood).
        print update"""
        rng = check_random_state(self.random_state)
        fe = self._free_energy(v)

        if issparse(v):
            v_ = v.toarray()
        else:
            v_ = v.copy()
        i_ = rng.randint(0, v.shape[1], v.shape[0])
        v_[np.arange(v.shape[0]), i_] = 1 - v_[np.arange(v.shape[0]), i_]
        fe_ = self._free_energy(v_)
	#print fe_
	#print fe
        return v.shape[1] * logistic_sigmoid(fe_ - fe, log=True)

