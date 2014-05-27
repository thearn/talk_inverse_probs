"""
Define psfMatrix classes corresponding to convolution with different
boundary conditions.

Example:
>>> from psfMatrix import *
>>> m = 5; n = 3
>>> psf = arange(1, n ** 2 + 1); psf = psf.reshape((n,n), order='F')
>>> x = arange(1, m ** 2 + 1); x = x.reshape((m, m), order='F')
>>> A = psfMatrix(psf,boundary='zero'); b = A * x; b
array([[  32.,  114.,  249.,  384.,  440.],
       [  68.,  219.,  444.,  669.,  734.],
       [  89.,  264.,  489.,  714.,  773.],
       [ 110.,  309.,  534.,  759.,  812.],
       [  96.,  252.,  417.,  582.,  600.]])
>>> b = A.transpose() * x; b
array([[ 128.,  276.,  441.,  606.,  320.],
       [ 202.,  411.,  636.,  861.,  436.],
       [ 241.,  456.,  681.,  906.,  457.],
       [ 280.,  501.,  726.,  951.,  478.],
       [ 184.,  318.,  453.,  588.,  280.]])
>>> A = psfMatrix(psf,boundary='periodic'); b = A * x; b
array([[ 639.,  264.,  489.,  714.,  789.],
       [ 594.,  219.,  444.,  669.,  744.],
       [ 639.,  264.,  489.,  714.,  789.],
       [ 684.,  309.,  534.,  759.,  834.],
       [ 669.,  294.,  519.,  744.,  819.]])
>>> b = A.transpose() * x; b
array([[ 351.,  426.,  651.,  876.,  501.],
       [ 336.,  411.,  636.,  861.,  486.],
       [ 381.,  456.,  681.,  906.,  531.],
       [ 426.,  501.,  726.,  951.,  576.],
       [ 381.,  456.,  681.,  906.,  531.]])
>>> A = psfMatrix(psf,boundary='reflexive'); b = A * x; b
array([[  87.,  192.,  417.,  642.,  837.],
       [ 114.,  219.,  444.,  669.,  864.],
       [ 159.,  264.,  489.,  714.,  909.],
       [ 204.,  309.,  534.,  759.,  954.],
       [ 237.,  342.,  567.,  792.,  987.]])
>>> b = A.transpose() * x; b
array([[  183.,   378.,   603.,   828.,   933.],
       [  216.,   411.,   636.,   861.,   966.],
       [  261.,   456.,   681.,   906.,  1011.],
       [  306.,   501.,   726.,   951.,  1056.],
       [  333.,   528.,   753.,   978.,  1083.]])
>>> A = psfMatrix(psf,boundary='antireflexive'); b = A * x; b
array([[  -51.,   174.,   399.,   624.,   849.],
       [   -6.,   219.,   444.,   669.,   894.],
       [   39.,   264.,   489.,   714.,   939.],
       [   84.,   309.,   534.,   759.,   984.],
       [  129.,   354.,   579.,   804.,  1029.]])
>>> b = A.transpose() * x; b
array([[  141.,   366.,   591.,   816.,  1041.],
       [  186.,   411.,   636.,   861.,  1086.],
       [  231.,   456.,   681.,   906.,  1131.],
       [  276.,   501.,   726.,   951.,  1176.],
       [  321.,   546.,   771.,   996.,  1221.]])
"""

from numpy import array, arange, bitwise_and, rot90
from scipy.signal import fftconvolve

from padarray import padarray
from syntheticBC import get_syn_bc, del_syn_border

__all__ = ['psfMatrix']

def syn_multiply(self, x):
    padsize = array(self.psf.shape) / 2
    y = fftconvolve(self.psf, x, 'full')
    y = del_syn_border(y, padsize, self.Xind, self.Yind)
    
    # remove first column and/or row if psf is of even size
    if not bitwise_and(self.psf.shape[0], 1):
        y = y[1:,:]
    if not bitwise_and(self.psf.shape[1], 1):
        y = y[:,1:]

    return y

class psfMatrix:
    """
    psfMatrix is a class for image convolution.
    The convolution is done through the overloaded multiplication.
    Since '* is not valid in python, we use ** for transpose
    multiplication.
    Example:
    >>> from psfMatrix import psfMatrix; from numpy import arange
    >>> m = 5; n = 3
    >>> psf = arange(1, n ** 2 + 1); psf = psf.reshape((n,n), order='F')
    >>> x_vec = arange(1, m ** 2 + 1); x = x_vec.reshape((m, m), order='F')
    >>> A = psfMatrix(psf,boundary='zero'); b = A * x; b
    array([[  32.,  114.,  249.,  384.,  440.],
           [  68.,  219.,  444.,  669.,  734.],
           [  89.,  264.,  489.,  714.,  773.],
           [ 110.,  309.,  534.,  759.,  812.],
           [  96.,  252.,  417.,  582.,  600.]])
    >>> b = A.transpose() * x; b
    array([[ 128.,  276.,  441.,  606.,  320.],
           [ 202.,  411.,  636.,  861.,  436.],
           [ 241.,  456.,  681.,  906.,  457.],
           [ 280.,  501.,  726.,  951.,  478.],
           [ 184.,  318.,  453.,  588.,  280.]])
    >>> b = A ** x; b
    array([[ 128.,  276.,  441.,  606.,  320.],
           [ 202.,  411.,  636.,  861.,  436.],
           [ 241.,  456.,  681.,  906.,  457.],
           [ 280.,  501.,  726.,  951.,  478.],
           [ 184.,  318.,  453.,  588.,  280.]])
    """
    def __init__(self, psf, boundary='periodic', b=None,
            Xind=None, Yind=None):
        self.psf = psf
        self.boundary = boundary
        self.center = array(psf.shape) / 2;
        if boundary == 'synthetic':
            if b is not None:
                self.Xind, self.Yind = get_syn_bc(b, self.center[0]) 
            elif (Xind is not None) and (Yind is not None):
                self.Xind = Xind
                self.Yind = Yind
            else:
                raise NameError,\
                        'For synthetic boundary conditions, either b '\
                        +' or Xind and Yind must be given.'
        else:
            self.Xind = None
            self.Yind = None
 
    def __mul__(self, x):
        padsize = array(self.psf.shape) / 2
        x_padded = padarray(x, padsize, self.boundary, 'both', 
                           self.Xind, self.Yind)
        y = fftconvolve(self.psf, x_padded, 'valid')
        
        # remove first column and/or row if psf is of even size
        if not bitwise_and(self.psf.shape[0], 1):
            y = y[1:,:]
        if not bitwise_and(self.psf.shape[1], 1):
            y = y[:,1:]
        return y
    
    def __pow__(self, x):
        padsize = array(self.psf.shape) / 2
        psf = rot90(self.psf, 2)
        x_padded = padarray(x, padsize, self.boundary, 'both')
        y = fftconvolve(psf, x_padded, 'valid')
        
        # remove last column and/or row if psf is of even size
        if not bitwise_and(self.psf.shape[0], 1):
            y = y[:-1,:]
        if not bitwise_and(self.psf.shape[1], 1):
            y = y[:,:-1]
        return y

    def transpose(self):
        from copy import deepcopy
        A_T = deepcopy(self)
        A_T.psf = rot90(A_T.psf, 2)
        if self.boundary == 'synthetic':
            from new import instancemethod
            A_T.__mul__ = instancemethod(syn_multiply, A_T, psfMatrix)
        return A_T

if __name__ == "__main__":
    import doctest
    doctest.testmod()
