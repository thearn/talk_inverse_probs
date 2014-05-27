# from numpy import arange, ix_, array, abs, real
from scipy.fftpack import fftn, ifftn
from gcvpack import gcv_tsvd, gcv_tik

try:
    # dct and idct will be available in SciPy 0.8.0
    from scipy.fftpack import dct, idct
    def dct2(x):
        y = dct(x, axis=0, norm='ortho')
        y = dct(y, axis=1, norm='ortho')
        return y
    def idct2(x):
        y = idct(x, axis=0, norm='ortho')
        y = idct(y, axis=1, norm='ortho')
        return y
except ImportError:
    from dctpack import dcts2 as dct2, idcts2 as idct2


from math import sqrt

from numpy import array, maximum, minimum, max, abs, zeros_like, fix

from padarray import padarray
from dctpack import circshift, dctshift

__all__ = ['fftPrec', 'dctPrec', 'identityPrec']

def circEig(PSF, center):
    """
            E = circEig(PSF, center)

    Compute the eigenvalues of the "Strang" circulant preconditioner,
    that is, find the eigenvalues of the circulant matrix that minimizes

                  || A - C ||
                             1

    where A is a psfMatrix.

    Input: 
        PSF  -  array containing a single PSF
     center  -  array containing the indices of the center of the PSF

    Output:
          E  -  array containing the complex eigenvalues of the
                circulant preconditioner
    """
    return fftn( circshift(PSF, -array(center)) )

def dctEig(PSF, center):
    """
            E = dctEig(PSF, center)

    Compute the eigenvalues of the DCT circulant preconditioner,
    that is, find the eigenvalues of the BTHTHB matrix that minimizes

                  || A - C ||
                             1

    where A is a psfMatrix.

    Input: 
        PSF  -  array containing a single PSF
     center  -  array containing the indices of the center of the PSF

    Output:
          E  -  array containing the complex eigenvalues of the
                circulant preconditioner
    """
    e1 = zeros_like(PSF)
    e1[0,0] = 1
    return dct2( dctshift(PSF, center) ) / dct2(e1)

def adjustPsfSize(psf, center, b):
    #
    # First determine if any of the dimensions of the PSF are larger than 
    # the corresponding dimensions of the image, b.
    # If so, we need to extract a subimage of the PSF.
    #
    b_shape = array(b.shape)
    psf_shape = array(psf.shape)
    
    t = fix( b_shape / 2 ).astype('int')
    t1 = maximum( center - t, 0 )           # starting index 
    t2 = minimum( t1 + b_shape, psf_shape)  # ending index + 1
    if b.ndim == 1:
        psf = psf[ t1[0] : t2[0] ]
    elif b.ndim == 2:
        psf = psf[ t1[0]:t2[0], t1[1]:t2[1] ]
    else:
        psf = psf[ t1[0]:t2[0], t1[1]:t2[1], t1[2]:t2[2] ]

    #
    # The center has moved, so let's find the new center
    center = center - t1

    #
    # Now determine if any of the dimensions of the PSF are smaller than
    # the corresponding dimensions of the image, b.  If so, we need to
    # pad the array.
    #
    psf = padarray(psf, b_shape - psf_shape, 'zero', 'post')

    return psf, center

def constructTsvdPrecMatrix(psf, center, b, tol=None,
                            EigFunction=circEig, Transform=fftn):
    """
        precMatData = constructPrecMatrix(psf, center, b, tol)
    
    Construct the data needed for a regularized circulant preconditioner
    
    Input:
        psf  -  array containing a PSF
     center  -  array containing the indices of the center of the PSF
          b  -  blurred image to be restored
    
    Optional Input:
        tol         -  truncation tolerance for preconditioner
        EigFunction -  function to obtain the eigenvalues of the
                       convolution matrix
        Transform   -  transform for spectral decomposition of  
    
    Output:
      precMatData  -  inverse of the (truncated) eigenvalues
    """
    psf, center = adjustPsfSize(psf, center, b)

    #
    # Now compute the eigenvalues of the circulant preconditioner:
    #
    E = EigFunction(psf, center)

    #
    # If we were not given a truncation tolerance, let's try to
    # find one:
    #
    if tol is None:
        bhat = Transform(b)
        tol = gcv_tsvd(E.ravel(),bhat.ravel())

    maxE = max( abs( E.ravel() ) )
    E = E / maxE;                  # scale the eigenvalues, so max is 1
    idx = abs(E) < tol;            # then compare to tol
    E = E * maxE;                  # scale back
    E[idx] = 1;                    # replace small eigenvalues by 1
    E = 1 / E;                    # invert

    precMatData = E
    alpha = None
    return precMatData, alpha, tol

def constructTikPrecMatrix(psf, center, b, alpha=None,
                            EigFunction=circEig, Transform=fftn):
    """
        precMatData = constructPrecMatrix(psf, center, b, alpha)
    
    Construct the data needed for a regularized circulant preconditioner
    
    Input:
        psf  -  array containing a PSF
     center  -  array containing the indices of the center of the PSF
          b  -  blurred image to be restored
    
    Optional Input:
        alpha       -  regularization parameter
        EigFunction -  function to obtain the eigenvalues of the
                       convolution matrix
        Transform   -  transform for spectral decomposition of  
    
    Output:
      precMatData  -  inverse of the (truncated) eigenvalues
    """
    psf, center = adjustPsfSize(psf, center, b)

    #
    # Now compute the eigenvalues of the circulant preconditioner:
    #
    E = EigFunction(psf, center)

    #
    # If we were not given the regularization parameter, let's try to
    # find one:
    #
    if alpha is None:
        bhat = Transform(b)
        alpha = gcv_tik(E.ravel(),bhat.ravel())

    precMatData = E.conj() / ( E.conj() * E + alpha ** 2 )
    tol = None
    return precMatData, alpha, tol

class Prec:
    """
    Base Preconditioner class. fftPrec and dctPrec are subclasses of this.
    """
    def __init__(self, A, b, tol, alpha):
        if self.regularization == 'tsvd':
            self.matdata, self.alpha, self.tol \
                    = constructTsvdPrecMatrix(A.psf, A.center, b, tol,
                                          self.EigFunction, self.Transform)
        else:
            self.matdata, self.alpha, self.tol \
                    = constructTikPrecMatrix(A.psf, A.center, b, alpha,
                                          self.EigFunction, self.Transform)
        self.isinverse = True # True when approximating A^{-1}

    def inverse(self):
        from copy import deepcopy
        P_inv = deepcopy(self)
        P_inv.matdata = 1 / self.matdata
        P_inv.isinverse = True if not self.isinverse else False
        return P_inv
    
    def __mul__(self, x):
        T = self.Transform
        T_inv = self.InverseTransform
        if x.ndim == 1 or min(x.shape) == 1:
            m = sqrt(x.size)
            X = x.reshape((m,m), order='F')
            Y = (T_inv(T(X) * self.matdata)).real
            y = Y.reshape(x.shape, order='F')
        else:
            y = (T_inv(T(x) * self.matdata)).real
        return y

class fftPrec(Prec):
    """
    The fftPrec class is based on a structure with three fields:
      matdata        - matrix data needed to do matrix-vector solves with
                       the preconditioner
      regularization - 'tsvd' or 'tikhonov' to indicate which
                       regularization is being used
      isinverse      - True  : the matrix is approximating A^{-1}
                       False : the matrix is approximating A
                       the default is True

    Calling Syntax:
         P = fftPrec(A, b, regularization='tsvd', tol=None, alpha=None)
      where 
         * A   is a psfMatrix object or a multiPsfMatrix
         * b   is the right hand side image for the system Ax=b that
               is being preconditioned
         * regularization 
               'tsvd' or 'tikhonov' to indicate which regularization to use
         * tol is a tolerance to "regularize" the preconditioner (e.g.,
               the Hanke, Nagy, Plemmons approach)
               If tol is not specified, a default will be chosen using
               the generalized cross validation method.
    Example:
    >>> from numpy import set_printoptions; set_printoptions(suppress=True)
    >>> from psfMatrix import *; from preconditioners import *                      
    >>> m = 256; n = 11; s = 3                                              
    >>> from scipy.misc import imread
    >>> x = (imread('Barbara.png')).astype('float64')
    >>> from psfpack import psfGauss; psf, center = psfGauss(n,s)
    >>> A = psfMatrix(psf,boundary='periodic'); b = A * x
    >>> P = fftPrec(A, b); P.matdata
    array([[ 1.00000000+0.j,  1.00192410-0.j,  1.00772413-0.j, ...,
             1.01748409-0.j,  1.00772413-0.j,  1.00192410-0.j],
           [ 1.00192410-0.j,  1.00385190-0.j,  1.00966309-0.j, ...,
             1.01944183-0.j,  1.00966309-0.j,  1.00385190-0.j],
           [ 1.00772413-0.j,  1.00966309-0.j,  1.01550792-0.j, ...,
             1.02534326-0.j,  1.01550792-0.j,  1.00966309-0.j],
           ..., 
           [ 1.01748409-0.j,  1.01944183-0.j,  1.02534326-0.j, ...,
             1.03527387-0.j,  1.02534326-0.j,  1.01944183-0.j],
           [ 1.00772413-0.j,  1.00966309-0.j,  1.01550792-0.j, ...,
             1.02534326-0.j,  1.01550792-0.j,  1.00966309-0.j],
           [ 1.00192410-0.j,  1.00385190-0.j,  1.00966309-0.j, ...,
             1.01944183-0.j,  1.00966309-0.j,  1.00385190-0.j]])
    >>> x_prec = P * b; x_prec
    array([[ 134.14601231,  125.95378917,  111.93546683, ...,  110.97380202,
             110.12441115,  104.82303998],
           [  68.00443435,   68.90002562,   80.14426001, ...,  113.09176057,
             109.87464787,  103.0879717 ],
           [ 164.84803184,  170.18048821,  179.87077497, ...,  109.90295271,
             108.04395154,  103.05880365],
           ..., 
           [  40.03487956,   44.06820879,   52.8670285 , ...,   99.91932979,
             102.13353591,   96.8826807 ],
           [  36.12334474,   34.8285707 ,   43.14187575, ...,   95.10218299,
             101.93165962,  102.97588658],
           [  37.79945391,   40.16204097,   39.94241564, ...,  106.9434264 ,
              98.95825328,  104.14970647]])
    """
    def __init__(self, A, b, regularization='tsvd', tol=None, alpha=None):
        self.EigFunction = circEig
        self.Transform = fftn
        self.InverseTransform = ifftn
        self.regularization = regularization
        Prec.__init__(self, A, b, tol, alpha)
        self.isinverse = True # True when approximating A^{-1}

    def transpose(self):
        from copy import deepcopy
        P_T = deepcopy(self)
        P_T.matdata = P_T.matdata.conj()
        return P_T
    
class dctPrec(Prec):
    """
    The dctPrec class is based on a structure with three fields:
      matdata        - matrix data needed to do matrix-vector solves with
                       the preconditioner
      regularization - 'tsvd' or 'tikhonov' to indicate which
                       regularization is being used
      isinverse      - True  : the matrix is approximating A^{-1}
                       False : the matrix is approximating A
                       the default is True

    Calling Syntax:
         P = dctPrec(A, b, regularization='tsvd', tol=None, alpha=None)
      where 
         * A   is a psfMatrix object or a multiPsfMatrix
         * b   is the right hand side image for the system Ax=b that
               is being preconditioned
         * regularization 
               'tsvd' or 'tikhonov' to indicate which regularization to use
         * tol is a tolerance to "regularize" the preconditioner (e.g.,
               the Hanke, Nagy, Plemmons approach)
               If tol is not specified, a default will be chosen using
               the generalized cross validation method.
    Example:
    >>> from numpy import set_printoptions; set_printoptions(suppress=True)
    >>> from psfMatrix import *; from preconditioners import *                      
    >>> m = 256; n = 11; s = 3                                              
    >>> from scipy.misc import imread
    >>> x = (imread('Barbara.png')).astype('float64')
    >>> from psfpack import psfGauss; psf, center = psfGauss(n,s)
    >>> A = psfMatrix(psf,boundary='reflexive'); b = A * x
    >>> P = dctPrec(A, b); P.matdata
    array([[    1.        ,     1.00048059,     1.0019241 , ...,
              -39.89990971,   -39.42068129,   -39.13776345],
           [    1.00048059,     1.00096142,     1.00240562, ...,
              -39.91908534,   -39.43962662,   -39.1565728 ],
           [    1.0019241 ,     1.00240562,     1.0038519 , ...,
              -39.97668114,   -39.49653064,   -39.21306843],
           ..., 
           [  -39.89990971,   -39.91908534,   -39.97668114, ...,
             1592.00279488,  1572.88162439,  1561.59322789],
           [  -39.42068129,   -39.43962662,   -39.49653064, ...,
             1572.88162434,  1553.99011372,  1542.83729949],
           [  -39.13776345,   -39.1565728 ,   -39.21306843, ...,
             1561.59322792,  1542.83729956,  1531.76452784]])
    >>> x_prec = P * b; x_prec
    array([[ 133.94796273,  126.12407696,  111.87911474, ...,  110.91339014,
             110.102363  ,  104.95539683],
           [  68.06094762,   68.85220196,   80.15213937, ...,  113.07248658,
             109.90266315,  103.04367148],
           [ 164.98773457,  170.03317428,  179.95475465, ...,  110.02553889,
             107.98833144,  103.00308718],
           ..., 
           [  39.92056681,   44.19121478,   52.80773193, ...,   99.88902657,
             102.13971346,   96.93819905],
           [  36.03912332,   34.90292168,   43.10705964, ...,   95.02074836,
             101.95760112,  103.02040301],
           [  38.00932882,   39.98043181,   40.01026385, ...,  107.04681888,
              98.95727366,  104.01725735]])
    """
    def __init__(self, A, b, regularization='tsvd', tol=None, alpha=None):
        self.EigFunction = dctEig
        self.Transform = dct2
        self.InverseTransform = idct2
        self.regularization = regularization
        Prec.__init__(self, A, b, tol, alpha)
        self.isinverse = True # True when approximating A^{-1}

    def transpose(self):
        return self
    
class identityPrec:
    """This is just a dummy identity preconditioner to be used in
    preconditioned iterative methods when there is no preconditioning."""
    def __init__(self, *args, **kwargs):
        pass
    
    def __mul__(self, x):
        return x

    def transpose(self):
        return self


if __name__ == "__main__":
    import doctest
    doctest.testmod()
