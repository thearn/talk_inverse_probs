#
# Author: Ying Wai (Daniel) Fan, November 2006
#

__all__ = ['tsvd_dct', 'tik_dct', 'tsvd_fft', 'tik_fft', 'tsvd_sep', 'tik_sep']

#     Reference: See Chapter 6, 
#     "Deblurring Images - Matrices, Spectra, and Filtering"
#     by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
#     SIAM, Philadelphia, 2006.

from numpy import zeros, shape, where, conj, reshape, asmatrix, asarray,clip,log10
from numpy import dot, matrix, real, array
from scipy.linalg import svd, kron
from scipy.fftpack import fft2, ifft2

from pylab import *
from padarray import padarray

from scipy.stats import histogram
from scipy.optimize import fmin,fmin_cg

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

from dctpack import dctshift, circshift
from gcvpack import gcv_tsvd, gcv_tik
from psfpack import kronDecomp

def tsvd_dct(B, PSF, center=None, tol=None):
    """TSVD_DCT Truncated SVD image deblurring using the DCT algorithm.

    X, tol = tsvd_dct(B, PSF, center, tol)

  Compute restoration using a DCT-based truncated spectral factorization.

  Input:
        B  Array containing blurred image.
      PSF  Array containing the point spread function.

  Optional Inputs:
   center  [row, col] = indices of center of PSF.
      tol  Regularization parameter (truncation tolerance).
             Default parameter chosen by generalized cross validation.

  Output:
        X  Array containing computed restoration.
      tol  Regularization parameter used to construct restoration."""

    #
    # compute the center of the PSF if it is not provided
    #
    if center is None:
        center = array(PSF.shape) / 2

    #
    # if PSF is smaller than B, pad it to the same size as B
    #
    if PSF.size < B.size:
        PSF = padarray(PSF, array(B.shape) - array(PSF.shape),
                       direction='post')
    
    #
    # Use the DCT to compute the eigenvalues of the symmetric 
    # BTTB + BTHB + BHTB + BHHB blurring matrix.
    #
    e1 = zeros(shape(PSF),'d')
    e1[0,0] = 1

    bhat = dct2(B)
    S = dct2( dctshift(PSF, center) ) / dct2(e1)

    #
    # If a regularization parameter is not given, use GCV to find one.
    #
    if tol is None:
        tol = gcv_tsvd(S.flatten('f'), bhat.flatten('f'))

    #
    # Compute the TSVD regularized solution.
    #
    idx = where(abs(S) >= tol)
    Sfilt = zeros(shape(bhat),'d')
    Sfilt[idx] = 1.0 / S[idx]
    X = idct2(bhat * Sfilt)

    return X, tol

def tik_dct(B, PSF, center=None, alpha=None,clipsol=False):
    """TIK_DCT Tikhonov image deblurring using the DCT algorithm.

    X, alpha = tik_dct(B, PSF, center, alpha)

    Compute restoration using a DCT-based Tikhonov filter, 
    with the identity matrix as the regularization operator.

    Input:
    B  Array containing blurred image.
    PSF  Array containing the point spread function.
  
    Optional Inputs:
    center  [row, col] = indices of center of PSF.
    alpha  Regularization parameter.
    Default parameter chosen by generalized cross validation.

    Output:
    X  Array containing computed restoration.
    alpha  Regularization parameter used to construct restoration. """
    #
    # compute the center of the PSF if it is not provided
    #
    if center is None:
        center = array(PSF.shape) / 2
    #
    # if PSF is smaller than B, pad it to the same size as B
    #
    if PSF.size < B.size:
        PSF = padarray(PSF, array(B.shape) - array(PSF.shape),
                       direction='post')
    #
    # Use the DCT to compute the eigenvalues of the symmetric 
    # BTTB + BTHB + BHTB + BHHB blurring matrix.
    #
    e1 = zeros(shape(PSF))
    e1[0,0] = 1
    bhat = dct2(B)
    '''
    ss=bhat.flatten()
    print ss[0]/ss[-1]
    hs=histogram(ss,numbins=50)[0]
    figure()
    plot(hs)
    return
    '''
    
    S = dct2( dctshift(PSF, center) ) / dct2(e1)
    
    
    #
    # If a regularization parameter is not given, use GCV to find one.
    #
    bhat = bhat.flatten('f')
    s = S.flatten('f')
    if alpha is None:
        alpha = gcv_tik(s, bhat)
        print 'alpha:',alpha
    #
    # Compute the Tikhonov regularized solution.
    #
    D = conj(s)*s + abs(alpha)**2
    bhat = conj(s) * bhat
    xhat = bhat / D
    xhat = reshape(asmatrix(xhat), shape(B),'f')
    xhat = asarray(xhat)
    X = idct2(xhat)
    
    if clipsol:
        N,M=X.shape
        m=5
        xin=X[N/m:-N/m,M/m:-M/m]
        mn,mx= xin.min(),xin.max()    
        X=clip(X,mn,mx)
    return X,alpha


def tsvd_fft(B, PSF, center=None, tol=None,i=None):
    """TSVD_FFT Truncated SVD image deblurring using the FFT algorithm.

    X, tol = tsvd_fft(B, PSF, center, tol)

  Compute restoration using an FFT-based truncated spectral factorization.

  Input:
        B  Array containing blurred image.
      PSF  Array containing the point spread function.
  
  Optional Inputs:
   center  [row, col] = indices of center of PSF.
      tol  Regularization parameter (truncation tolerance).
             Default parameter chosen by generalized cross validation.

  Output:
        X  Array containing computed restoration.
      tol  Regularization parameter used to construct restoration."""
    #
    # compute the center of the PSF if it is not provided
    #
    if center is None:
        center = array(PSF.shape) / 2

    #
    # if PSF is smaller than B, pad it to the same size as B
    #
    if PSF.size < B.size:
        PSF = padarray(PSF, array(B.shape) - array(PSF.shape),
                       direction='post')
    
    #
    # Use the FFT to compute the eigenvalues of the BCCB blurring matrix.
    #
    S = fft2( circshift(PSF, 0-center) )
    #
    # If a regularization parameter is not given, use GCV to find one.
    #
    bhat = fft2(B)
    if i !=None:
        ev=abs(S.flatten())
        ev.sort()
        ev=ev[::-1]
        tol=ev[i] 
    elif tol is None:
        tol = gcv_tsvd(S.flatten('f'), bhat.flatten('f'))
	

    #
    # Compute the TSVD regularized solution.
    #
    
    idx = where(abs(S) >= tol)
    Sfilt = zeros(shape(bhat),'d')
    Sfilt[idx] = 1 / S[idx]
    X = real(ifft2(bhat * Sfilt))

    return X, sorted(S.flatten())[::-1]

    

def tik_fft(B, PSF, center=None, alpha=None,sigma=None):
    """TIK_FFT Tikhonov image deblurring using the FFT algorithm.

    X, alpha = tik_fft(B, PSF, center, alpha)

    Compute restoration using an FFT-based Tikhonov filter, 
    with the identity matrix as the regularization operator.

    Input:
    B  Array containing blurred image.
    PSF  Array containing the point spread function.
  
    Optional Inputs:
    center  [row, col] = indices of center of PSF.
    alpha  Regularization parameter.
    Default parameter chosen by generalized cross validation.
    
    Output:
    X  Array containing computed restoration.
    alpha  Regularization parameter used to construct restoration.
    """
    #
    # compute the center of the PSF if it is not provided
    #
    if center is None:
        center = array(PSF.shape) / 2
    #
    # if PSF is smaller than B, pad it to the same size as B
    #
    if PSF.size < B.size:
        PSF = padarray(PSF, array(B.shape) - array(PSF.shape),
                       direction='post')
    #
    # Use the FFT to compute the eigenvalues of the BCCB blurring matrix.
    #
    S = fft2( circshift(PSF, 0-center) )
    s = S.flatten('f')
    bhat = fft2(B)
    bhat = bhat.flatten('f')
    #
    # If a regularization parameter is not given, use GCV to find one.
    #
##    if alpha is None:
    def tik_fft_obj(alpha,B,PSF,sigma):
        return abs(norm(tik_fft(B, PSF,alpha=abs(alpha)),2)-sigma) 
    
    if alpha is None:
        alpha = gcv_tik(s, bhat)
        print "alpha:", alpha,log10(alpha)

    #
    # Compute the Tikhonov regularized solution.
    #
    D = conj(s)*s + abs(alpha**2)
    bhat = conj(s) * bhat
    xhat = bhat / D
    xhat = reshape(asmatrix(xhat), shape(B), 'f')
    X = real(ifft2(xhat))
    return X,alpha


def tsvd_sep(B, PSF, center=None, tol=None, BC='zero'):
    """TSVD_SEP Truncated SVD image deblurring using Kronecker decomposition.

    X, tol = tsvd_sep(B, PSF, center, tol, BC)

  Compute restoration using a Kronecker product decomposition and
  a truncated SVD.

  Input:
        B  Array containing blurred image.
      PSF  Array containing the point spread function.
  
  Optional Inputs:
   center  [row, col] = indices of center of PSF.
      tol  Regularization parameter (truncation tolerance).
             Default parameter chosen by generalized cross validation.
       BC  String indicating boundary condition.
             ('zero', 'reflexive', or 'periodic'; default is 'zero'.)

  Output:
        X  Array containing computed restoration.
      tol  Regularization parameter used to construct restoration."""
    #
    # compute the center of the PSF if it is not provided
    #
    if center is None:
        center = array(PSF.shape) / 2

    #
    # if PSF is smaller than B, pad it to the same size as B
    #
    if PSF.size < B.size:
        PSF = padarray(PSF, array(B.shape) - array(PSF.shape),
                       direction='post')

    #
    # First compute the Kronecker product terms, Ar and Ac, where
    # A = kron(Ar, Ac).  Note that if the PSF is not separable, this
    # step computes a Kronecker product approximation to A.
    #
    Ar, Ac = kronDecomp(PSF, center, BC)

    #
    # Compute SVD of the blurring matrix.
    #
    Ur, sr, Vr = svd(Ar)
    Uc, sc, Vc = svd(Ac)

    bhat = dot( dot( Uc.transpose(), B ), Ur )
    bhat = bhat.flatten('f')
    s = kron(matrix(sr),matrix(sc))
    s = array(s.flatten('f'))
    
    #
    # If a regularization parameter is not given, use GCV to find one.
    #
    if tol is None:
        tol = gcv_tsvd( s, bhat.flatten('f') )

    #
    # Compute the TSVD regularized solution.
    #
    idx = where(abs(s) >= tol)
    Sfilt = zeros(shape(bhat),'d')
    Sfilt[idx] = 1 / s[idx]
    Bhat = reshape(bhat * Sfilt , shape(B))
    Bhat = Bhat.transpose()
    X = dot( dot(Vc.transpose(),Bhat), Vr )

    return X, tol


def tik_sep(B, PSF, center=None, alpha=None, BC=None):
    """TIK_SEP Tikhonov image deblurring using the Kronecker decomposition.

    X, alpha = tik_sep(B, PSF, center, alpha, BC)

  Compute restoration using a Kronecker product decomposition and a
  Tikhonov filter, with the identity matrix as the regularization operator.

  Input:
        B  Array containing blurred image.
      PSF  Array containing the point spread function.
   center  [row, col] = indices of center of PSF.
  
  Optional Inputs:
    alpha  Regularization parameter.
             Default parameter chosen by generalized cross validation.
       BC  String indicating boundary condition.
             ('zero', 'reflexive', or 'periodic')
           Default is 'zero'.

  Output:
        X  Array containing computed restoration.
    alpha  Regularization parameter used to construct restoration."""
    #
    # compute the center of the PSF if it is not provided
    #
    if center is None:
        center = array(PSF.shape) / 2

    #
    # if PSF is smaller than B, pad it to the same size as B
    #
    if PSF.size < B.size:
        PSF = padarray(PSF, array(B.shape) - array(PSF.shape),
                       direction='post')

    #
    # First compute the Kronecker product terms, Ar and Ac, where
    # the blurring matrix  A = kron(Ar, Ac).  
    # Note that if the PSF is not separable, this
    # step computes a Kronecker product approximation to A.
    #
    Ar, Ac = kronDecomp(PSF, center, BC)

    #
    # Compute SVD of the blurring matrix.
    #
    Ur, sr, Vr = svd(Ar)
    Uc, sc, Vc = svd(Ac)

    bhat = dot( dot(Uc.transpose(), B), Ur )
    bhat = bhat.flatten('f')
    s = kron(matrix(sr),matrix(sc))
    s = array(s.flatten('f'))

    #
    # If a regularization parameter is not given, use GCV to find one.
    #
    if alpha is None:
        alpha = gcv_tik(s, bhat)

    #
    # Compute the Tikhonov regularized solution.
    #
    D = abs(s)**2 + abs(alpha)**2
    bhat = s * bhat
    xhat = bhat / D
    xhat = reshape(xhat, shape(B))
    xhat = xhat.transpose()
    X = dot( dot(Vc.transpose(),xhat), Vr)

    return X, alpha
