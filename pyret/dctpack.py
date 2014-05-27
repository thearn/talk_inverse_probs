#
# Author: Ying Wai (Daniel) Fan, November 2006
#

__all__ = ['circshift', 'dcts', 'dcts2', 'idcts', 'idcts2', 'dctshift']

# Reference: "Deblurring Images - Matrices, Spectra, and Filtering"
#            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
#            SIAM, Philadelphia, 2006.

from math import pi, sqrt

from numpy import arange, ix_, shape, hstack, reshape, vstack, flipud
from numpy import real, zeros, matrix, diag, ones, exp
from numpy.fft import fft, ifft
from numpy.matlib import repmat

def circshift(X, shift):
    """ shift X circularly"""
    m, n = X.shape
    index_x = (arange(m) - shift[0]) % m
    index_y = (arange(n) - shift[1]) % n
    return X[ix_(index_x, index_y)]


def dcts(x):
    """DCTS Model implementation of discrete cosine transform.

         y = dcts(x);

  Compute the discrete cosine transform of x.  
  This is a very simple implementation.  If the Signal Processing
  Toolbox is available, then you should use the function dct.

  Input:
        x  column vector, or a matrix.  If x is a matrix then dcts(x)
           computes the DCT of each column

  Output:
        y  contains the discrete cosine transform of x.

 Reference: See Chapter 4, 
            "Deblurring Images - Matrices, Spectra, and Filtering"
            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
            SIAM, Philadelphia, 2006.
   
 If an FFT routine is available, then it can be used to compute
 the DCT.  Since the FFT is part of the standard MATLAB distribution,
 we use this approach.  For further details on the formulas, see:


            "Computational Frameworks for the Fast Fourier Transform"
            by C. F. Van Loan, SIAM, Philadelphia, 1992.

            "Fundamentals of Digital Image Processing"
            by A. Jain, Prentice-Hall, NJ, 1989."""
    n, m = shape(x)
    omega = exp(-1j*pi/(2*n))
    d = hstack((1/sqrt(2), omega**arange(1,n))) / sqrt(2*n)
    d = reshape(d,(n,1))
    d = repmat(d,1,m)
    xt = vstack((x, flipud(x)))
    yt = fft(xt,axis=0)
    y = real(d * yt[0:n,:])
    return y


def dcts2(x):
    """DCTS2 Model implementation of 2-D discrete cosine transform.

         y = dcts2(x);

  Compute the two-dimensional discrete cosine transform of x.  
  This is a very simple implementation.  If the Image Processing Toolbox 
  is available, then you should use the function dct2.

  Input:
        x  array

  Output:
        y  contains the two-dimensional discrete cosine transform of x.

 Reference: See Chapter 4, 
            "Deblurring Images - Matrices, Spectra, and Filtering"
            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
            SIAM, Philadelphia, 2006.
   
            See also:
            "Computational Frameworks for the Fast Fourier Transform"
            by C. F. Van Loan, SIAM, Philadelphia, 1992.

            "Fundamentals of Digital Image Processing"
            by A. Jain, Prentice-Hall, NJ, 1989.

 The two-dimensional DCT is obtained by computing a one-dimensional DCT of
 the columns, followed by a one-dimensional DCT of the rows."""
    y = dcts(dcts(x).transpose()).transpose()
    return y


def idcts(x):
    """IDCTS Model implementation of inverse discrete cosine transform.

function y = idcts(x)

         y = idcts(x);

  Compute the inverse discrete cosine transform of x.  
  This is a very simple implementation.  If the Signal Processing
  Toolbox is available, then you should use the function idct.

  Input:
        x  column vector, or a matrix.  If x is a matrix then idcts
           computes the IDCT of each column.

  Output:
        y  contains the inverse discrete cosine transform of x.

 Reference: See Chapter 4, 
            "Deblurring Images - Matrices, Spectra, and Filtering"
            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
            SIAM, Philadelphia, 2006.
   
   
 If an inverse FFT routine is available, then it can be used to compute
 the inverse DCT.  Since the inverse FFT is part of the standard MATLAB 
 distribution, we use this approach.  For further details on the formulas,
 see
            "Computational Frameworks for the Fast Fourier Transform"
            by C. F. Van Loan, SIAM, Philadelphia, 1992.

            "Fundamentals of Digital Image Processing"
            by A. Jain, Prentice-Hall, NJ, 1989."""
    n, m = shape(x)
    omega = exp(1j*pi/(2*n))
    d = sqrt(2*n) * omega**arange(0,n)
    d[0] = d[0] * sqrt(2)
    d = reshape(d,(n,1))
    d = repmat(d,1,m)
    xt = vstack((d*x, zeros((1,m)), -1j*d[1:n,:]*flipud(x[1:n,:])))
    yt = ifft(xt,axis=0)
    y = real(yt[0:n,:])
    return y


def idcts2(x):
    """IDCTS2 Model implementation of 2-D inverse discrete cosine transform.

function y = idcts2(x)

         y = idcts2(x);

  Compute the inverse two-dimensional discrete cosine transform of x.  
  This is a very simple implementation.  If the Image Processing Toolbox 
  is available, then you should use the function idct2.

  Input:
        x  array

  Output:
        y  contains the two-dimensional inverse discrete cosine
           transform of x.

 Reference: See Chapter 4, 
            "Deblurring Images - Matrices, Spectra, and Filtering"
            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
            SIAM, Philadelphia, 2006.
   
            See also:
            "Computational Frameworks for the Fast Fourier Transform"
            by C. F. Van Loan, SIAM, Philadelphia, 1992.

            "Fundamentals of Digital Image Processing"
            by A. Jain, Prentice-Hall, NJ, 1989.

 The two-dimensional inverse DCT is obtained by computing a one-dimensional 
 inverse DCT of the columns, followed by a one-dimensional inverse DCT of 
 the rows."""
    y = idcts(idcts(x).transpose()).transpose()
    return y


def dctshift(PSF, center):
    """DCTSHIFT Create array containing the first column of a blurring matrix.

         Ps = dctshift(PSF, center);

  Create an array containing the first column of a blurring matrix
  when implementing reflexive boundary conditions.

  Input:
      PSF  Array containing the point spread function.
   center  [row, col] = indices of center of PSF.

  Output:
       Ps  Array (vector) containing first column of blurring matrix.

 Reference: See Chapter 4, 
            "Deblurring Images - Matrices, Spectra, and Filtering"
            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
            SIAM, Philadelphia, 2006."""
    m,n = shape(PSF)
    i = center[0]
    j = center[1]
    k = min(i,m-i-1,j,n-j-1)
    #
    # The PSF gives the entries of a central column of the blurring matrix.
    # The first column is obtained by reordering the entries of the PSF; for
    # a detailed description of this reordering, see the reference cited
    # above.
    #
    PP = matrix(PSF[i-k:i+k+1,j-k:j+k+1])
    Z1 = matrix(diag(ones((k+1,1),'d').flatten(),k))
    Z2 = matrix(diag(ones((k,1),'d').flatten(),k+1))
    PP = Z1*PP*Z1.T + Z1*PP*Z2.T + Z2*PP*Z1.T + Z2*PP*Z2.T
    Ps = zeros((m,n),'d')
    Ps[0:2*k+1,0:2*k+1] = PP
    return Ps
