#
# Author: Ying Wai (Daniel) Fan, November 2006
#

__all__ = ['psfGauss', 'psfMotion', 'padPSF', 'kronDecomp']

# Reference: "Deblurring Images - Matrices, Spectra, and Filtering"
#            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
#            SIAM, Philadelphia, 2006.

from math import sqrt, pi

from numpy import double, mgrid, fix, exp, array, size, zeros, concatenate
from numpy import finfo, diag, ones, fliplr,sqrt,where,log
from scipy.linalg import svd, toeplitz, hankel,norm
from numpy import dot,transpose
from padarray import *
from scipy.signal import convolve2d

def psfFocus(dim, sigma):
    m = dim[0]
    n = dim[1]
    sigma = 1.1*double(sigma) # make sure sigma is floating-point number
    #
    # Set up grid points to evaluate the Gaussian function.
    #
    X,Y = mgrid[0:m,0:n]
    X = X - fix(m/2)
    Y = Y - fix(n/2)
    #
    # Compute the Gaussian, and normalize the PSF.
    # Scaling by 1 /( sqrt((2*pi)**2)*sigma**2 ) is skipped here, as it has
    # no effect to the normalization.
    #
    #PSF = exp( (-abs(Y)-abs(X))/sigma) 
    #PSF=1/(pi*sigma*(1+(Y/sigma)**2+(X/sigma)**2))
    i=where(X**2+Y**2<sigma)
    PSF=0.0*X*Y
    PSF[i]=1.0
    PSF = PSF / sum(PSF.ravel())
    #
    # Get center ready for output.
    #
    center = array(divmod(PSF.argmax(), n))
    return PSF, center

def psfM(dim,l):
    m = dim[0]
    n = dim[1]

    X,Y = mgrid[0:m,0:n]
    X = X - fix(m/2)
    Y = Y - fix(n/2)
    #
    PSF=zeros(dim)
    #PSF[where(X==Y)]=1
    #PSF[where(abs(X)>l)]=0
    PSF[m/2,n/2] = 1
    PSF[m/2+1,n/2+1] = 1
    #PSF[m/2+2,n/2+2] = 1
    PSF = PSF / sum(PSF.ravel())
    #from pylab import *
    #imshow(PSF)
    #show()
    #quit()
    #
    # Get center ready for output.
    #
    center = array(divmod(PSF.argmax(), n))
    return PSF, center

def psfLap(dim, sigma):
    m = dim[0]
    n = dim[1]
    sigma = 1.1*double(sigma) # make sure sigma is floating-point number
    #
    # Set up grid points to evaluate the Gaussian function.
    #
    X,Y = mgrid[0:m,0:n]
    X = X - fix(m/2)
    Y = Y - fix(n/2)
    #
    # Compute the Gaussian, and normalize the PSF.
    # Scaling by 1 /( sqrt((2*pi)**2)*sigma**2 ) is skipped here, as it has
    # no effect to the normalization.
    #
    #PSF = exp( (-abs(Y)-abs(X))/sigma) 
    #PSF=1/(pi*sigma*(1+(Y/sigma)**2+(X/sigma)**2))
    PSF=1/(2*sigma)*exp(-abs(X)/sigma-abs(Y)/sigma)
    PSF = PSF / sum(PSF.ravel())
    #
    # Get center ready for output.
    #
    center = array(divmod(PSF.argmax(), n))
    return PSF, center
	
def psfPar(dim, sigma):
    m = dim[0]
    n = dim[1]
    sigma = 1.1*double(sigma) # make sure sigma is floating-point number
    #
    # Set up grid points to evaluate the Gaussian function.
    #
    X,Y = mgrid[0:m,0:n]
    X = X - fix(m/2)
    Y = Y - fix(n/2)
    #
    # Compute the Gaussian, and normalize the PSF.
    # Scaling by 1 /( sqrt((2*pi)**2)*sigma**2 ) is skipped here, as it has
    # no effect to the normalization.
    #
    #PSF = exp( (-abs(Y)-abs(X))/sigma) 
    PSF=1/(pi*sigma*(1+(Y/sigma)**2+(X/sigma)**2))
    #PSF=1/(pi**2)*((sigma**2)/((X**2+sigma**2)*(Y**2+sigma**2)))
    PSF = PSF / sum(PSF.ravel())
    #
    # Get center ready for output.
    #
    center = array(divmod(PSF.argmax(), n))
    return PSF, center

def psfGR1(dim,sigma):
    m = dim[0]
    n = dim[1]
    mm=max([m,n])
    from pylab import imshow,show,figure
    XX=array(range(0,mm))
    XX=array([XX-mm/2.])
    XX=exp(-XX**2/(2 * sigma**2))
    XM= dot(transpose(XX),XX)
    if m>n:
        XM=XM[:,(m-n)/2:-(m-n)/2]
    elif n>m:
        XM=XM[(n-m)/2:-(n-m)/2,:]
    PSF = XM / sum(XM.ravel())
    center = array(divmod(PSF.argmax(), n))
    return PSF, center    
def psfGauss(dim, sigma,inn=False):
    m = dim[0]
    n = dim[1]
    sigma = double(sigma) 
    X,Y = mgrid[0:m,0:n]
    X = X - fix(m/2)
    Y = Y - fix(n/2)
    
    PSF = exp( -(X**2 + Y**2) / (2 * sigma**2) ) 
    PSF = PSF / sum(PSF.ravel())
    center = array(divmod(PSF.argmax(), n))
    if inn:
        nn= int(8*sigma)
        
        inner=PSF[center[0]-nn:center[0]+nn+1,center[1]-nn:center[1]+nn+1]
        n,m=inner.shape
        
        center2 = array(divmod(inner.argmax(), n))
        
        m,n=convolve2d(PSF,inner,mode='valid').shape

        X,Y = mgrid[0:m,0:n]
        X = X - fix(m/2)
        Y = Y - fix(n/2)
        
        PSF2 = exp( -(X**2 + Y**2) / (2 * sigma**2) ) 
        PSF2 = PSF2 / sum(PSF2.ravel())   
        centerr= array(divmod(PSF2.argmax(), n))     
        return PSF2,centerr,inner,center2
    else:
        return PSF,center

def psfMotion(dim, direction='diagonal'):
    """psfMotion Array with point spread function for motion blur.

    PSF, center = psfMotion(dim, direction)

    Construct a motion blur point spread function. 

    Input:
    dim       : Desired dimension of the PSF array.  For example,
    direction : direction of the motion blur. It can be 'horizontal',
                'vertical', 'diagonal', 'antidiagonal'. 
                The default is 'diagonal'.

    Output:
    PSF  Array containing the point spread function.
    center  [row, col] gives index of center of PSF"""
    m = int(dim)
    center = array([m/2, m/2])

    if direction == 'diagonal':
        PSF = diag(ones(m,'d')) / m;
    elif direction == 'horizontal':
        PSF = zeros((m,m))
        PSF[m/2, :] = 1./m;
    elif direction == 'vertical':
        PSF = zeros((m,m))
        PSF[:, m/2] = 1./m;
    elif direction == 'antidiagonal':
        PSF = diag(ones(m,'d')) / m;
        PSF = fliplr(PSF)
    else:
        raise NameError, "direction must be either 'horizontal', " + \
                "'vertical', 'diagonal' or 'antidiagonal'."
    return PSF, center

def padPSF(PSF, m, n):
    """PADPSF Pad a PSF array with zeros to make it bigger.

    function P = padPSF(PSF, m, n)

        P = padPSF(PSF, m);
        P = padPSF(PSF, m, n);
        P = padPSF(PSF, [m,n]);

      Pad PSF with zeros to make it an m-by-n array. 
    
      If the PSF is an array with dimension smaller than the blurred image,
      then deblurring codes may require padding first, such as:
          PSF = padPSF(PSF, size(B));
      where B is the blurred image array.
    
      Input:
          PSF  Array containing the point spread function.
         m, n  Desired dimension of padded array.  
                 If only m is specified, and m is a scalar, then n = m.
    
      Output:
            P  Padded m-by-n array."""
    #
    # Pad the PSF with zeros.
    #
    P = zeros((m, n),'d')
    P[0:size(PSF,0), 0:size(PSF,1)] = PSF
    return P


def kronDecomp(P, center, BC):
    def buildToep(c, k):
        """ Build a banded Toeplitz matrix from a central column and an index
        denoting the central column."""

        n = len(c)
        col = zeros(n,'d')
        col[0:n-k] = c[k:n]
        row = zeros(n,'d')
        row[0:k+1] = c[k::-1]
        T = toeplitz(col, row)
        return T

    def buildCirc(c, k):
        """Build a banded circulant matrix from a central column and an index
     denoting the central column."""

        n = len(c)
        col = concatenate((c[k:], c[:k]))
        row = concatenate((c[k::-1], c[:k:-1]))
        C = toeplitz(col, row)
        return C

    def buildHank(c, k):
        """ Build a Hankel matrix for separable PSF and reflexive boundary
     conditions."""

        n = len(c)
        col = zeros(n,'d')
        col[0:n-k-1] = c[k+1:n]
        row = zeros(n,'d')
        row[n-k:n] = c[0:k]
        H = hankel(col, row)
        return H

    ############################## main function ##############################
    #
    # Find the two largest singular values and corresponding singular vectors
    # of the PSF -- these are used to see if the PSF is separable.
    #
    U, S, V = svd(P)
    V = V.transpose()                 # since P = USV not P=USV' as in MATLAB
    eps = finfo(float).eps
    if ( S[1] / S[0] > sqrt(eps) ):
        print('The PSF, P is not separable; using separable approximation.')
    # 
    # Since the PSF has nonnegative entries, we would like the vectors of the
    # rank-one decomposition of the PSF to have nonnegative components.  That
    # is, the singular vectors corresponding to the largest singular value of P
    # should have nonnegative entries. The next few statements check this, and 
    # change sign if necessary.
    #
    minU = abs(min(U[:,0]))
    maxU = max(abs(U[:,0]))
    if minU == maxU:
        U = -U
        V = -V
    # 
    # The matrices Ar and Ac are defined by vectors r and c, respectively.
    # These vectors can be computed as follows:
    #
    c = sqrt(S[0])*U[:,0]
    r = sqrt(S[0])*V[:,0]
    #
    # The structure of Ar and Ac depends on the imposed boundary condition.
    #
    if BC == 'zero':
        # Build Toeplitz matrices here
        Ar = buildToep(r, center[1])
        Ac = buildToep(c, center[0])
    elif BC == 'reflexive':
        # Build Toeplitz-plus-Hankel matrices here
        Ar = buildToep(r, center[1]) + buildHank(r, center[1])
        Ac = buildToep(c, center[0]) + buildHank(c, center[0])
    elif BC == 'periodic':
        # Build circulant matrices here
        Ar = buildCirc(r, center[1])
        Ac = buildCirc(c, center[0])
    else:
        print('Invalid boundary condition.')

    return Ar, Ac

if __name__ == "__main__":
    from pylab import *
    '''
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt    
    '''
    psf, cent,psf2,cent2 = psfGauss([100,100],3)
    figure()
    imshow(psf)
    figure()
    imshow(psf2)
    show()
