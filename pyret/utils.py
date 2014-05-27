__all__ = ['psnr', 'relative_error']

from numpy.linalg import norm

def psnr(A, B):
    """
    Compute the peak signal-to-noise ratio (PSNR) between two images.
        
    PSNR is defined as

        PSNR = 20 * log10( b / rms )

    where b is the largest possible value of the image (typically 255) and
    rms is the root-meean-square difference between the two images.
    """
    from math import sqrt, log10
    from numpy import mean
    b = 255
    rms = sqrt(mean((A - B) ** 2))
    return 20 * log10( b / rms )

def relative_error(P, T):
    """
    Compute the relative error between prediction P and target T.

        relative error = norm(P - T) / norm(T)
    """
    from scipy.linalg import norm
    return norm(P - T) / norm(T)
