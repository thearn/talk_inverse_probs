#
# Author: Ying Wai (Daniel) Fan, November 2006
#

__all__ = ['gcv_tik', 'gcv_tsvd', 'gcv_gtik']

# Reference: "Deblurring Images - Matrices, Spectra, and Filtering"
#            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
#            SIAM, Philadelphia, 2006.

from numpy import argsort, flipud, take, size, zeros, inf, argmin
from scipy.optimize import fminbound,brute


def gcv_tik(s, bhat):
    """GCV_TIK Choose GCV parameter for Tikhonov image deblurring.
    
    function alpha = gcv_tik(s, bhat)
    
    alpha = gcv_tik(s, bhat);
    
    This function uses generalized cross validation (GCV) to choose
    a regularization parameter for Tikhonov filtering.
    
    Input:
    s  Vector containing singular or spectral values
    of the blurring matrix.
    bhat  Vector containing the spectral coefficients of the blurred
    image.
    
    Output:
    alpha  Regularization parameter."""
    def GCV(alpha, s, bhat):
        """This is a nested function that evaluates the GCV function for
        Tikhonov filtering.  It is called by fminbnd."""
        phi_d = 1 / (abs(s)**2 + alpha**2)
        G = sum( abs(bhat*phi_d) ** 2 ) / (sum(phi_d)**2)
        return G
    alpha = fminbound(GCV, min(abs(s)), max(abs(s)), (s, bhat));
    #alpha=brute(GCV,[ (0, 10)],Ns=1000,args=(s,bhat))
    return alpha


def gcv_tsvd(s, bhat):
    """GCV_TSVD Choose GCV parameter for TSVD image deblurring.
    
    function tol = gcv_tsvd(s, bhat)
    
    tol = gcv_tsvd(s, bhat);
    
    This function uses generalized cross validation (GCV) to choose
    a truncation parameter for TSVD regularization.
    
    Input:
    s  Vector containing singular or spectral values.
    bhat  Vector containing the spectral coefficients of the blurred
    image.
    
    Output:
    tol  Truncation parameter; all abs(s) < tol should be truncated.
    
    Reference: See Chapter 6, 
    "Deblurring Images - Matrices, Spectra, and Filtering"
    by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
    SIAM, Philadelphia, 2006."""
	#
    # Sort absolute values of singular/spectral values in descending order.
	#
    s = abs(s)
    idx = argsort(s)
    idx = flipud(idx)
    s = take(s,idx)
    bhat = abs( take(bhat,idx) )
    n = size(s)
	#
    # The GCV function G for TSVD has a finite set of possible values 
    # corresponding to the truncation levels.  It is computed using
    # rho, a vector containing the squared 2-norm of the residual for 
    # all possible truncation parameters tol.
	#
    rho = zeros(n-1,'d')
    rho[n-2] = bhat[n-1]**2
    G = zeros(n-1,'d')
    G[n-2] = rho[n-2]
    for k in range(n-3,-1,-1):
	rho[k] = rho[k+1] + bhat[k+1]**2
	G[k] = rho[k]/(n - k)**2
	# Ensure that the parameter choice will not be fooled by pairs of
	# equal singular values.
    for k in range(0,n-2):
	if (s[k]==s[k+1]):
	    G[k] = inf
	#
		# Now find the minimum of the discrete GCV function.
	#
    reg_min = argmin(G)
	#
		# reg_min is the truncation index, and tol is the truncation parameter.
		# That is, any singular values < tol are truncated.
	#
    tol = s[reg_min]
    return tol
	    
	    
###################################################


def gcv_gtik(sa, sd, bhat):
    """GCV_GTIK Choose GCV parameter for gtik_fft deblurring function.

    function alpha = gcv_gtik(sa, sd, bhat)

    alpha = gcv_gtik(sa, sd, bhat)

    This function uses generalized cross validation (GCV) to choose
    a regularization parameter for generalized Tikhonov filtering.

    Input:
    sa  Vector containing singular or spectral values of the
    blurring matrix.
    sd  Vector containing singular of spectral values of the
    regularization operator.
    bhat  Vector containing the spectral coefficients of the blurred
    image.

    Output:
    alpha  Regularization parameter.

    Reference: See Chapter 7, 
    "Deblurring Images - Matrices, Spectra, and Filtering"
    by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
    SIAM, Philadelphia, 2006."""

    def GCV(alpha, sa, sd, bhat):
	"""This is a nested function that evaluates the GCV function for
	Tikhonov filtering.  It is called by fminbnd."""
    
	denom = abs(sa)**2 + alpha**2 * abs(sd)**2
	#
	#  NOTE: It is possible to get division by zero if using a derivative
	#        operator for the regularization operator.
	#
	phi_d = abs(sd)**2 / denom
	G = sum(abs(bhat*phi_d)**2) / (sum(phi_d)**2)
	return G
    
    alpha = fminbound(GCV, min(abs(sa)), max(abs(sa)), (sa, sd, bhat))
    return alpha
