#!/usr/bin/env python

from numpy import zeros, arange, ix_, mod 
from syntheticBC import add_syn_border

__all__ = ['padarray']

def padarray(A, padsize, method='zero', direction='both',
      Xind=None, Yind=None):
    """
    pads array A using the specified method and direction. 

    method can be one of these strings: 
       'zero'           Pads with zeros
       'perodic'        Pads with circular repetitiion of elements
       'symmetric'      Pads array with mirror reflections of itself 
       'antisymmetric'  Pads array with point reflections of itself 
       'synthteic'      Pads array with synthetic boundary conditions

    direction can be one of the following strings.  
       'pre'            Pads before the first array element along each
                        dimension .
       'post'           Pads after the last array element along each
                        dimension. 
       'both'           Pads before the first array element and after the
                        last array element
    Example:
    >>> from numpy import arange
    >>> from padarray import padarray
    >>> n = 5
    >>> A = arange(n ** 2)
    >>> A.shape = (n, n)
    >>> padsize = (2,3)
    >>> B = padarray(A,padsize,'periodic','both')
    >>> B
    array([[17, 18, 19, 15, 16, 17, 18, 19, 15, 16, 17],
           [22, 23, 24, 20, 21, 22, 23, 24, 20, 21, 22],
           [ 2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2],
           [ 7,  8,  9,  5,  6,  7,  8,  9,  5,  6,  7],
           [12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12],
           [17, 18, 19, 15, 16, 17, 18, 19, 15, 16, 17],
           [22, 23, 24, 20, 21, 22, 23, 24, 20, 21, 22],
           [ 2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2],
           [ 7,  8,  9,  5,  6,  7,  8,  9,  5,  6,  7]])
    """

    a0, a1 = A.shape
    p0, p1 = padsize

    if method == 'zero':
        if direction == 'pre':
            B = zeros((a0 + p0, a1 + p1))
            B[p0 : p0 + a0, p1 : p1 + a1] = A          
        elif direction == 'post':
            B = zeros((a0 + p0, a1 + p1))
            B[0 : a0, 0 : a1] = A          
        elif direction == 'both':
            B = zeros((a0 + 2*p0, a1 + 2*p1))
            B[p0 : p0 + a0, p1 : p1 + a1] = A          

    elif method == 'periodic':
        if direction == 'pre':
            B = A[ix_(range(-p0, a0), range(-p1, a1))]
        elif direction == 'post':
            B = A[ix_(mod(range(0,a0 + p0), a0), mod(range(0,a1 + p1), a1))]
        elif direction == 'both':
            B = A[ix_(mod(range(-p0,a0+p0), a0), mod(range(-p1,a1+p1), a1))]

    elif method == 'reflexive':
        if direction == 'pre':
            B = A[ix_(range(p0-1, -1, -1) + range(0, a0),
                      range(p1-1, -1, -1) + range(0, a1))]
        elif direction == 'post':
            B = A[ix_(range(0, a0) + range(a0-1, a0-p0-1, -1),
                      range(0, a1) + range(a1-1, a1-p1-1, -1))]
        elif direction == 'both':
            B = A[ix_(range(p0-1,-1,-1)+range(0,a0)+range(a0-1,a0-p0-1,-1),
                      range(p1-1,-1,-1)+range(0,a1)+range(a1-1,a1-p1-1,-1))]
    
    elif method == 'antireflexive':
        if direction == 'pre':
            B0 = A[ix_([0]*p0 + range(0, a0),
                       [0]*p1 + range(0, a1))]
            B1 = A[ix_(range(p0, 0, -1) + range(0, a0),
                       range(p1, 0, -1) + range(0, a1))]
        elif direction == 'post':
            B0 = A[ix_(range(0, a0) + [a0-1]*p0,
                       range(0, a1) + [a1-1]*p1)]
            B1 = A[ix_(range(0, a0) + range(a0-2, a0-p0-2, -1),
                       range(0, a1) + range(a1-2, a1-p1-2, -1))]
        elif direction == 'both':
            B0 = A[ix_([0]*p0 + range(0, a0) + [a0-1]*p0,
                       [0]*p1 + range(0, a1) + [a1-1]*p1)]
            B1 = A[ix_(range(p0,0,-1)+range(0,a0)+range(a0-2,a0-p0-2,-1),
                       range(p1,0,-1)+range(0,a1)+range(a1-2,a1-p1-2,-1))]
        B = 2 * B0 - B1
    elif method == 'synthetic':
        B = add_syn_border(A, padsize, Xind, Yind)
    return B 


n = 5
A = arange(n ** 2)
A.shape = (n, n)
padsize = (2,3)
B = padarray(A,padsize,'periodic','both')
B

def _test():
    import doctest, psfMatrix
    return doctest.testmod(psfMatrix)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
