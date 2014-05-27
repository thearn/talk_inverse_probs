from numpy import Inf, ix_, arange, maximum, minimum, ndarray, zeros, r_
from numpy import rot90, meshgrid
from scipy.linalg import norm

__all__ = ['get_syn_bc', 'add_syn_border', 'del_syn_border']

def get_min_d(u,u_padded,i,j,xind,yind,xind2,yind2):
    min_d = Inf
    for k in xind2:
        for l in yind2:
            # loop to find the minimum pixel difference
            d = norm( u_padded[ix_(i+xind,j+yind)] - u[ix_(k+xind,l+yind)] )
            if d < min_d:
                min_d = d
                min_k = k
                min_l = l
    return min_k, min_l

def pad_layer(Xind, Yind, u_padded,
              layer,imsize,padsize,overlapsize,searchsize,patchsize):
     o1 = overlapsize[0]
     o2 = (overlapsize[1] - 1) / 2
     s1 = searchsize[0]
     s2 = (searchsize[1] - 1) / 2
     p1 = patchsize[0]
     p2 = patchsize[1]

     layer_start = padsize - layer
     layer_end   = imsize - padsize + layer - 1

     last_layer_start = layer_start + p1
     last_layer_end   = layer_end - p1

     dom_start = last_layer_start
     dom_end   = last_layer_end

     # top 1st layer
     i = layer_start
     
     for j in arange(layer_start, layer_end+1, p2):
         xind  = arange(o1) + p1
         xind2 = arange(s1) + dom_start
         if j-s2 <= dom_start:
             c = dom_start
             d = dom_start + 2*s2
         elif j+s2+p2-1 >= dom_end:
             c = dom_end - p2 + 1 - 2*s2
             d = dom_end - p2 + 1
         else:
             c = j-s2 
             d = j+s2
         yind2 = arange(c, d+1)
         # if length(yind2) < s2
         #     error('yind2 too short')
         # if length(yind2) > 2*s2+1
         #     error('yind2 too long')
         a = minimum(j-last_layer_start, c-dom_start)
         a = minimum( maximum(a,0), o2)
         b = minimum(last_layer_end-j, dom_end-d)
         b = minimum( maximum(b,0), o2)
         yind = arange(-a, b+1)
         # if length(yind) < o2
         #     error('yind too short')
         # if length(yind) > 2*o2+1
         #     error('yind too long')
         min_k, min_l = get_min_d(u_padded,u_padded,i,j,xind,yind,xind2,yind2)
         range1 = arange(p1)
         range2 = arange(p2)
         Xind[ix_(i+range1, j+range2)] = Xind[ix_(min_k+range1, min_l+range2)]
         Yind[ix_(i+range1, j+range2)] = Yind[ix_(min_k+range1, min_l+range2)]
         u_padded[ix_(i+range1, j+range2)] = u_padded[ix_(min_k+range1, min_l+range2)]
     return Xind, Yind, u_padded

def get_syn_bc(u, padsize, overlapsize=[5,11], searchsize=[3,5], patchsize=[2,2]):
    """Get synthetic boundary conditions. We assume u is a square image and
    padsize a scalar."""
    # pick a patch by comparing ssd

    # parameters
    #if ~exist('imsize')
       #imsize  = 256
    #end

    #if ~exist('padsize') 
    #padsize = 5
    #end

    if isinstance(padsize, ndarray) or isinstance(padsize, tuple):
        padsize = padsize[0]

    usize = u.shape[0]
    imsize = usize + 2 * padsize
    

    p1 = patchsize[0]
    p2 = patchsize[1]
    
    # initialization
    u_padded = zeros((imsize, imsize))
    u_padded[ix_(arange(padsize, padsize + usize), arange(padsize, padsize +
        usize))] = u

    layer = 0
    Yind, Xind = meshgrid(arange(imsize), arange(imsize))

    for layer in r_[p1 : (padsize-1)/p1*p1 + 1 : p1 , padsize]:
        for rotaion in xrange(4):
             Xind, Yind, u_padded \
               = pad_layer(rot90(Xind,-1),rot90(Yind,-1),rot90(u_padded,-1),
                     layer,imsize,padsize,overlapsize,searchsize,patchsize)
    return Xind, Yind

def add_syn_border(u,padsize,Xind,Yind):
    # add_border of width 5 to get an output image of size [256,256] 

    m, n = Xind.shape
    p1, p2 = padsize

    v = zeros((m, n))
    v[p1 : m-p1, p2 : n-p2] = u
    for i in r_[xrange(p1), xrange(m-p1, m)]:
        for j in xrange(n):
            v[i, j] = v[Xind[i, j],Yind[i, j]]
    for i in xrange(p1, m-p1):
        for j in r_[xrange(p2), xrange(n-p2, n)]:
            v[i, j] = v[Xind[i, j],Yind[i, j]]
    return v

def del_syn_border(u,padsize,Xind,Yind):
    # del_border is the dual of add_border

    m, n = Xind.shape
    p1, p2 = padsize

    # "unpadding"
    for i in r_[xrange(p1), xrange(m-p1, m)]:
        for j in xrange(n):
            u[Xind[i,j],Yind[i,j]] = u[Xind[i,j],Yind[i,j]] + u[i,j]
    for i in xrange(p1, m-p1):
        for j in r_[arange(p2), arange(n-p2, n)]:
            u[Xind[i,j],Yind[i,j]] = u[Xind[i,j],Yind[i,j]] +  u[i,j]

    # cropping
    v = u[ p1 : m-p1, p2 : n-p2]

    return v
