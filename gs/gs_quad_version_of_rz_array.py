def gs_quad_version_of_rz_array(r, z):

    '''Returns: (X,Q)'''
    # From arrays r and z generate coordinate array X(n,2) and quad
    # indexing Q(iquad,4)

    # The quad points are ordered  (i1,i2)=[(1,1),(1,2),(2,2),(2,1)]
    # This corresponds to ccw in (z,r) plane if i1 is r-like and i2 is z-like
    
    from numpy import size, zeros, ravel, shape, meshgrid, array

    n = size(r)
    
    X = zeros((n,2))
    X[:,0] = ravel(r)
    X[:,1] = ravel(z)
    
    n1 = shape(r)[0]
    n2 = shape(r)[1]

    (i1,i2) = meshgrid(range(n1 - 1), range(n2 - 1), indexing='ij')
    i1 = ravel(i1)
    i2 = ravel(i2)
    
    # The index of point (i1,i2) = n2*i1 + i2
    idx = lambda i1,i2: n2*i1 + i2
    Q = array([idx(i1,i2),idx(i1,i2+1),idx(i1+1,i2+1),idx(i1+1,i2)]).T

    return (X,Q)
