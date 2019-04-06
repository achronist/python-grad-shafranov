def gs_boundary(r,quads):
    # function [bp,dl] = boundary(r,quads)
    # locate boundary points and compute tangent vectors
    # r(np,2) = points, tris(nc,3) = linear triangles
    # bp(np,1) = true on boundary points, false elsewhere
    # dl(np,2) = tangent length vector on boundary points
    
    from numpy import spacing, shape, zeros, sqrt
    
    eps = spacing(1)
    (np,dim) = shape(r)
    (nq,deg) = shape(quads)
    if dim != 2: raise ValueError('mesh not 2-D')
    if deg != 4: raise ValueError('mesh not quadrilaterals')
    dl = zeros((np,2))
    bp = zeros((np,1), dtype='bool')
    # Loop over quads
    for i in range(nq):
        # k(4) = corner indices
        k = quads[i,:]
        # c(4,2) = corner length vectors
        c1 = zeros((4,2))
        c1[0,:] = r[k[1],:] - r[k[2],:]
        c1[1,:] = r[k[2],:] - r[k[3],:]
        c1[2,:] = r[k[3],:] - r[k[0],:]
        c1[3,:] = r[k[0],:] - r[k[1],:]
        c2 = zeros((4,2))
        c2[0,:] = r[k[2],:] - r[k[3],:]
        c2[1,:] = r[k[3],:] - r[k[0],:]
        c2[2,:] = r[k[0],:] - r[k[1],:]
        c2[3,:] = r[k[1],:] - r[k[2],:]
        # accumulate c
        dl[k,:] = dl[k,:] + c1 + c2
    # locate boundary points
    dl2 = dl[:,0]**2 + dl[:,1]**2
    dl2crit = max(dl2)*sqrt(eps)
    bp = dl2 > dl2crit
    return (bp,dl)