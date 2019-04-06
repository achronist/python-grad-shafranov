def gs_delstar(r, quads):
    '''Calculate mimetic matrix of delstar = -(r/a)*w
      w(np,np) is sparse symmetric positive definite
      a(np) = vector of vertex areas
      
    Arguments:
        r     r[:,0] np radial coordinates
              r[:,1] np axial coordinates
        quads   quads[i,:] = four indexes into r
        
    Returns (w, area_vertex, isp, jsp, dsp)'''
    
    from numpy import spacing, shape, zeros, maximum, sum, size,\
        dot, array, stack, matmul, ravel
    from scipy import sparse
    
    # Cross product of 2-d vectors
    cp = lambda a,b: a[0]*b[1] - a[1]*b[0]

    # Function to calculate product of matrix with its tranpose
    P = lambda M: matmul(M, M.T)
    
    (np,dim) = shape(r)
    (nq,deg) = shape(quads)
    
    if dim != 2: raise ValueError('mesh not 2-D')
    if deg != 4: raise ValueError('mesh not linear quads')
    
    # Index and value for sparse matrix construction
    isp = zeros((nq,4,4))
    jsp = zeros((nq,4,4))
    dsp = zeros((nq,4,4))

    dl = zeros((4,2))

    # 8 areas, each corresponding to a triangle, 2 of which add up
    # to give the contribution to the vertex centered areas for each
    # quadrilateral element.
    area_triangle = zeros((4,2))
    
    # diagonal across each vertex centered piece within each quad
    L = zeros((4,2))
    area_vertex = zeros(np) # vertex-centered areas
    r_cent_tri = zeros((4,2)) # centroid locations for the 8 triangles
    seq = (3,0,1,2) # cyclic permutation of length indices-to be used further down
    
    # Loop over quads
    for i in range(nq):
        # k = corner indices
        k = quads[i,:]
        
        # dl = edge length vectors
        dl[0,:] = r[k[1],:] - r[k[0],:]
        dl[1,:] = r[k[2],:] - r[k[1],:]
        dl[2,:] = r[k[3],:] - r[k[2],:]
        dl[3,:] = r[k[0],:] - r[k[3],:]
        
        # radius at edge centers
        r23 = (r[k[2],0] + r[k[1],0])/2
        r34 = (r[k[3],0] + r[k[2],0])/2
        r14 = (r[k[3],0] + r[k[0],0])/2
        r12 = (r[k[1],0] + r[k[0],0])/2
                        
        # Locating the centroid of the quad. element 
        R_cent_quad = sum(r[k,:], axis=0)/4
         
        # Calculating the cross products needed for poloidal field computation
        # and storing them as four different areas
        area = zeros(4)
        for jj in range(4):
            area[jj] = abs(cp(dl[seq[jj],:],dl[jj,:]))
        
        # bpk(4,2) = corner field matrix
        bp1 = zeros((4,2))
        if r12 > 0:
            bp1[1,:] = -dl[3,:]/r12
        if r14 > 0:
            bp1[3,:] = -dl[0,:]/r14
        bp1[0,:] = -(bp1[1,:] + bp1[3,:])
        bp1 = bp1/area[0]
    
        bp2 = zeros((4,2))
        if r12 > 0:
            bp2[0,:] = -dl[1,:]/r12
        if r23 > 0:
            bp2[2,:] = -dl[0,:]/r23
        bp2[1,:] = -(bp2[0,:] + bp2[2,:])
        bp2 = bp2/area[1]

        bp3 = zeros((4,2))
        if r23 > 0:
            bp3[1,:] = -dl[2,:]/r23
        if r34 > 0:
            bp3[3,:] = -dl[1,:]/r34
        bp3[2,:] = -(bp3[1,:] + bp3[3,:])
        bp3 = bp3/area[2]

        bp4 = zeros((4,2))
        if r14 > 0:
            bp4[0,:] = -dl[2,:]/r14
        if r34 > 0:
            bp4[2,:] = -dl[3,:]/r34
        bp4[3,:] = -(bp4[0,:] + bp4[2,:])
        bp4 = bp4/area[3]
        
        # corner volumes
        # Calculating the vertex centered-volume element for each corner.
        # dv=2*pi*centroid_of_vertex_centered_area*vertex_centered area
        # Note the centroid of the vertex_centered area is NOT the same as the
        # radial coordinate of the vertex itself, the coding below calculates the
        # four vertex_centered areas within each quad. element.

        # defining each corner diagonal, L which is the vector that goes from each
        # corner of the quadrilateral to the centroid of the quadrilateral. It points
        # inward toward the centroid
        for j in range(4):
            L[j,:] = R_cent_quad - r[k[j],:]
   
        # Calculating the 8 triangular areas: 4x2 array
        area_triangle[0,0] = abs(cp(L[0,:], dl[3,:]))/4
        area_triangle[0,1] = abs(cp(L[0,:], dl[0,:]))/4
        area_triangle[1,0] = abs(cp(L[1,:], dl[0,:]))/4
        area_triangle[1,1] = abs(cp(L[1,:], dl[1,:]))/4
        area_triangle[2,0] = abs(cp(L[2,:], dl[1,:]))/4
        area_triangle[2,1] = abs(cp(L[2,:], dl[2,:]))/4
        area_triangle[3,0] = abs(cp(L[3,:], dl[2,:]))/4
        area_triangle[3,1] = abs(cp(L[3,:], dl[3,:]))/4
 
        # triangle centroids: 4x2 array
        r_cent_tri[0,0] = dot(array([7,1,1,3]), r[k,0])/12
        r_cent_tri[0,1] = dot(array([7,3,1,1]), r[k,0])/12
        r_cent_tri[1,0] = dot(array([3,7,1,1]), r[k,0])/12
        r_cent_tri[1,1] = dot(array([1,7,3,1]), r[k,0])/12
        r_cent_tri[2,0] = dot(array([1,3,7,1]), r[k,0])/12
        r_cent_tri[2,1] = dot(array([1,1,7,3]), r[k,0])/12
        r_cent_tri[3,0] = dot(array([1,1,3,7]), r[k,0])/12
        r_cent_tri[3,1] = dot(array([3,1,1,7]), r[k,0])/12
    
        # centroids for quarter pieces of the quad element->finding the
        # area-averaged radii for each quarter piece that conribute to
        # vertex-centered area.
        # radii=r_cent_tri*area_triangle/Area_quads
        r_quad = r_cent_tri*area_triangle
    
        Area_quads = sum(area_triangle, axis=1)
        radii = sum(r_quad, axis=1)/Area_quads
  
        # the four volume elements are
        dv = radii*Area_quads
  
        # accumulate energy matrix
        kkkk = stack((k,k,k,k))
        isp[i,:,:] = kkkk.T
        jsp[i,:,:] = kkkk
        dsp[i,:,:] = P(bp1)*dv[0] + P(bp2)*dv[1] + P(bp3)*dv[2] + P(bp4)*dv[3]
  
        # accumulate vertex areas
        area_vertex[k] = area_vertex[k] + Area_quads   

    # Assemble sparse matrix
    w = sparse.coo_matrix((ravel(dsp),(ravel(isp),ravel(jsp))))
    w = w.tocsc()
    
    return (w,area_vertex,isp,jsp,dsp)
