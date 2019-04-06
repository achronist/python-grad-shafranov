def gssetup(m):
    
    '''Given Mesh object m set up a GS object'''
    
    from gs import GS
    from gs import gs_quad_version_of_rz_array, gs_boundary, gs_delstar

    (X,Q) = gs_quad_version_of_rz_array(m.r, m.z)
    
    (bp,dl) = gs_boundary(X, Q)
    (ds,va) = gs_delstar(X, Q)[:2]

    gs = GS()
    gs.X = X
    gs.Q = Q
    gs.bp = bp
    gs.dl = dl
    gs.ds = ds
    gs.va = va
    
    return gs
