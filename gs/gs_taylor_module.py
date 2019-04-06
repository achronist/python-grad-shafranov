def gs_taylor_module(X, Q, bp, ds=None, va=None):
    '''Inputs:
      X = grid array, first column is r, second column is z
      Q = array for quadralaterals
      bp = boundary points,
      ds,va [OPTIONAL] the delstar operator
     Returns:
      psi = poloidal flux with peak value unity, long vector form
      psimax = max(abs(psi)) before normalization
      magaxis = coordinate of magnetic axis in the X and psi vectors
      lambda = eigenvalue
    '''

    from numpy import shape, sum, sqrt, real, zeros, argmax
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import eigs
    from gs import gs_delstar

    (np,dim) = shape(X)
    
    # compute delstar matrix, if not provided
    if ds is None:
        (ds,va) = gs_delstar(X, Q)[:2]
    
    # select homogeneous (non boundary) part
    ndof = sum(~bp)
    dsh = ds[~bp,:][:,~bp]
    vah = spdiags(va[~bp]/X[~bp,0], 0, ndof, ndof)

    # find eigenvalue of smallest magnitude
    # solves eigenequation  dsh * ef = vah * ef * ev
    # For k=1 returns ndof*1 column array
    (ev,ef) = eigs(dsh, k=1, M=vah.tocsc(), which='SM')
    
    lam = sqrt(real(ev[0]))

    psi = zeros(np)
    psi[~bp] = real(ef)

    # normalize psi to peak value unity
    magaxis = argmax(abs(psi))
    psimax = abs(psi[magaxis])
    psi = psi/psi[magaxis]

    return (psi,psimax,magaxis,lam)


