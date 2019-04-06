def gspoly(gs):

    '''Solve Grad-Shafranov equation using normalized form

 Input gs is structure containing fields:
   X         position: X[i,0] = R, X[i,1] = Z, i = 0,..,nvertex-1
   Q         Q[j,:] = indexes in X of corners of quad j, ccw order
   bp        true for boundary vertexes
   psi       initial guess for psi at each vertex, also boundary values
   Fpoly     polynomial coefficients for F(psi)
   ppoly     polynomial coefficients for p(psi)
   omega     rotation pararmeter
   grav      gravity parameter (z-direction)
   offset    shaft current times mu0 (for 1 weber CT)
   Iternums  number of Picard iterations
   ds,va     factored delstar operator [optional]
   zmin      no current for z<zmin  [optional]
   rmax      no current for r>rmax [optional]
   psiSeparatrix   no current for psi<psiSeparatrix [optional]
   rjpmin    no current for rjp<rjpmin [optional]

 Returns as output the input structure with new and updated fields:

   psi       psi[i] = poloidal flux at vertex i
   psimax    maximum values of psi
   magaxis   index of magnetic axis defined as vertex with max psi
   rax       radius of magnetic axis
   lambar    average value of lam
   F         F[i] = R*Bphi at vertex i
   p         p[i] = pressure at vertex i
   dFdpsi    dFdpsi[i] = dF/dpsi at vertex i
   ds,va     factored delstar operator [generated if not provided]
   dlambarsqr  Convergence factor for last it'''

    from numpy import array, polyval, polyder, Infinity, isfinite,\
        size, shape, argmax, exp, sqrt, argsort, zeros_like, isnan
    from numpy.linalg import norm
    from scipy.sparse import spdiags
    from scipy.integrate import cumtrapz
    from scipy.sparse.linalg import spsolve
    
    # Second index of X
    r_ = 0
    z_ = 1

   ## Add defaults for missing fields

    if not hasattr(gs, 'Fpoly'):
        gs.Fpoly = array((gs.taylor.lam,0))
    
    if not hasattr(gs, 'ppoly'):
        gs.ppoly = array((0,))
    
    if not hasattr(gs, 'Iternums'):
        gs.Iternums = 10

    ## Extract parameters

    X = gs.X
    Q = gs.Q
    bp = gs.bp

    ## More obvious notation for (r,z) coordinates
    
    r = X[:,r_]
    z = X[:,z_]

    if hasattr(gs, 'verbose'):
        verbose = gs.verbose
    else:
        verbose = True       

    ## Get initial guess for psi
    
    if hasattr(gs, 'psi'):
        psi = gs.psi
    elif hasattr(gs, 'taylor'):
        psi = gs.taylor.psi
    else:
        psi = zeros(size(X,0),1)
        
    ## Polynomials for F and p

    Fpoly = gs.Fpoly
    ppoly = gs.ppoly

    # Angular velocity of rigid-body rotation   
    if hasattr(gs, 'omega'):
        omega = gs.omega
    else:
        omega = 0
    
    # Gravitational acceleration
    if hasattr(gs, 'grav'):
        grav = gs.grav
    else:
        grav = 0

    # Cutoff (no current for psi < psiSeparatrix)
    if hasattr(gs, 'psiSeparatrix'):
        psiSeparatrix = gs.psiSeparatrix
    else:
        psiSeparatrix = -Infinity

    if hasattr(gs, 'rjpmin'):
        rjpmin = gs.rjpmin
    else:
        rjpmin = -Infinity

    if hasattr(gs, 'zmin'):
        zmin = gs.zmin
    else:
        zmin = -Infinity

    if hasattr(gs, 'rmax'):
        rmax = gs.rmax
    else:
        rmax = Infinity

    if hasattr(gs, 'lambar'):
        lambar = gs.lambar
    elif hasattr(gs, 'taylor'):
        lambar = gs.taylor.lam
    else:
        lambar = 1

    Iternums = gs.Iternums

    ## Convergence tolerance

    if hasattr(gs, 'tol'):
        tol = par.tol
    else:
        tol = 1e-12

    assert size(shape(psi)) == 1, 'psi must be vector'
    assert size(psi) == shape(X)[0], 'psi is wrong size'

    ## initial location of magnetic axis (maximum of psi)
    magaxis = argmax(psi)
    psimax = psi[magaxis]

    # Normalize
    psi=psi/psi[magaxis]
    
    # Radius at magnetic axis
    rax = r[magaxis]

    ## compute factored delstar matrix (ds,va), if not provided
    # The delstar operator is provided in factored form (ds,va)

    if hasattr(gs, 'ds'):
        ds = gs.ds
        va = gs.va
    else:
        (ds,va) = gs_delstar(X, Q)

    ## boundary contribution
    bc = ds[~bp,:][:,bp].dot(psi[bp])

    ## select homogeneous (non boundary) part
    dsh = ds[~bp,:][:,~bp]
    ndof = sum(~bp)

    vah = spdiags(va[~bp]/r[~bp], 0, ndof, ndof)

    ## Shaft current offset (unscaled part of F)

    offset = Fpoly[-1]

    if isfinite(psiSeparatrix):
        offset = polyval(Fpoly, psiSeparatrix)

    Fpoly[-1] = Fpoly[-1] - offset

    ## Current profile

    dFdpsipoly = polyder(Fpoly)

    w = lambda psi: polyval(Fpoly, psi)
    dwdpsi = lambda psi: polyval(dFdpsipoly, psi)

    ## Pressure profile

    dpdpsipoly = polyder(ppoly)
    pscalefunc = 1

    ## Picard iteration

    lambarsqr = lambar**2

    for itnum in range(Iternums):
        # Current term
        F = offset + lambar*w(psi)
        dFdpsi = lambar*dwdpsi(psi)
        termF=F*dFdpsi
        # Pressure term
        # pscalefunc = pressure scaling function, here we take it to be 
        # unity at (z,r)=(rax,0).
        pscalefunc = exp(omega**2*(r**2 - rax**2) + grav*z)
        termp = polyval(dpdpsipoly, psi/psimax)*r**2*pscalefunc
        # Here is the right-hand side of the GS equation:
        rjp = termF + termp
        # Special regions
        # Setting psiSeparatrix=0 and doing this test changes the convergence
        # so we use as default psiSeparatrix=-inf.
        sel = (psi < psiSeparatrix) | (z < zmin) | (r > rmax) | (rjp < rjpmin)
        rjp[sel] = 0
        # Set contributing terms so that output is internally consistent
        dFdpsi[sel] = 0
        F[sel] = offset
        termp[sel] = 0
        # Solve the GS equation with split operator
        psiold = psi
        psi[~bp] = spsolve(dsh, vah.dot(rjp[~bp]) - bc)
        # Locate magnetic axis (where psi has its maximum value)
        magaxis = argmax(psi)
        psimax = psi[magaxis]
        print('new psimax:', psimax)
        lambarsqrold = lambarsqr
        # Renormalize lambarsqr
        lambarsqr = lambarsqr/psimax
        lambar = sqrt(lambarsqr)
        psi[~bp] = psi[~bp]/psimax
        # rax = radius of magnetic axis
        rax = r[magaxis]
        dpsi = psi - psiold
        print('%d norm(dpsi)=%.3e d(lambarsqr)=%.3e' % 
              (itnum,norm(dpsi),lambarsqr-lambarsqrold))
        # Check for convergence
        if abs(lambarsqr - lambarsqrold) < tol:
            break
 
    ## Sort and desort vectors i and k
    i = argsort(psi)
    k = zeros_like(i)
    for j in range(size(i)):
        k[i[j]] = j

    ## Generate pressure field
    # With rotation the flux function p0(psi) is a reference pressure because
    # the true pressure is non-uniform on a flux surface.
    # The true pressure is p=p0*pfunc
    # The function pfunc encodes the position dependence.
    # The pressure term (termp) is r^2*p0*pfunc (units are mu0=1).
    dp0dpsi = zeros_like(psi)
    sel = r > 0
    dp0dpsi[sel] = termp[sel]/(r[sel]**2*pscalefunc[sel])
    p0 = cumtrapz(psi[i], dp0dpsi[i], initial=0)
    p0 = p0[k]
    if omega != 0:
        # Arbitrary offset to get pressure off zero
        p0 = p0 + max(p0)
    # Impose pressure shaping (centrifugal and gravity) to get spatially
    # varying pressure p(r,z)=p0(psi(r,z))*pfunc(r,z)
    p = p0*pscalefunc

    ## Save results to structure for return

    gs.psi = psi
    gs.F = F
    gs.p = p
    gs.dFdpsi = dFdpsi
    gs.rjp = rjp
    gs.rax = rax
    gs.dpsi = dpsi
    gs.psiold = psiold
    gs.dlambarsqr = lambarsqr - lambarsqrold

    return gs
