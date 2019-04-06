def gstaylor(gs, verbose=False):
    '''
    % Given gs structure, returns a gs structure containing new information:
    %   gs.taylor.psi        Taylor state poloidal flux field, 1 at magnetic axis
    %   gs.taylor.lam        Taylor state eigenvalue
    %   gs.taylor.rax.       Radius of magnetic axis
    % Also sets
    %   gs.psi   equal to gs.taylor.psi for use as initial guess in
    %            subsequent use of gs structure
    %   gs.F     Taylor state F=lam*psi to enable use of gsfield for gstaylor output
    '''

    from gs import Taylor
    from gs import gs_taylor_module
    
    (psi,psimax,magaxis,lam) = gs_taylor_module(gs.X,gs.Q,gs.bp,gs.ds,gs.va)
    rax = gs.X[magaxis,0]

    if verbose:
        print('magaxis = %d' % magaxis)
        print('(z,r) = %g,%g' % (gs.X[magaxis,1],gs.X[magaxis,0]))
        print('psimax = %g' % psimax)
        print('Taylor lambda = %g' % lam)
    
    gs.taylor = Taylor(psi, lam, rax)

    gs.psi = psi
    gs.F = lam*psi

    return gs
