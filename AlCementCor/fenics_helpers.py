import fenics as fe

def as_3D_tensor(X):
    return fe.as_tensor([[X[0], X[3], 0],
                         [X[3], X[1], 0],
                         [0, 0, X[2]]])


def local_project(v, V, dxm, u=None):
    dv = fe.TrialFunction(V)
    v_ = fe.TestFunction(V)
    a_proj = fe.inner(dv, v_) * dxm
    b_proj = fe.inner(v, v_) * dxm
    solver = fe.LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = fe.Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return