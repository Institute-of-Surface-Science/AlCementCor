import fenics as fe


def as_3D_tensor(X):
    """
    Convert a given vector to a 3D tensor.

    Parameters:
    - X: Vector to be converted

    Returns:
    - 3D tensor representation of X.
    """
    return fe.as_tensor([[X[0], X[3], 0],
                         [X[3], X[1], 0],
                         [0, 0, X[2]]])


def local_project(v, V, dxm, u=None):
    """
    Project a given vector onto a function space locally.

    Parameters:
    - v: Vector to be projected
    - V: Target function space
    - dxm: Measure for integration
    - u: Optional function to be updated with the result

    Returns:
    - Resultant function if `u` is None, otherwise updates `u` and returns None.
    """
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
        return None


def assign_values_based_on_boundaries(function_space, partionings, values) -> 'fe.Function':
    """
    Assign values based on provided boundaries and return an interpolated function.

    Args:
        function_space: FEniCS function space.
        partionings: List of x-coordinates where values change. Should be in increasing order.
        values: List of values to be assigned within the intervals defined by the boundaries.

    Returns:
        A FEniCS function with values assigned based on the given boundaries.
    """

    assert len(partionings) + 1 == len(values), "Number of values should be one more than number of boundaries."

    class BoundaryBasedValue(fe.UserExpression):
        """User-defined expression for FEniCS to set values based on given boundaries."""

        def __init__(self, boundaries, values, **kwargs):
            super().__init__(**kwargs)
            self.boundaries = boundaries
            self.values = values

        def eval(self, value, x):
            """Evaluate the function based on position and set values accordingly."""
            for i, boundary in enumerate(self.boundaries):
                if x[0] <= boundary:
                    value[0] = self.values[i]
                    return
            value[0] = self.values[-1]

        def value_shape(self):
            """Return the shape of the value (scalar in this case)."""
            return ()

    # Instantiate the user expression and interpolate
    boundary_expr = BoundaryBasedValue(partionings, values)
    return fe.interpolate(boundary_expr, function_space)
