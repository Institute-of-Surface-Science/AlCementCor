import ufl
import fenics as fe


class LinearElastoPlasticModel:
    def __init__(self, simulation_config, mesh):
        """Initialize the model with given configurations and mesh."""
        self._simulation_config = simulation_config
        self._mesh = mesh

        self._setup_function_spaces()
        self._setup_displacement_functions()
        self._setup_stress_functions()
        self._setup_other_functions()

    def _setup_function_spaces(self):
        """Set up the function spaces required for the simulation."""
        self.deg_stress = self._simulation_config.finite_element_degree_stress
        self.V = fe.VectorFunctionSpace(self._mesh, "CG", self._simulation_config.finite_element_degree_u)
        self.DG = fe.FunctionSpace(self._mesh, "DG", 0)
        We = fe.VectorElement("Quadrature", self._mesh.ufl_cell(), degree=self.deg_stress, dim=4, quad_scheme='default')
        self.W = fe.FunctionSpace(self._mesh, We)
        self.W0 = fe.FunctionSpace(self._mesh,
                                   fe.FiniteElement("Quadrature", self._mesh.ufl_cell(), degree=self.deg_stress,
                                                    quad_scheme='default'))
        self.P0 = fe.FunctionSpace(self._mesh, "DG", 0)

    def _setup_displacement_functions(self):
        """Set up functions related to displacements."""
        self.u, self.du, self.Du = [fe.Function(self.V, name=n) for n in
                                    ["Total displacement", "Iteration correction", "Current increment"]]

    def _setup_stress_functions(self):
        """Set up functions related to stress."""
        self.sig, self.sig_old, self.n_elas = [fe.Function(self.W) for _ in range(3)]

    def _setup_other_functions(self):
        """Set up miscellaneous functions."""
        func_names = ["Beta", "Cumulative plastic strain", "Hydrostatic stress", "local sig0", "local mu",
                      "local lmbda", "local hardening factor"]
        self.beta, self.p, self.sig_hyd, self.sig_0_local, self.mu_local, self.lmbda_local, self.C_linear_h_local = [
            fe.Function(self.W0, name=n) for n in func_names]
        self.sig_hyd_avg, self.sig_0_test, self.lmbda_test = [fe.Function(self.P0, name=n) for n in
                                                              ["Avg. Hydrostatic stress", "test", "test2"]]
        self.metadata = {"quadrature_degree": self.deg_stress, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)
        self.v = fe.TrialFunction(self.V)
        self.u_ = fe.TestFunction(self.V)


