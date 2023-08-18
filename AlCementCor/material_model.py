import ufl
import fenics as fe
import numpy as np

from AlCementCor.fenics_helpers import as_3D_tensor


def eps(displacement: fe.Function):
    """
    Calculate the strain tensor (epsilon).

    Parameters:
    - displacement: The displacement tensor

    Returns:
    - strain tensor
    """
    e = fe.sym(fe.nabla_grad(displacement))
    return fe.as_tensor([[e[0, 0], e[0, 1], 0],
                         [e[0, 1], e[1, 1], 0],
                         [0, 0, 0]])


def sigma(strain, lmbda_local_DG: fe.Function, mu_local_DG: fe.Function):
    """
    Calculate the Cauchy Stress Tensor.

    Parameters:
    - strain: Strain tensor

    Returns:
    - Cauchy Stress Tensor
    """
    return lmbda_local_DG * fe.tr(strain) * fe.Identity(3) + 2 * mu_local_DG * strain


def sigma_tang(e, n_elas: fe.Function, mu_local_DG: fe.Function,
               C_linear_h_local_DG: fe.Function, beta: fe.Function,
               lmbda_local_DG: fe.Function):
    N_elas = as_3D_tensor(n_elas)
    return sigma(e, lmbda_local_DG, mu_local_DG) - 3 * mu_local_DG * (
            3 * mu_local_DG / (3 * mu_local_DG + C_linear_h_local_DG) - beta) * fe.inner(
        N_elas, e) * N_elas - 2 * mu_local_DG * beta * fe.dev(e)


def sigma_v(strain, lmbda_local_DG: fe.Function, mu_local_DG: fe.Function) -> fe.Function:
    """
    Calculate the Von-Mises Stress.

    Parameters:
    - strain: Strain tensor

    Returns:
    - Von-Mises Stress
    """
    s = fe.dev(sigma(strain, lmbda_local_DG, mu_local_DG))
    return fe.sqrt(3 / 2. * fe.inner(s, s))


class LinearElastoPlasticModel:
    def __init__(self, simulation_config: 'SimulationConfig', mesh: 'MeshType',
                 substrate_props: 'MaterialProps', layer_props: 'MaterialProps') -> None:
        """Initialize the model with given configurations and mesh."""
        self._simulation_config = simulation_config
        self._mesh = mesh

        # todo: move to config file
        self.strain_rate = fe.Constant(0.000001)

        # Material properties
        self.substrate_props = substrate_props
        self.layer_props = layer_props

        # Function spaces
        self.deg_stress: int = -1
        self.V = None
        self.DG = None
        self.W = None
        self.W0 = None
        self.P0 = None

        # Displacement functions
        self.u = None
        self.du = None
        self.Du = None

        # Stress functions
        self.sig = None
        self.sig_old = None
        self.n_elas = None

        # Local properties
        self.mu_local_DG = None
        self.lmbda_local_DG = None
        self.local_linear_hardening_DG = None

        # Other functions and parameters
        self.beta = None
        self.p = None
        self.sig_hyd = None
        self.sig_0_local = None
        self.mu_local = None
        self.lmbda_local = None
        self.C_linear_h_local = None
        self.sig_hyd_avg = None
        self.sig_0_test = None
        self.lmbda_test = None
        self.metadata = None
        self.dxm = None
        self.v = None
        self.u_ = None

        # Newton equations
        self.newton_lhs = None
        self.newton_rhs = None

        self._setup()

    def _setup(self):
        self._setup_function_spaces()
        self._setup_displacement_functions()
        self._setup_stress_functions()
        self._setup_other_functions()
        self._setup_local_properties()
        self._setup_newton_equations()

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

    def _setup_local_properties(self):
        """Setup local properties of the model."""
        self.mu_local_DG = fe.Function(self.DG)
        self._assign_local_values(self.substrate_props.shear_modulus, self.layer_props.shear_modulus, self.mu_local_DG)

        self.lmbda_local_DG = fe.Function(self.DG)
        self._assign_local_values(self.substrate_props.first_lame_parameter, self.layer_props.first_lame_parameter,
                                  self.lmbda_local_DG)

        self.local_linear_hardening_DG = fe.Function(self.DG)
        self._assign_local_values(self.substrate_props.linear_isotropic_hardening,
                                  self.layer_props.linear_isotropic_hardening,
                                  self.local_linear_hardening_DG)

    def _assign_local_values(self, values: float, outer_values: float, local_DG: fe.Function) -> None:
        """Assign values based on the specified condition."""
        dofmap = self.DG.tabulate_dof_coordinates()[:]
        vec = np.full(dofmap.shape[0], values)
        vec[dofmap[:, 0] > self._simulation_config.width] = outer_values
        local_DG.vector()[:] = vec

    def _setup_newton_equations(self):
        """Setup Newton equations for the model."""
        self.newton_lhs = fe.inner(eps(self.v), sigma_tang(eps(self.u_), self.n_elas, self.mu_local_DG,
                                                           self.local_linear_hardening_DG, self.beta,
                                                           self.lmbda_local_DG)) * self.dxm
        self.newton_rhs = -fe.inner(eps(self.u_), as_3D_tensor(self.sig)) * self.dxm

    @property
    def mesh(self) -> 'MeshType':
        return self._mesh

    @property
    def simulation_config(self) -> 'SimulationConfig':
        return self._simulation_config
