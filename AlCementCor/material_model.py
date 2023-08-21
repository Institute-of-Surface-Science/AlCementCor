import ufl
import fenics as fe
import numpy as np

from AlCementCor.fenics_helpers import as_3D_tensor


def compute_strain_tensor(displacement: fe.Function):
    """
    Calculate the strain tensor (epsilon) based on a given displacement tensor.

    Note:
    - Assumes Plane-Strain conditions (third component in the strain is 0)

    Parameters:
    - displacement: The displacement tensor

    Returns:
    - strain tensor
    """
    symmetric_gradient = fe.sym(fe.nabla_grad(displacement))

    return fe.as_tensor([
        [symmetric_gradient[0, 0], symmetric_gradient[0, 1], 0],
        [symmetric_gradient[0, 1], symmetric_gradient[1, 1], 0],
        [0, 0, 0]
    ])


def compute_stress(strain_tensor, lambda_coefficient: fe.Function, shear_modulus: fe.Function) -> fe.Function:
    """
    Calculate the Cauchy Stress Tensor for linear elasticity.

    Parameters:
    - strain_tensor: 2D or 3D strain tensor representing deformation.
    - lambda_coefficient: Lamé's first parameter.
    - shear_modulus: Material's shear modulus (or Lamé's second parameter).

    Returns:
    - Cauchy Stress Tensor
    """

    # Calculate isotropic (volumetric) stress contribution
    isotropic_stress = lambda_coefficient * fe.tr(strain_tensor) * fe.Identity(3)

    # Calculate deviatoric (shape-changing) stress contribution
    deviatoric_stress = 2 * shear_modulus * strain_tensor

    # Combine isotropic and deviatoric contributions
    total_stress = isotropic_stress + deviatoric_stress

    return total_stress


def compute_tangential_stress(strain_tensor, normal_elasticity: fe.Function, shear_modulus: fe.Function,
                              linear_hardening_coeff: fe.Function, beta_coeff: fe.Function,
                              lambda_coefficient: fe.Function) -> fe.Function:
    """
    Compute the tangential stress tensor based on given parameters.

    Parameters:
    - strain_tensor: 2D or 3D strain tensor.
    - normal_elasticity: Normal direction in the elasticity space.
    - shear_modulus: Material's shear modulus (or Lamé's second parameter).
    - linear_hardening_coeff: Linear hardening coefficient.
    - beta_coeff: Beta coefficient.
    - lambda_coefficient: Lamé's first parameter.

    Returns:
    - Tangential stress tensor
    """

    # Convert the normal elasticity to a 3D tensor format
    three_dim_normal_elas = as_3D_tensor(normal_elasticity)

    # Compute base stress
    base_stress = compute_stress(strain_tensor, lambda_coefficient, shear_modulus)

    # Compute contributions to the tangential stress
    normal_elasticity_contribution = 3 * shear_modulus * (
                3 * shear_modulus / (3 * shear_modulus + linear_hardening_coeff) - beta_coeff)
    normal_elasticity_contribution *= fe.inner(three_dim_normal_elas, strain_tensor) * three_dim_normal_elas

    deviatoric_contribution = 2 * shear_modulus * beta_coeff * fe.dev(strain_tensor)

    # Combine base stress with additional stress contributions
    tangential_stress = base_stress - normal_elasticity_contribution - deviatoric_contribution

    return tangential_stress


def sigma_v(strain, lmbda_local_DG: fe.Function, mu_local_DG: fe.Function) -> fe.Function:
    """
    Calculate the Von-Mises Stress.

    Parameters:
    - strain: Strain tensor

    Returns:
    - Von-Mises Stress
    """
    s = fe.dev(compute_stress(strain, lmbda_local_DG, mu_local_DG))
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
        self.newton_lhs = fe.inner(compute_strain_tensor(self.v), compute_tangential_stress(compute_strain_tensor(self.u_), self.n_elas, self.mu_local_DG,
                                                           self.local_linear_hardening_DG, self.beta,
                                                           self.lmbda_local_DG)) * self.dxm
        self.newton_rhs = -fe.inner(compute_strain_tensor(self.u_), as_3D_tensor(self.sig)) * self.dxm

    @property
    def mesh(self) -> 'MeshType':
        return self._mesh

    @property
    def simulation_config(self) -> 'SimulationConfig':
        return self._simulation_config
