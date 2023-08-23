import ufl
import fenics as fe
import numpy as np

from AlCementCor.bnd import SquareStrainRate, FunctionDisplacementBoundaryCondition
from AlCementCor.config import SimulationConfig
from AlCementCor.fenics_helpers import as_3D_tensor, local_project, assign_values_based_on_boundaries
from AlCementCor.material_model_config import LinearElastoPlasticConfig
from AlCementCor.postproc import plot
from AlCementCor.time_controller import PITimeController


def compute_strain_tensor(displacement: fe.Function):
    """
    Calculate the strain tensor (epsilon) based on a given displacement field.

    The strain tensor, in continuum mechanics, quantifies how a material deforms under the influence of forces.
    It is defined as half the sum of the gradient of the displacement field and its transpose.

    In this function, the Plane-Strain condition is assumed. This condition is a simplifying assumption where
    out-of-plane strains are assumed to be zero. It's often used in plate bending or situations where one
    dimension is significantly smaller than the other two.

    Parameters:
    - displacement: The displacement function representing how points in the material move. This function is
                    usually vector-valued.

    Returns:
    - A 3x3 tensor representing the strain tensor, where the third component (out-of-plane) is zero due to the
      Plane-Strain assumption.
    """

    # The symmetric gradient (or symmetrized gradient) of the displacement is computed.
    # This symmetrization is the mathematical formulation of the definition of strain:
    # ε = 1/2 (∇u + (∇u)ᵀ)
    # Where u is the displacement field and ∇u is its gradient.
    symmetric_gradient = fe.sym(fe.nabla_grad(displacement))

    # Construct the 3x3 strain tensor using the computed symmetric gradient.
    # Given the plane-strain assumption, the third row and column are all zeros.
    return fe.as_tensor([
        [symmetric_gradient[0, 0], symmetric_gradient[0, 1], 0],
        [symmetric_gradient[0, 1], symmetric_gradient[1, 1], 0],
        [0, 0, 0]
    ])


def compute_stress(strain_tensor, lambda_coefficient: fe.Function, shear_modulus: fe.Function) -> fe.Function:
    """
    Calculate the Cauchy Stress Tensor for linear elasticity.

    In the context of linear elasticity, the stress tensor relates to the strain tensor through the material's
    elasticity parameters. Specifically, it divides into two main contributions: the isotropic (volumetric) stress
    and the deviatoric (shape-changing) stress.

    Parameters:
    - strain_tensor: 2D or 3D strain tensor representing deformation. It quantifies how the material deforms.
    - lambda_coefficient: Lamé's first parameter, which is related to the material's bulk modulus and
                          Poisson's ratio. It characterizes the material's resistance to uniform compression or dilation.
    - shear_modulus: Material's shear modulus (or Lamé's second parameter). It quantifies the material's resistance
                     to shape changes (shearing).

    Returns:
    - Cauchy Stress Tensor, which describes the internal forces within a material subjected to external loads.
    """

    # The isotropic (or volumetric) stress contribution arises due to a change in volume of the material.
    # It is proportional to the trace of the strain tensor (sum of its diagonal elements), which represents
    # the fractional change in volume. The Identity(3) term produces a 3x3 identity tensor.
    # This part of the stress tends to change the material's volume without changing its shape.
    isotropic_stress = lambda_coefficient * fe.tr(strain_tensor) * fe.Identity(3)

    # The deviatoric (or shape-changing) stress contribution is due to the shape change of the material.
    # It is proportional to the actual strain tensor itself and determines the shear stresses within the material.
    # This part of the stress tends to deform the material without a change in volume.
    deviatoric_stress = 2 * shear_modulus * strain_tensor

    # The total stress tensor is the sum of the isotropic and deviatoric contributions. In linear elasticity,
    # these two contributions are additive.
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


def compute_von_mises_stress_from_strain(strain_tensor, lambda_coefficient: fe.Function, shear_modulus: fe.Function) -> fe.Function:
    """
    Calculate the Von-Mises Stress.

    The Von-Mises stress is a scalar representation of a complex state of stress in a material. It provides
    a single value that can be compared to the material's yield strength to determine if yielding occurs.
    It is particularly useful in the context of ductile material failure.

    Parameters:
    - strain_tensor: The strain tensor describes the deformation of a material element in terms of elongations
                     and rotations. For linear elasticity, it linearly relates to the stress tensor.
    - lambda_coefficient: Lamé's first parameter, a material property related to its bulk modulus and Poisson's ratio.
                          It influences the volumetric response of the material.
    - shear_modulus: Also known as the Lamé's second parameter, it quantifies the material's resistance to shearing
                     (shape-changing) deformations.

    Returns:
    - Von-Mises Stress: A scalar stress value derived from the stress tensor, useful for failure predictions.
    """

    # First, we compute the full Cauchy stress tensor for the material based on its strain tensor and
    # material properties (lambda_coefficient and shear_modulus).
    # Then, the deviatoric component of this stress tensor is extracted. The deviatoric stress represents the
    # shape-changing aspect of stress, discarding the volumetric (hydrostatic) part.
    deviatoric_stress = fe.dev(compute_stress(strain_tensor, lambda_coefficient, shear_modulus))

    return compute_von_mises_stress(deviatoric_stress)


def compute_von_mises_stress(sig: fe.Function) -> fe.Function:
    """
    Calculate and project the Von-Mises Stress onto a DG(0) function space.

    The Von-Mises stress is a scalar value often used in mechanical engineering to
    predict yielding of materials under complex loading conditions. This function first
    calculates the Von-Mises stress from a given stress tensor and then projects the
    resulting scalar field onto a specified finite element function space. Projecting onto
    a DG(0) space means the stress will be represented as piecewise constant values over
    the mesh elements. This can be essential for visualization or when post-processing
    stress distributions.

    Parameters:
    - sig: Stress tensor representing the internal forces in the material. It's typically
           derived from a balance of forces in the material under deformation.

    Returns:
    - Von-Mises Stress: A scalar stress value derived from the stress tensor, useful for failure predictions.
    """

    # Convert the provided stress tensor (which might be 2D) into a 3D tensor. This step ensures
    # compatibility with 3D operations, even if the actual problem is 2D.
    sig_n = as_3D_tensor(sig)

    # Extract the deviatoric part of the stress tensor. The deviatoric stress tensor represents
    # the differential or shape-changing aspects of stress, separate from the hydrostatic or
    # volumetric part.
    s = fe.dev(sig_n)

    # Compute the magnitude of the Von-Mises stress. The formula is derived from the deviatoric
    # stress tensor and represents the effective stress that can be compared with material yield
    # strengths.
    return fe.sqrt(3 / 2. * fe.inner(s, s))



def ppos(x):
    """
    Macaulay's Bracket for <f_elastic>+.
    Returns a value only positive for x > 0.
    """
    return (x + abs(x)) / 2.


# https://www.dynasupport.com/tutorial/computational-plasticity/radial-return
# https://www.dynasupport.com/tutorial/computational-plasticity/generalizing-the-yield-function
# https://www.dynasupport.com/tutorial/computational-plasticity/the-consistent-tangent-matrix
def proj_sig(deps, old_sig, old_p, sig_0_local, C_linear_h_local, mu_local, lmbda_local_DG, mu_local_DG):
    # update stress from change in strain (deps)
    sig_n = as_3D_tensor(old_sig)
    sig_elas = sig_n + compute_stress(deps, lmbda_local_DG, mu_local_DG)

    # trial stress
    s = fe.dev(sig_elas)
    # von-Mises stress or equivalent trial stress
    sig_eq = fe.sqrt(3 / 2. * fe.inner(s, s))

    # prevent division by zero
    # if np.mean(local_project(old_p, P0).vector()[:]) < 1E-12:
    #    old_p += 1E-12

    # https://doc.comsol.com/5.5/doc/com.comsol.help.sme/sme_ug_theory.06.29.html#3554826
    # Calculate flow stress based on hardening law
    # linear hardening
    # k_linear = C_sig0 + C_linear_isotropic_hardening * old_p
    k_linear = sig_0_local + C_linear_h_local * old_p

    #     # Ludwik hardening
    #     k_ludwik = C_sig0 + C_nlin_ludwik * pow(old_p + 1E-12, C_exponent_ludwik)

    #     # swift hardening
    #     k_swift = C_sig0 * pow(1 + old_p/C_swift_eps0, C_exponent_swift)

    # yield surface/ if trial stress <= yield_stress + H * old_p: elastic
    f_elas = sig_eq - k_linear
    # f_elas = sig_eq - k_ludwik
    # f_elas = sig_eq - k_swift

    # change of plastic strain =0 when f_elas < 0
    # in elastic case = 0
    dp = ppos(f_elas) / (3 * mu_local + C_linear_h_local)
    # dp_old = ppos(f_elas) / (3 * C_mu + C_linear_isotropic_hardening)
    # dp = ppos(f_elas) / (3 * C_mu)
    # print("dp", np.mean(local_project(dp, P0).vector()[:]), np.max(local_project(dp, P0).vector()[:]))
    # print("dp_old", np.mean(local_project(dp_old, P0).vector()[:]), np.max(local_project(dp_old, P0).vector()[:]))

    # normal vector on yield surface?
    # in elastic case = 0
    n_elas = s * ppos(f_elas) / (sig_eq * f_elas)

    # radial return mapping?
    # in elastic case = 0
    beta = 3 * mu_local * dp / sig_eq

    # updated cauchy stress tensor
    # in elastic case = sig_elas
    new_sig = sig_elas - beta * s

    # Hydrostatic stress
    sig_hyd = (1. / 3) * fe.tr(new_sig)

    return fe.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
        fe.as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]), \
        beta, dp, sig_hyd


class LinearElastoPlasticIntegrator:
    def __init__(self, model: 'SimulationModel'):
        self.model = model
        # todo: make settable
        self.results_file = self.setup_results_file("plasticity_results.xdmf")

        # todo: make settable
        self.max_iters, self.tolerance = 10, 1e-8  # Newton-Raphson procedure parameters
        self.time_step = self.model.integration_time_limit / self.model.total_timesteps

        self.time_controller = PITimeController(self.time_step, 1e-2 * self.tolerance)
        self.time = 0.0

    def setup_results_file(self, filename: str):
        results_file = fe.XDMFFile(filename)
        results_file.parameters["flush_output"] = True
        results_file.parameters["functions_share_mesh"] = True
        return results_file

    def single_time_step_integration(self):
        self.time += self.time_step
        for condition in self.model.boundary.conditions:
            condition.update_time(self.time_step)

        A, Res = fe.assemble_system(self.model.newton_lhs, self.model.newton_rhs, self.model.boundary.bc)
        newton_res_norm, plastic_strain_update = self.run_newton_raphson(A, Res)

        if newton_res_norm > 1 or np.isnan(newton_res_norm):
            raise ValueError("ERROR: Calculation diverged!")

        self.update_and_store_results(plastic_strain_update)
        self.time_step = self.time_controller.update(newton_res_norm)

    def update_and_store_results(self, dp_):
        u, Du, p, sig, sig_old, sig_hyd, sig_hyd_avg = self.model.u, self.model.Du, self.model.p, self.model.sig, self.model.sig_old, self.model.sig_hyd, self.model.sig_hyd_avg

        u.assign(u + Du)
        p.assign(p + local_project(dp_, self.model.W0, self.model.dxm))
        sig_old.assign(sig)
        sig_hyd_avg.assign(fe.project(sig_hyd, self.model.P0))

        self.results_file.write(u, self.time)
        p_avg = fe.Function(self.model.P0, name="Plastic strain")
        p_avg.assign(fe.project(p, self.model.P0))
        self.results_file.write(p_avg, self.time)

    def run_newton_raphson(self, system_matrix, residual):
        """
        Solve a nonlinear problem using the Newton-Raphson method.
        """

        # Initial residual norm
        initial_residual_norm = residual.norm("l2")

        # Initialize iteration counter and current residual norm
        iteration_counter = 0
        current_residual_norm = initial_residual_norm

        # Start iterations
        while iteration_counter == 0 or (
                initial_residual_norm > 0 and current_residual_norm / initial_residual_norm > self.tolerance and iteration_counter < self.max_iters):
            # Solve the linear system
            fe.solve(system_matrix, self.model.du.vector(), residual, "mumps")

            # Update solution
            self.model.Du.assign(self.model.Du + self.model.du)
            strain_change = compute_strain_tensor(self.model.Du)

            # Project the new stress
            stress_update, elastic_strain_update, back_stress_update, pressure_change, hydrostatic_stress_update = proj_sig(
                strain_change, self.model.sig_old, self.model.p, self.model.local_initial_stress,
                self.model.local_linear_hardening,
                self.model.local_shear_modulus, self.model.lmbda_local_DG, self.model.mu_local_DG)

            # Update field values
            local_project(stress_update, self.model.W, self.model.dxm, self.model.sig)
            local_project(elastic_strain_update, self.model.W, self.model.dxm, self.model.n_elas)
            local_project(back_stress_update, self.model.W0, self.model.dxm, self.model.beta)
            self.model.sig_hyd.assign(local_project(hydrostatic_stress_update, self.model.W0, self.model.dxm))

            # Assemble system
            system_matrix, residual = fe.assemble_system(self.model.newton_lhs, self.model.newton_rhs,
                                                         self.model.boundary.bc_iter)

            # Update residual norm
            current_residual_norm = residual.norm("l2")
            print(f"Residual: {current_residual_norm}")

            # Increment iteration counter
            iteration_counter += 1

        return current_residual_norm, pressure_change


class LinearElastoPlasticModel:

    def __init__(self, config_file: "LinearElastoPlasticConfig"):
        self._simulation_config = config_file

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

        # Geometry setup
        self._mesh = self._simulation_config.mesh
        self.l_x = self._simulation_config.l_x
        self.l_y = self._simulation_config.l_y

        self._setup()

        # Set up boundary conditions
        self.boundary = LinearElastoPlasticBnd(self._simulation_config, self.V)

        # Define boundary and values
        partioning = [self._simulation_config.width]  # Assuming the width is the "boundary" between substrate and layer

        # Assign layer values
        substrate_properties = self._simulation_config.substrate_properties
        layer_properties = self._simulation_config.layer_properties

        self.local_initial_stress = assign_values_based_on_boundaries(
            self.W0, partioning,
            [substrate_properties.yield_strength, layer_properties.yield_strength]
        )

        self.local_shear_modulus = assign_values_based_on_boundaries(
            self.W0, partioning,
            [substrate_properties.shear_modulus, layer_properties.shear_modulus]
        )

        self.local_linear_hardening = assign_values_based_on_boundaries(
            self.W0, partioning,
            [substrate_properties.linear_isotropic_hardening, layer_properties.linear_isotropic_hardening]
        )

    def _setup(self) -> None:
        self._setup_function_spaces()
        self._setup_displacement_functions()
        self._setup_stress_functions()
        self._setup_other_functions()
        self._setup_local_properties()
        self._setup_newton_equations()

    def _setup_function_spaces(self) -> None:
        """Set up the function spaces required for the simulation."""
        mesh_cell = self._mesh.ufl_cell()
        self.deg_stress = self._simulation_config.finite_element_degree_stress
        deg_u = self._simulation_config.finite_element_degree_u

        # Create vector and scalar function spaces
        self.V = fe.VectorFunctionSpace(self._mesh, "CG", deg_u)
        self.P0 = fe.FunctionSpace(self._mesh, "DG", 0)

        # Setup quadrature spaces
        quad_element = fe.FiniteElement("Quadrature", mesh_cell, degree=self.deg_stress, quad_scheme='default')
        self.W0 = fe.FunctionSpace(self._mesh, quad_element)

        vec_element = fe.VectorElement("Quadrature", mesh_cell, degree=self.deg_stress, dim=4, quad_scheme='default')
        self.W = fe.FunctionSpace(self._mesh, vec_element)

    def _setup_displacement_functions(self):
        """Initialize functions for total, correction, and current increment displacements."""
        self.u = fe.Function(self.V, name="Total displacement")
        self.du = fe.Function(self.V, name="Iteration correction")
        self.Du = fe.Function(self.V, name="Current increment")

    def _setup_stress_functions(self):
        """Initialize functions for stress, previous stress, and elastic domain normal."""
        self.sig = fe.Function(self.W, name="Stress")
        self.sig_old = fe.Function(self.W, name="Previous stress")
        self.n_elas = fe.Function(self.W, name="Elastic domain normal")

    def _setup_other_functions(self):
        """Initialize miscellaneous functions."""
        self.beta = fe.Function(self.W0, name="Beta")
        self.p = fe.Function(self.W0, name="Cumulative plastic strain")
        self.sig_hyd = fe.Function(self.W0, name="Hydrostatic stress")
        self.sig_0_local = fe.Function(self.W0, name="local sig0")
        self.mu_local = fe.Function(self.W0, name="local mu")
        self.lmbda_local = fe.Function(self.W0, name="local lmbda")
        self.C_linear_h_local = fe.Function(self.W0, name="local hardening factor")

        self.sig_hyd_avg = fe.Function(self.P0, name="Avg. Hydrostatic stress")
        self.sig_0_test = fe.Function(self.P0, name="test")
        self.lmbda_test = fe.Function(self.P0, name="test2")

        self.metadata = {"quadrature_degree": self.deg_stress, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)
        self.v = fe.TrialFunction(self.V)
        self.u_ = fe.TestFunction(self.V)

    def _setup_local_properties(self):
        """Setup local properties of the model."""
        self.mu_local_DG = fe.Function(self.P0)
        self._assign_local_values(self._simulation_config.substrate_properties.shear_modulus,
                                  self._simulation_config.layer_properties.shear_modulus,
                                  self.mu_local_DG)

        self.lmbda_local_DG = fe.Function(self.P0)
        self._assign_local_values(self._simulation_config.substrate_properties.first_lame_parameter,
                                  self._simulation_config.layer_properties.first_lame_parameter,
                                  self.lmbda_local_DG)

        self.local_linear_hardening_DG = fe.Function(self.P0)
        self._assign_local_values(self._simulation_config.substrate_properties.linear_isotropic_hardening,
                                  self._simulation_config.layer_properties.linear_isotropic_hardening,
                                  self.local_linear_hardening_DG)

    def _assign_local_values(self, values: float, outer_values: float, local_DG: fe.Function) -> None:
        """Assign values based on the specified condition."""
        width = self._simulation_config.width
        dofmap = self.P0.tabulate_dof_coordinates()[:]
        vec = np.full(dofmap.shape[0], values)
        vec[dofmap[:, 0] > width] = outer_values
        local_DG.vector()[:] = vec

    def _setup_newton_equations(self):
        """Setup Newton equations for the model."""
        self.newton_lhs = fe.inner(compute_strain_tensor(self.v),
                                   compute_tangential_stress(compute_strain_tensor(self.u_), self.n_elas,
                                                             self.mu_local_DG,
                                                             self.local_linear_hardening_DG, self.beta,
                                                             self.lmbda_local_DG)) * self.dxm
        self.newton_rhs = -fe.inner(compute_strain_tensor(self.u_), as_3D_tensor(self.sig)) * self.dxm

    @property
    def mesh(self) -> 'MeshType':
        """Provide access to mesh."""
        return self._mesh

    @property
    def simulation_config(self) -> LinearElastoPlasticConfig:
        """Provide access to simulation configuration."""
        return self._simulation_config

    @property
    def integration_time_limit(self):
        return self.simulation_config.simulation_config.integration_time_limit

    @property
    def total_timesteps(self):
        return self.simulation_config.simulation_config.total_timesteps


class LinearElastoPlasticBnd:
    def __init__(self, simulation_config, V):
        self.simulation_config = simulation_config
        self.mesh = simulation_config.mesh
        self.V = V
        self.l_x = simulation_config.l_x
        self.l_y = simulation_config.l_y
        self.bc, self.bc_iter, self.conditions = self.setup_displacement_bnd()

    def setup_displacement_bnd(self):
        # Define boundary location conditions
        def is_bottom_boundary(x, on_boundary):
            return on_boundary and fe.near(x[1], 0.0)

        def is_top_boundary(x, on_boundary):
            return on_boundary and fe.near(x[1], self.l_y)

        def is_left_boundary(x, on_boundary):
            return on_boundary and fe.near(x[0], 0.0)

        def is_right_boundary(x, on_boundary):
            return on_boundary and fe.near(x[0], self.l_x)

        # Define the boundary conditions
        # bnd_length = l_x
        # displacement_func = LinearDisplacementX((-C_strain_rate, 0.0), bnd_length)
        # displacement_func = ConstantStrainRate((-C_strain_rate, 0.0))
        # bottom_condition = FunctionDisplacementBoundaryCondition(V, is_bottom_boundary, displacement_func)
        # bottom_condition = NoDisplacementBoundaryCondition(V, is_bottom_boundary)
        # if two_layers:
        #     bottom_condition = ConstantStrainRateBoundaryCondition(V, is_bottom_boundary, -C_strain_rate)
        # else:
        #     bottom_condition = NoDisplacementBoundaryCondition(V, is_bottom_boundary)

        # top_condition = ConstantStrainRateBoundaryCondition(V, is_top_boundary, C_strain_rate)
        # bnd_length = 100.0
        # displacement_func = LinearDisplacementX(-C_strain_rate, bnd_length)
        # displacement_func = ConstantStrainRate((-C_strain_rate, 0.0))
        # top_condition = FunctionDisplacementBoundaryCondition(V, is_top_boundary, displacement_func)
        # top_condition = NoDisplacementBoundaryCondition(V, is_top_boundary)

        # bnd_length = l_y
        # displacement_func = SinglePointDisplacement((0.0, 6.0), (-C_strain_rate, 0.0))
        # displacement_func = ConstantStrainRate((-C_strain_rate, 0.0))
        displacement_func = SquareStrainRate((-self.simulation_config.strain_rate, 0.0), 0.0, self.l_y)
        left_condition = FunctionDisplacementBoundaryCondition(self.V, is_left_boundary, displacement_func)

        # displacement_func = SinglePointDisplacement((4.2, 6.0), (-C_strain_rate, 0.0))
        # displacement_func = ConstantStrainRate((-C_strain_rate, 0.0))
        # right_condition = FunctionDisplacementBoundaryCondition(V, is_right_boundary, displacement_func)

        # Create the conditions list
        conditions = [left_condition]

        # Generate the Dirichlet boundary conditions
        bc = [condition.get_condition() for condition in conditions]

        # Generate homogenized boundary conditions
        bc_iter = [condition.get_homogenized_condition() for condition in conditions]

        return bc, bc_iter, conditions
