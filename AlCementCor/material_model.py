import ufl
import fenics as fe
import numpy as np
from functools import partial
from typing import Any, Tuple, List
from enum import Enum
from AlCementCor.bnd import DisplacementElastoPlasticBnd, StressElastoPlasticBnd
from AlCementCor.config import SimulationConfig
from AlCementCor.fenics_helpers import as_3D_tensor, local_project, assign_values_based_on_boundaries
from AlCementCor.material_model_config import LinearElastoPlasticConfig
from AlCementCor.misc import check_divergence
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


def compute_von_mises_stress_from_strain(strain_tensor, lambda_coefficient: fe.Function,
                                         shear_modulus: fe.Function) -> fe.Function:
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


class HardeningModel(Enum):
    LINEAR = 'linear'
    LUDWIK = 'ludwik'
    SWIFT = 'swift'


def compute_hardening(plastic_strain: float, initial_stress: float, hardening_params: dict,
                      model_type: HardeningModel = HardeningModel.LINEAR) -> (float, float):
    """
    Compute the flow stress and its derivative for a given plastic strain based on the selected hardening model.

    Parameters:
    - plastic_strain (float): The accumulated plastic strain.
    - initial_stress (float): The stress value at zero plastic strain (initial yield stress).
    - hardening_params (dict): Parameters for the selected hardening model.
    - model_type (HardeningModel): Enum indicating the type of hardening model. Default is LINEAR.

    Returns:
    - flow_stress (float): The flow stress for the given plastic strain.
    - stress_derivative (float): The derivative of the flow stress with respect to the plastic strain.

    Note:
    - Linear model: flow_stress = initial_stress + linear_coefficient * plastic_strain
    - Ludwik model: flow_stress = initial_stress + ludwik_nonlinear_coefficient * (plastic_strain + TOL) ^ ludwik_exponent
    - Swift model: flow_stress = initial_stress * (1 + plastic_strain / swift_epsilon_0) ^ swift_exponent
    """

    # Small tolerance to prevent division by zero or raise to power issues
    TOLERANCE = 1E-12

    flow_stress = 0.0
    stress_derivative = 0.0

    # Linear hardening model
    if model_type == HardeningModel.LINEAR:
        # The linear hardening model represents a linear relationship between stress and plastic strain.
        # Applicable to materials where the increase in flow stress is proportional to plastic deformation.
        # Mathematical representation:
        # sigma = sigma_0 + H * epsilon_p
        # where H is the hardening modulus.
        flow_stress = initial_stress + hardening_params.get('linear_coefficient', 0) * plastic_strain
        stress_derivative = hardening_params.get('linear_coefficient', 0)

    # Ludwik hardening model
    elif model_type == HardeningModel.LUDWIK:
        # The Ludwik model is suitable for metals exhibiting power-law strain hardening.
        # It is widely used for modeling cold work metal plasticity.
        # Mathematical representation:
        # sigma = sigma_0 + K * epsilon_p^n
        # where K and n are material constants.
        nonlinear_coefficient = hardening_params.get('ludwik_nonlinear_coefficient', 0)
        exponent = hardening_params.get('ludwik_exponent', 0)

        # Flow stress computation using the Ludwik model formula
        flow_stress = initial_stress + nonlinear_coefficient * (plastic_strain + TOLERANCE) ** exponent

        # Derivative with respect to the plastic strain for the Ludwik model
        stress_derivative = nonlinear_coefficient * exponent * (plastic_strain + TOLERANCE) ** (exponent - 1)

    # Swift hardening model
    elif model_type == HardeningModel.SWIFT:
        # The Swift hardening model is a power-law based model but with the inclusion of a normalization term.
        # It's applicable to metals that experience significant hardening during deformation.
        # Mathematical representation:
        # sigma = sigma_0 * (1 + epsilon_p / epsilon_0)^n
        # where epsilon_0 is a reference strain and n is a material constant.
        epsilon_0 = hardening_params.get('swift_epsilon_0', 0)
        exponent = hardening_params.get('swift_exponent', 0)

        # Flow stress computation using the Swift model formula
        flow_stress = initial_stress * (1 + plastic_strain / epsilon_0) ** exponent

        # Derivative with respect to the plastic strain for the Swift model
        stress_derivative = initial_stress * exponent * (1 + plastic_strain / epsilon_0) ** (exponent - 1) / epsilon_0

    # Unsupported model type
    else:
        raise ValueError(f"Unsupported hardening model: {model_type.value}")

    return flow_stress, stress_derivative


def project_stress(incremental_strain, previous_stress, shear_modulus, lambda_DG, mu_DG, yield_stress,
                   hardening_derivative):
    """
    Return-mapping algorithm for elastoplasticity, which projects trial stresses back to the yield surface
    in case of plastic deformation. The method is essential for numerical stability and accuracy in
    plasticity simulations.

    Parameters:
    - incremental_strain (list or tensor): Incremental strain tensor.
    - previous_stress (list or tensor): Stress tensor from the previous iteration or time step.
    - shear_modulus (float): Shear modulus of the material.
    - lambda_DG (float): Lambda parameter for the Discontinuous Galerkin (DG) method.
    - mu_DG (float): Mu parameter for the DG method.
    - yield_stress (float): Current value of the yield stress or hardening parameter.
    - hardening_derivative (float): Derivative of the yield stress with respect to plastic strain.

    Returns:
    - tuple: Contains updated stress tensor, normal to the yield surface, radial_return_factor, change in plastic strain, and volumetric stress.
    """

    # Convert the previous stress tensor to a 3D format for ease of calculations.
    stress_3D = as_3D_tensor(previous_stress)

    # Elastic predictor step: Based on the assumption of purely elastic deformation, compute an intermediate or 'trial' stress.
    # σ_trial = σ_old + D:ε_increment, where D is the elastic stiffness matrix.
    trial_stress = stress_3D + compute_stress(incremental_strain, lambda_DG, mu_DG)

    # Extract the deviatoric component (shape-changing part) of the trial stress.
    # This helps in determining if the material yields due to shear deformation.
    # s = σ - (1/3)*tr(σ)*I
    deviatoric_stress = fe.dev(trial_stress)

    # Von Mises equivalent stress gives a scalar measure of stress magnitude, irrespective of the stress state's direction.
    # σ_eq = sqrt(3/2) * ||s||
    equivalent_stress = fe.sqrt(3 / 2. * fe.inner(deviatoric_stress, deviatoric_stress))

    # The yield function evaluates the difference between the trial stress and the current yield stress.
    # If positive, it indicates that the material has yielded.
    yield_function_value = equivalent_stress - yield_stress

    # Using the yield function's value and the hardening derivative, compute the amount of plastic strain developed.
    # Δp = f / (3*G + H'), where H' is the hardening derivative.
    plastic_strain_increment = ppos(yield_function_value) / (3 * shear_modulus + hardening_derivative)

    # The normal to the yield surface in stress space provides the direction of maximum resistance to deformation.
    yield_normal = deviatoric_stress * ppos(yield_function_value) / (equivalent_stress * yield_function_value)

    # Radial return maps the trial stress back to the yield surface along the normal direction.
    # Δσ = - 3*G*Δp*n
    radial_return_factor = 3 * shear_modulus * plastic_strain_increment / equivalent_stress

    # Correcting the trial stress ensures that the updated stress state remains on or inside the yield surface.
    corrected_stress = trial_stress - radial_return_factor * deviatoric_stress

    # Hydrostatic (volumetric) stress component represents the uniform compression or dilation in the material.
    volumetric_stress = (1. / 3) * fe.tr(corrected_stress)

    return fe.as_vector(
        [corrected_stress[0, 0], corrected_stress[1, 1], corrected_stress[2, 2], corrected_stress[0, 1]]), \
        fe.as_vector([yield_normal[0, 0], yield_normal[1, 1], yield_normal[2, 2], yield_normal[0, 1]]), \
        radial_return_factor, plastic_strain_increment, volumetric_stress


class LinearElastoPlasticIntegrator:
    def __init__(self, model: 'SimulationModel'):
        self.model = model
        self._MAX_ITERS = 10  # Newton-Raphson procedure parameters
        self._TOLERANCE = 1e-8
        self.time_step = self.model.integration_time_limit / self.model.total_timesteps
        self.time_controller = PITimeController(self.time_step, 0.01 * self._TOLERANCE)
        self.time = 0.0

    def single_time_step_integration(self):
        """Perform a single time step integration."""
        self.time += self.time_step
        self._update_boundary_conditions()

        system_matrix, residual = fe.assemble_system(self.model.newton_lhs, self.model.newton_rhs,
                                                     self.model.boundary_deformation.bc)
        newton_res_norm, plastic_strain_update = self._run_newton_raphson(system_matrix, residual)

        check_divergence(newton_res_norm)

        self._update_model_values(plastic_strain_update)
        self.time_step = self.time_controller.update(newton_res_norm)

    def _update_boundary_conditions(self):
        """Update boundary conditions based on the current time step."""
        self.model.update_bnd(self.time_step, self.time)


    def _update_model_values(self, plastic_strain_update: float):
        """Update model's internal values after a successful Newton-Raphson step."""
        self.model.total_displacement.assign(self.model.total_displacement + self.model.current_displacement_increment)

        self.model.cum_plstic_strain.assign(
            self.model.cum_plstic_strain + local_project(plastic_strain_update, self.model.scalar_quad_space,
                                                         self.model.dxm))
        self.model.old_stress.assign(self.model.stress)
        self.model.sig_hyd_avg.assign(fe.project(self.model.hydrostatic_stress, self.model.P0))

    def _run_newton_raphson(self, system_matrix, residual):
        """
        Solve a nonlinear problem using the Newton-Raphson method.
        """

        # Initial residual norm
        initial_residual_norm = residual.norm("l2")

        # Initialize iteration counter and current residual norm
        iteration_counter = 0
        current_residual_norm = initial_residual_norm

        m = self.model

        # Start iterations
        while iteration_counter == 0 or (
                initial_residual_norm > 0 and current_residual_norm / initial_residual_norm > self._TOLERANCE and iteration_counter < self._MAX_ITERS):
            # Solve the linear system
            fe.solve(system_matrix, m.displacement_correction.vector(), residual, "mumps")

            # Update solution
            m.current_displacement_increment.assign(m.current_displacement_increment + m.displacement_correction)
            strain_change = compute_strain_tensor(m.current_displacement_increment)

            # Parameters for the hardening model
            hardening_params = {
                'linear_coefficient': m.local_linear_hardening
            }

            k, dk_dp = compute_hardening(m.cum_plstic_strain, m.local_initial_stress, hardening_params)

            # Project the new stress
            stress_update, elastic_strain_update, back_stress_update, pressure_change, hydrostatic_stress_update = project_stress(
                strain_change, m.old_stress, m.local_shear_modulus, m.lmbda_local_DG, m.mu_local_DG, k, dk_dp)

            # Update field values
            local_project(stress_update, m.tensor_quad_space, m.dxm, m.stress)
            local_project(elastic_strain_update, m.tensor_quad_space, m.dxm, m.n_elas)
            local_project(back_stress_update, m.scalar_quad_space, m.dxm, m.beta)
            m.hydrostatic_stress.assign(local_project(hydrostatic_stress_update, m.scalar_quad_space, m.dxm))

            # Assemble system
            system_matrix, residual = fe.assemble_system(m.newton_lhs, m.newton_rhs, m.boundary_deformation.bc_iter)

            # Update residual norm
            current_residual_norm = residual.norm("l2")
            print(f"Residual: {current_residual_norm}")

            # Increment iteration counter
            iteration_counter += 1

        return current_residual_norm, pressure_change


class LinearElastoPlasticModel:

    def __init__(self, config_file: str):
        self._config = LinearElastoPlasticConfig(config_file)

        # Function spaces
        self.stress_degree: int = -1
        self.vector_space = None
        self.tensor_quad_space = None
        self.scalar_quad_space = None
        self.P0 = None

        # Displacement functions
        self.total_displacement = None
        self.displacement_correction = None
        self.current_displacement_increment = None

        # Stress functions
        self.stress = None
        self.old_stress = None
        self.n_elas = None

        # Local properties
        self.mu_local_DG = None
        self.lmbda_local_DG = None
        self.local_linear_hardening_DG = None

        # Other functions and parameters
        self.beta = None
        self.cum_plstic_strain = None
        self.hydrostatic_stress = None
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
        self._mesh = self._config.mesh
        self.l_x = self._config.l_x
        self.l_y = self._config.l_y

        self._setup()

        # Set up boundary conditions
        self.boundary_deformation = DisplacementElastoPlasticBnd(self._config, self.vector_space)
        self.boundary_stress = StressElastoPlasticBnd(self._config, self.vector_space)
        self._setup_newton_equations()

        # Define boundary and values
        partioning = [self._config.width]  # Assuming the width is the "boundary" between substrate and layer

        # Assign layer values
        substrate_properties = self._config.substrate_properties
        layer_properties = self._config.layer_properties

        self.local_initial_stress = assign_values_based_on_boundaries(
            self.scalar_quad_space, partioning,
            [substrate_properties.yield_strength, layer_properties.yield_strength]
        )

        self.local_shear_modulus = assign_values_based_on_boundaries(
            self.scalar_quad_space, partioning,
            [substrate_properties.shear_modulus, layer_properties.shear_modulus]
        )

        self.local_linear_hardening = assign_values_based_on_boundaries(
            self.scalar_quad_space, partioning,
            [substrate_properties.linear_isotropic_hardening, layer_properties.linear_isotropic_hardening]
        )

    def _setup(self) -> None:
        self._setup_function_spaces()
        self._setup_displacement_functions()
        self._setup_stress_functions()
        self._setup_other_functions()
        self._setup_local_properties()

    def _setup_function_spaces(self) -> None:
        """Set up the function spaces required for the simulation."""
        mesh_cell = self._mesh.ufl_cell()
        self.stress_degree = self._config.finite_element_degree_stress
        deg_u = self._config.finite_element_degree_u

        # Create vector and scalar function spaces
        self.vector_space = fe.VectorFunctionSpace(self._mesh, "CG", deg_u)
        self.P0 = fe.FunctionSpace(self._mesh, "DG", 0)

        # Setup quadrature spaces
        quad_element = fe.FiniteElement("Quadrature", mesh_cell, degree=self.stress_degree, quad_scheme='default')
        self.scalar_quad_space = fe.FunctionSpace(self._mesh, quad_element)

        vec_element = fe.VectorElement("Quadrature", mesh_cell, degree=self.stress_degree, dim=4, quad_scheme='default')
        self.tensor_quad_space = fe.FunctionSpace(self._mesh, vec_element)

    def _setup_displacement_functions(self):
        """Initialize functions for total, correction, and current increment displacements."""
        self.total_displacement = fe.Function(self.vector_space, name="Total displacement")
        self.displacement_correction = fe.Function(self.vector_space, name="Iteration correction")
        self.current_displacement_increment = fe.Function(self.vector_space, name="Current increment")

    def _setup_stress_functions(self):
        """Initialize functions for stress, previous stress, and elastic domain normal."""
        self.stress = fe.Function(self.tensor_quad_space, name="Stress")
        self.old_stress = fe.Function(self.tensor_quad_space, name="Previous stress")
        self.n_elas = fe.Function(self.tensor_quad_space, name="Elastic domain normal")

    def _setup_other_functions(self):
        """Initialize miscellaneous functions."""
        self.beta = fe.Function(self.scalar_quad_space, name="Beta")
        self.cum_plstic_strain = fe.Function(self.scalar_quad_space, name="Cumulative plastic strain")
        self.hydrostatic_stress = fe.Function(self.scalar_quad_space, name="Hydrostatic stress")
        self.sig_0_local = fe.Function(self.scalar_quad_space, name="local sig0")
        self.mu_local = fe.Function(self.scalar_quad_space, name="local mu")
        self.lmbda_local = fe.Function(self.scalar_quad_space, name="local lmbda")
        self.C_linear_h_local = fe.Function(self.scalar_quad_space, name="local hardening factor")

        self.sig_hyd_avg = fe.Function(self.P0, name="Avg. Hydrostatic stress")
        self.sig_0_test = fe.Function(self.P0, name="test")
        self.lmbda_test = fe.Function(self.P0, name="test2")

        self.metadata = {"quadrature_degree": self.stress_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)
        self.v = fe.TrialFunction(self.vector_space)
        self.u_ = fe.TestFunction(self.vector_space)

    def _setup_local_properties(self):
        """Setup local properties of the model."""
        self.mu_local_DG = fe.Function(self.P0)
        self._assign_local_values(self._config.substrate_properties.shear_modulus,
                                  self._config.layer_properties.shear_modulus,
                                  self.mu_local_DG)

        self.lmbda_local_DG = fe.Function(self.P0)
        self._assign_local_values(self._config.substrate_properties.first_lame_parameter,
                                  self._config.layer_properties.first_lame_parameter,
                                  self.lmbda_local_DG)

        self.local_linear_hardening_DG = fe.Function(self.P0)
        self._assign_local_values(self._config.substrate_properties.linear_isotropic_hardening,
                                  self._config.layer_properties.linear_isotropic_hardening,
                                  self.local_linear_hardening_DG)

    def _assign_local_values(self, values: float, outer_values: float, local_DG: fe.Function) -> None:
        """Assign values based on the specified condition."""
        width = self._config.width
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

        stress_rhs = self.boundary_stress.get_stress_rhs(
            self.u_)  # Assuming self.boundary_conditions is an instance of LinearElastoPlasticBnd
        self.newton_rhs = (-fe.inner(compute_strain_tensor(self.u_), as_3D_tensor(self.stress)) * self.dxm
                           + stress_rhs)

    def update_bnd(self, time_step, time):
        self.boundary_deformation.update_bnd(time_step, time)
        self.boundary_stress.update_bnd(time_step, time)

    @property
    def mesh(self) -> 'MeshType':
        """Provide access to mesh."""
        return self._mesh

    @property
    def model_config(self) -> LinearElastoPlasticConfig:
        """Provide access to model configuration."""
        return self._config

    @property
    def simulation_config(self) -> SimulationConfig:
        """Provide access to simulation configuration."""
        return self._config.simulation_config

    @property
    def integration_time_limit(self):
        return self.model_config.simulation_config.integration_time_limit

    @property
    def total_timesteps(self):
        return self.model_config.simulation_config.total_timesteps

    @property
    def substrate_properties(self):
        return self.model_config.substrate_properties

    @property
    def layer_properties(self):
        return self.model_config.layer_properties
