import ufl
import fenics as fe
import numpy as np

from AlCementCor.bnd import SquareStrainRate, FunctionDisplacementBoundaryCondition
from AlCementCor.config import SimulationConfig
from AlCementCor.fenics_helpers import as_3D_tensor, local_project
from AlCementCor.info import summarize_and_print_config
from AlCementCor.input_file import process_abaqus_input_file, ExternalInput
from AlCementCor.interpolate import interpolate_displacements
from AlCementCor.material_properties import MaterialProperties
from AlCementCor.postproc import plot, plot_strain_displacement, plot_displacement, plot_movement
from AlCementCor.time_controller import PITimeController


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


def compute_von_mises_stress(strain_tensor, lambda_coefficient: fe.Function, shear_modulus: fe.Function) -> fe.Function:
    """
    Calculate the Von-Mises Stress.

    Parameters:
    - strain_tensor: Strain tensor
    - lambda_coefficient: Lamé's first parameter
    - shear_modulus: Material's shear modulus (or Lamé's second parameter)

    Returns:
    - Von-Mises Stress
    """

    # Compute the deviatoric component of the stress tensor
    deviatoric_stress = fe.dev(compute_stress(strain_tensor, lambda_coefficient, shear_modulus))

    # Compute the magnitude of the deviatoric stress
    deviatoric_magnitude = fe.sqrt(1.5 * fe.inner(deviatoric_stress, deviatoric_stress))

    return deviatoric_magnitude


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
        self.time_step = self.model.simulation_config._simulation_config.integration_time_limit / self.model.simulation_config._simulation_config.total_timesteps

        self.time_controller = None
        self.time = None
        self.displacement_over_time = None
        self.mean_stress_over_time = None
        self.displacement_list = None
        self.max_stress_over_time = None

    def setup_results_file(self, filename: str):
        results_file = fe.XDMFFile(filename)
        results_file.parameters["flush_output"] = True
        results_file.parameters["functions_share_mesh"] = True
        return results_file

    def run_time_integration(self):
        # Initialization
        self.initialize_time_variables()
        self.time_controller = PITimeController(self.time_step, 1e-2 * self.tolerance)

        # Time-stepping loop
        iteration = 0
        while self.time < self.model.simulation_config._simulation_config.integration_time_limit:
            iteration += 1
            self.time_step_integration(self.model.boundary.conditions, self.model.boundary.bc, self.model.boundary.bc_iter, self.model.local_initial_stress, self.model.local_linear_hardening, self.model.local_shear_modulus, self.max_iters, self.tolerance, self.results_file, self.model.l_x, self.model.l_y, iteration)
            print(f"Step: {iteration}, time: {self.time} s")
            print(f"displacement: {self.model.strain_rate.values()[0] * self.time} mm")

            self.displacement_over_time += [(np.abs(self.model.u(self.model.l_x / 2, self.model.l_y)[1]) / self.model.l_y, self.time)]

    def initialize_time_variables(self):
        self.time = 0
        self.displacement_over_time = [(0, 0)]
        self.max_stress_over_time = [0]
        self.mean_stress_over_time = [0]
        self.displacement_list = [0]

    def time_step_integration(self, conditions, bc, bc_iter, local_initial_stress, local_linear_hardening,
                              local_shear_modulus, max_iters, tolerance, results_file, l_x, l_y, iteration):
        self.time += self.time_controller.time_step
        for condition in conditions:
            condition.update_time(self.time_controller.time_step)

        A, Res = fe.assemble_system(self.model.newton_lhs, self.model.newton_rhs, bc)

        newton_res_norm, plastic_strain_update = self.run_newton_raphson(
            A, Res, self.model.newton_lhs, self.model.newton_rhs, bc_iter, self.model.du, self.model.Du,
            self.model.sig_old, self.model.p,
            local_initial_stress, local_linear_hardening,
            local_shear_modulus, self.model.lmbda_local_DG, self.model.mu_local_DG, self.model.sig, self.model.n_elas,
            self.model.beta,
            self.model.sig_hyd, self.model.W, self.model.W0, self.model.dxm, max_iters, tolerance)

        if newton_res_norm > 1 or np.isnan(newton_res_norm):
            raise ValueError("ERROR: Calculation diverged!")

        self.update_and_store_results(
            iteration, self.model.Du, plastic_strain_update, self.model.sig, self.model.sig_old, self.model.sig_hyd,
            self.model.sig_hyd_avg,
            self.model.p, self.model.W0, self.model.dxm, self.model.P0, self.model.u, l_x, l_y, self.time, results_file,
            self.max_stress_over_time,
            self.mean_stress_over_time, self.displacement_list
        )

        self.time_controller.update(newton_res_norm)

    def update_and_store_results(self, i, Du, dp_, sig, sig_old, sig_hyd, sig_hyd_avg, p, W0, dxm, P0, u, l_x, l_y,
                                 time,
                                 file_results, stress_max_t, stress_mean_t, disp_t):
        # update displacement
        u.assign(u + Du)
        # update plastic strain
        p.assign(p + local_project(dp_, W0, dxm))

        # update stress fields
        sig_old.assign(sig)
        sig_hyd_avg.assign(fe.project(sig_hyd, P0))

        # # s11, s12, s21, s22 = sig.split(deepcopy=True)
        # # avg_stress_y = np.average(s22.vector()[:])
        # # avg_stress = np.average(sig.vector()[:])
        sig_n = as_3D_tensor(sig)
        s = fe.dev(sig_n)

        # calculate the von-mises equivalent stress
        sig_eq = fe.sqrt(3 / 2. * fe.inner(s, s))
        sig_eq_p = local_project(sig_eq, P0, dxm)

        if i % 10 == 0:
            plot(i, u, sig_eq_p)

        # calculate and project the von-mises stress for later use
        stress_max_t.extend([np.abs(np.amax(sig_eq_p.vector()[:]))])
        stress_mean_t.extend([np.abs(np.mean(sig_eq_p.vector()[:]))])

        # append the y-displacement at the center of the bar
        disp_t.append(u(l_x / 2, l_y)[1])

        file_results.write(u, time)
        p_avg = fe.Function(P0, name="Plastic strain")
        p_avg.assign(fe.project(p, P0))
        file_results.write(p_avg, time)

    def check_convergence(self, nRes, nRes0, tol, niter, Nitermax):
        return niter == 0 or (nRes0 > 0 and nRes / nRes0 > tol and niter < Nitermax)

    def run_newton_raphson(self, system_matrix, residual, newton_form, residual_form, boundary_conditions,
                           solution_change, total_change, old_stress, hydrostatic_pressure, local_initial_stress,
                           local_linear_hardening, local_shear_modulus, local_lame_first, local_shear_modulus_dg,
                           stress, elastic_strain, back_stress, hydrostatic_stress, function_space, null_space,
                           dx_measure, max_iterations, tolerance):
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
                initial_residual_norm > 0 and current_residual_norm / initial_residual_norm > tolerance and iteration_counter < max_iterations):
            # Solve the linear system
            fe.solve(system_matrix, solution_change.vector(), residual, "mumps")

            # Update solution
            total_change.assign(total_change + solution_change)
            strain_change = compute_strain_tensor(total_change)

            # Project the new stress
            stress_update, elastic_strain_update, back_stress_update, pressure_change, hydrostatic_stress_update = proj_sig(
                strain_change, old_stress, hydrostatic_pressure, local_initial_stress, local_linear_hardening,
                local_shear_modulus, local_lame_first, local_shear_modulus_dg)

            # Update field values
            local_project(stress_update, function_space, dx_measure, stress)
            local_project(elastic_strain_update, function_space, dx_measure, elastic_strain)
            local_project(back_stress_update, null_space, dx_measure, back_stress)
            hydrostatic_stress.assign(local_project(hydrostatic_stress_update, null_space, dx_measure))

            # Assemble system
            system_matrix, residual = fe.assemble_system(newton_form, residual_form, boundary_conditions)

            # Update residual norm
            current_residual_norm = residual.norm("l2")
            print(f"Residual: {current_residual_norm}")

            # Increment iteration counter
            iteration_counter += 1

        return current_residual_norm, pressure_change


class LinearElastoPlasticModel:


    def __init__(self, config_file: str):
        self._simulation_config = LinearElastoPlasticConfig(config_file)

        # todo: move to config file
        self.strain_rate = fe.Constant(0.000001)

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

        # Assign layer values
        self.local_initial_stress = self.assign_layer_values(self._simulation_config.substrate_properties.yield_strength,
                                                             self._simulation_config.layer_properties.yield_strength)
        self.local_shear_modulus = self.assign_layer_values(self._simulation_config.substrate_properties.shear_modulus,
                                                            self._simulation_config.layer_properties.shear_modulus)
        self.local_linear_hardening = self.assign_layer_values(self._simulation_config.substrate_properties.linear_isotropic_hardening,
                                                               self._simulation_config.layer_properties.linear_isotropic_hardening)



    # assign local values to the layers
    def assign_layer_values(self, inner_value: float, outer_value: float) -> 'fe.Function':
        """Assign values based on layers."""

        class set_layer(fe.UserExpression):
            def __init__(self, inner_value, outer_value, simulation_config, **kwargs):
                self.inner = inner_value
                self.outer = outer_value
                self.width = simulation_config.width
                super().__init__(**kwargs)

            def eval(self, value, x):
                if x[0] > self.width:
                    value[0] = self.outer
                else:
                    value[0] = self.inner

            def value_shape(self):
                return ()

        layer = set_layer(inner_value, outer_value, self.simulation_config._simulation_config)
        return fe.interpolate(layer, self.W0)

    def _setup(self) -> None:
        self._setup_function_spaces()
        self._setup_displacement_functions()
        self._setup_stress_functions()
        self._setup_other_functions()
        self._setup_local_properties()
        self._setup_newton_equations()

    def _setup_function_spaces(self) -> None:
        """Set up the function spaces required for the simulation."""
        self.deg_stress = self._simulation_config._simulation_config.finite_element_degree_stress
        self.V = fe.VectorFunctionSpace(self._mesh, "CG", self._simulation_config._simulation_config.finite_element_degree_u)
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
        self._assign_local_values(self._simulation_config.substrate_properties.shear_modulus, self._simulation_config.layer_properties.shear_modulus,
                                  self.mu_local_DG)

        self.lmbda_local_DG = fe.Function(self.DG)
        self._assign_local_values(self._simulation_config.substrate_properties.first_lame_parameter,
                                  self._simulation_config.layer_properties.first_lame_parameter,
                                  self.lmbda_local_DG)

        self.local_linear_hardening_DG = fe.Function(self.DG)
        self._assign_local_values(self._simulation_config.substrate_properties.linear_isotropic_hardening,
                                  self._simulation_config.layer_properties.linear_isotropic_hardening,
                                  self.local_linear_hardening_DG)

    def _assign_local_values(self, values: float, outer_values: float, local_DG: fe.Function) -> None:
        """Assign values based on the specified condition."""
        dofmap = self.DG.tabulate_dof_coordinates()[:]
        vec = np.full(dofmap.shape[0], values)
        vec[dofmap[:, 0] > self._simulation_config._simulation_config.width] = outer_values
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
    def simulation_config(self) -> 'SimulationConfig':
        """Provide access to simulation configuration."""
        return self._simulation_config


class LinearElastoPlasticConfig:
    def __init__(self, config_file: str):
        self.mesh = None
        self.l_x = None
        self.l_y = None
        # todo: move to config file
        self.strain_rate = fe.Constant(0.000001)

        self._simulation_config, self._substrate_properties, self._layer_properties = self.load_simulation_config(config_file)
        summarize_and_print_config(self._simulation_config, [self._substrate_properties, self._layer_properties])
        self.setup_geometry()

    def load_simulation_config(self, file_name):
        """Loads and initializes a SimulationConfig object from a JSON configuration file."""
        simulation_config = SimulationConfig(file_name)
        if simulation_config.field_input_file:
            input_file = process_abaqus_input_file(simulation_config.field_input_file, plot=True)
            simulation_config.width = input_file[ExternalInput.WIDTH.value]
            simulation_config.length = input_file[ExternalInput.LENGTH.value]

            center_yz_points_outside, center_yz_points_inside = self.determine_center_plane(input_file)

            coordinates_on_center_plane = []
            for outside_points_t, inside_points_t in zip(center_yz_points_outside, center_yz_points_inside):
                combined_points_t = outside_points_t + inside_points_t
                coordinates_on_center_plane.append(combined_points_t)

            displacement_x = input_file[ExternalInput.DISPLACEMENTX.value]
            displacement_y = input_file[ExternalInput.DISPLACEMENTY.value]
            displacement_z = input_file[ExternalInput.DISPLACEMENTZ.value]

            plot_strain_displacement(input_file)

            x_coordinates = input_file[ExternalInput.X.value]
            y_coordinates = input_file[ExternalInput.Y.value]
            z_coordinates = input_file[ExternalInput.Z.value]

            displacement_x_center, displacement_y_center = interpolate_displacements(
                coordinates_on_center_plane, x_coordinates, y_coordinates, z_coordinates, displacement_x,
                displacement_y,
                displacement_z)

            # Call the plot function
            plot_displacement(displacement_x_center, displacement_y_center)

            # Call the plot function
            plot_movement(coordinates_on_center_plane, displacement_x_center, displacement_y_center)

            # # Assuming time_values are extracted from your data
            # time_values = result[ExternalInput.TIME.value]
            #
            # # Create an InPlaneInterpolator instance for both in-plane displacements
            # in_plane_interpolator = InPlaneInterpolator(time_values, coordinates_on_center_plane, displacement_x_center,
            #                                             displacement_y_center)
            #
            # # Now you can query this interpolator for displacements at any time and point in the plane
            # query_time = 25.0  # for example
            # query_point = (5.0, 2.0, 3.0)  # for example, assuming it's within the bounds of your data
            # disp_at_query_time_and_point_1, disp_at_query_time_and_point_2 = in_plane_interpolator.get_displacement_at_point(
            #     query_time, query_point)

        substrate_properties = MaterialProperties('material_properties.json', simulation_config.material)
        if simulation_config.use_two_material_layers:
            layer_properties = MaterialProperties('material_properties.json', simulation_config.layer_material)
        return simulation_config, substrate_properties, layer_properties

    def determine_center_plane(self, result):
        x_coordinates = result[ExternalInput.X.value]
        y_coordinates = result[ExternalInput.Y.value]
        z_coordinates = result[ExternalInput.Z.value]

        # Extract the outside and inside point indices
        outside_indices = result[ExternalInput.OUTSIDE_P.value]
        inside_indices = result[ExternalInput.INSIDE_P.value]

        def get_center_yz_points(indices, x_coordinates, y_coordinates, z_coordinates, tolerance=1e-3):
            center_yz_points_per_timestep = []
            for t in range(x_coordinates.shape[1]):
                x_values_t = x_coordinates[indices, t]
                y_values_t = y_coordinates[indices, t]
                z_values_t = z_coordinates[indices, t]
                center_x_t = np.median(x_values_t)
                center_yz_points_t = [(x, y, z) for x, y, z in zip(x_values_t, y_values_t, z_values_t) if
                                      abs(x - center_x_t) < tolerance]
                if len(center_yz_points_t) != 3:
                    print("missing points")
                    exit(-1)
                center_yz_points_per_timestep.append(center_yz_points_t)
            return center_yz_points_per_timestep

        center_yz_points_outside = get_center_yz_points(outside_indices, np.array(x_coordinates),
                                                        np.array(y_coordinates), np.array(z_coordinates))
        center_yz_points_inside = get_center_yz_points(inside_indices, np.array(x_coordinates), np.array(y_coordinates),
                                                       np.array(z_coordinates))

        return center_yz_points_outside, center_yz_points_inside

    def setup_geometry(self):
        """Sets up the geometry of the simulation, including layer thickness and mesh initialization."""
        l_layer_x = self._simulation_config.layer_1_thickness if self._simulation_config.use_two_material_layers else 0.0
        l_layer_y = 0.0
        self.l_x = self._simulation_config.width + l_layer_x
        # todo: hardcode
        # multiplier_y = 3.0
        # l_y = multiplier_y * simulation_config.length + l_layer_y
        self.l_y = self._simulation_config.length + l_layer_y
        self.mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(self.l_x, self.l_y),
                                     self._simulation_config.mesh_resolution_x,
                                     self._simulation_config.mesh_resolution_y)

    @property
    def substrate_properties(self) -> 'PropertiesType':
        return self._substrate_properties

    @property
    def layer_properties(self) -> 'PropertiesType':
        return self._layer_properties

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

