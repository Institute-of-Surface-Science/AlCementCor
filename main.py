import ufl
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import warnings
import argparse
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from scipy.interpolate import RegularGridInterpolator

from AlCementCor.bnd import *
from AlCementCor.info import *
from AlCementCor.input_file import *
from AlCementCor.material_model import *
from AlCementCor.material_properties import *
from AlCementCor.interpolate import *
from AlCementCor.postproc import plot_strain_displacement, plot_movement, plot_displacement

fe.parameters["form_compiler"]["representation"] = 'quadrature'
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


def setup_displacement_bnd(model, two_layers, C_strain_rate, l_x, l_y):
    # Define boundary location conditions
    def is_bottom_boundary(x, on_boundary):
        return on_boundary and fe.near(x[1], 0.0)

    def is_top_boundary(x, on_boundary):
        return on_boundary and fe.near(x[1], l_y)

    def is_left_boundary(x, on_boundary):
        return on_boundary and fe.near(x[0], 0.0)

    def is_right_boundary(x, on_boundary):
        return on_boundary and fe.near(x[0], l_x)

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
    displacement_func = SquareStrainRate((-C_strain_rate, 0.0), 0.0, l_y)
    left_condition = FunctionDisplacementBoundaryCondition(model.V, is_left_boundary, displacement_func)

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

def determine_center_plane(result):
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


def load_simulation_config(file_name):
    """Loads and initializes a SimulationConfig object from a JSON configuration file."""
    simulation_config = SimulationConfig(file_name)
    if simulation_config.field_input_file:
        input_file = process_abaqus_input_file(simulation_config.field_input_file, plot=True)
        simulation_config.width = input_file[ExternalInput.WIDTH.value]
        simulation_config.length = input_file[ExternalInput.LENGTH.value]

        center_yz_points_outside, center_yz_points_inside = determine_center_plane(input_file)

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
            coordinates_on_center_plane, x_coordinates, y_coordinates, z_coordinates, displacement_x, displacement_y,
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


def setup_geometry(simulation_config):
    """Sets up the geometry of the simulation, including layer thickness and mesh initialization."""
    l_layer_x = simulation_config.layer_1_thickness if simulation_config.use_two_material_layers else 0.0
    l_layer_y = 0.0
    l_x = simulation_config.width + l_layer_x
    # todo: hardcode
    # multiplier_y = 3.0
    # l_y = multiplier_y * simulation_config.length + l_layer_y
    l_y = simulation_config.length + l_layer_y
    mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(l_x, l_y), simulation_config.mesh_resolution_x,
                            simulation_config.mesh_resolution_y)
    return mesh, l_x, l_y

def assign_local_values(values, outer_values, local_DG, DG, simulation_config):
    dofmap = DG.tabulate_dof_coordinates()[:]
    vec = np.zeros(dofmap.shape[0])
    vec[:] = values
    vec[dofmap[:, 0] > simulation_config.width] = outer_values
    local_DG.vector()[:] = vec


# assign local values to the layers
def assign_layer_values(inner_value, outer_value, W0, simulation_config):
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

    layer = set_layer(inner_value, outer_value, simulation_config)
    return fe.interpolate(layer, W0)


# def update_and_store_results(i, Du, dp_, sig, sig_old, sig_hyd, sig_hyd_avg, p, W0, dxm, P0, u, l_x, l_y, time,
#                              file_results, stress_max_t, stress_mean_t, disp_t):
#     # update displacement
#     u.assign(u + Du)
#     # update plastic strain
#     p.assign(p + local_project(dp_, W0, dxm))
#
#     # update stress fields
#     sig_old.assign(sig)
#     sig_hyd_avg.assign(fe.project(sig_hyd, P0))
#
#     # # s11, s12, s21, s22 = sig.split(deepcopy=True)
#     # # avg_stress_y = np.average(s22.vector()[:])
#     # # avg_stress = np.average(sig.vector()[:])
#     sig_n = as_3D_tensor(sig)
#     s = fe.dev(sig_n)
#
#     # calculate the von-mises equivalent stress
#     sig_eq = fe.sqrt(3 / 2. * fe.inner(s, s))
#     sig_eq_p = local_project(sig_eq, P0, dxm)
#
#     if i % 10 == 0:
#         plot(i, u, sig_eq_p)
#
#     # calculate and project the von-mises stress for later use
#     stress_max_t.extend([np.abs(np.amax(sig_eq_p.vector()[:]))])
#     stress_mean_t.extend([np.abs(np.mean(sig_eq_p.vector()[:]))])
#
#     # append the y-displacement at the center of the bar
#     disp_t.append(u(l_x / 2, l_y)[1])
#
#     file_results.write(u, time)
#     p_avg = fe.Function(P0, name="Plastic strain")
#     p_avg.assign(fe.project(p, P0))
#     file_results.write(p_avg, time)


def cli_interface():
    parser = argparse.ArgumentParser(description=logo(), formatter_class=PreserveWhiteSpaceArgParseFormatter)

    # Add an argument for configuration file, default to 'simulation_config.json'
    parser.add_argument('-c', '--config', type=str, default='simulation_config.json',
                        help='Path to the simulation configuration JSON file.')

    # Add an argument for version information
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0',
                        help='Show program\'s version number and exit.')

    args = parser.parse_args()
    return args


def main() -> None:
    """Main function to run the simulation."""

    args = cli_interface()

    # Load configuration and material properties
    config, substrate_props, layer_props = load_simulation_config(args.config)
    summarize_and_print_config(config, [substrate_props, layer_props])

    # Geometry setup
    mesh, l_x, l_y = setup_geometry(config)

    model = LinearElastoPlasticModel(config, mesh, substrate_props, layer_props)

    # Set up boundary conditions
    bc, bc_iter, conditions = setup_displacement_bnd(model, config.use_two_material_layers, model.strain_rate, l_x, l_y)

    results_file = fe.XDMFFile("plasticity_results.xdmf")
    results_file.parameters["flush_output"] = True
    results_file.parameters["functions_share_mesh"] = True

    max_iters, tolerance = 10, 1e-8  # parameters of the Newton-Raphson procedure
    time_step = config.integration_time_limit / (config.total_timesteps)

    # Assign layer values
    W0 = model.W0
    local_initial_stress = assign_layer_values(substrate_props.yield_strength, layer_props.yield_strength, W0, config)
    local_shear_modulus = assign_layer_values(substrate_props.shear_modulus, layer_props.shear_modulus, W0, config)
    local_linear_hardening = assign_layer_values(substrate_props.linear_isotropic_hardening,
                                                 layer_props.linear_isotropic_hardening, W0, config)

    model.run_time_integration(time_step, conditions, bc, bc_iter, local_initial_stress,
                             local_linear_hardening, local_shear_modulus, max_iters, tolerance, results_file, l_x, l_y)

    # Initialize result and time step lists
    # displacement_over_time = [(0, 0)]
    # max_stress_over_time = [0]
    # mean_stress_over_time = [0]
    # displacement_list = [0]
    #
    # time = 0
    # iteration = 0
    #
    # time_controller = PITimeController(time_step, 1e-2 * tolerance)
    #
    # while time < config.integration_time_limit:
    #     time += time_step
    #     iteration += 1
    #     for condition in conditions:
    #         condition.update_time(time_step)
    #     A, Res = fe.assemble_system(model.newton_lhs, model.newton_rhs, bc)
    #     print(f"Step: {iteration + 1}, time: {time} s")
    #     print(f"displacement: {model.strain_rate.values()[0] * time} mm")
    #
    #     newton_res_norm, plastic_strain_update = run_newton_raphson(
    #         A, Res, model.newton_lhs, model.newton_rhs, bc_iter, model.du, model.Du, model.sig_old, model.p, local_initial_stress, local_linear_hardening,
    #         local_shear_modulus, model.lmbda_local_DG, model.mu_local_DG, model.sig, model.n_elas, model.beta, model.sig_hyd, model.W, model.W0, model.dxm, max_iters,
    #         tolerance)
    #
    #     if newton_res_norm > 1 or np.isnan(newton_res_norm):
    #         raise ValueError("ERROR: Calculation diverged!")
    #
    #     update_and_store_results(
    #         iteration, model.Du, plastic_strain_update, model.sig, model.sig_old, model.sig_hyd, model.sig_hyd_avg, model.p, model.W0, model.dxm, model.P0, model.u, l_x, l_y, time, results_file, max_stress_over_time, mean_stress_over_time, displacement_list
    #     )
    #
    #     displacement_over_time += [(np.abs(model.u(l_x / 2, l_y)[1]) / l_y, time)]
    #
    #     time_step = time_controller.update(newton_res_norm)


if __name__ == "__main__":
    main()

# ax = fe.plot(u, mode="displacement")
# cbar = plt.colorbar(ax)
# plt.show()

# plt.plot(results[:, 0], results[:, 1], "-o")
# plt.xlabel("Displacement of inner boundary")
# plt.ylabel(r"Applied pressure $q/q_{lim}$")
# # plt.show()
# plt.savefig("test_disp_pressure.svg")

# ax = fe.plot(sig_hyd_avg)
# cbar = plt.colorbar(ax)
# plt.show()

# # data extracted from fig. 6 paper1
# ref_strain = [4.5917e-7, 0.0022967, 0.0068471, 0.0077457, 0.008868360277136257,
#               0.011511, 0.024988, 0.040012, 0.059011, 0.071602, 0.083751,
#               0.10429, 0.12108, 0.13963, 0.16105, 0.17960]
# ref_stress = [0.10395, 99.376, 250.10, 267.26, 275.09019989024057,
#               279.21, 284.41, 290.64, 296.36, 299.48, 301.04, 302.60, 303.12, 302.60,
#               301.04, 296.88]
#
# # data extracted from fig. 10 paper 3
# ref_strain2 = [0, 0.001072706, 0.001966627, 0.002324195, 0.003218117, 0.004469607, 0.007508939,
#                0.009833135, 0.015554231, 0.027711561, 0.038796186, 0.056495828, 0.070262217,
#                0.085637664, 0.10852205, 0.131585221, 0.155542312, 0.175029797, 0.178605483]
#
# ref_stress2 = [0, 117.4641148, 172.0095694, 222.9665072, 236.2440191, 246.291866, 253.8277512,
#                260.2870813, 264.9521531, 277.15311, 285.0478469, 294.7368421, 299.0430622,
#                302.9904306, 307.2966507, 305.861244, 306.5789474, 300.8373206, 288.6363636]
#
# # for linear deformation E = stress/strain -> strain = stress/E
# linear_strain = np.multiply(np.array(stress_max_t), 1.0 / C_E.values()[0])
# linear_deformation = linear_strain * l_y
#
# print("max vm stress", np.max(stress_max_t))
# print("mean vm stress", np.max(stress_mean_t))
#
# # Plot the stress-deformation curve
# plt.plot(disp_t, stress_max_t, "-o", label="sim", markevery=5)
# plt.plot(linear_deformation, stress_max_t, label="linear")
# plt.xlabel("displacement [mm]")
# plt.ylabel(r"max. von-Mises stress [MPa]")
# plt.legend()
# plt.savefig("test_deformation_stress.svg")
#
# # Plot the stress-strain curve
# plt.figure()
# # plt.plot(np.array(disp_t) / l_y, stress_max_t, "-o", label="sim-max", markevery=5)
# plt.plot(np.array(disp_t) / l_y, stress_mean_t, "-o", label="sim", markevery=5)
# plt.plot(linear_strain, stress_max_t, label="linear")
# plt.plot(ref_strain, ref_stress, label="exp")
# plt.plot(ref_strain2, ref_stress2, label="exp2")
# # plt.plot(linear_strain, stress_mean_t, label="linear")
# plt.xlabel("strain [-]")
# plt.ylabel(r"max. von-Mises stress [MPa]")
# plt.legend()
# plt.savefig("test_strain_stress.svg")

# It can also be checked that the analytical limit load is also well reproduced
# when considering a zero hardening modulus.
#
# -----------
# References
# -----------
#
# .. [BON2014] Marc Bonnet, Attilio Frangi, Christian Rey.
#  *The finite element method in solid mechanics.* McGraw Hill Education, pp.365, 2014
# paper 1:Flow and fracture behavior of aluminum alloy 6082-T6 at different tensile strain rates and triaxialities
# paper 2:Fatigue Study of the Pre-Corroded 6082-T6 Aluminum Alloy in Saline Atmosphere
# Alejandro Fernández Muñoz 1 , José Luis Mier Buenhombre 2, *, Ana Isabel García-Diez 2 ,
# Carolina Camba Fabal 2 and Juan José Galán Díaz 2
# paper3: Analysis of manufacturing parameters on the shear strength of aluminium adhesive single-lap joints
# A.M. Pereira a,∗, J.M. Ferreira b, F.V. Antunes b, P.J. Bártolo
