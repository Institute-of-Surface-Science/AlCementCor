import ufl
import fenics as fe
import warnings
import argparse
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

from AlCementCor.info import *
from AlCementCor.material_model import *

fe.parameters["form_compiler"]["representation"] = 'quadrature'
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


def parse_command_line_args():
    """
    Command-line interface setup function.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=logo(),
                                     formatter_class=PreserveWhiteSpaceArgParseFormatter)

    parser.add_argument('-c', '--config', type=str, default='simulation_config.json',
                        help='Path to the simulation configuration JSON file.')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0',
                        help="Show program's version number and exit.")

    parser.add_argument('-m', '--max-steps', type=int, default=10000,
                        help='Maximum number of time steps.')

    return parser.parse_args()


def postprocess(model: 'LinearElastoPlasticModel', timestep_count: int) -> None:
    """
    Post-process and visualize the results of the simulation.

    Parameters:
    - model: Model containing relevant information about the simulation.
    - timestep_count: Current timestep number.
    """
    # max_stress_over_time = [0]
    # mean_stress_over_time = [0]
    # displacement_over_time = [(0, 0)]

    von_mises_stress = local_project(compute_von_mises_stress(model.sig), model.P0, model.dxm)

    if timestep_count % 10 == 0:
        plot(timestep_count, model.u, von_mises_stress)

    # displacement_over_time.append((displacement_at_center_top[1], integrator.time))
    # max_stress_over_time.extend([np.abs(np.amax(sig_eq_p.vector()[:]))])
    # mean_stress_over_time.extend([np.abs(np.mean(sig_eq_p.vector()[:]))])


def info_out(model: 'LinearElastoPlasticModel', timestep_count: int, time: float) -> None:
    """
    Display information about the current state of the simulation.

    Parameters:
    - integrator: Integrator being used in the simulation.
    - model: Model containing relevant information about the simulation.
    - timestep_count: Current timestep number.
    """
    displacement_at_center_top = model.u(model.l_x / 2.0, model.l_y)
    # print(displacement_at_center_top)
    # disp = np.abs(displacement_at_center_top[1]) / model.l_y
    print(f"Step: {timestep_count}, time: {time} s")
    print(f"displacement: {displacement_at_center_top[1]} mm")


def write_output(results_file, model, time):
    results_file.write(model.u, time)
    p_avg = fe.Function(model.P0, name="Plastic strain")
    p_avg.assign(fe.project(model.p, model.P0))
    results_file.write(p_avg, time)


def create_output_file(filename):
    results_file = fe.XDMFFile(filename)
    results_file.parameters["flush_output"] = True
    results_file.parameters["functions_share_mesh"] = True
    return results_file


def main() -> None:
    args = parse_command_line_args()

    simulation_config = LinearElastoPlasticConfig(args.config)
    model = LinearElastoPlasticModel(simulation_config)
    integrator = LinearElastoPlasticIntegrator(model)

    summarize_and_print_config(simulation_config.simulation_config,
                               materials=[model.substrate_properties, model.layer_properties])

    # todo: make settable
    results_file = create_output_file("plasticity_results.xdmf")

    time = 0
    timestep_count = 1
    max_time = simulation_config.integration_time_limit

    # Main time integration loop
    while time < max_time and args.max_steps > timestep_count:
        integrator.single_time_step_integration()
        time = integrator.time

        info_out(model, timestep_count, time)

        postprocess(model, timestep_count)

        write_output(results_file, model, time)

        timestep_count += 1


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
