import argparse
import warnings

from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

from AlCementCor.info import *
from AlCementCor.material_model import *
from AlCementCor.postproc import *

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

    von_mises_stress = local_project(compute_von_mises_stress(model.stress), model.P0, model.dxm)

    if timestep_count % 10 == 0:
        plot(timestep_count, model.total_displacement, von_mises_stress)

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
    displacement_at_center_top = model.total_displacement(model.l_x / 2.0, model.l_y)
    # print(displacement_at_center_top)
    # disp = np.abs(displacement_at_center_top[1]) / model.l_y
    print(f"Step: {timestep_count}, time: {time} s")
    print(f"displacement: {displacement_at_center_top[1]} mm")


def write_output(results_file, model, time):
    results_file.write(model.total_displacement, time)
    p_avg = fe.Function(model.P0, name="Plastic strain")
    p_avg.assign(fe.project(model.cum_plstic_strain, model.P0))
    results_file.write(p_avg, time)


def create_output_file(filename):
    results_file = fe.XDMFFile(filename)
    results_file.parameters["flush_output"] = True
    results_file.parameters["functions_share_mesh"] = True
    return results_file


def main() -> None:
    args = parse_command_line_args()

    model = LinearElastoPlasticModel(args.config)
    simulation_config = model.simulation_config
    integrator = LinearElastoPlasticIntegrator(model)

    summarize_and_print_config(simulation_config, materials=[model.substrate_properties, model.layer_properties])

    results_file = create_output_file(simulation_config.output_file)

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
