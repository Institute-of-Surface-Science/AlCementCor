import ufl
import fenics as fe
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import warnings
import argparse
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from matplotlib.lines import Line2D

from AlCementCor.bnd import *
from AlCementCor.config import *
from AlCementCor.info import *
from AlCementCor.input_file import *
from AlCementCor.material_properties import *
from AlCementCor.time_controller import PITimeController

fe.parameters["form_compiler"]["representation"] = 'quadrature'
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


def setup_boundary_conditions(V, two_layers, C_strain_rate, l_y):
    # Define boundary location conditions
    def is_bottom_boundary(x, on_boundary):
        return on_boundary and fe.near(x[1], 0.0)

    def is_top_boundary(x, on_boundary):
        return on_boundary and fe.near(x[1], l_y)

    # Define the boundary conditions
    bottom_condition = NoDisplacementBoundaryCondition(V, is_bottom_boundary)
    # if two_layers:
    #     bottom_condition = ConstantStrainRateBoundaryCondition(V, is_bottom_boundary, -C_strain_rate)
    # else:
    #     bottom_condition = NoDisplacementBoundaryCondition(V, is_bottom_boundary)

    # top_condition = ConstantStrainRateBoundaryCondition(V, is_top_boundary, C_strain_rate)
    bnd_length = 5.0
    displacement_func = DisplacementExpressionY(C_strain_rate, bnd_length)
    top_condition = FunctionDisplacementBoundaryCondition(V, is_top_boundary, displacement_func)

    # Create the conditions list
    conditions = [bottom_condition, top_condition]

    # Generate the Dirichlet boundary conditions
    bc = [condition.get_condition() for condition in conditions]

    # Generate homogenized boundary conditions
    bc_iter = [condition.get_homogenized_condition() for condition in conditions]

    return bc, bc_iter, conditions


def load_simulation_config(file_name):
    """Loads and initializes a SimulationConfig object from a JSON configuration file."""
    simulation_config = SimulationConfig(file_name)
    if simulation_config.field_input_file:
        result = process_input_tensors(simulation_config.field_input_file, plot=True)
        simulation_config.width = result[ExternalInput.WIDTH.value]
        simulation_config.length = result[ExternalInput.LENGTH.value]

    substrate_properties = MaterialProperties('material_properties.json', simulation_config.material)
    if simulation_config.use_two_material_layers:
        layer_properties = MaterialProperties('material_properties.json', simulation_config.layer_material)
    return simulation_config, substrate_properties, layer_properties


def setup_geometry(simulation_config):
    """Sets up the geometry of the simulation, including layer thickness and mesh initialization."""
    l_layer_x = simulation_config.layer_1_thickness if simulation_config.use_two_material_layers else 0.0
    l_layer_y = 0.0
    l_x = simulation_config.width + l_layer_x
    l_y = simulation_config.length + l_layer_y
    mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(l_x, l_y), simulation_config.mesh_resolution_x,
                            simulation_config.mesh_resolution_y)
    return mesh, l_x, l_y


def setup_numerical_stuff(simulation_config, mesh):
    """Sets up the numerical parameters for the simulation, including the function spaces."""
    deg_stress = simulation_config.finite_element_degree_stress
    V = fe.VectorFunctionSpace(mesh, "CG", simulation_config.finite_element_degree_u)
    u, du, Du = [fe.Function(V, name=n) for n in ["Total displacement", "Iteration correction", "Current increment"]]
    DG = fe.FunctionSpace(mesh, "DG", 0)
    We = fe.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
    W = fe.FunctionSpace(mesh, We)
    sig, sig_old, n_elas = [fe.Function(W) for _ in range(3)]
    func_names = ["Beta", "Cumulative plastic strain", "Hydrostatic stress", "local sig0",
                  "local mu", "local lmbda", "local hardening factor"]
    W0 = fe.FunctionSpace(mesh,
                          fe.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default'))
    beta, p, sig_hyd, sig_0_local, mu_local, lmbda_local, C_linear_h_local = [fe.Function(W0, name=n) for n in
                                                                              func_names]
    P0 = fe.FunctionSpace(mesh, "DG", 0)
    sig_hyd_avg, sig_0_test, lmbda_test = [fe.Function(P0, name=n) for n in
                                           ["Avg. Hydrostatic stress", "test", "test2"]]
    return V, u, du, Du, W, sig, sig_old, n_elas, W0, beta, p, sig_hyd, sig_0_local, mu_local, lmbda_local, C_linear_h_local, P0, sig_hyd_avg, sig_0_test, lmbda_test, DG, deg_stress


# def plot_vm(i, sig_eq_p):
#     plt.figure()
#     # plt.plot(results[:, 0], results[:, 1], "-o")
#     ax = fe.plot(sig_eq_p)
#     cbar = plt.colorbar(ax)
#     plt.xlabel("x")
#     plt.ylabel("y$")
#     # plt.show()
#     plt.savefig("vm" + str(i) + ".svg")
#     plt.close()

def plot(iteration, u, sig_eq_p, title="Von-Mises Stress", cbar_label="Von-Mises Stress", cmap="viridis", quiver_steps=5):
    """
    Function to plot and save the Von Mises stress distribution

    Parameters:
    i : integer
        Iteration index for saving the file
    sig_eq_p : fenics.Function
        The function you want to plot
    title : str, optional
        Title of the plot
    cbar_label : str, optional
        Label for the colorbar
    cmap : str, optional
        Colormap used for the plot

    Returns:
    None
    """

    fig = plt.figure(figsize=(10, 8))

    mesh = sig_eq_p.function_space().mesh()
    x = mesh.coordinates()[:, 0]
    y = mesh.coordinates()[:, 1]
    triangles = mesh.cells()

    scalars = sig_eq_p.compute_vertex_values(mesh)
    triangulation = tri.Triangulation(x, y, triangles)

    c = plt.tripcolor(triangulation, scalars, shading='flat', cmap=cmap)

    plt.title(title, fontsize=20)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # create quiver plot
    X = np.linspace(np.min(x), np.max(x), quiver_steps)
    Y = np.linspace(np.min(y), np.max(y), quiver_steps)
    U = np.zeros((quiver_steps, quiver_steps))
    V = np.zeros((quiver_steps, quiver_steps))

    for i in range(quiver_steps):
        for j in range(quiver_steps):
            U[j, i], V[j, i] = u(X[i], Y[j])

    # Reduce the arrow head size
    #Q = plt.quiver(X, Y, U, V, color='r', headwidth=4, headlength=4, headaxislength=4, scale_units='width', scale=10)
    Q = plt.quiver(X, Y, U, V, color='r', pivot='mid')

    # Annotate the quivers
    # for i in range(quiver_steps):
    #     for j in range(quiver_steps):
    #         if (i + j) % 2 == 0:  # Skip some quivers for clarity
    #             label = f'({U[j, i]:.2f}, {V[j, i]:.2f})'
    #             plt.annotate(label, (X[i], Y[j]), textcoords="offset points", xytext=(-10, -10), ha='center',
    #                          fontsize=8)

    # Add a single quiver for the legend
    # qk = plt.quiverkey(Q, 0.9, 0.1, 1, r'$1 \, m$', labelpos='E', coordinates='figure')


    # Extend the x and y limits
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    plt.xlim(np.min(x) - 0.1 * x_range, np.max(x) + 0.1 * x_range)
    plt.ylim(np.min(y) - 0.1 * y_range, np.max(y) + 0.1 * y_range)

    # Create custom legend
    custom_lines = [Line2D([0], [0], color='r', lw=2)]
    plt.legend(custom_lines, ['deformation'])

    cbar = plt.colorbar(c)
    cbar.set_label(cbar_label, size=16)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    plt.savefig("vm" + str(iteration) + ".png", dpi=300)
    plt.close()


# Util
########################################################################
def as_3D_tensor(X):
    return fe.as_tensor([[X[0], X[3], 0],
                         [X[3], X[1], 0],
                         [0, 0, X[2]]])


def local_project(v, V, dxm, u=None):
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
        return


# Math!
########################################################################

def eps(displacement):
    """
    Calculate the strain tensor (epsilon)
    :param displacement: The displacement tensor
    :return: strain tensor
    """
    e = fe.sym(fe.nabla_grad(displacement))
    return fe.as_tensor([[e[0, 0], e[0, 1], 0],
                         [e[0, 1], e[1, 1], 0],
                         [0, 0, 0]])


def sigma(strain, lmbda_local_DG, mu_local_DG):
    """
    Calculate the Cauchy Stress Tensor
    :param strain: Strain (epsilon)
    :return: Cauchy Stress Tensor
    """
    return lmbda_local_DG * fe.tr(strain) * fe.Identity(3) + 2 * mu_local_DG * strain


def sigma_tang(e, n_elas, mu_local_DG, C_linear_h_local_DG, beta, lmbda_local_DG):
    N_elas = as_3D_tensor(n_elas)
    return sigma(e, lmbda_local_DG, mu_local_DG) - 3 * mu_local_DG * (
            3 * mu_local_DG / (3 * mu_local_DG + C_linear_h_local_DG) - beta) * fe.inner(
        N_elas, e) * N_elas - 2 * mu_local_DG * beta * fe.dev(e)


# Von-Mises Stress
def sigma_v(strain, lmbda_local_DG, mu_local_DG):
    s = fe.dev(sigma(strain, lmbda_local_DG, mu_local_DG))
    return fe.sqrt(3 / 2. * fe.inner(s, s))


# Macaulays Bracket for <f_elastic>+
# only positive for x > 0
ppos = lambda x: (x + abs(x)) / 2.


# https://www.dynasupport.com/tutorial/computational-plasticity/radial-return
# https://www.dynasupport.com/tutorial/computational-plasticity/generalizing-the-yield-function
# https://www.dynasupport.com/tutorial/computational-plasticity/the-consistent-tangent-matrix
def proj_sig(deps, old_sig, old_p, sig_0_local, C_linear_h_local, mu_local, lmbda_local_DG, mu_local_DG):
    # update stress from change in strain (deps)
    sig_n = as_3D_tensor(old_sig)
    sig_elas = sig_n + sigma(deps, lmbda_local_DG, mu_local_DG)

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


def check_convergence(nRes, nRes0, tol, niter, Nitermax):
    return niter == 0 or (nRes0 > 0 and nRes / nRes0 > tol and niter < Nitermax)


def run_newton_raphson(system_matrix, residual, newton_form, residual_form, boundary_conditions,
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
        strain_change = eps(total_change)

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


def update_and_store_results(i, Du, dp_, sig, sig_old, sig_hyd, sig_hyd_avg, p, W0, dxm, P0, u, l_x, l_y, time,
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

    # Set up numerical parameters
    strain_rate = fe.Constant(0.000001)  # 0.01/s

    # Set up numerical variables and functions
    (V, u, du, Du, W, sig, sig_old, n_elas, W0, beta, p, sig_hyd, local_initial_stress,
     local_shear_modulus, lmbda_local, local_linear_hardening, P0, sig_hyd_avg,
     sig_0_test, lmbda_test, DG, deg_stress) = setup_numerical_stuff(config, mesh)

    # Set up boundary conditions
    bc, bc_iter, conditions = setup_boundary_conditions(V, config.use_two_material_layers, strain_rate, l_y)

    metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
    dxm = ufl.dx(metadata=metadata)

    v = fe.TrialFunction(V)
    u_ = fe.TestFunction(V)

    # calculate local mu
    mu_local_DG = fe.Function(DG)
    assign_local_values(substrate_props.shear_modulus, layer_props.shear_modulus, mu_local_DG, DG, config)

    lmbda_local_DG = fe.Function(DG)
    assign_local_values(substrate_props.first_lame_parameter, layer_props.first_lame_parameter, lmbda_local_DG, DG,
                        config)

    local_linear_hardening_DG = fe.Function(DG)
    assign_local_values(substrate_props.linear_isotropic_hardening, layer_props.linear_isotropic_hardening,
                        local_linear_hardening_DG, DG, config)

    newton_lhs = fe.inner(eps(v), sigma_tang(eps(u_), n_elas, mu_local_DG, local_linear_hardening_DG, beta,
                                             lmbda_local_DG)) * dxm
    newton_rhs = -fe.inner(eps(u_), as_3D_tensor(sig)) * dxm

    results_file = fe.XDMFFile("plasticity_results.xdmf")
    results_file.parameters["flush_output"] = True
    results_file.parameters["functions_share_mesh"] = True
    P0 = fe.FunctionSpace(mesh, "DG", 0)

    max_iters, tolerance = 10, 1e-8  # parameters of the Newton-Raphson procedure
    time_step = config.integration_time_limit / (config.total_timesteps)

    # Assign layer values
    local_initial_stress = assign_layer_values(substrate_props.yield_strength, layer_props.yield_strength, W0, config)
    local_shear_modulus = assign_layer_values(substrate_props.shear_modulus, layer_props.shear_modulus, W0, config)
    local_linear_hardening = assign_layer_values(substrate_props.linear_isotropic_hardening,
                                                 layer_props.linear_isotropic_hardening, W0, config)

    # Initialize result and time step lists
    displacement_over_time = [(0, 0)]
    max_stress_over_time = [0]
    mean_stress_over_time = [0]
    displacement_list = [0]

    time = 0
    iteration = 0

    time_controller = PITimeController(time_step, 1e-2 * tolerance)

    while time < config.integration_time_limit:
        time += time_step
        iteration += 1
        for condition in conditions:
            condition.update_time(time_step)
        A, Res = fe.assemble_system(newton_lhs, newton_rhs, bc)
        print(f"Step: {iteration + 1}, time: {time} s")
        print(f"displacement: {strain_rate.values()[0] * time} mm")

        newton_res_norm, plastic_strain_update = run_newton_raphson(
            A, Res, newton_lhs, newton_rhs, bc_iter, du, Du, sig_old, p, local_initial_stress, local_linear_hardening,
            local_shear_modulus, lmbda_local_DG, mu_local_DG, sig, n_elas, beta, sig_hyd, W, W0, dxm, max_iters,
            tolerance)

        if newton_res_norm > 1 or np.isnan(newton_res_norm):
            raise ValueError("ERROR: Calculation diverged!")

        update_and_store_results(
            iteration, Du, plastic_strain_update, sig, sig_old, sig_hyd, sig_hyd_avg, p, W0, dxm, P0, u, l_x, l_y, time,
            results_file, max_stress_over_time, mean_stress_over_time, displacement_list
        )

        displacement_over_time += [(np.abs(u(l_x / 2, l_y)[1]) / l_y, time)]

        time_step = time_controller.update(newton_res_norm)


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
