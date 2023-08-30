import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.lines import Line2D

from AlCementCor.input_file import ExternalInput


def plot_strain_displacement(result):
    displacement_x = result[ExternalInput.DISPLACEMENTX.value]
    displacement_y = result[ExternalInput.DISPLACEMENTY.value]
    displacement_z = result[ExternalInput.DISPLACEMENTZ.value]

    strain_x = result[ExternalInput.STRAINX.value]
    strain_y = result[ExternalInput.STRAINY.value]
    strain_z = result[ExternalInput.STRAINZ.value]

    # Number of nodes
    no_nodes = displacement_x.shape[0]

    # Create subplots to compare displacement and strain for each coordinate (x, y, z)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    coordinates = ['X', 'Y', 'Z']
    displacements = [displacement_x, displacement_y, displacement_z]
    strains = [strain_x, strain_y, strain_z]

    for i, (coord, disp, strain) in enumerate(zip(coordinates, displacements, strains)):
        for node in range(no_nodes):
            # Plot displacements
            axes[i, 0].plot(disp[node], label=f'Node {node}')
            axes[i, 0].set_title(f'Displacement in {coord}-direction')
            axes[i, 0].set_xlabel('Time steps')
            axes[i, 0].set_ylabel('Displacement')
            axes[i, 0].legend()

            # Plot strains
            axes[i, 1].plot(strain[node], label=f'Node {node}')
            axes[i, 1].set_title(f'Strain in {coord}-direction')
            axes[i, 1].set_xlabel('Time steps')
            axes[i, 1].set_ylabel('Strain')
            axes[i, 1].legend()

    plt.tight_layout()
    plt.savefig("disp_strain.png")


def plot_movement(coordinates_on_center_plane, displacement_x_center, displacement_y_center):
    # Assuming the structure of coordinates_on_center_plane is [timestep][node_id][coordinate]
    # Extracting original x and y coordinates from the first timestep
    original_x_coordinates = [point[1] for point in coordinates_on_center_plane[0]]
    original_y_coordinates = [point[2] for point in coordinates_on_center_plane[0]]

    # Compute the average displacements over all time steps for each node
    avg_displacement_x = np.mean(displacement_x_center, axis=1)
    avg_displacement_y = np.mean(displacement_y_center, axis=1)

    plt.figure(figsize=(10, 10))

    # Plot original points
    plt.scatter(original_x_coordinates, original_y_coordinates, color='green', label='Original Position')

    # Quiver plot for average displacements
    plt.quiver(original_x_coordinates, original_y_coordinates,
               avg_displacement_x, avg_displacement_y,
               angles='xy', scale_units='xy', scale=1, color='blue')

    plt.title("Visualization of Average Movement for Center Plane Nodes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # This ensures that the lengths of the arrows are proportional
    plt.savefig("movement_center.png")


def plot_displacement(displacement_x_center, displacement_y_center):
    timesteps = range(len(displacement_x_center[0]))  # assuming all nodes have the same number of timesteps

    num_nodes = len(displacement_x_center)

    plt.figure(figsize=(15, num_nodes * 5))  # adjusting the figure size based on number of nodes

    for i in range(num_nodes):
        # Plotting displacement_x_center for node i
        plt.subplot(num_nodes, 2, 2 * i + 1)
        plt.plot(timesteps, displacement_x_center[i], marker='o', linestyle='-', color='blue')
        plt.title(f"Displacement X Center for Node {i} vs Time")
        plt.xlabel("Time")
        plt.ylabel(f"Displacement X for Node {i}")

        # Plotting displacement_y_center for node i
        plt.subplot(num_nodes, 2, 2 * i + 2)
        plt.plot(timesteps, displacement_y_center[i], marker='o', linestyle='-', color='red')
        plt.title(f"Displacement Y Center for Node {i} vs Time")
        plt.xlabel("Time")
        plt.ylabel(f"Displacement Y for Node {i}")

    plt.tight_layout()
    plt.savefig("disp_center.png")


def plot(iteration, u, sig_eq_p, title="Von-Mises Stress and Deformation", cbar_label="Von-Mises Stress",
         cmap="viridis", quiver_steps=5):
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
    # Q = plt.quiver(X, Y, U, V, color='r', headwidth=4, headlength=4, headaxislength=4, scale_units='width', scale=10)
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
