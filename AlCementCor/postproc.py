from AlCementCor.input_file import ExternalInput
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.lines import Line2D
import numpy as np

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
