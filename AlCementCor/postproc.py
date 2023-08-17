from AlCementCor.input_file import ExternalInput
import matplotlib.pyplot as plt


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
