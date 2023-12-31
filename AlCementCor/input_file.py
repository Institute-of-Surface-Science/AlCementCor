import json
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


class InputFileKeys(Enum):
    """
    An enumeration that provides string constants for accessing fields in a JSON file
    containing node information for a simulation.

    Attributes:
    -----------
    NODES : str
        The key to access node data.
    X : str
        The key to access X coordinates of nodes.
    Y : str
        The key to access Y coordinates of nodes.
    Z : str
        The key to access Z coordinates of nodes.
    VARIABLES_TO_LOAD : list of str
        The list of keys to access various other variables.
    """
    NODES = "nodes"
    X = "X"
    Y = "Y"
    Z = "Z"
    TIME = "time"
    LOGARITHMIC_STRAIN_11 = "LE11"
    LOGARITHMIC_STRAIN_12 = "LE12"
    LOGARITHMIC_STRAIN_22 = "LE22"
    LOGARITHMIC_STRAIN_23 = "LE23"
    LOGARITHMIC_STRAIN_31 = "LE31"
    LOGARITHMIC_STRAIN_33 = "LE33"
    LOGARITHMIC_STRAIN_TENSOR = ["LE11", "LE12", "LE22", "LE23", "LE31", "LE33"]
    STRESS_TENSOR = ["S11", "S12", "S22", "S23", "S31", "S33"]
    VARIABLES_TO_LOAD = ["LE11", "LE12", "LE22", "LE23", "LE31", "LE33",
                         "S11", "S12", "S22", "S23", "S31", "S33",
                         "T1", "T2", "T3"]


class ExternalInput(Enum):
    """
    An enumeration that provides string constants for storing computed variables in the `loaded_vars` dictionary.

    Attributes:
    -----------
    WIDTH : str
        The key to store the calculated thickness.
    LENGTH : str
        The key to store the calculated length.
    DISPLACEMENT : str
        The key to store the calculated displacement.
    X : str
        The key to store the X coordinates.
    Y : str
        The key to store the Y coordinates.
    Z : str
        The key to store the Z coordinates.
    """
    WIDTH = "width"
    LENGTH = "length"
    DISPLACEMENT = "displacement"
    DISPLACEMENTX = "displacement_x"
    DISPLACEMENTY = "displacement_y"
    DISPLACEMENTZ = "displacement_z"
    RELDISPLACEMENT = "rel_displacement"
    X = "X"
    Y = "Y"
    Z = "Z"
    OUTSIDE_P = "outside_points"
    INSIDE_P = "inside_points"
    TIME = "time"
    STRAINX = "LE11"
    STRAINY = "LE22"
    STRAINZ = "LE33"


def process_abaqus_input_file(filename, plot=False):
    """
    Load a JSON file containing node information, calculate the thickness and length of the area,
    and optionally plot the nodes in 3D.

    Parameters:
    filename (str): Path to the JSON file to be processed.
    plot (bool): Whether to plot the node data in 3D. Defaults to False.

    Returns:
    dict: A dictionary containing the calculated thickness, length, and displacements,
          along with the extracted LE, S and T values.
    """
    with open(filename) as f:
        data = json.load(f)

    # Extract node data
    node_data = data[InputFileKeys.NODES.value]

    # Get X, Y, and Z coordinates of all nodes for all timesteps
    x_coordinates = [node_data[node_id][InputFileKeys.X.value] for node_id in node_data.keys()]
    y_coordinates = [node_data[node_id][InputFileKeys.Y.value] for node_id in node_data.keys()]
    z_coordinates = [node_data[node_id][InputFileKeys.Z.value] for node_id in node_data.keys()]

    # Determine the number of points and timesteps
    num_points = len(x_coordinates)
    num_timesteps = len(x_coordinates[0])

    # Initialize displacement arrays with zeros
    displacement_x = np.zeros((num_points, num_timesteps))
    displacement_y = np.zeros_like(displacement_x)
    displacement_z = np.zeros_like(displacement_x)

    # Calculate displacement for subsequent timesteps
    for t in range(1, num_timesteps):
        # Calculate translation for this time step
        translation_x = np.mean([xc[t] - xc[t - 1] for xc in x_coordinates])
        translation_y = np.mean([yc[t] - yc[t - 1] for yc in y_coordinates])
        translation_z = np.mean([zc[t] - zc[t - 1] for zc in z_coordinates])

        for i, (x, y, z) in enumerate(zip(x_coordinates, y_coordinates, z_coordinates)):
            # Subtract translation from coordinates to get displacement
            dx = (x[t] - x[t - 1]) - translation_x
            dy = (y[t] - y[t - 1]) - translation_y
            dz = (z[t] - z[t - 1]) - translation_z

            displacement_x[i, t] = dx
            displacement_y[i, t] = dy
            displacement_z[i, t] = dz

    print(np.shape(displacement_y))

    # Extract Logarithmic Strain Tensors
    log_strain_tensors = []
    for node_id in node_data.keys():
        node_strain = []
        for le_key in InputFileKeys.LOGARITHMIC_STRAIN_TENSOR.value:
            node_strain.append(node_data[node_id][le_key])
        log_strain_tensors.append(node_strain)
    log_strain_tensors = np.array(log_strain_tensors)

    # # Extract stress Tensors
    # stress_tensors = []
    # for node_id in node_data.keys():
    #     node_stress = []
    #     for skey in InputFileKeys.STRESS_TENSOR.value:
    #         node_stress.append(node_data[node_id][skey])
    #     stress_tensors.append(node_stress)
    # stress_tensors = np.array(stress_tensors)

    # # Calculating relative coordinates
    # min_index = min(range(len(x_coordinates)),
    #                 key=lambda i: (abs(x_coordinates[i][0]), abs(y_coordinates[i][0]), abs(z_coordinates[i][0])))
    #
    # relative_x_coordinates = [[x - x_coordinates[min_index][i] for i, x in enumerate(coords)] for coords in
    #                           x_coordinates]
    # relative_y_coordinates = [[y - y_coordinates[min_index][i] for i, y in enumerate(coords)] for coords in
    #                           y_coordinates]
    # relative_z_coordinates = [[z - z_coordinates[min_index][i] for i, z in enumerate(coords)] for coords in
    #                           z_coordinates]
    #
    # rel_displacement = []
    # for x, y, z in zip(relative_x_coordinates, relative_y_coordinates, relative_z_coordinates):
    #     rel_displacement.append([np.sqrt((x[i+1] - x[i]) ** 2 + (y[i+1] - y[i]) ** 2 + (z[i+1] - z[i]) ** 2)
    #                          for i in range(len(x) - 1)])

    # Get coordinates for the first timestep for further calculations
    x_coordinates_0 = [x[0] for x in x_coordinates]
    y_coordinates_0 = [y[0] for y in y_coordinates]
    z_coordinates_0 = [z[0] for z in z_coordinates]

    number_of_timesteps = np.shape(x_coordinates)[1]

    # Combine coordinates into a single array
    coordinates_0 = np.column_stack([x_coordinates_0, y_coordinates_0, z_coordinates_0])

    # Convert Logarithmic Strain Tensors to approximate displacements
    # displacements_from_strain = []
    # for node_strain, x0, y0, z0 in zip(log_strain_tensors, x_coordinates_0, y_coordinates_0, z_coordinates_0):
    #     displacements = [np.array([0, 0, 0])]  # Initialize with zero displacement at timestep 0
    #     for t in range(len(node_strain[0]) - 1):
    #         strain_t = np.array([le[t + 1] - le[t] for le in node_strain])
    #         strain_matrix = np.array([
    #             [strain_t[0], strain_t[1], strain_t[4]],
    #             [strain_t[1], strain_t[2], strain_t[5]],
    #             [strain_t[4], strain_t[5], strain_t[3]]
    #         ])
    #         displacement_t = np.dot(strain_matrix, [x0, y0, z0])
    #         displacements.append(displacement_t)
    #     displacements_from_strain.append(displacements)
    #
    # displacements_from_strain = np.array(displacements_from_strain)

    displacements_from_strain = []
    for node_strain, x0, y0, z0 in zip(log_strain_tensors, x_coordinates_0, y_coordinates_0, z_coordinates_0):
        displacements = [np.array([0, 0, 0])]  # Initialize with zero displacement at timestep 0
        displacement_t = np.array([0, 0, 0])
        for t in range(1, len(node_strain[0])):
            strain_increment = np.array([node_strain[i][t] - node_strain[i][t - 1] for i in [0, 2, 5]])
            displacement_increment = strain_increment * [x0, y0, z0] + displacement_t
            displacements.append(displacement_increment)
            displacement_t = displacement_increment  # Update cumulative displacement
            if len(displacements_from_strain) + 1 == 16:
                print("disp:", displacement_t, "strain ", strain_increment)
        displacements_from_strain.append(displacements)

    displacements_from_strain = np.array(displacements_from_strain)

    print(np.shape(displacements_from_strain))
    print(displacements_from_strain[15, :, :])

    # Perform clustering to differentiate between inside and outside points
    kmeans = KMeans(n_clusters=2, random_state=0).fit(coordinates_0[:, :2])
    labels = kmeans.labels_

    # get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # determine which cluster corresponds to "outside" (larger y values) and which to "inside"
    outside_cluster = np.argmax(np.abs(cluster_centers[:, 1]))
    inside_cluster = 1 - outside_cluster

    # split points based on labels
    outside_indices = np.where(labels == outside_cluster)[0]
    inside_indices = np.where(labels == inside_cluster)[0]

    # Calculate the width of the volume
    y_coordinates_centroids = kmeans.cluster_centers_[:, 1]
    width = np.abs(np.diff(y_coordinates_centroids))

    # Calculate the length of the area
    length = np.ptp(x_coordinates_0)

    # If 'plot' is True, plot the node data in 3D
    if plot:
        fig = plt.figure(figsize=(12, 6))

        # Subplot 1 - Initial Node Distribution
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(*coordinates_0.T, marker='o', c=labels.astype(float), cmap='viridis')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Initial Node Distribution')

        # Subplot 2 - Node Movement Over Time
        ax2 = fig.add_subplot(122, projection='3d')
        final_coordinates = np.column_stack(([node_data[node_id]["X"][-1] for node_id in node_data.keys()],
                                             [node_data[node_id]["Y"][-1] for node_id in node_data.keys()],
                                             [node_data[node_id]["Z"][-1] for node_id in node_data.keys()]))

        scatter2 = ax2.scatter(*coordinates_0.T, marker='o', c=labels.astype(float), cmap='viridis')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Final Node Distribution')

        for i in range(len(coordinates_0)):
            ax2.plot([coordinates_0[i, 0], coordinates_0[i, 0] + displacement_x[i][-1]],
                     [coordinates_0[i, 1], coordinates_0[i, 1] + displacement_y[i][-1]],
                     [coordinates_0[i, 2], final_coordinates[i, 2]], 'gray',
                     label='Node displacement' if i == 0 else "")

        # Make sure legend is not repeated
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())

        # Add a colorbar with appropriate labels
        cbar = plt.colorbar(scatter2, ax=[ax1, ax2], pad=0.10)
        cbar.set_label('Cluster')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Inside', 'Outside'])

        plt.savefig("coord3d_0.png", dpi=300)

    # Check if 'TIME' key exists in node_data
    time = None
    if InputFileKeys.TIME.value not in data.keys():
        # Create the 'TIME' key with values increasing by 10 seconds for each time step
        time = np.arange(0, number_of_timesteps * 10, 10)
    else:
        # If 'TIME' key exists, load it as usual
        time = np.array(data[InputFileKeys.TIME.value])

    # Load other variables
    loaded_vars = {}
    for var in InputFileKeys.VARIABLES_TO_LOAD.value:
        try:
            loaded_vars[var] = np.array([node_data[node_id][var] for node_id in node_data.keys()])
        except KeyError:
            loaded_vars[var] = None

    # Add calculated values to the loaded_vars dictionary
    loaded_vars.update({
        ExternalInput.WIDTH.value: width[0],
        ExternalInput.LENGTH.value: length,
        # ExternalInput.DISPLACEMENTX.value: np.array(displacement_x),
        # ExternalInput.DISPLACEMENTY.value: np.array(displacement_y),
        # ExternalInput.DISPLACEMENTZ.value: np.array(displacement_z),
        ExternalInput.DISPLACEMENTX.value: np.array(displacements_from_strain[:, :, 0]),
        ExternalInput.DISPLACEMENTY.value: np.array(displacements_from_strain[:, :, 1]),
        ExternalInput.DISPLACEMENTZ.value: np.array(displacements_from_strain[:, :, 2]),
        # ExternalInput.RELDISPLACEMENT.value: np.array(rel_displacement),
        ExternalInput.X.value: np.array(x_coordinates),
        ExternalInput.Y.value: np.array(y_coordinates),
        ExternalInput.Z.value: np.array(z_coordinates),
        ExternalInput.OUTSIDE_P.value: outside_indices,
        ExternalInput.INSIDE_P.value: inside_indices,
        ExternalInput.TIME.value: time
    })

    return loaded_vars


def unpack_coordinates(input_file):
    return input_file[ExternalInput.X.value], input_file[ExternalInput.Y.value], input_file[ExternalInput.Z.value]
