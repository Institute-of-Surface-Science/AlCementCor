import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from enum import Enum


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
    X = "X"
    Y = "Y"
    Z = "Z"


def process_input_tensors(filename, plot=False):
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

    # Calculate displacement from initial position for each node
    displacement = [np.sqrt((x[0] - x[-1]) ** 2 + (y[0] - y[-1]) ** 2 + (z[0] - z[-1]) ** 2)
                    for x, y, z in zip(x_coordinates, y_coordinates, z_coordinates)]

    # Get coordinates for the first timestep for further calculations
    x_coordinates_0 = [x[0] for x in x_coordinates]
    y_coordinates_0 = [y[0] for y in y_coordinates]
    z_coordinates_0 = [z[0] for z in z_coordinates]

    # Combine coordinates into a single array
    coordinates_0 = np.column_stack([x_coordinates_0, y_coordinates_0, z_coordinates_0])

    # Perform clustering to differentiate between inside and outside points
    kmeans = KMeans(n_clusters=2, random_state=0).fit(coordinates_0[:, :2])
    labels = kmeans.labels_

    # Calculate the thickness of the aluminium layer
    y_coordinates_centroids = kmeans.cluster_centers_[:, 1]
    thickness_aluminium = np.abs(np.diff(y_coordinates_centroids))
    print(f"Aluminium layer thickness: {thickness_aluminium[0]}")

    # Calculate the length of the area
    length = np.ptp(x_coordinates_0)
    print(f"Area length: {length}")

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

        scatter2 = ax2.scatter(*final_coordinates.T, marker='o', c=labels.astype(float), cmap='viridis')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Final Node Distribution')

        for i in range(len(coordinates_0)):
            ax2.plot([coordinates_0[i, 0], final_coordinates[i, 0]],
                     [coordinates_0[i, 1], final_coordinates[i, 1]],
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

    # Load other variables
    loaded_vars = {}
    for var in InputFileKeys.VARIABLES_TO_LOAD.value:
        try:
            loaded_vars[var] = np.array([node_data[node_id][var] for node_id in node_data.keys()])
        except KeyError:
            loaded_vars[var] = None

    # Add calculated values to the loaded_vars dictionary
    loaded_vars.update({
        ExternalInput.WIDTH.value: thickness_aluminium[0],
        ExternalInput.LENGTH.value: length,
        ExternalInput.DISPLACEMENT.value: displacement,
        ExternalInput.X.value: np.array(x_coordinates),
        ExternalInput.Y.value: np.array(y_coordinates),
        ExternalInput.Z.value: np.array(z_coordinates),
    })

    return loaded_vars