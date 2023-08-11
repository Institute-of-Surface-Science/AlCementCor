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
