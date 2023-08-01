import ufl
import fenics as fe
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.cluster import KMeans
from scipy.interpolate import griddata, RegularGridInterpolator, interp1d
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from jsonschema import validate, ValidationError
from enum import Enum

fe.parameters["form_compiler"]["representation"] = 'quadrature'
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


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
    node_data = data["nodes"]

    # Get X, Y, and Z coordinates of all nodes for all timesteps
    x_coordinates = [node_data[node_id]["X"] for node_id in node_data.keys()]
    y_coordinates = [node_data[node_id]["Y"] for node_id in node_data.keys()]
    z_coordinates = [node_data[node_id]["Z"] for node_id in node_data.keys()]

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
    vars_to_load = ["LE11", "LE12", "LE22", "LE23", "LE31", "LE33",
                    "S11", "S12", "S22", "S23", "S31", "S33",
                    "T1", "T2", "T3"]
    loaded_vars = {}
    for var in vars_to_load:
        try:
            loaded_vars[var] = np.array([node_data[node_id][var] for node_id in node_data.keys()])
        except KeyError:
            loaded_vars[var] = None

    # Add calculated values to the loaded_vars dictionary
    loaded_vars.update({
        "thickness": thickness_aluminium[0],
        "length": length,
        "displacement": displacement,
        "X": np.array(x_coordinates),
        "Y": np.array(y_coordinates),
        "Z": np.array(z_coordinates),
    })

    return loaded_vars


def interpolate_values(coordinates, values, new_coordinates):
    """
    Interpolate values to a new set of coordinates.

    Parameters:
    coordinates (ndarray): The input coordinates.
    values (ndarray): The values at the input coordinates.
    new_coordinates (ndarray): The coordinates where we want to estimate the values.

    Returns:
    ndarray: The interpolated values at the new coordinates.
    """
    return griddata(coordinates, values, new_coordinates, method='linear')


# Function to interpolate a time series
def interpolate_timeseries(old_coordinates, timeseries, new_coordinates):
    n_time_steps = timeseries.shape[1]
    new_timeseries = []
    for t in range(n_time_steps):
        new_values_t = interpolate_values(old_coordinates, timeseries[:, t], new_coordinates)
        new_timeseries.append(new_values_t)
    return np.array(new_timeseries).T  # Shape: len(new_coordinates) x n_time_steps


def interpolate_in_time_and_space(old_coordinates, new_coordinates, old_times, new_times, old_values):
    """
    Interpolate values in both space and time.

    Parameters:
    old_coordinates (array-like): Original spatial coordinates, shape (num_points, 3).
    new_coordinates (array-like): New spatial coordinates where values are to be interpolated, shape (num_points_new, 3).
    old_times (array-like): Original time points, shape (num_times,).
    new_times (array-like): New time points where values are to be interpolated, shape (num_times_new,).
    old_values (array-like): Original values, shape (num_points, num_times).

    Returns:
    numpy.ndarray: Interpolated values at new_coordinates and new_times, shape (num_points_new, num_times_new).
    """
    num_points_new, num_times_new = len(new_coordinates), len(new_times)
    interpolated_values = np.empty((num_points_new, num_times_new))

    # Temporal interpolation function for each spatial point
    interp_funcs = [interp1d(old_times, old_values[i, :]) for i in range(len(old_coordinates))]

    # Spatial interpolation function for each time point
    for t in range(num_times_new):
        old_values_t = np.array([func(new_times[t]) for func in interp_funcs])
        interp_func_space = RegularGridInterpolator(old_coordinates, old_values_t)
        interpolated_values[:, t] = interp_func_space(new_coordinates)

    return interpolated_values


class SimulationConfig:
    def __init__(self, config):
        self.use_two_layers = config['simulation_parameters']['use_two_layers']
        self.time_integration_endpoint = config['simulation_parameters']['time_integration_endpoint']
        self.number_of_timesteps = config['simulation_parameters']['number_of_timesteps']
        self.selected_hardening_model = config['simulation_parameters']['selected_hardening_model']


# Load the configuration file
with open('simulation_config.json') as json_file:
    config = json.load(json_file)

# Create an instance of SimulationConfig
simulation_config = SimulationConfig(config)

# Now you can access the values like this
two_layers = simulation_config.use_two_layers
endTime = simulation_config.time_integration_endpoint
no_of_timesteps = simulation_config.number_of_timesteps
selected_hardening_model = simulation_config.selected_hardening_model

result = process_input_tensors('CementOutput.json', plot=True)

# Access thickness and length directly from the result dictionary
thickness_al = result['thickness']
length = result['length']

# Define old coordinates
old_coordinates = np.array([result['X'], result['Y'], result['Z']]).T


# # Define new coordinates
# new_coordinates = np.random.rand(10, 3)  # Example: 10 random 3D coordinates
#
# # Interpolate all values
# interp_LE11 = interpolate_timeseries(old_coordinates, result['LE11'], new_coordinates)
# interp_LE12 = interpolate_timeseries(old_coordinates, result['LE12'], new_coordinates)
# interp_LE22 = interpolate_timeseries(old_coordinates, result['LE22'], new_coordinates)
# interp_LE23 = interpolate_timeseries(old_coordinates, result['LE23'], new_coordinates)
# interp_LE31 = interpolate_timeseries(old_coordinates, result['LE31'], new_coordinates)
# interp_LE33 = interpolate_timeseries(old_coordinates, result['LE33'], new_coordinates)
#
# interp_S11 = interpolate_timeseries(old_coordinates, result['S11'], new_coordinates)
# interp_S12 = interpolate_timeseries(old_coordinates, result['S12'], new_coordinates)
# interp_S22 = interpolate_timeseries(old_coordinates, result['S22'], new_coordinates)
# interp_S23 = interpolate_timeseries(old_coordinates, result['S23'], new_coordinates)
# interp_S31 = interpolate_timeseries(old_coordinates, result['S31'], new_coordinates)
# interp_S33 = interpolate_timeseries(old_coordinates, result['S33'], new_coordinates)
#
# interp_T1 = interpolate_timeseries(old_coordinates, result['T1'], new_coordinates) if result['T1'] is not None else None
# interp_T2 = interpolate_timeseries(old_coordinates, result['T2'], new_coordinates) if result['T2'] is not None else None
# interp_T3 = interpolate_timeseries(old_coordinates, result['T3'], new_coordinates) if result['T3'] is not None else None
#
# interp_displacement = interpolate_timeseries(old_coordinates, result['displacement'], new_coordinates)

# # Define new times and coordinates
# new_times = np.linspace(0, 10, 100)  # just an example, use your actual new times
# new_coordinates = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # just an example, use your actual new coordinates
#
# # List of variables to interpolate
# variables_to_interpolate = ['LE11', 'LE12', 'LE22', 'LE23', 'LE31', 'LE33', 'S11', 'S12', 'S22', 'S23', 'S31', 'S33', 'T1', 'T2', 'T3']
#
# # Interpolate each variable and store the interpolated values in a new dictionary
# interpolated_values = {}
# for var in variables_to_interpolate:
#     old_values = result[var]
#     interpolated_values[var] = interpolate_in_time_and_space(result['coordinates'], new_coordinates, result['times'], new_times, old_values)

def load_material_properties(json_file, material):
    """
    Load material properties from a JSON file for a specific material.
    If Shear_modulus, First_Lame_parameter, Tangent_modulus, Linear_isotropic_hardening,
    or Nonlinear_Ludwik_parameter is not provided, they are calculated using the following relationships:

    - Shear_modulus (G): G = E / (2 * (1 + ν)) (Equation 1)
    - First_Lame_parameter (λ): λ = E * ν / ((1 + ν) * (1 - 2 * ν)) (Equation 2)
    - Tangent_modulus (Et): Et = E / 100 (Equation 3)
    - Linear_isotropic_hardening (H): H = E * Et / (E - Et) (Equation 4)
    - Nonlinear_Ludwik_parameter (nlin): nlin = 0.9 * H (Equation 5)

    Where:
    E is Young's modulus,
    ν is Poisson's ratio,
    Et is the Tangent modulus, and
    H is the Linear isotropic hardening.

    Parameters:
    json_file (str): Path to the JSON file to be loaded.
    material (str): The specific material to load properties for.

    Returns:
    dict: A dictionary containing the loaded material properties.
    """
    # Load and validate schema
    with open('material_properties.schema') as f:
        schema = json.load(f)

    # Load JSON file
    with open(json_file) as file:
        all_materials = json.load(file)

    # Validate the JSON file against the schema
    try:
        validate(instance=all_materials, schema=schema)
    except ValidationError as e:
        print(f"Validation error: {e.message}")

    # Check if the material is in the JSON file
    if material not in all_materials:
        print(f"Material {material} not found in the JSON file.")
        return

    # Extract specific material properties
    material_properties = all_materials[material]["properties"]

    # Retrieve or calculate each property
    E = material_properties.get("Youngs_modulus")
    nu = material_properties.get("Poissons_ratio")

    if E is not None and nu is not None:
        material_properties.setdefault("Shear_modulus", E / (2.0 * (1 + nu)))
        material_properties.setdefault("First_Lame_parameter", E * nu / ((1 + nu) * (1 - 2 * nu)))

    if E is not None:
        material_properties.setdefault("Tangent_modulus", E / 100.0)

    Et = material_properties.get("Tangent_modulus")
    if E is not None and Et is not None:
        material_properties.setdefault("Linear_isotropic_hardening", E * Et / (E - Et))

    H = material_properties.get("Linear_isotropic_hardening")
    if H is not None:
        material_properties.setdefault("Nonlinear_Ludwik_parameter", 0.9 * H)

    return material_properties


class Property(Enum):
    YOUNGS_MODULUS = "Youngs_modulus"
    POISSONS_RATIO = "Poissons_ratio"
    YIELD_STRENGTH = "Yield_strength"
    SHEAR_MODULUS = "Shear_modulus"
    FIRST_LAME_PARAMETER = "First_Lame_parameter"
    TANGENT_MODULUS = "Tangent_modulus"
    LINEAR_ISOTROPIC_HARDENING = "Linear_isotropic_hardening"
    NONLINEAR_LUDWIK_PARAMETER = "Nonlinear_Ludwik_parameter"
    EXPONENT_LUDWIK = "Exponent_Ludwik"
    SWIFT_EPSILON0 = "Swift_epsilon0"
    EXPONENT_SWIFT = "Exponent_Swift"


class MaterialProperties:
    def __init__(self, properties):
        self.properties = properties

    def get(self, key):
        """
        Retrieve a material property.

        Parameters:
        key (str): The key of the property to be retrieved.

        Returns:
        The requested material property.

        Raises:
        KeyError: If the provided key does not exist in the properties.
        """
        try:
            return self.properties[key]
        except KeyError:
            print(
                f"Key '{key}' not found in material properties. Available keys are: {', '.join(self.properties.keys())}")
            raise


# Load material properties
properties_al_dict = load_material_properties('material_properties.json', 'Al6082-T6')
properties_al = MaterialProperties(properties_al_dict)

properties_ceramic_dict = load_material_properties('material_properties.json', 'Aluminium-Ceramic')
properties_ceramic = MaterialProperties(properties_ceramic_dict)

# Access properties
C_E = properties_al.get(Property.YOUNGS_MODULUS.value)
C_nu = properties_al.get(Property.POISSONS_RATIO.value)
C_sig0 = properties_al.get(Property.YIELD_STRENGTH.value)
C_mu = properties_al.get(Property.SHEAR_MODULUS.value)
lmbda = properties_al.get(Property.FIRST_LAME_PARAMETER.value)
C_Et = properties_al.get(Property.TANGENT_MODULUS.value)
C_linear_isotropic_hardening = properties_al.get(Property.LINEAR_ISOTROPIC_HARDENING.value)
C_nlin_ludwik = properties_al.get(Property.NONLINEAR_LUDWIK_PARAMETER.value)
C_exponent_ludwik = properties_al.get(Property.EXPONENT_LUDWIK.value)
C_swift_eps0 = properties_al.get(Property.SWIFT_EPSILON0.value)
C_exponent_swift = properties_al.get(Property.EXPONENT_SWIFT.value)

C_E_outer = properties_ceramic.get(Property.YOUNGS_MODULUS.value)
C_nu_outer = properties_ceramic.get(Property.POISSONS_RATIO.value)
C_sig0_outer = properties_ceramic.get(Property.YIELD_STRENGTH.value)
C_mu_outer = properties_ceramic.get(Property.SHEAR_MODULUS.value)
lmbda_outer = properties_ceramic.get(Property.FIRST_LAME_PARAMETER.value)
C_Et_outer = properties_ceramic.get(Property.TANGENT_MODULUS.value)
C_linear_isotropic_hardening_outer = properties_ceramic.get(Property.LINEAR_ISOTROPIC_HARDENING.value)

# Geometry of the domain
##########################################
l_inner_x = 5.0  # mm
l_inner_y = 90.0  # mm

l_outer_x = 2.0  # mm
l_outer_y = 0.0  # mm

if two_layers:
    l_inner_x = thickness_al  # mm
    l_inner_y = length  # mm

    l_outer_x = 0.2  # mm
    l_outer_y = 0.0  # mm

l_x = l_inner_x + l_outer_x
l_y = l_inner_y + l_outer_y

# Discretization of the domain
n_x = 200
n_y = 100
n_z = 2  # Number of elements

# C_strain_rate = fe.Constant(0.0001)  # 1/s
C_strain_rate = fe.Constant(0.000001)  # 0.01/s

# Initialize the mesh
mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(l_x, l_y), n_x, n_y)

# Declare Numerical Stuff
deg_u, deg_stress = 2, 2

V = fe.VectorFunctionSpace(mesh, "CG", deg_u)
u, du, Du = [fe.Function(V, name=n) for n in ["Total displacement", "Iteration correction", "Current increment"]]

DG = fe.FunctionSpace(mesh, "DG", 0)
We = fe.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
W = fe.FunctionSpace(mesh, We)
sig, sig_old, n_elas = [fe.Function(W) for _ in range(3)]

func_names = ["Beta", "Cumulative plastic strain", "Hydrostatic stress", "local sig0",
              "local mu", "local lmbda", "local hardening factor"]
W0 = fe.FunctionSpace(mesh, fe.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default'))
beta, p, sig_hyd, sig_0_local, mu_local, lmbda_local, C_linear_h_local = [fe.Function(W0, name=n) for n in func_names]

P0 = fe.FunctionSpace(mesh, "DG", 0)
sig_hyd_avg, sig_0_test, lmbda_test = [fe.Function(P0, name=n) for n in ["Avg. Hydrostatic stress", "test", "test2"]]


# Boundary condition setup
def top(x, on_boundary):
    return on_boundary and fe.near(x[1], l_y)


def bottom(x, on_boundary):
    return on_boundary and fe.near(x[1], 0.0)


boundary_conditions = {
    'top': top,
    'bottom': bottom
}

displacement_conditions = {
    'top': fe.Expression("strain*t", t=0.0, degree=0, strain=C_strain_rate),
    'bottom': fe.Expression("-strain*t", t=0.0, degree=0, strain=C_strain_rate)
}

# top, bottom = boundary_condition(x, l_y), boundary_condition(x, 0.0)
# u_D = fe.Expression("strain*t", t=0.0, degree=0, strain=C_strain_rate)
# u_mD = fe.Expression("-strain*t", t=0.0, degree=0, strain=C_strain_rate)
bc1 = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), boundary_conditions['bottom'])
bc1_i = bc1

if two_layers:
    bc1 = fe.DirichletBC(V.sub(1), displacement_conditions['bottom'], boundary_conditions['bottom'])
    # reset displacement to 0
    bc1_i = fe.DirichletBC(V.sub(1), 0, boundary_conditions['bottom'])

# TOP
#########
# set the top to displace with a constant strain rate
bc2 = fe.DirichletBC(V.sub(1), displacement_conditions['top'], boundary_conditions['top'])

# reset displacement to 0
bc2_i = fe.DirichletBC(V.sub(1), 0, boundary_conditions['top'])

# set boundary condition
bc = [bc1, bc2]
# homogenized bc
bc_iter = [bc1_i, bc2_i]


# Util
########################################################################
def as_3D_tensor(X):
    return fe.as_tensor([[X[0], X[3], 0],
                         [X[3], X[1], 0],
                         [0, 0, X[2]]])


def local_project(v, V, u=None):
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
    # e= 0.5 * (fe.nabla_grad(u) + fe.nabla_grad(u).T)
    return fe.as_tensor([[e[0, 0], e[0, 1], 0],
                         [e[0, 1], e[1, 1], 0],
                         [0, 0, 0]])


def sigma(strain):
    """
    Calculate the Cauchy Stress Tensor
    :param strain: Strain (epsilon)
    :return: Cauchy Stress Tensor
    """
    # return lmbda * fe.tr(strain) * fe.Identity(3) + 2 * C_mu * strain
    # return lmbda * fe.tr(strain) * fe.Identity(3) + 2 * mu_local_V * strain #todo: line is broken
    # return lmbda * fe.tr(strain) * fe.Identity(3) + 2 * mu_local_DG * strain
    return lmbda_local_DG * fe.tr(strain) * fe.Identity(3) + 2 * mu_local_DG * strain


def sigma_tang(e):
    N_elas = as_3D_tensor(n_elas)
    # return sigma(e) - 3 * C_mu * (3 * C_mu / (3 * C_mu + C_linear_isotropic_hardening) - beta) * fe.inner(N_elas,
    #                                                                            e) * N_elas - 2 * C_mu * beta * fe.dev(e)

    return sigma(e) - 3 * mu_local_DG * (3 * mu_local_DG / (3 * mu_local_DG + C_linear_h_local_DG) - beta) * fe.inner(
        N_elas,
        e) * N_elas - 2 * mu_local_DG * beta * fe.dev(e)


# Von-Mises Stress
def sigma_v(strain):
    s = fe.dev(sigma(strain))
    return fe.sqrt(3 / 2. * fe.inner(s, s))


# Macaulays Bracket for <f_elastic>+
# only positive for x > 0
ppos = lambda x: (x + abs(x)) / 2.


# https://www.dynasupport.com/tutorial/computational-plasticity/radial-return
# https://www.dynasupport.com/tutorial/computational-plasticity/generalizing-the-yield-function
# https://www.dynasupport.com/tutorial/computational-plasticity/the-consistent-tangent-matrix
def proj_sig(deps, old_sig, old_p):
    # update stress from change in strain (deps)
    sig_n = as_3D_tensor(old_sig)
    sig_elas = sig_n + sigma(deps)

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
    beta = 3 * C_mu * dp / sig_eq

    # updated cauchy stress tensor
    # in elastic case = sig_elas
    new_sig = sig_elas - beta * s

    # Hydrostatic stress
    sig_hyd = (1. / 3) * fe.tr(new_sig)

    return fe.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
        fe.as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]), \
        beta, dp, sig_hyd


metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = ufl.dx(metadata=metadata)

v = fe.TrialFunction(V)
u_ = fe.TestFunction(V)


class set_layer_2(fe.UserExpression):
    # Return the diffusion coefficient at a point scaled by level set function
    def __init__(self, inner_value, outer_value, **kwargs):
        self.inner = inner_value
        self.outer = outer_value
        super().__init__(**kwargs)

    def eval(self, value, x):
        if x[0] > l_inner_x:
            value[0] = self.outer
            value[1] = self.outer
        else:
            value[0] = self.inner
            value[1] = self.inner

    def value_shape(self):
        return (2,)


# calculate local mu
# test2 = set_layer_2(C_mu, C_mu)
# mu_local_V = fe.interpolate(test2, V)
mu_local_DG = fe.Function(DG)
dofmap = DG.tabulate_dof_coordinates()[:]
mu_vec = np.zeros(dofmap.shape[0])
mu_vec[:] = C_mu
mu_vec[dofmap[:, 0] > l_inner_x] = C_mu_outer
mu_local_DG.vector()[:] = mu_vec
# print(dofmap)
# print(dofmap.shape)
# u_DG = fe.XDMFFile("u.xdmf")
# u_DG << fe.project(mu_local_DG, V)
# # mu_local_DG = fe.project(test2, DG)
# exit()

lmbda_local_DG = fe.Function(DG)
dofmap = DG.tabulate_dof_coordinates()[:]
lmbda_vec = np.zeros(dofmap.shape[0])
lmbda_vec[:] = lmbda
lmbda_vec[dofmap[:, 0] > l_inner_x] = lmbda_outer
lmbda_local_DG.vector()[:] = lmbda_vec

# lmbda_test.assign(fe.project(lmbda_local_DG, P0))

# plt.figure()
# # plt.plot(results[:, 0], results[:, 1], "-o")
# ax = fe.plot(lmbda_test)
# cbar = plt.colorbar(ax)
# plt.xlabel("x")
# plt.ylabel("y$")
# # plt.show()
# plt.savefig("testlmbda.svg")
# plt.close()
# exit()

C_linear_h_local_DG = fe.Function(DG)
dofmap = DG.tabulate_dof_coordinates()[:]
C_linear_h_vec = np.zeros(dofmap.shape[0])
C_linear_h_vec[:] = C_linear_isotropic_hardening
C_linear_h_vec[dofmap[:, 0] > l_inner_x] = C_linear_isotropic_hardening_outer
C_linear_h_local_DG.vector()[:] = C_linear_h_vec

a_Newton = fe.inner(eps(v), sigma_tang(eps(u_))) * dxm
# res = -inner(eps(u_), as_3D_tensor(sig)) * dxm + F_ext(u_)
res = -fe.inner(eps(u_), as_3D_tensor(sig)) * dxm

cellV = local_project(fe.CellVolume(mesh), P0)
# print("minV:", np.amin(cellV.vector()[:]), "maxV", np.amax(cellV.vector()[:]))
# print("H", H.values()[0])

file_results = fe.XDMFFile("plasticity_results.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
P0 = fe.FunctionSpace(mesh, "DG", 0)
p_avg = fe.Function(P0, name="Plastic strain")

Nitermax, tol = 100, 1e-8  # parameters of the Newton-Raphson procedure
time_step = endTime / (no_of_timesteps)


# assign local values to the layers
class set_yield(fe.UserExpression):
    # Return the diffusion coefficient at a point scaled by level set function
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, value, x):
        if x[0] > l_inner_x:
            value[0] = C_sig0_outer
        else:
            value[0] = C_sig0

    def value_shape(self):
        return ()


class set_layer(fe.UserExpression):
    # Return the diffusion coefficient at a point scaled by level set function
    def __init__(self, inner_value, outer_value, **kwargs):
        self.inner = inner_value
        self.outer = outer_value
        super().__init__(**kwargs)

    def eval(self, value, x):
        if x[0] > l_inner_x:
            value[0] = self.outer
        else:
            value[0] = self.inner

    def value_shape(self):
        return ()


# test = set_yield()
# sig_0_local = fe.interpolate(test, W0)

test = set_layer(C_sig0, C_sig0_outer)
sig_0_local = fe.interpolate(test, W0)

test = set_layer(C_mu, C_mu_outer)
mu_local = fe.interpolate(test, W0)

test = set_layer(C_linear_isotropic_hardening, C_linear_isotropic_hardening_outer)
C_linear_h_local = fe.interpolate(test, W0)

# #mu_local_V = fe.project(mu_local, V)
# test2 = set_layer_2(C_mu, C_mu)
# mu_local_V = fe.interpolate(test2, V)

# sig_0_test.assign(fe.project(sig_0_local, P0))

# plt.figure()
# # plt.plot(results[:, 0], results[:, 1], "-o")
# ax = fe.plot(sig_0_test)
# cbar = plt.colorbar(ax)
# plt.xlabel("x")
# plt.ylabel("y$")
# # plt.show()
# plt.savefig("test.svg")
# plt.close()

results = []
stress_max_t = []
stress_max_t += [0]
stress_mean_t = []
stress_mean_t += [0]
disp_t = []
disp_t += [0]
time_step_adjusted = False
time = 0
i = 0
not_adjusted_count = 0
# for (i, time) in enumerate(time_steps):
while time < endTime:
    time += time_step
    i += 1

    # update the displacement boundary
    displacement_conditions['top'].t = time_step
    if two_layers:
        displacement_conditions['bottom'].t = time_step

    # assemble system
    A, Res = fe.assemble_system(a_Newton, res, bc)

    # calculate residual
    nRes0 = Res.norm("l2")
    nRes = nRes0

    # reset Du to 0
    Du.interpolate(fe.Constant((0, 0)))

    print("Step:", str(i + 1), "time", time, "s")
    print("displacement", C_strain_rate.values()[0] * time, "mm")

    niter = 0
    # n_resold = 1000
    while niter == 0 or (nRes0 > 0 and nRes / nRes0 > tol and niter < Nitermax):
        # solve for du
        fe.solve(A, du.vector(), Res, "mumps")

        # update displacement gradient
        Du.assign(Du + du)

        # calculate strain gradient
        deps = eps(Du)

        # update variables
        sig_, n_elas_, beta_, dp_, sig_hyd_ = proj_sig(deps, sig_old, p)
        local_project(sig_, W, sig)
        local_project(n_elas_, W, n_elas)
        local_project(beta_, W0, beta)
        sig_hyd.assign(local_project(sig_hyd_, W0))

        # update system
        A, Res = fe.assemble_system(a_Newton, res, bc_iter)

        nRes = Res.norm("l2")
        print("    Residual:", nRes)
        # n_resold = nRes
        niter += 1

    # exit if diverged
    if nRes > 1 or np.isnan(nRes):
        print("ERROR: diverged!")
        exit(-1)

    # update displacement
    u.assign(u + Du)
    # update plastic strain
    p.assign(p + local_project(dp_, W0))

    # update stress fields
    sig_old.assign(sig)
    sig_hyd_avg.assign(fe.project(sig_hyd, P0))

    s11, s12, s21, s22 = sig.split(deepcopy=True)
    avg_stress_y = np.average(s22.vector()[:])
    avg_stress = np.average(sig.vector()[:])
    sig_n = as_3D_tensor(sig)
    s = fe.dev(sig_n)

    # calculate the von-mises stress
    sig_eq = fe.sqrt(3 / 2. * fe.inner(s, s))
    sig_eq_p = local_project(sig_eq, P0)

    if i % 10 == 0:
        plt.figure()
        # plt.plot(results[:, 0], results[:, 1], "-o")
        ax = fe.plot(sig_eq_p)
        cbar = plt.colorbar(ax)
        plt.xlabel("x")
        plt.ylabel("y$")
        # plt.show()
        plt.savefig("vm" + str(i) + ".svg")
        plt.close()

    # project the von-mises stress for plotting
    stress_max_t += [np.abs(np.amax(sig_eq_p.vector()[:]))]
    stress_mean_t += [np.abs(np.mean(sig_eq_p.vector()[:]))]

    # displacement at the middle of the bar in y-direction
    disp_t += [u(l_x / 2, l_y)[1]]

    # print("maximum stress ", np.amax(sig.vector()[:]))
    # print("maximum stress(y-y) ", np.amax(s22.vector()[:]))
    # print("plastic strain ", np.amax(p.vector()[:]))

    file_results.write(u, time)
    p_avg.assign(fe.project(p, P0))
    file_results.write(p_avg, time)
    results += [(np.abs(u(l_x / 2, l_y)[1]) / l_y, time)]

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

# data extracted from fig. 6 paper1
ref_strain = [4.5917e-7, 0.0022967, 0.0068471, 0.0077457, 0.008868360277136257,
              0.011511, 0.024988, 0.040012, 0.059011, 0.071602, 0.083751,
              0.10429, 0.12108, 0.13963, 0.16105, 0.17960]
ref_stress = [0.10395, 99.376, 250.10, 267.26, 275.09019989024057,
              279.21, 284.41, 290.64, 296.36, 299.48, 301.04, 302.60, 303.12, 302.60,
              301.04, 296.88]

# data extracted from fig. 10 paper 3
ref_strain2 = [0, 0.001072706, 0.001966627, 0.002324195, 0.003218117, 0.004469607, 0.007508939,
               0.009833135, 0.015554231, 0.027711561, 0.038796186, 0.056495828, 0.070262217,
               0.085637664, 0.10852205, 0.131585221, 0.155542312, 0.175029797, 0.178605483]

ref_stress2 = [0, 117.4641148, 172.0095694, 222.9665072, 236.2440191, 246.291866, 253.8277512,
               260.2870813, 264.9521531, 277.15311, 285.0478469, 294.7368421, 299.0430622,
               302.9904306, 307.2966507, 305.861244, 306.5789474, 300.8373206, 288.6363636]

# for linear deformation E = stress/strain -> strain = stress/E
linear_strain = np.multiply(np.array(stress_max_t), 1.0 / C_E.values()[0])
linear_deformation = linear_strain * l_y

print("max vm stress", np.max(stress_max_t))
print("mean vm stress", np.max(stress_mean_t))

# Plot the stress-deformation curve
plt.plot(disp_t, stress_max_t, "-o", label="sim", markevery=5)
plt.plot(linear_deformation, stress_max_t, label="linear")
plt.xlabel("displacement [mm]")
plt.ylabel(r"max. von-Mises stress [MPa]")
plt.legend()
plt.savefig("test_deformation_stress.svg")

# Plot the stress-strain curve
plt.figure()
# plt.plot(np.array(disp_t) / l_y, stress_max_t, "-o", label="sim-max", markevery=5)
plt.plot(np.array(disp_t) / l_y, stress_mean_t, "-o", label="sim", markevery=5)
plt.plot(linear_strain, stress_max_t, label="linear")
plt.plot(ref_strain, ref_stress, label="exp")
plt.plot(ref_strain2, ref_stress2, label="exp2")
# plt.plot(linear_strain, stress_mean_t, label="linear")
plt.xlabel("strain [-]")
plt.ylabel(r"max. von-Mises stress [MPa]")
plt.legend()
plt.savefig("test_strain_stress.svg")

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
