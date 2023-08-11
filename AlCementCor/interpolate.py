import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator, interp1d
from scipy.interpolate import LinearNDInterpolator


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
    """
    Interpolates a time series between old coordinates to new coordinates.

    Parameters:
    ----------
    old_coordinates : array_like
        Original coordinates (1D array) corresponding to the provided time series.

    timeseries : array_like
        Time series to be interpolated. It is a 2D array where each row represents
        the value of the timeseries at the corresponding old coordinate and each
        column represents a time step.

    new_coordinates : array_like
        New coordinates (1D array) to which the timeseries will be interpolated.

    Returns:
    -------
    numpy.ndarray
        A 2D array of the interpolated time series. Each row corresponds to a new
        coordinate and each column corresponds to a time step.

    Notes:
    ------
    The function uses linear interpolation to generate new values.

    The interpolation function is assumed to be 'interpolate_values', which must be
    previously defined and take three arguments: old_coordinates, values, and new_coordinates.

    """
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

class InPlaneInterpolator:
    def __init__(self, time_values, center_yz_points, displacement_in_plane_1, displacement_in_plane_2):
        # Create interpolators for the in-plane displacements
        self.interpolator_1 = self._create_interpolator(time_values, center_yz_points, displacement_in_plane_1)
        self.interpolator_2 = self._create_interpolator(time_values, center_yz_points, displacement_in_plane_2)

    def _create_interpolator(self, time_values, center_yz_points, displacement_values):
        # Construct a 3D grid of points (time, x, y, z)
        time_grid, x_grid, y_grid, z_grid = np.meshgrid(time_values, center_yz_points[:, :, 0],
                                                        center_yz_points[:, :, 1], center_yz_points[:, :, 2],
                                                        indexing='ij')
        points = np.column_stack([time_grid.flatten(), x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])

        # Create and return the interpolator
        return RegularGridInterpolator((time_values, center_yz_points[:, :, 0].flatten(),
                                        center_yz_points[:, :, 1].flatten(), center_yz_points[:, :, 2].flatten()),
                                       displacement_values.flatten(), bounds_error=False)

    def get_displacement_at_point(self, time_value, query_point):
        # Query the interpolators at the given time and point
        query = np.array([time_value] + list(query_point))
        displacement_at_point_1 = self.interpolator_1(query)
        displacement_at_point_2 = self.interpolator_2(query)

        return displacement_at_point_1, displacement_at_point_2


def interpolate_displacements(center_yz_points, x_coordinates, y_coordinates, z_coordinates, displacement_x,
                              displacement_y, displacement_z):
    no_center_nodes = len(center_yz_points[0])
    no_timesteps = len(x_coordinates[0])
    displacement_x_center = np.empty((no_center_nodes, no_timesteps))
    displacement_y_center = np.empty((no_center_nodes, no_timesteps))

    for t in range(no_timesteps):
        # Extracting coordinates for this timestep
        coordinates_t = np.column_stack([x_coordinates[:, t], y_coordinates[:, t], z_coordinates[:, t]])

        # Extracting center_yz_points for this timestep
        center_yz_points_t = [tuple(yz) for yz in center_yz_points[t]]

        # Start with the first two points from center_yz_points_t
        A = np.array(center_yz_points_t[0])
        B = np.array(center_yz_points_t[1])
        C = None

        # Find a third point that is not collinear with A and B
        for point in center_yz_points_t[2:]:
            C = np.array(point)
            AB = B - A
            AC = C - A
            cross_product = np.cross(AB, AC)
            if np.linalg.norm(cross_product) > 1e-6:  # tolerance for numerical errors
                break
        else:
            print("No non-collinear point found")
            exit(-1)

        # Normalize AB and AC
        AB /= np.linalg.norm(AB)
        AC -= np.dot(AB, AC) * AB  # Make AC orthogonal to AB
        AC /= np.linalg.norm(AC)

        # Interpolation functions for this timestep
        interp_x = LinearNDInterpolator(coordinates_t, displacement_x[:, t])
        interp_y = LinearNDInterpolator(coordinates_t, displacement_y[:, t])
        interp_z = LinearNDInterpolator(coordinates_t, displacement_z[:, t])

        # Interpolated displacements for this timestep, in-plane coordinates
        for p, yz in enumerate(center_yz_points_t):
            dx = interp_x(yz[0], yz[1], yz[2])
            dy = interp_y(yz[0], yz[1], yz[2])
            dz = interp_z(yz[0], yz[1], yz[2])
            displacement = np.array([dx, dy, dz])
            disp_in_plane_1 = np.dot(displacement, AB)
            disp_in_plane_2 = np.dot(displacement, AC)
            displacement_x_center[p, t] = disp_in_plane_1
            displacement_y_center[p, t] = disp_in_plane_2

    return displacement_x_center, displacement_y_center
