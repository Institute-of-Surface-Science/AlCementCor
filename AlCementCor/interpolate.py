import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata, RegularGridInterpolator, interp1d


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
        """
        Initializes the interpolators for in-plane displacements.

        Parameters:
            time_values (array): Time values at each step.
            center_yz_points (array): Y-Z coordinates of the center points.
            displacement_in_plane_1, displacement_in_plane_2 (array): Displacements in the plane.
        """
        self.interpolator_1 = self._create_interpolator(time_values, center_yz_points, displacement_in_plane_1)
        self.interpolator_2 = self._create_interpolator(time_values, center_yz_points, displacement_in_plane_2)

    def _create_interpolator(self, time_values, center_yz_points, displacement_values):
        """
        Constructs an interpolator for the given values.

        Parameters:
            time_values (array): Time values at each step.
            center_yz_points (array): Y-Z coordinates of the center points.
            displacement_values (array): Displacement values.

        Returns:
            RegularGridInterpolator: The created interpolator.
        """

        # Construct a 4D grid of points (time, x, y, z)
        time_grid, x_grid, y_grid, z_grid = np.meshgrid(
            time_values,
            center_yz_points[:, :, 0],
            center_yz_points[:, :, 1],
            center_yz_points[:, :, 2],
            indexing='ij'
        )
        points = np.column_stack([time_grid.flatten(), x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])

        # Create and return the interpolator
        return RegularGridInterpolator(
            (time_values, x_grid.flatten(), y_grid.flatten(), z_grid.flatten()),
            displacement_values.flatten(),
            bounds_error=False
        )

    def get_displacement_at_point(self, time_value, query_point):
        """
        Queries the interpolators at the given time and point.

        Parameters:
            time_value (float): The time value for the query.
            query_point (array): The spatial coordinates for the query.

        Returns:
            tuple: The interpolated displacements at the given time and point.
        """
        query = np.concatenate(([time_value], query_point))
        displacement_1 = self.interpolator_1(query)
        displacement_2 = self.interpolator_2(query)

        return displacement_1, displacement_2


def interpolate_displacements(center_points, x_coords, y_coords, z_coords, x_displacements,
                              y_displacements, z_displacements):
    """
    Interpolates the displacements of given center points in a plane over time and projects them onto the plane.

    Parameters:
        center_points (list): List of center points of the plane.
        x_coords, y_coords, z_coords (array): Coordinates of the points.
        x_displacements, y_displacements, z_displacements (array): Displacements in each direction.

    Returns:
        tuple: Interpolated displacements in the x and y directions projected onto the plane.
    """
    num_center_nodes, num_timesteps = len(center_points[0]), x_coords.shape[1]
    displacements_in_plane_x = np.empty((num_center_nodes, num_timesteps))
    displacements_in_plane_y = np.empty((num_center_nodes, num_timesteps))

    center_points = np.array(center_points)

    for timestep in range(num_timesteps):
        # Combine coordinates for this timestep
        coordinates_at_timestep = np.column_stack((x_coords[:, timestep], y_coords[:, timestep], z_coords[:, timestep]))

        # Retrieve the center points for this timestep
        center_points_at_timestep = center_points[timestep]

        # Define points A and B and calculate normalized vector AB
        point_A, point_B = center_points_at_timestep[:2]
        vector_AB = point_B - point_A
        vector_AB /= np.linalg.norm(vector_AB)

        # Find a non-collinear point C and calculate normalized vector AC
        vector_AC = None
        for point in center_points_at_timestep[2:]:
            point_C = point
            vector_AC = point_C - point_A
            cross_product = np.cross(vector_AB, vector_AC)
            if np.linalg.norm(cross_product) > 1e-6:
                vector_AC -= np.dot(vector_AB, vector_AC) * vector_AB
                vector_AC /= np.linalg.norm(vector_AC)
                break
        else:
            raise ValueError("No non-collinear point found")

        # Create interpolation functions for this timestep
        interp_funcs = [LinearNDInterpolator(coordinates_at_timestep, displacement[:, timestep])
                        for displacement in [x_displacements, y_displacements, z_displacements]]

        # Interpolate the displacements for each center point
        for p, point in enumerate(center_points_at_timestep):
            interpolated_displacement = np.array([func(*point) for func in interp_funcs])
            displacements_in_plane_x[p, timestep] = np.dot(interpolated_displacement, vector_AB)
            displacements_in_plane_y[p, timestep] = np.dot(interpolated_displacement, vector_AC)

    return displacements_in_plane_x, displacements_in_plane_y
