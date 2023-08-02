import numpy as np
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