"""Methods for numba-accelerated bilinear and trilinear interpolation."""

import numpy as np
from numba import njit, prange

PARALLEL_MODE = True


def _is_linearly_spaced(array, tol=1e-8):
    """
    Check if a numpy array is linearly spaced within some numerical precision.

    Parameters
    ----------
    array : np.ndarray
        The input array to check.
    tol : float
        The numerical tolerance for checking linear spacing.

    Returns
    -------
    bool
        True if the array is linearly spaced within the given tolerance, False otherwise.
    """
    if len(array) < 2:
        return True

    # Calculate the differences between consecutive elements
    diffs = np.diff(array)

    # Check if all differences are approximately equal within the given tolerance
    return np.all(np.abs(diffs - diffs[0]) <= tol)


@njit(parallel=PARALLEL_MODE)
def _fast_linear_interpolate(desired_pos, orig_coords, orig_values):
    """
    Perform numba-accelerated linear interpolation on a 1D array of values.

    Parameters
    ----------
    desired_pos : np.ndarray
        The desired positions for interpolation.
    orig_coords : np.ndarray
        The original coordinates. These should be monotonic but need not be linearly spaced.
    orig_values : np.ndarray
        The values at the original coordinates.

    Returns
    -------
    np.ndarray
        The interpolated values at the desired positions.
    """
    # Check if the original coordinates are decreasing and reverse them if necessary
    if orig_coords[0] > orig_coords[-1]:
        orig_coords = orig_coords[::-1]
        orig_values = orig_values[::-1]

    # Flatten the desired positions
    desired_pos = desired_pos.flatten()
    n_points = desired_pos.size
    result = np.empty(n_points)

    for idx in prange(n_points):
        x = desired_pos[idx]

        # Find the indices of the grid points surrounding x
        x1_idx = np.searchsorted(orig_coords, x) - 1
        x2_idx = x1_idx + 1

        # Boundary check to ensure we do not go out of bounds
        if x1_idx < 0 or x2_idx >= len(orig_coords):
            result[idx] = np.nan
            continue

        # Coordinates for surrounding points
        x1 = orig_coords[x1_idx]
        x2 = orig_coords[x2_idx]

        # Values at surrounding points
        Q1 = orig_values[x1_idx]
        Q2 = orig_values[x2_idx]

        # Perform linear interpolation
        result[idx] = (Q1 * (x2 - x) + Q2 * (x - x1)) / (x2 - x1)

    return result


@njit(parallel=PARALLEL_MODE)
def _fast_linear_interpolate_rectilinear(desired_pos, orig_coords, orig_values):
    """
    Perform numba-accelerated linear interpolation on a 1D array of values assuming a linearly spaced input grid.

    Parameters
    ----------
    desired_pos : np.ndarray
        The desired positions for interpolation.
    orig_coords : np.ndarray
        The original coordinates. These must be linearly spaced.
    orig_values : np.ndarray
        The values at the original coordinates.

    Returns
    -------
    np.ndarray
        The interpolated values at the desired positions.
    """
    # Check if the original coordinates are decreasing and reverse them if necessary
    if orig_coords[0] > orig_coords[-1]:
        orig_coords = orig_coords[::-1]
        orig_values = orig_values[::-1]

    # Flatten the desired positions
    desired_pos = desired_pos.flatten()
    n_points = desired_pos.size
    result = np.empty(n_points)

    # Calculate the step size
    step = (orig_coords[-1] - orig_coords[0]) / (len(orig_coords) - 1)

    for idx in prange(n_points):
        x = desired_pos[idx]

        # Calculate the indices of the grid points surrounding x
        x1_idx = int((x - orig_coords[0]) / step)
        x2_idx = x1_idx + 1

        # Boundary check to ensure we do not go out of bounds
        if x1_idx < 0 or x2_idx >= len(orig_coords):
            result[idx] = np.nan
            continue

        # Coordinates for surrounding points
        x1 = orig_coords[x1_idx]
        x2 = orig_coords[x2_idx]

        # Values at surrounding points
        Q1 = orig_values[x1_idx]
        Q2 = orig_values[x2_idx]

        # Perform linear interpolation
        result[idx] = (Q1 * (x2 - x) + Q2 * (x - x1)) / (x2 - x1)

    return result


@njit(parallel=PARALLEL_MODE)
def _fast_bilinear_interpolate(
    desired_pos_dim0,
    desired_pos_dim1,
    orig_coords_dim0,
    orig_coords_dim1,
    orig_values,
):
    """
    Perform numba-accelerated bilinear interpolation on a 2D grid of values.

    Parameters
    ----------
    desired_pos_dim0 : np.ndarray
        The desired positions along the first dimension.
    desired_pos_dim1 : np.ndarray
        The desired positions along the second dimension.
        Should have the same shape as `desired_pos_dim0`.
    orig_coords_dim0 : np.ndarray
        The original coordinates along the first dimension.
        These should be monotonic but need not be linearly spaced.
    orig_coords_dim1 : np.ndarray
        The original coordinates along the second dimension.
        These should be monotonic but need not be linearly spaced.
    orig_values : np.ndarray
        The values at the original grid points.

    Returns
    -------
    np.ndarray
        The interpolated values at the desired positions.
    """
    # Check if the original coordinates are decreasing and reverse them if necessary
    if orig_coords_dim0[0] > orig_coords_dim0[-1]:
        orig_coords_dim0 = orig_coords_dim0[::-1]
        orig_values = orig_values[::-1, :]
    if orig_coords_dim1[0] > orig_coords_dim1[-1]:
        orig_coords_dim1 = orig_coords_dim1[::-1]
        orig_values = orig_values[:, ::-1]

    # Flatten the desired positions
    desired_shape = desired_pos_dim0.shape  # Store for later
    desired_pos_dim0 = desired_pos_dim0.flatten()
    desired_pos_dim1 = desired_pos_dim1.flatten()

    n_points = desired_pos_dim0.size
    result = np.empty(n_points)

    for idx in prange(n_points):
        x = desired_pos_dim0[idx]
        y = desired_pos_dim1[idx]

        # Find the indices of the grid points surrounding (x, y)
        x1_idx = np.searchsorted(orig_coords_dim0, x) - 1
        x2_idx = x1_idx + 1
        y1_idx = np.searchsorted(orig_coords_dim1, y) - 1
        y2_idx = y1_idx + 1

        # Boundary check to ensure we do not go out of bounds
        if (
            x1_idx < 0
            or x2_idx >= len(orig_coords_dim0)
            or y1_idx < 0
            or y2_idx >= len(orig_coords_dim1)
        ):
            result[idx] = np.nan
            continue

        # Coordinates for surrounding points
        x1 = orig_coords_dim0[x1_idx]
        x2 = orig_coords_dim0[x2_idx]
        y1 = orig_coords_dim1[y1_idx]
        y2 = orig_coords_dim1[y2_idx]

        # Values at surrounding points
        Q11 = orig_values[x1_idx, y1_idx]
        Q12 = orig_values[x1_idx, y2_idx]
        Q21 = orig_values[x2_idx, y1_idx]
        Q22 = orig_values[x2_idx, y2_idx]

        # Perform bilinear interpolation
        result[idx] = (
            Q11 * (x2 - x) * (y2 - y)
            + Q21 * (x - x1) * (y2 - y)
            + Q12 * (x2 - x) * (y - y1)
            + Q22 * (x - x1) * (y - y1)
        ) / ((x2 - x1) * (y2 - y1))

    return result.reshape(desired_shape)


@njit(parallel=PARALLEL_MODE)
def _fast_bilinear_interpolate_rectilinear(
    desired_pos_dim0,
    desired_pos_dim1,
    orig_coords_dim0,
    orig_coords_dim1,
    orig_values,
):
    """
    Perform numba-accelerated bilinear interpolation on a rectilinear 2D grid of values.

    Parameters
    ----------
    desired_pos_dim0 : np.ndarray
        The desired positions along the first dimension.
    desired_pos_dim1 : np.ndarray
        The desired positions along the second dimension.
        Should have the same shape as `desired_pos_dim0`.
    orig_coords_dim0 : np.ndarray
        The original coordinates along the first dimension.
        These must be linearly spaced.
    orig_coords_dim1 : np.ndarray
        The original coordinates along the second dimension.
        These must be linearly spaced.
    orig_values : np.ndarray
        The values at the original grid points.

    Returns
    -------
    np.ndarray
        The interpolated values at the desired positions.
    """
    # Check if the original coordinates are decreasing and reverse them if necessary
    if orig_coords_dim0[0] > orig_coords_dim0[-1]:
        orig_coords_dim0 = orig_coords_dim0[::-1]
        orig_values = orig_values[::-1, :]
    if orig_coords_dim1[0] > orig_coords_dim1[-1]:
        orig_coords_dim1 = orig_coords_dim1[::-1]
        orig_values = orig_values[:, ::-1]

    # Flatten the desired positions
    desired_shape = desired_pos_dim0.shape  # Store for later
    desired_pos_dim0 = desired_pos_dim0.flatten()
    desired_pos_dim1 = desired_pos_dim1.flatten()

    n_points = desired_pos_dim0.size
    result = np.empty(n_points)

    # Calculate the step sizes
    step_dim0 = (orig_coords_dim0[-1] - orig_coords_dim0[0]) / (
        len(orig_coords_dim0) - 1
    )
    step_dim1 = (orig_coords_dim1[-1] - orig_coords_dim1[0]) / (
        len(orig_coords_dim1) - 1
    )

    for idx in prange(n_points):
        x = desired_pos_dim0[idx]
        y = desired_pos_dim1[idx]

        # Calculate the indices of the grid points surrounding (x, y)
        x1_idx = int((x - orig_coords_dim0[0]) / step_dim0)
        x2_idx = x1_idx + 1
        y1_idx = int((y - orig_coords_dim1[0]) / step_dim1)
        y2_idx = y1_idx + 1

        # Boundary check to ensure we do not go out of bounds
        if (
            x1_idx < 0
            or x2_idx >= len(orig_coords_dim0)
            or y1_idx < 0
            or y2_idx >= len(orig_coords_dim1)
        ):
            result[idx] = np.nan
            continue

        # Coordinates for surrounding points
        x1 = orig_coords_dim0[x1_idx]
        x2 = orig_coords_dim0[x2_idx]
        y1 = orig_coords_dim1[y1_idx]
        y2 = orig_coords_dim1[y2_idx]

        # Values at surrounding points
        Q11 = orig_values[x1_idx, y1_idx]
        Q12 = orig_values[x1_idx, y2_idx]
        Q21 = orig_values[x2_idx, y1_idx]
        Q22 = orig_values[x2_idx, y2_idx]

        # Perform bilinear interpolation
        result[idx] = (
            Q11 * (x2 - x) * (y2 - y)
            + Q21 * (x - x1) * (y2 - y)
            + Q12 * (x2 - x) * (y - y1)
            + Q22 * (x - x1) * (y - y1)
        ) / ((x2 - x1) * (y2 - y1))

    return result.reshape(desired_shape)


@njit(parallel=PARALLEL_MODE)
def _fast_trilinear_interpolate(
    desired_pos_dim0,
    desired_pos_dim1,
    desired_pos_dim2,
    orig_coords_dim0,
    orig_coords_dim1,
    orig_coords_dim2,
    orig_values,
    progress_proxy=None,
):
    """
    Perform numba-accelerated trilinear interpolation on a 3D grid of values.

    Parameters
    ----------
    desired_pos_dim0 : np.ndarray
        The desired positions along the first dimension.
    desired_pos_dim1 : np.ndarray
        The desired positions along the second dimension.
        Should have the same shape as `desired_pos_dim0`.
    desired_pos_dim2 : np.ndarray
        The desired positions along the third dimension.
        Should have the same shape as `desired_pos_dim0`.
    orig_coords_dim0 : np.ndarray
        The original coordinates along the first dimension.
        These should be monotonic but need not be linearly spaced.
    orig_coords_dim1 : np.ndarray
        The original coordinates along the second dimension.
        These should be monotonic but need not be linearly spaced.
    orig_coords_dim2 : np.ndarray
        The original coordinates along the third dimension.
        These should be monotonic but need not be linearly spaced.
    orig_values : np.ndarray
        The values at the original grid points.
    progress_proxy : ProgressProxy, optional
        A numba-progress ProgressBar proxy to update the progress of the interpolation.

    Returns
    -------
    np.ndarray
      The interpolated values at the desired positions.
    """
    # Check if the original coordinates are decreasing and reverse them if necessary
    if orig_coords_dim0[0] > orig_coords_dim0[-1]:
        orig_coords_dim0 = orig_coords_dim0[::-1]
        orig_values = orig_values[::-1, :, :]
    if orig_coords_dim1[0] > orig_coords_dim1[-1]:
        orig_coords_dim1 = orig_coords_dim1[::-1]
        orig_values = orig_values[:, ::-1, :]
    if orig_coords_dim2[0] > orig_coords_dim2[-1]:
        orig_coords_dim2 = orig_coords_dim2[::-1]
        orig_values = orig_values[:, :, ::-1]

    # Flatten the desired positions
    desired_shape = desired_pos_dim0.shape  # Store for later
    desired_pos_dim0 = desired_pos_dim0.flatten()
    desired_pos_dim1 = desired_pos_dim1.flatten()
    desired_pos_dim2 = desired_pos_dim2.flatten()

    n_points = desired_pos_dim0.size
    result = np.empty(n_points)

    for idx in prange(n_points):
        if progress_proxy is not None and (idx % 100 == 0 or idx == n_points - 1):
            progress_proxy.update(100)

        x = desired_pos_dim0[idx]
        y = desired_pos_dim1[idx]
        z = desired_pos_dim2[idx]

        # Find the indices of the grid points surrounding (x, y, z)
        x1_idx = np.searchsorted(orig_coords_dim0, x) - 1
        x2_idx = x1_idx + 1
        y1_idx = np.searchsorted(orig_coords_dim1, y) - 1
        y2_idx = y1_idx + 1
        z1_idx = np.searchsorted(orig_coords_dim2, z) - 1
        z2_idx = z1_idx + 1

        # Boundary check to ensure we do not go out of bounds
        if (
            x1_idx < 0
            or x2_idx >= len(orig_coords_dim0)
            or y1_idx < 0
            or y2_idx >= len(orig_coords_dim1)
            or z1_idx < 0
            or z2_idx >= len(orig_coords_dim2)
        ):
            result[idx] = np.nan
            continue

        # Coordinates for surrounding points
        x1 = orig_coords_dim0[x1_idx]
        x2 = orig_coords_dim0[x2_idx]
        y1 = orig_coords_dim1[y1_idx]
        y2 = orig_coords_dim1[y2_idx]
        z1 = orig_coords_dim2[z1_idx]
        z2 = orig_coords_dim2[z2_idx]

        # Values at surrounding points
        Q111 = orig_values[x1_idx, y1_idx, z1_idx]
        Q112 = orig_values[x1_idx, y1_idx, z2_idx]
        Q121 = orig_values[x1_idx, y2_idx, z1_idx]
        Q122 = orig_values[x1_idx, y2_idx, z2_idx]
        Q211 = orig_values[x2_idx, y1_idx, z1_idx]
        Q212 = orig_values[x2_idx, y1_idx, z2_idx]
        Q221 = orig_values[x2_idx, y2_idx, z1_idx]
        Q222 = orig_values[x2_idx, y2_idx, z2_idx]

        # Perform trilinear interpolation
        result[idx] = (
            Q111 * (x2 - x) * (y2 - y) * (z2 - z)
            + Q211 * (x - x1) * (y2 - y) * (z2 - z)
            + Q121 * (x2 - x) * (y - y1) * (z2 - z)
            + Q221 * (x - x1) * (y - y1) * (z2 - z)
            + Q112 * (x2 - x) * (y2 - y) * (z - z1)
            + Q212 * (x - x1) * (y2 - y) * (z - z1)
            + Q122 * (x2 - x) * (y - y1) * (z - z1)
            + Q222 * (x - x1) * (y - y1) * (z - z1)
        ) / ((x2 - x1) * (y2 - y1) * (z2 - z1))

    return result.reshape(desired_shape)


@njit(parallel=PARALLEL_MODE)
def _fast_trilinear_interpolate_rectilinear(
    desired_pos_dim0,
    desired_pos_dim1,
    desired_pos_dim2,
    orig_coords_dim0,
    orig_coords_dim1,
    orig_coords_dim2,
    orig_values,
    progress_proxy=None,
):
    """
    Perform numba-accelerated trilinear interpolation on a rectilinear 3D grid of values.

    Parameters
    ----------
    desired_pos_dim0 : np.ndarray
        The desired positions along the first dimension.
    desired_pos_dim1 : np.ndarray
        The desired positions along the second dimension.
        Should have the same shape as `desired_pos_dim0`.
    desired_pos_dim2 : np.ndarray
        The desired positions along the third dimension.
        Should have the same shape as `desired_pos_dim0`.
    orig_coords_dim0 : np.ndarray
        The original coordinates along the first dimension.
        These must be linearly spaced.
    orig_coords_dim1 : np.ndarray
        The original coordinates along the second dimension.
        These must be linearly spaced.
    orig_coords_dim2 : np.ndarray
        The original coordinates along the third dimension.
        These must be linearly spaced.
    orig_values : np.ndarray
        The values at the original grid points.
    progress_proxy : ProgressProxy, optional
        A numba-progress ProgressBar proxy to update the progress of the interpolation.

    Returns
    -------
    np.ndarray
        The interpolated values at the desired positions.
    """
    # Check if the original coordinates are decreasing and reverse them if necessary
    if orig_coords_dim0[0] > orig_coords_dim0[-1]:
        orig_coords_dim0 = orig_coords_dim0[::-1]
        orig_values = orig_values[::-1, :, :]
    if orig_coords_dim1[0] > orig_coords_dim1[-1]:
        orig_coords_dim1 = orig_coords_dim1[::-1]
        orig_values = orig_values[:, ::-1, :]
    if orig_coords_dim2[0] > orig_coords_dim2[-1]:
        orig_coords_dim2 = orig_coords_dim2[::-1]
        orig_values = orig_values[:, :, ::-1]

    # Flatten the desired positions
    desired_shape = desired_pos_dim0.shape  # Store for later
    desired_pos_dim0 = desired_pos_dim0.flatten()
    desired_pos_dim1 = desired_pos_dim1.flatten()
    desired_pos_dim2 = desired_pos_dim2.flatten()

    n_points = desired_pos_dim0.size
    result = np.empty(n_points)

    # Calculate the step sizes
    step_dim0 = (orig_coords_dim0[-1] - orig_coords_dim0[0]) / (
        len(orig_coords_dim0) - 1
    )
    step_dim1 = (orig_coords_dim1[-1] - orig_coords_dim1[0]) / (
        len(orig_coords_dim1) - 1
    )
    step_dim2 = (orig_coords_dim2[-1] - orig_coords_dim2[0]) / (
        len(orig_coords_dim2) - 1
    )

    for idx in prange(n_points):
        if progress_proxy is not None and (idx % 100 == 0 or idx == n_points - 1):
            progress_proxy.update(100)

        x = desired_pos_dim0[idx]
        y = desired_pos_dim1[idx]
        z = desired_pos_dim2[idx]

        # Calculate the indices of the grid points surrounding (x, y, z)
        x1_idx = int((x - orig_coords_dim0[0]) / step_dim0)
        x2_idx = x1_idx + 1
        y1_idx = int((y - orig_coords_dim1[0]) / step_dim1)
        y2_idx = y1_idx + 1
        z1_idx = int((z - orig_coords_dim2[0]) / step_dim2)
        z2_idx = z1_idx + 1

        # Boundary check to ensure we do not go out of bounds
        if (
            x1_idx < 0
            or x2_idx >= len(orig_coords_dim0)
            or y1_idx < 0
            or y2_idx >= len(orig_coords_dim1)
            or z1_idx < 0
            or z2_idx >= len(orig_coords_dim2)
        ):
            result[idx] = np.nan
            continue

        # Coordinates for surrounding points
        x1 = orig_coords_dim0[x1_idx]
        x2 = orig_coords_dim0[x2_idx]
        y1 = orig_coords_dim1[y1_idx]
        y2 = orig_coords_dim1[y2_idx]
        z1 = orig_coords_dim2[z1_idx]
        z2 = orig_coords_dim2[z2_idx]

        # Values at surrounding points
        Q111 = orig_values[x1_idx, y1_idx, z1_idx]
        Q112 = orig_values[x1_idx, y1_idx, z2_idx]
        Q121 = orig_values[x1_idx, y2_idx, z1_idx]
        Q122 = orig_values[x1_idx, y2_idx, z2_idx]
        Q211 = orig_values[x2_idx, y1_idx, z1_idx]
        Q212 = orig_values[x2_idx, y1_idx, z2_idx]
        Q221 = orig_values[x2_idx, y2_idx, z1_idx]
        Q222 = orig_values[x2_idx, y2_idx, z2_idx]

        # Perform trilinear interpolation
        result[idx] = (
            Q111 * (x2 - x) * (y2 - y) * (z2 - z)
            + Q211 * (x - x1) * (y2 - y) * (z2 - z)
            + Q121 * (x2 - x) * (y - y1) * (z2 - z)
            + Q221 * (x - x1) * (y - y1) * (z2 - z)
            + Q112 * (x2 - x) * (y2 - y) * (z - z1)
            + Q212 * (x - x1) * (y2 - y) * (z - z1)
            + Q122 * (x2 - x) * (y - y1) * (z - z1)
            + Q222 * (x - x1) * (y - y1) * (z - z1)
        ) / ((x2 - x1) * (y2 - y1) * (z2 - z1))

    return result.reshape(desired_shape)
