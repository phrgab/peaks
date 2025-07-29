"""Functions that utilise machine learning techniques to denoise data."""

# Brendan Edwards 31/10/2023

import xarray as xr
from sklearn.decomposition import PCA

from peaks.core.accessors.accessor_methods import register_accessor


@register_accessor(xr.DataArray)
def SM_PCA(data, PCs=10):
    """Perform a principal component analysis on a spatial map, and then reconstruct the spatial map using the reduced
    dimensionality dataset and PCA eigenvectors. Results in a reduction in the information retained by the spatial map,
    but this lost information can just be noise. Thus, this function can act to markedly denoise spatial mapping data.
    The effect becomes more apparent for lower numbers of principal components, but too low will oversimplify data.

    Parameters
    ------------
    data : xarray.DataArray
        The spatial mapping data to perform a principal component analysis on.

    PCs : int, optional
        The number of principal components used to perform PCA. Defaults to 10.

    Returns
    ------------
    Reconstructed_SM : xarray.DataArray
        The reconstructed spatial map.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        SM = load('SM.ibw')

        # Use a 10 principal components PCA to reduce the information contained a spatial map, effectively denoising it
        reconstructed_SM1 = SM.SM_PCA()

        # Use a 30 principal components PCA to reduce the information contained a spatial map, effectively denoising it
        reconstructed_SM2 = SM.SM_PCA(PCs=30)

    """

    # Prevent unwanted overwriting of original data
    data = data.copy(deep=True)

    # Represent spatial mapping data as a tabular pandas DataFrame
    df = data.ML_pre_proc(extract="dispersion", scale=False)

    # Perform PCA
    pca = PCA(n_components=PCs)
    principal_components = pca.fit_transform(df)

    # Reconstruct the spatial map using the reduced dimensionality dataset and PCA eigenvectors
    reconstructed_data = pca.inverse_transform(principal_components).reshape(
        len(data.x1), len(data.x2), data.shape[2], data.shape[3]
    )

    # Represent reconstructed data using xarray format
    coords = (
        data.dims
    )  # Used so function works for any eV/k_par/theta_par combination/order
    reconstructed_SM = xr.DataArray(
        reconstructed_data,
        dims=("x1", "x2", coords[2], coords[3]),
        coords={
            "x1": data.x1,
            "x2": data.x2,
            coords[2]: data.coords[coords[2]],
            coords[3]: data.coords[coords[3]],
        },
    )
    reconstructed_SM.attrs = data.attrs

    return reconstructed_SM
