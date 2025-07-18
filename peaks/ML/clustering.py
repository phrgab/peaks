"""Unsupervised clustering functions."""

# Brendan Edwards 31/10/2023

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import clear_output
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from peaks.core.accessors.accessor_methods import register_accessor
from peaks.core.display.plotting import plot_grid
from peaks.utils import analysis_warning


@register_accessor(xr.DataArray)
def ML_pre_proc(
    data, extract="dispersion", E=0, dE=0, k=0, dk=0, scale=False, norm=False
):
    """Represent spatial mapping data as a tabular pandas DataFrame where each spatial position is a feature.
    This DataFrame format is used to represent data in machine learning functions.

    Parameters
    ------------
    data : xarray.DataArray
        The spatial mapping data to change to a tabular pandas dataframe.

    extract : str, optional
        Determines what is extracted from spatial mapping data. Defaults to dispersion. Valid entries are:
            dispersion
            MDC
            EDC
        Selecting MDC/EDC will rapidly increase calculation time.

    E : float, optional
        Energy of MDCs to extract. Defaults to 0.

    dE : float, optional
        MDC Integration range (represents the total range, i.e. integrates over +/- dE/2). Defaults to 0.

    k : float, optional
        k or theta_par value of EDCs to extract. Defaults to 0.

    dk : float, optional
        EDC integration range (represents the total range, i.e. integrates over +/- dk/2). Defaults to 0.

    scale : bool, optional
        Whether to apply standard scaling to data (i.e. center all values in each dimension around zero with unit
        standard deviation). Defaults to False.

    norm : bool, optional
        Whether to normalise the data at each spatial position. Defaults to False.

    Returns
    ------------
    df : pandas.DataFrame
        The spatial mapping data represented as a tabular pandas dataframe.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        SM = load('SM.ibw')

        # Get spatial mapping data in a tabular pandas dataframe format
        SM_df1 = SM.ML_pre_proc()

        # Get spatial mapping MDCs in a tabular pandas dataframe format
        SM_df2 = SM.ML_pre_proc(extract='MDC', E=73.42, dE=0.02)

    """

    # List to store spatial mapping data
    sample_data = []

    # Loop through spatial positions
    x1_vals = data.x1.data
    x2_vals = data.x2.data
    for i in range(len(x1_vals)):
        for j in range(len(x2_vals)):
            current_disp = data.isel(x1=i).isel(x2=j)

            # Apply normalisation at each spatial position if requested
            if norm:
                current_disp = current_disp.norm()

            # If we want to represent each feature to be a full dispersion
            if extract == "dispersion":
                # Get total number of dimensions (pixels) of dispersion
                dimensions = 1
                for coord_dim in current_disp.shape:
                    dimensions = dimensions * coord_dim

                # Reshape current dispersion into tabular form and append row to data
                reshaped_disp = current_disp.data.reshape(dimensions)
                sample_data.append(reshaped_disp)

            # If we want to represent each feature to be an MDC
            elif extract == "MDC":
                # Extract MDC from dispersion
                current_MDC = current_disp.MDC(E=E, dE=dE)
                sample_data.append(current_MDC.data)

            # If we want to represent each feature to be an EDC
            elif extract == "EDC":
                # Extract EDC from dispersion
                current_EDC = current_disp.EDC(k=k, dk=dk)
                sample_data.append(current_EDC.data)

            # Else user has entered an invalid method argument
            else:
                raise Exception("Method must be dispersion, MDC or EDC.")

    # Create tabular pandas dataframe
    df = pd.DataFrame(data=np.array(sample_data))

    if scale:
        # Apply optional standard scaling (center all values in each dimension around zero with unit standard deviation)
        std_scaler = StandardScaler()
        df = pd.DataFrame(data=std_scaler.fit_transform(df))

    return df


def perform_k_means(data, k=3, n_init="auto"):
    """Perform k-means clustering using scikit-learn.

    Parameters
    ------------
    data : pandas.DataFrame
        The data represented as a tabular pandas dataframe to perform clustering analysis on.

    k : int, optional
        Number of clusters. Defaults to 3.

    n_init : int, string, optional
        Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia. Required since the kmeans algorithm can fall into
        local minima, so repeats are required to check for this. Defaults to 'auto' (note: if an outdated scikit-learn
        package is installed where 'auto' is not yet implemented, an error will arise. In this case set n_init=10, or
        similar).

    Returns
    ------------
    model : sklearn.cluster._kmeans.KMeans
        K-means clustering analysis model information.

    labels : numpy.ndarray
        Array of spatially-dependent cluster assignments.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        SM = load('SM.ibw')

        # Get spatial mapping data in a tabular pandas dataframe format
        SM_df = SM.ML_pre_proc()

        # Perform k-means clustering analysis for 3 clusters
        model1, labels1 = perform_k_means(SM_df)

        # Perform k-means clustering analysis for 4 clusters
        model2, labels2 = perform_k_means(SM_df, k=4)

    """

    # Create a k-means clustering model
    model = KMeans(n_clusters=k, n_init=n_init)

    # Fit the model to the sampling data
    model.fit(data)

    # Predict the labels of the samples, i.e. which cluster they belong to
    labels = model.predict(data)

    return model, labels


@register_accessor(xr.DataArray)
def clusters_explore(
    data,
    cluster_range=None,
    n_init="auto",
    use_PCA=True,
    PCs=3,
    extract="dispersion",
    E=0,
    dE=0,
    k=0,
    dk=0,
    scale=False,
    norm=False,
):
    """Perform an exploratory k-means clustering analysis on a spatial map for a range of number of clusters.

    Parameters
    ------------
    data : xarray.DataArray
        The spatial mapping data to perform an exploratory k-means clustering analysis on.

    cluster_range : range, optional
        Range of number of clusters to perform k-means clustering analysis for. Defaults to range(1,7).

    n_init : int, string, optional
        Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia. Required since the kmeans algorithm can fall into
        local minima, so repeats are required to check for this. Defaults to 'auto' (note: if an outdated scikit-learn
        package is installed where 'auto' is not yet implemented, an error will arise. In this case set n_init=10, or
        similar).

    use_PCA : bool, optional
        Whether to apply a principal component analysis to the data. Defaults to True.

    PCs: int, optional
        Number of principal components. Defaults to 3.

    extract : str, optional
        Determines what is extracted from spatial mapping data. Defaults to dispersion. Valid entries are:
            dispersion
            MDC
            EDC
        Selecting MDC/EDC will rapidly increase calculation time.

    E : float, optional
        Energy of MDCs to extract. Defaults to 0.

    dE : float, optional
        MDC Integration range (represents the total range, i.e. integrates over +/- dE/2). Defaults to 0.

    k : float, optional
        k or theta_par value of EDCs to extract. Defaults to 0.

    dk : float, optional
        EDC integration range (represents the total range, i.e. integrates over +/- dk/2). Defaults to 0.

    scale : bool, optional
        Whether to apply standard scaling to data (i.e. center all values in each dimension around zero with unit
        standard deviation). Defaults to False.

    norm : bool, optional
        Whether to normalise the data at each spatial position. Defaults to False.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        SM = load('SM.ibw')

        # Perform an exploratory k-means clustering analysis for numbers of clusters ranging from 1 to 6, applying a
        # principal component analysis to the spatial mapping data using 3 principal components
        SM.clusters_explore()

        # Perform an exploratory k-means clustering analysis on spatial mapping MDCs for numbers of clusters ranging
        # from 1 to 10
        SM.clusters_explore(cluster_range=range(1,11), use_PCA=False, extract='MDC', E=73.42, dE=0.02)

    """
    if cluster_range is None:
        # Default range of number of clusters to test
        cluster_range = range(1, 7)

    # Prevent unwanted overwriting of original data
    data = data.copy(deep=True)

    # Represent spatial mapping data as a tabular pandas dataframe
    df = data.ML_pre_proc(
        extract=extract, E=E, dE=dE, k=k, dk=dk, scale=scale, norm=norm
    )

    # Perform principal component analysis if PCA is True
    if use_PCA:
        pca = PCA(n_components=PCs)  # Define PCA model
        principal_components = pca.fit_transform(
            df
        )  # Fit PCA model to data and get principal components
        df = pd.DataFrame(
            data=principal_components
        )  # Get principal components in tabular pandas dataframe format

    # Perform k-means clustering analysis for the range of number of clusters (k) requested
    k_titles = ["k=" + str(k) for k in cluster_range]  # Titles for plots
    inertias = []  # Empty list to store model inertias (a metric that defines spread of a cluster)
    classification_maps = []  # Empty list to store classification maps (spatial maps of cluster labels)
    for num_clusters in tqdm(cluster_range, desc="Calculating", colour="CYAN"):
        model, labels = perform_k_means(
            data=df, k=num_clusters, n_init=n_init
        )  # Perform k-means clustering
        inertias.append(model.inertia_)
        classification_map_data = labels.reshape(
            len(data.x1), len(data.x2)
        )  # Reshape 1D labels to 2D
        classification_map = xr.DataArray(
            classification_map_data,
            dims=("x1", "x2"),
            coords={"x1": data.x1, "x2": data.x2},
        )
        classification_maps.append(classification_map)

    inertias_xarray = xr.DataArray(
        inertias, dims="num_clusters", coords={"num_clusters": cluster_range}
    )  # Create xarray of inertias

    # Find optimal number of clusters
    for item in abs(inertias_xarray.differentiate("num_clusters")).norm():
        # If rate of decrease in inertia is below 20% of the initial decrease, found optimal number of clusters
        if float(item) < 0.2:
            recommended_num_clusters = int(item.num_clusters)
            break
        # Else continue through loop
        else:
            recommended_num_clusters = int(item.num_clusters)

    # Remove progress bar
    clear_output()

    # Plot model inertia against number of clusters, and indicate recommended number of clusters
    inertias_xarray.plot(marker="o", figsize=(15, 4))
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.axvline(recommended_num_clusters, c="black", linestyle="--")
    plt.title(r"Optimal k$\approx$" + str(recommended_num_clusters))
    plt.show()

    # Plot classification map dependence on number of clusters
    plot_grid(classification_maps, ncols=3, titles=k_titles, cmap="cividis", y="x2")
    plt.show()


@register_accessor(xr.DataArray)
def clusters(
    data,
    num_clusters=3,
    n_init="auto",
    use_PCA=True,
    PCs=3,
    extract="dispersion",
    E=0,
    dE=0,
    k=0,
    dk=0,
    scale=False,
    norm=False,
    robust=False,
    vmin=None,
    vmax=None,
):
    """Perform a k-means clustering analysis on a spatial map.

    Parameters
    ------------
    data : xarray.DataArray
        The spatial mapping data to perform an exploratory k-means clustering analysis on.

    num_clusters : int, optional
        Number of clusters to perform k-means clustering analysis for. Defaults to 3.

    n_init : int, string, optional
        Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia. Required since the kmeans algorithm can fall into
        local minima, so repeats are required to check for this. Defaults to 'auto' (note: if an outdated scikit-learn
        package is installed where 'auto' is not yet implemented, an error will arise. In this case set n_init=10, or
        similar).

    use_PCA : bool, optional
        Whether to apply a principal component analysis to the data. Defaults to True.

    PCs: int, optional
        Number of principal components. Defaults to 3.

    extract : str, optional
        Determines what is extracted from spatial mapping data. Defaults to dispersion. Valid entries are:
            dispersion
            MDC
            EDC
        Selecting MDC/EDC will rapidly increase calculation time.

    E : float, optional
        Energy of MDCs to extract. Defaults to 0.

    dE : float, optional
        MDC Integration range (represents the total range, i.e. integrates over +/- dE/2). Defaults to 0.

    k : float, optional
        k or theta_par value of EDCs to extract. Defaults to 0.

    dk : float, optional
        EDC integration range (represents the total range, i.e. integrates over +/- dk/2). Defaults to 0.

    scale : bool, optional
        Whether to apply standard scaling to data (i.e. center all values in each dimension around zero with unit
        standard deviation). Defaults to False.

    norm : bool, optional
        Whether to normalise the data at each spatial position. Defaults to False.

    robust : bool, optional
        Whether the argument robust=True is passed to the plots. Defaults to False.

    vmin : float, optional
        Matplotlib vmin value used in plots of dispersions. Defaults to None.

    vmax : float, optional
        Matplotlib vmax value used in plots of dispersions. Defaults to None.

    Returns
    ------------
    classification_map : xarray.DataArray
        Spatial map of cluster labels.

    cluster_center_disps : list
        Dispersions for each cluster center.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        SM = load('SM.ibw')

        # Perform a k-means clustering analysis using 3 clusters, applying a principal component analysis to the spatial
        # mapping data using 4 principal components
        cluster_map1, cluster_disps1 = SM.clusters(num_clusters=3, PCs=4)

        # Perform a k-means clustering analysis using 3 clusters on spatial mapping MDCs
        cluster_map2, cluster_disps2 = SM.clusters(num_clusters=3, use_PCA=False, extract='MDC', E=73.42, dE=0.02)

    """

    # Prevent unwanted overwriting of original data
    data = data.copy(deep=True)

    # Represent spatial mapping data as a tabular pandas dataframe
    df = data.ML_pre_proc(
        extract=extract, E=E, dE=dE, k=k, dk=dk, scale=scale, norm=norm
    )

    # Perform principal component analysis if PCA is True
    if use_PCA:
        pca = PCA(n_components=PCs)  # Define PCA model
        principal_components = pca.fit_transform(
            df
        )  # Fit PCA model to data and get principal components
        df = pd.DataFrame(
            data=principal_components
        )  # Get principal components in tabular pandas dataframe format

    # Perform k-means clustering analysis for the number of clusters (k) requested
    model, labels = perform_k_means(data=df, k=num_clusters, n_init=n_init)
    classification_map_data = labels.reshape(
        len(data.x1), len(data.x2)
    )  # Reshape 1D labels to 2D
    classification_map = xr.DataArray(
        classification_map_data,
        dims=("x1", "x2"),
        coords={"x1": data.x1, "x2": data.x2},
    )

    # Extract cluster center of each cluster (not done using cluster centers so that we can get dispersions if MDC/EDC
    # extraction is used)
    cluster_average_disps = []  # Empty list to store average dispersions
    # Loop through cluster labels
    for cluster in range(num_clusters):
        current_disps = []
        # Loop through spatial positions
        for i in range(len(data.x1)):
            for j in range(len(data.x2)):
                # If current spatial position is assigned to current cluster
                if float(classification_map.isel(x1=i).isel(x2=j)) == cluster:
                    current_disps.append(data.isel(x1=i).isel(x2=j).copy(deep=True))
        # Sum dispersions with the same cluster label
        total_disp = current_disps[0]
        for i in range(1, len(current_disps)):
            total_disp += current_disps[i]
        cluster_average_disps.append(total_disp / len(current_disps))

    # Find maximum vmax of the averaged cluster dispersions so intensity variations between terminations can be observed
    if not robust and not vmax:
        max_cluster_disps_vmax = 0
        for disp in cluster_average_disps:
            if float(disp.max()) > max_cluster_disps_vmax:
                max_cluster_disps_vmax = float(disp.max())

    # Plot integrated spectral weight and k-means clustering labels spatial maps
    fig, axes = plt.subplots(figsize=(12, 5), ncols=2)
    data.tot().plot(
        ax=axes[0], cmap="cividis", y="x2", cbar_kwargs={"label": "Intensity"}
    )
    classification_map.plot(
        ax=axes[1],
        cmap="cividis",
        y="x2",
        cbar_kwargs={"ticks": range(num_clusters), "label": "Cluster"},
    )
    axes[0].set_title("Integrated spectral weight")
    axes[1].set_title("k-means clustering (k=" + str(num_clusters) + ")")
    plt.tight_layout()
    plt.show()

    # Plot average dispersion of each cluster
    cluster_average_titles = [
        "Cluster " + str(i) for i in range(num_clusters)
    ]  # Titles for plots
    if vmax:
        plot_grid(
            cluster_average_disps,
            titles=cluster_average_titles,
            cmap="binary",
            y="eV",
            ncols=num_clusters,
            vmin=vmin,
            vmax=vmax,
            cbar_kwargs={"label": None},
        )
    else:
        if not robust:
            plot_grid(
                cluster_average_disps,
                titles=cluster_average_titles,
                cmap="binary",
                y="eV",
                ncols=num_clusters,
                vmin=vmin,
                vmax=max_cluster_disps_vmax,
                cbar_kwargs={"label": None},
            )
        else:
            plot_grid(
                cluster_average_disps,
                titles=cluster_average_titles,
                cmap="binary",
                y="eV",
                ncols=num_clusters,
                robust=True,
                cbar_kwargs={"label": None},
            )
    plt.tight_layout()
    plt.show()

    # If a PCA was used on full dispersions, plot also the reconstructed cluster centers
    if use_PCA and extract == "dispersion":
        # Get non-spatial coordinates
        coords = list(data.dims)
        coords.remove("x1")
        coords.remove("x2")

        # Get reconstructed cluster center dispersions from reduced dimensionality dataset
        reconstructed_cluster_centers = pca.inverse_transform(
            model.cluster_centers_
        ).reshape(num_clusters, len(data.coords[coords[0]]), len(data.coords[coords[1]]))
        reconstructed_cluster_centers_disps = []
        for center in reconstructed_cluster_centers:
            disp_xarray = xr.DataArray(
                center,
                dims=(coords[0], coords[1]),
                coords={
                    coords[0]: data.coords[coords[0]],
                    coords[1]: data.coords[coords[1]],
                },
            )
            reconstructed_cluster_centers_disps.append(disp_xarray)

        # Find maximum vmax of the reconstructed cluster center dispersions so intensity variations between
        # terminations can be observed
        if not robust and not vmax:
            max_reconstructed_cluster_disps_vmax = 0
            for disp in reconstructed_cluster_centers_disps:
                if float(disp.max()) > max_reconstructed_cluster_disps_vmax:
                    max_reconstructed_cluster_disps_vmax = float(disp.max())

        # reconstructed cluster center dispersions
        reconstructed_cluster_centers_titles = [
            "Cluster " + str(i) + " (reconstructed)" for i in range(num_clusters)
        ]  # Titles for plots
        if vmax:
            plot_grid(
                reconstructed_cluster_centers_disps,
                titles=reconstructed_cluster_centers_titles,
                cmap="binary",
                y="eV",
                ncols=num_clusters,
                vmin=vmin,
                vmax=vmax,
                cbar_kwargs={"label": None},
            )
        else:
            if not robust:
                plot_grid(
                    reconstructed_cluster_centers_disps,
                    titles=reconstructed_cluster_centers_titles,
                    cmap="binary",
                    y="eV",
                    ncols=num_clusters,
                    vmin=vmin,
                    vmax=max_reconstructed_cluster_disps_vmax,
                    cbar_kwargs={"label": None},
                )
            else:
                plot_grid(
                    reconstructed_cluster_centers_disps,
                    titles=reconstructed_cluster_centers_titles,
                    cmap="binary",
                    y="eV",
                    ncols=num_clusters,
                    robust=True,
                    cbar_kwargs={"label": None},
                )
        plt.tight_layout()
        plt.show()

    return classification_map, cluster_average_disps


@register_accessor(xr.DataArray)
def PCA_explore(
    data,
    PCs_range=None,
    threshold=0.95,
    extract="dispersion",
    E=0,
    dE=0,
    k=0,
    dk=0,
    scale=False,
    norm=False,
):
    """Perform an exploratory principal component analysis on a spatial map for a range of principal components.

    Parameters
    ------------
    data : xarray.DataArray
        The spatial mapping data to perform an exploratory principal component analysis on.

    PCs_range : range, optional
        Range of number of principal components to perform PCA for. Defaults to range(1,6).

    threshold : float, optional
        Required threshold for the explained variance fraction of the dimensionally reduced dataset. Defaults to 0.95.

    extract : str, optional
        Determines what is extracted from spatial mapping data. Defaults to dispersion. Valid entries are:
            dispersion
            MDC
            EDC
        Selecting MDC/EDC will rapidly increase calculation time.

    E : float, optional
        Energy of MDCs to extract. Defaults to 0.

    dE : float, optional
        MDC Integration range (represents the total range, i.e. integrates over +/- dE/2). Defaults to 0.

    k : float, optional
        k or theta_par value of EDCs to extract. Defaults to 0.

    dk : float, optional
        EDC integration range (represents the total range, i.e. integrates over +/- dk/2). Defaults to 0.

    scale : bool, optional
        Whether to apply standard scaling to data (i.e. center all values in each dimension around zero with unit
        standard deviation). Defaults to False.

    norm : bool, optional
        Whether to normalise the data at each spatial position. Defaults to False.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        SM = load('SM.ibw')

        # Perform an exploratory PCA on spatial mapping data for numbers of principal components ranging from 1 to 5
        SM.PCA_explore()

        # Perform an exploratory PCA on spatial mapping MDCs for numbers of principal components ranging from 1 to 10
        SM.PCA_explore(PCs_range=range(1,11), extract='MDC', E=73.42, dE=0.02)

    """
    if PCs_range is None:
        # Default range of principal components to test
        PCs_range = range(1, 6)

    # Prevent unwanted overwriting of original data
    data = data.copy(deep=True)

    # Represent spatial mapping data as a tabular pandas dataframes
    df = data.ML_pre_proc(
        extract=extract, E=E, dE=dE, k=k, dk=dk, scale=scale, norm=norm
    )

    summed_var_ratio = []  # Empty list to store explained variance fractions

    # Loop through the range of principal components to test, and perform PCA at each
    for num_principal_components in tqdm(PCs_range, desc="Calculating", colour="CYAN"):
        model = PCA(n_components=num_principal_components)  # Define PCA model
        model.fit(df)  # Fit PCA model to dataset
        summed_var_ratio.append(
            np.sum(model.explained_variance_ratio_)
        )  # Append the total explained variance fractions of each model
    clear_output()  # Remove progress bar

    # Check minimum number of principal axes required to exceed explained variance fraction threshold
    num_principal_axes = "Unknown"
    for i, item in enumerate(summed_var_ratio):
        if item >= threshold:
            num_principal_axes = PCs_range[i]
            break

    # If explained variance could not exceed threshold, display a warning
    if num_principal_axes == "Unknown":
        analysis_warning(
            "Explained variance could not exceed threshold. Either reduce threshold or "
            "increase PCA_range.",
            title="Analysis info",
            warn_type="danger",
        )

    # Plot results of exploratory PCA, and suggest minimum required number of principal axes (if the explained
    # variance threshold has been exceeded)
    plt.figure(figsize=(8, 5))
    plt.plot(PCs_range, summed_var_ratio, "o-")
    plt.axhline(threshold, c="black", linestyle="--")
    if num_principal_axes != "Unknown":
        plt.axvline(num_principal_axes, c="black", linestyle="--")
    plt.xlabel("Principal axes")
    plt.ylabel("Explained variance fraction")
    plt.title(r"Minimum number of required principal axes$=$" + str(num_principal_axes))
    plt.show()
