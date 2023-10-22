# Functions for automated cluster analysis
# PK 29/4/22

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans
from sklearn.decomposition import PCA

from peaks.core.utils import update_hist
from peaks.core.utils.OOP_method import add_methods

@add_methods(xr.DataArray)
def clustering_pre_proc(data):
    '''Pre-process map data into correct format for k-means and fuzzy c-means clustering analysis. Performing
    this as a pre-step provides a speed up for multiple iterations of the clustering analysis. Alternatively,
    if the raw map data is passed to the clustering analysis, this function is then run from there.

    Input:
            data - Spatial map data in raw format (xr.DataArray)

    Return:
            Dataset containing integrated MDC and integrated EDC stacked into a 3D output '''

    # Restack data into appropriate form for k-means algorithm
    iMDC = data.mean('eV', keep_attrs=True).stack(z=("x1", "x2"))  # Integrated MDC
    ang_dim = list(filter(lambda n: n != 'x1' and n != 'x2' and n != 'eV', data.dims))[0]  # Extract relevant angular dimension
    iEDC = data.mean(ang_dim, keep_attrs=True).stack(z=("x1", "x2"))  # Integrated EDC

    stacked_ds = xr.Dataset(dict(iMDC=iMDC, iEDC=iEDC))
    stacked_ds.attrs = data.attrs
    stacked_ds.iEDC.attrs['ang_dim'] = ang_dim
    if data.dims.index('x1') > data.dims.index('x2'):
        stacked_ds.iEDC.attrs['x2x1'] = True
        stacked_ds.iMDC.attrs['x2x1'] = True

    return stacked_ds



@add_methods(xr.DataArray)
@add_methods(xr.Dataset)
def k_means(data, nk=5, n_init=20, random_state=500, plot=True, **kwargs):
    '''k-means clustering analysis of spatial map data, as implemented within sklearn.
     The analysis is performed on integrated MDCs and EDCs of a passed spatial map DataArray.

        Input:
            data - Spatial map data in raw format (xr.DataArray)
                 - Alternatively pass an xr.Dataset with data pre-processed using clustering_pre_proc
                    (speed-up for multiple runs)
            nk - the number of clusters to form (int, default=5)
            n_init - number of time the k-means algorithm will be run with different centroid seeds
             (int, default=20)
            random_state - Determines random number generation for centroid initialization. Use an int to
             make the randomness deterministic.
            plot - whether to return a plot of the spatial map cluster analysis (bool, default: True)
            plot_kwargs - matplotlib parameters to pass to plotting (dict)
            kwargs - additional parameters to pass to the KMeans algorithm defined at sklearn.cluster.KMeans

        Returns:
            km_ds - dataset containing integrated MDC and integrated EDC cluster analysis (xr.Dataset)'''

    # Pull out plotting kwargs
    if 'plot_kwargs' in kwargs:
        plot_kwargs = kwargs.pop('plot_kwargs')
    else:
        plot_kwargs = {}

    try:  # Check if pre-processed data has been provided
        iMDC = data.iMDC
        iEDC = data.iEDC
        ang_dim = data.iEDC.ang_dim  # Relevant angular dimension
    except:  # Else raw map provided so process from there
        data_stack = clustering_pre_proc(data)
        iMDC = data_stack.iMDC
        iEDC = data_stack.iEDC
        ang_dim = data_stack.iEDC.ang_dim  # Relevant angular dimension

    # Perform the k-means clustering
    km = KMeans(n_clusters=nk, n_init=n_init, random_state=random_state, **kwargs)
    km_model_iMDC = km.fit(iMDC.data.T)
    km_results_iMDC = km_model_iMDC.labels_
    km_model_iEDC = km.fit(iEDC.data.T)
    km_results_iEDC = km_model_iEDC.labels_

    # Put km clusters data into DataArrays
    km_iMDC = xr.DataArray(km_results_iMDC,
                           dims = "z",
                           coords = {"z": iMDC.z})
    km_iMDC = km_iMDC.unstack()  # Unstack data into original form
    km_iMDC.attrs['nk'] = nk
    km_iEDC = xr.DataArray(km_results_iEDC,
                           dims="z",
                           coords={"z": iEDC.z})
    km_iEDC.data = km_results_iEDC
    km_iEDC = km_iEDC.unstack()  # Unstack data into original form
    km_iEDC.attrs['nk'] = nk

    # Put them into the same ordering as in the original wave
    if iEDC.attrs['x2x1']:
        km_iMDC = km_iMDC.transpose('x2', 'x1')
        km_iEDC = km_iEDC.transpose('x2', 'x1')

    # Combine both into a single dataset
    km_ds = xr.Dataset(dict(iMDC=km_iMDC, iEDC=km_iEDC))
    km_ds.attrs = data.attrs

    if plot == True:
        # Plot the maps
        if 'figsize' in plot_kwargs:
            figsize = plot_kwargs.pop('figsize')
        else:
            figsize = (10, 5)
        if 'cmap' not in plot_kwargs:
            plot_kwargs['cmap'] =  'Accent'

        fix, ax = plt.subplots(ncols=2, figsize=figsize)
        km_iMDC.plot(ax=ax[0], vmin=-0.5, vmax=nk - 0.5, levels=nk + 1,
                     cbar_kwargs={'ticks': np.linspace(0, nk - 1, nk)}, **plot_kwargs)
        ax[0].set_title('Integrated MDC k-means analysis')
        km_iEDC.plot(ax=ax[1], vmin=-0.5, vmax=nk - 0.5, levels=nk + 1,
                     cbar_kwargs={'ticks': np.linspace(0, nk - 1, nk)}, **plot_kwargs)
        ax[1].set_title('Integrated EDC k-means analysis')

        plt.tight_layout()

    # Update history string
    hist = 'k-means clustering analysis applied to data with parameters: nk=' + str(nk) + ', n_init=' + str(n_init) + ', random_state=' + str(random_state) + ', ' + str(kwargs)
    hist += 'Energy range of data set used for integrated DCs: E=' + str(np.round(data.eV.min().data,3)) +' - ' + str(np.round(data.eV.max().data,3)) + ', '
    hist += str(ang_dim) + '=' + str(np.round(data[ang_dim].min().data,3)) +' - ' + str(np.round(data[ang_dim].max().data,3))

    # Update history
    update_hist(km_ds, hist)

    return km_ds

@add_methods(xr.DataArray)
@add_methods(xr.Dataset)
def fc_means(data, nk=5, m=1.5, threshold=0.95, plot=True, **kwargs):
    '''Fuzzy c-means clustering analysis of spatial map data, as implemented within skfuzzy.cluster.cmeans.
         The analysis is performed on integrated MDCs and EDCs of a passed spatial map DataArray.

            Input:
                data - Spatial map data in raw format (xr.DataArray)
                     - Alternatively pass an xr.Dataset with data pre-processed using clustering_pre_proc
                        (speed-up for multiple runs)
                nk - the number of clusters to form (int, default=5)
                m - fuzzifier (float, default=1.5)
                threshold - probability threshold for assigning a pixel to a particular cluster
                plot - whether to return a plot of the spatial map cluster analysis (bool, default: True)
                plot_kwargs - matplotlib parameters to pass to plotting of the collapsed cluster maps (dict)
                plot2_kwargs - matplotlib parameters to pass to plotting of the probability cluster maps (dict)
                kwargs - additional parameters to pass to the cmeans algorithm defined at skfuzzy.cluster.cmeans

            Returns:
                km_ds - dataset containing integrated MDC and integrated EDC cluster analysis (xr.Dataset)'''

    # Pull out plotting kwargs
    if 'plot_kwargs' in kwargs:
        plot_kwargs = kwargs.pop('plot_kwargs')
    else:
        plot_kwargs = {}
    if 'plot2_kwargs' in kwargs:
        plot2_kwargs = kwargs.pop('plot2_kwargs')
    else:
        plot2_kwargs = {}

    try:  # Check if pre-processed data has been provided
        iMDC = data.iMDC
        iEDC = data.iEDC
        ang_dim = data.iEDC.ang_dim  # Relevant angular dimension
    except:  # Else raw map provided so process from there
        data_stack = clustering_pre_proc(data)
        iMDC = data_stack.iMDC
        iEDC = data_stack.iEDC
        ang_dim = data_stack.iEDC.ang_dim  # Relevant angular dimension

    # Set some default parameters if not provided in the kwargs
    if 'error' not in kwargs:
        kwargs['error'] = 0.003
    if 'maxiter' not in kwargs:
        kwargs['maxiter'] = 10000
    if 'seed' not in kwargs:
        kwargs['seed'] = 10

    # iMDC analysis
    cntr, u, u0, d, jm, p, fpc = cmeans(iMDC.data, nk, m, **kwargs)

    # Put resulting c-partiaioned matrix into DataArray
    fcm_iMDC = xr.DataArray(u,
                            dims=("cluster", "z"),
                            coords={"cluster": np.arange(0, nk, 1), "z": iMDC.z})

    fcm_iMDC = fcm_iMDC.unstack()  # Unstack data into original form
    fcm_iMDC.attrs['nk'] = nk
    fcm_iMDC.attrs['m'] = m

    # iEDC analysis
    cntr, u, u0, d, jm, p, fpc = cmeans(iEDC.data, nk, m, **kwargs)

    # Put resulting c-partiaioned matrix into DataArray
    fcm_iEDC = xr.DataArray(u,
                            dims=("cluster", "z"),
                            coords={"cluster": np.arange(0, nk, 1), "z": iEDC.z})

    fcm_iEDC = fcm_iMDC.unstack()  # Unstack data into original form
    fcm_iEDC.attrs['nk'] = nk
    fcm_iEDC.attrs['m'] = m

    # Put them into the same ordering as in the original wave
    if iEDC.attrs['x2x1']:
        fcm_iMDC = fcm_iMDC.transpose('cluster', 'x2', 'x1')
        fcm_iEDC = fcm_iEDC.transpose('cluster', 'x2', 'x1')

    # Make a best estimate cluster attribution based on the probabilities - assign to cluster i when p>threshold
    # Define unity da on same axis as map
    da = xr.DataArray(
        np.ones(fcm_iMDC.isel(cluster=0).shape),
        dims=fcm_iMDC.isel(cluster=0).dims,
        coords=fcm_iMDC.isel(cluster=0).coords)

    # Make da's to hold the cluster assignment
    cluster_iMDC = fcm_iMDC.copy(deep=True)
    cluster_iEDC = cluster_iMDC.copy(deep=True)

    # Iterate through the clusters and make assignments
    for i in fcm_iMDC.cluster.data:
        cluster_iMDC.loc[{'cluster': i}] = da.where(fcm_iMDC.isel(cluster=i) > threshold) * (i + 1)
        cluster_iEDC.loc[{'cluster': i}] = da.where(fcm_iEDC.isel(cluster=i) > threshold) * (i + 1)

    # Collapse cluster assignments onto a single map
    cluster_iMDC_tot = cluster_iMDC.sum(dim='cluster') - 1  # Make cluster numbers match above ones
    cluster_iMDC_tot = cluster_iMDC_tot.where(cluster_iMDC_tot != -1)  # Replace -1 values with NaNs
    cluster_iMDC_tot.attrs['nk'] = nk
    cluster_iMDC_tot.attrs['m'] = m

    cluster_iEDC_tot = cluster_iEDC.sum(dim='cluster') - 1  # Make cluster numbers match above ones
    cluster_iEDC_tot = cluster_iEDC_tot.where(cluster_iEDC_tot != -1)  # Replace -1 values with NaNs
    cluster_iEDC_tot.attrs['nk'] = nk
    cluster_iEDC_tot.attrs['m'] = m

    # Combine all into a single dataset
    fcm_ds = xr.Dataset(dict(iMDC_prob=fcm_iMDC, iEDC_prob=fcm_iEDC, iMDC=cluster_iMDC_tot, iEDC=cluster_iEDC_tot))
    fcm_ds.attrs = data.attrs

    if plot == True:
        # Plot the collapsed maps
        if 'figsize' in plot_kwargs:
            figsize = plot_kwargs.pop('figsize')
        else:
            figsize = (10, 5)
        if 'cmap' not in plot_kwargs:
            plot_kwargs['cmap'] =  'Accent'

        fix, ax = plt.subplots(ncols=2, figsize=figsize)
        cluster_iMDC_tot.plot(ax=ax[0], vmin=-0.5, vmax=nk - 0.5, levels=nk + 1,
                              cbar_kwargs={'ticks': np.linspace(0, nk - 1, nk)}, **plot_kwargs)
        ax[0].set_title('Integrated MDC fuzzy-c-means analysis')
        cluster_iEDC_tot.plot(ax=ax[1], vmin=-0.5, vmax=nk - 0.5, levels=nk + 1,
                              cbar_kwargs={'ticks': np.linspace(0, nk - 1, nk)}, **plot_kwargs)
        ax[1].set_title('Integrated EDC fuzzy-c-means analysis')

        plt.tight_layout()
        plt.show()

        # Plot the probability maps
        if 'col_wrap' not in plot2_kwargs:
            plot2_kwargs['col_wrap'] = 5

        print('MDC cluster probabilities:')
        fcm_iMDC.plot(col="cluster", **plot2_kwargs)
        plt.show()
        print('EDC cluster probabilities:')
        fcm_iEDC.plot(col="cluster", **plot2_kwargs)

    return fcm_ds


@add_methods(xr.DataArray)
@add_methods(xr.Dataset)
def pca_map(data, nPC=5, plot=True, **kwargs):
    '''Pricipal component decomposition of spatial map data, as implemented within sklearn.decomposition.PCA.
         The analysis is performed on integrated MDCs and EDCs of a passed spatial map DataArray.

            Input:
                data - Spatial map data in raw format (xr.DataArray)
                     - Alternatively pass an xr.Dataset with data pre-processed using clustering_pre_proc
                        (speed-up for multiple runs)
                nPC - the number of princial components (int, default=8)
                plot - whether to return a plot of the PC data (bool, default: True)
                plot_kwargs - matplotlib parameters to pass to plotting of the PC maps (dict)
                kwargs - additional parameters to pass to the PCA algorithm defined at sklearn.decomposition.PCA

            Returns:
                pca - dataset containing principlal components and their spatial distribution (xr.Dataset)'''

    # Pull out plotting kwargs
    if 'plot_kwargs' in kwargs:
        plot_kwargs = kwargs.pop('plot_kwargs')
    else:
        plot_kwargs = {}

    try:  # Check if pre-processed data has been provided
        iMDC = data.iMDC
        iEDC = data.iEDC
        ang_dim = data.iEDC.ang_dim  # Relevant angular dimension
    except:  # Else raw map provided so process from there
        data_stack = clustering_pre_proc(data)
        iMDC = data_stack.iMDC
        iEDC = data_stack.iEDC
        ang_dim = data_stack.iEDC.ang_dim  # Relevant angular dimension

    # iMDC PCA analysis
    pca_MDC = PCA(n_components=nPC)
    H_MDC = pca_MDC.fit_transform(iMDC.data)
    components_MDC = pca_MDC.components_.T
    PC_coords = np.arange(1, nPC +1, 1)

    PC_iMDC = xr.DataArray(
        H_MDC,
        dims = [ang_dim, 'PC'],
        coords = {ang_dim: iMDC[ang_dim].data, 'PC': PC_coords},
        )
    PC_iMDC_map = xr.DataArray(components_MDC,
                   dims = ['z','PC'],
                   coords = {'z':iMDC.z, 'PC': PC_coords}
        )
    PC_iMDC_map = PC_iMDC_map.unstack()

    # Contribution Ratio & Cumulative Contribution Ratio (CR & CCR)
    cr = pca_MDC.explained_variance_ratio_
    ccr = np.cumsum(cr)
    cr_iMDC = xr.DataArray(
        cr,
        dims=['PC'],
        coords={'PC': PC_coords}
    )
    ccr_iMDC = xr.DataArray(
        ccr,
        dims=['PC'],
        coords={'PC': PC_coords}
    )

    # iEDC PCA analysis
    pca_EDC = PCA(n_components=nPC)
    H_EDC = pca_EDC.fit_transform(iEDC.data)
    components_EDC = pca_EDC.components_.T

    PC_iEDC = xr.DataArray(
        H_EDC,
        dims=['eV', 'PC'],
        coords={'eV': iEDC.eV.data, 'PC': PC_coords}
    )
    PC_iEDC_map = xr.DataArray(components_EDC,
                              dims=['z', 'PC'],
                              coords={'z': iEDC.z, 'PC': PC_coords}
                               )
    PC_iEDC_map = PC_iEDC_map.unstack()

    # Contribution Ratio & Cumulative Contribution Ratio (CR & CCR)
    cr = pca_EDC.explained_variance_ratio_
    ccr = np.cumsum(cr)
    cr_iEDC = xr.DataArray(
        cr,
        dims=['PC'],
        coords={'PC': PC_coords}
    )
    ccr_iEDC = xr.DataArray(
        ccr,
        dims=['PC'],
        coords={'PC': PC_coords}
    )

    # Put maps into the same ordering as in the original da
    if iEDC.attrs['x2x1']:
        PC_iMDC_map = PC_iMDC_map.transpose('PC', 'x2', 'x1')
        PC_iEDC_map = PC_iEDC_map.transpose('PC', 'x2', 'x1')

    # Combine all into a single dataset
    pca_ds = xr.Dataset(dict(pc_iMDC=PC_iMDC, pc_iEDC=PC_iEDC, iMDC=PC_iMDC_map, iEDC=PC_iEDC_map, cr_iMDC=cr_iMDC, ccr_iMDC=ccr_iMDC, cr_iEDC=cr_iEDC, ccr_iEDC=ccr_iEDC))
    pca_ds.attrs = data.attrs

    if plot == True:
        # Plot the data
        # MDC decomposition
        print('Principal component decomposition from iMDCs:')
        fig, ax = plt.subplots(ncols=2, figsize=(10,5))
        #CR/CCR
        ax[0].bar(cr_iMDC.PC.data, cr_iMDC.data, align='center', width=1, color='thistle', edgecolor='purple')
        ccr_iMDC.plot(ax=ax[0], marker='o')
        ax[0].set_ylabel('CR/CCR')

        #PCs
        PC_iMDC.plot.line(x=ang_dim, ax=ax[1])
        plt.tight_layout()
        plt.show()

        #iMDC PC maps (spatial distribution of PCs)
        if 'col_wrap' not in plot_kwargs:
            plot_kwargs['col_wrap'] = 5
        PC_iMDC_map.plot(col='PC', **plot_kwargs)
        plt.show()

        #Score plots (plot PCs vs each other)
        fig, ax = plt.subplots(ncols=3, figsize=(12, 3.5))
        ax[0].scatter(PC_iMDC_map.sel(PC=1).data, PC_iMDC_map.sel(PC=2).data)
        ax[0].set_xlabel('PC1')
        ax[0].set_ylabel('PC2')
        ax[1].scatter(PC_iMDC_map.sel(PC=1).data, PC_iMDC_map.sel(PC=3).data)
        ax[1].set_xlabel('PC1')
        ax[1].set_ylabel('PC3')
        ax[2].scatter(PC_iMDC_map.sel(PC=2).data, PC_iMDC_map.sel(PC=3).data)
        ax[2].set_xlabel('PC2')
        ax[2].set_ylabel('PC3')

        plt.tight_layout()
        plt.show()

        print('Principal component decomposition from iEDCs:')
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        # CR/CCR
        ax[0].bar(cr_iEDC.PC.data, cr_iEDC.data, align='center', width=1, color='thistle', edgecolor='purple')
        ccr_iEDC.plot(ax=ax[0], marker='o')
        ax[0].set_ylabel('CR/CCR')

        # PCs
        PC_iEDC.plot.line(x='eV', ax=ax[1])

        plt.tight_layout()
        plt.show()

        # iEDC PC maps
        if 'col_wrap' not in plot_kwargs:
            plot_kwargs['col_wrap'] = 5
        PC_iEDC_map.plot(col='PC', **plot_kwargs)
        plt.show()

        # Score plots (plot PCs vs each other)
        fig, ax = plt.subplots(ncols=3, figsize=(12, 3.5))
        ax[0].scatter(PC_iEDC_map.sel(PC=1).data, PC_iEDC_map.sel(PC=2).data)
        ax[0].set_xlabel('PC1')
        ax[0].set_ylabel('PC2')
        ax[1].scatter(PC_iEDC_map.sel(PC=1).data, PC_iEDC_map.sel(PC=3).data)
        ax[1].set_xlabel('PC1')
        ax[1].set_ylabel('PC3')
        ax[2].scatter(PC_iEDC_map.sel(PC=2).data, PC_iEDC_map.sel(PC=3).data)
        ax[2].set_xlabel('PC2')
        ax[2].set_ylabel('PC3')

        plt.tight_layout()
        plt.show()

    return pca_ds

@add_methods(xr.DataArray)
def cluster_extract(data, cluster_map, cluster_no='all', output='disp'):
    '''Tool for extracting data masked by cluster number based on automated cluster analysis methods.

            Input:
                data - Original spatial map data (xr.DataArray)
                cluster_map - result of cluster analysis, spatial map where each pixel is assigned a cluster
                 number (xr.DataArray)
                cluster_no - definition of which clusters to project data onto:
                    'all' - all of them (default)
                    int - specific number
                    list or array - list of desired clusters
                output - what integrations are desired for output data:
                    'disp' - dispersions are returned (default)
                    'map' - spatial maps are returned
                    'all' - full 4d array returned projected onto cluster

            Returns:
                data_out - results of cluster analysis projections (xr.DataArray) '''

    # Put desired clusters to extract into a list or array format
    if cluster_no == 'all':
        cluster_no = np.arange(0, cluster_map.nk, 1)
    elif type(cluster_no) is int:
        cluster_no = [cluster_no]

    # Number of clusters to extract
    nk = len(cluster_no)

    # Setup for what output is required
    if output == 'disp':  # Dispersion
        coords_int = ['x1', 'x2']
    elif output == 'map':  # Spatial map
        coords_int = ['eV', 'theta_par']
    else:  # Otherwise return the full array
        coords_int = None

    if coords_int is not None:
        # Mask the data for the first cluster and integrate over relevant dimensions to get the relevant output
        data_out = data.where(cluster_map == cluster_no[0]).mean(coords_int, keep_attrs=True)

        # Make a new cluster dimension
        data_out = data_out.expand_dims({'cluster': nk}).assign_coords(
            {'cluster': np.asarray(cluster_no).astype(int)}).copy(deep=True)

        if nk != 1:  # Fill in data for the other clusters required
            for i in cluster_no[1::]:
                data_out.loc[{'cluster': i}] = data.where(cluster_map == i).mean(coords_int, keep_attrs=True)

    else:
        print(1)
        # Make output data
        data_out = data.expand_dims({'cluster': nk}).assign_coords(
            {'cluster': np.asarray(cluster_no).astype(int)}).copy(deep=True)

        # Perform the masing
        for i in cluster_no:
            data_out.loc[{'cluster': i}] = data.where(cluster_map == i)

    return data_out

@add_methods(xr.DataArray)
def cluster_plot(cluster_map, **kwargs):
    '''Helper function for plotting cluster map with some sensible choices for the colour bar.

            Input:
                data - result of cluster analysis, spatial map where each pixel is assigned a cluster
                 number (xr.DataArray)
                kwargs - matplotlib parameters to pass to plotting

            Returns:
                plot '''

    nk = cluster_map.nk

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Accent'

    cluster_map.plot(vmin=-0.5, vmax=nk - 0.5, levels=nk + 1, cbar_kwargs={'ticks': np.linspace(0, nk - 1, nk)}, **kwargs)

