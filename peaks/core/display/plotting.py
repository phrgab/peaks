"""Static in-line plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import pint_xarray  # noqa: F401
import xarray as xr
from cycler import cycler
from IPython.display import display
from matplotlib import cm, colors
from numpy.fft import fft

from peaks.core.utils.misc import analysis_warning


def plot_grid(
    data,
    ncols=3,
    nrows=None,
    titles=None,
    sharex=False,
    sharey=False,
    figsize=None,
    vmin=None,
    vmax=None,
    cmap=None,
    **plotting_kwargs,
):
    """Plots an array of 2D DataArrays on a grid.

    Parameters
    ------------
    data : list or xarray.DataTree
         A list or :class:`xarray.DataTree` containing 2D scans (in :class:`xarray.DataArray` format) to be plotted.
         If a datatree, the tree should be hollow, and the leaves should be :class:`xarray.Dataset`'s with a single
         data variable, `data`

    ncols : int, optional
        Number of columns. Ignored if nrows is specified. Defaults to 3 (or lower if <3 plots).

    nrows : int, optional
        Number of rows. Overwrites ncols. Defaults to None (i.e. ncols is used).

    titles : list, optional
        List of subtitles to be supplied on the plots. Length must match length of the data list. Defaults to None.

    sharex : bool, optional
        Whether to have the plots share an x-axis. Defaults to False.

    sharey : bool, optional
        Whether to have the plots share a y-axis. Defaults to False.

    figsize : tuple, optional
        Size of figure to be plotted. Defaults to None.

    vmin : float or int or list, optional
        Minimum value for the colorbar. Use a list to specify separate values for each plot. If a list is passed, the
        length must match the number of plots. Defaults to None.

    vmax : float or int or list, optional
        Maximum value for the colorbar. Use a list to specify separate values for each plot. If a list is passed, the
        length must match the number of plots. Defaults to None.

    cmap : str or list, optional
        Matplotlib colormap to use for the plots. Use a list to specify separate colormaps for each plot. If a list is
        passed, the length must match the number of plots.

    **plotting_kwargs
        Additional standard matplotlib calls arguments to pass to the plot.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp1 = load('disp1.ibw')
        disp2 = load('disp2.ibw')
        disp3 = load('disp3.ibw')
        disp4 = load('disp4.ibw')

        disps_to_plot = [disp1, disp2, disp3, disp4]
        disp_titles = ['10 K', '20 K', '30 K', '40 K']

        # Plot dispersions on a 2D grid
        plot_grid(disps_to_plot)

        # Plot dispersions on a 2D grid, where each dispersion has a title, all dispersions share an x-axis, and
        # colour bars are not shown
        plot_grid(disps_to_plot, titles=disp_titles, sharex=True, add_colorbar=False)

    """

    # Number of plots
    nplots = len(data)

    # Check list-like inputs are the correct length
    for item in [vmin, vmax, cmap]:
        if isinstance(item, (list, np.ndarray)):
            if len(item) != nplots:
                raise ValueError(
                    f"Length of supplied {item} list does not match number of plots. Either specify a list of length "
                    f"{nplots} or pass a single value."
                )

    # Check default columns is sensible
    if nplots < ncols:
        ncols = nplots

    # Number of rows required
    if nrows:  # If nrows specified in call
        ncols = int(np.ceil(nplots / nrows))  # Overwrite default columns call
    else:
        nrows = int(np.ceil(nplots / ncols))  # Set nrows based on ncols and nplots

    # If figsize not specified, set a default based on number of rows and columns
    if not figsize:
        figsize = (5 * ncols, 5 * nrows)

    # Make the figure layout
    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=figsize, sharex=sharex, sharey=sharey
    )

    # Clear any remaining subplot axes
    if nrows * ncols > nplots:
        for i in range(nplots, nrows * ncols):
            # Work out grid positions
            j0 = int(np.floor((i) / ncols))
            j1 = int(i - ncols * j0)
            axes[j0][j1].axis("off")

    # Check for whether plot titles are to be displayed
    plot_titles = False
    if titles:
        if len(titles) == nplots:  # Check the correct number are provided
            plot_titles = True
        else:  # If not, don't use them and give a warning
            analysis_warning(
                "Length of supplied titles list does not match number of plots. Supplied titles have not been used.",
                title="Plotting info",
                warn_type="danger",
            )

    # Make the plots
    def _plot_single(da, count, vmin, vmax, cmap):
        additional_plotting_kwargs = {
            k: v
            for k, v in {"vmin": vmin, "vmax": vmax, "cmap": cmap}.items()
            if v is not None
        }
        if nrows < 2 or ncols == 1:  # 1D grid
            ax = axes[count]
        else:  # 2D grid
            j0, j1 = divmod(count, ncols)
            ax = axes[j0][j1]
        da.plot(ax=ax, **plotting_kwargs, **additional_plotting_kwargs)  # Plot data
        if plot_titles:  # If plot titles to be displayed, update them here
            ax.set_title(titles[count])

    def _get_value_from_list(value, count):
        return value[count] if isinstance(value, list) else value

    if isinstance(data, list):
        for count, value in enumerate(data):
            _plot_single(
                value,
                count,
                _get_value_from_list(vmin, count),
                _get_value_from_list(vmax, count),
                _get_value_from_list(cmap, count),
            )
    elif isinstance(data, xr.DataTree):
        # If titles not passed as a specific list, we will populate them from node names
        if not plot_titles:
            titles = []
            plot_titles = True
        offset = 0
        for count, node in enumerate(data.subtree):
            if not node.is_empty:
                titles.append(node.name)
                _plot_single(
                    node.data,
                    count - offset,
                    _get_value_from_list(vmin, count - offset),
                    _get_value_from_list(vmax, count - offset),
                    _get_value_from_list(cmap, count - offset),
                )
            else:
                offset += 1

    # Tidy up the layout
    plt.tight_layout()


def plot_DCs(
    DCs,
    titles=None,
    cmap="coolwarm",
    color=None,
    offset=0,
    norm=False,
    stack_dim="auto",
    figsize=(6, 6),
    linewidth=1,
    ax=None,
    **plotting_kwargs,
):
    """Plot a DC stack with the colours varying according to a colormap.

    Parameters
    ------------
    DCs : xarray.DataArray, list
         DCs to plot, either as a list of single DC DataArrays, or a single DataArray with the stack of DCs included.

    titles : list, optional
        List of DC labels to be used in the legend. Length must match number of DCs to plot. Defaults to None.

    cmap : str, optional
        Matplotlib cmap to use for line colors. Defaults to coolwarm.

    color : str, optional
        Single matplotlib color to use for all lines. Takes precedence over cmap if defined. Defaults to None.

    offset : float, optional
        Vertical offsets between subsequent DCs, represented as a fraction of the maximum peak height. Defaults to 0.

    norm : bool, optional
        Whether to normalise the DCs to 1. Defaults to False.

    stack_dim : str, optional
        Which dimension to stack DCs along. Ignored if a list of DCs is passed. Defaults to 'auto' which takes the
        smallest dimension.

    figsize : tuple, optional
        Size of figure to be plotted. Defaults to None.

    linewidth : float, optional
        Width of lines to be plotted. Defaults to 1.

    ax : numpy.ndarray, optional
        Specific matplotlib axis call. Defaults to None.

    **plotting_kwargs
        Additional standard matplotlib calls arguments to pass to the plot.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        disp1 = pks.load('disp1.ibw')

        EDCs_to_plot = disp.EDC(k=[0, 0.1, 0.2, 0.3, 0.4, 0.5], dk=0.01)

        # Plot EDCs
        EDCs_to_plot.plot_DCs()

        MDC_1 = disp.MDC(E=95.61, dE=0.01)
        MDC_2 = disp.MDC(E=95.62, dE=0.01)
        MDC_3 = disp.MDC(E=95.63, dE=0.01)

        # Plot normalised MDCs with a 5% (of maximum height) offset, and add legend
        plot_DCs([MDC_1, MDC_2, MDC_3], titles = ['EF', 'EF - 0.01 eV', 'EF -0.02 eV'], norm=True, offset=0.05)

    """

    # If a list of DCs is supplied, combine them into a single DataArray
    if isinstance(DCs, list):
        DC_array = xr.concat(DCs, dim="DC_no")
        stack_dim = "DC_no"

    elif isinstance(DCs, xr.DataArray):
        # If a DataArray is instead supplied, finding the stacking dimension if not defined
        DC_array = DCs.copy(deep=True)  # Make a copy so we can safely modify
        if stack_dim == "auto":
            # Assume the stacking dimension is the smallest dimension size
            if DC_array.shape[0] < DC_array.shape[1]:
                stack_dim = DC_array.dims[0]
            else:
                stack_dim = DC_array.dims[1]

    elif isinstance(DCs, xr.DataTree):
        DC_list = [DC.data.copy(deep=True) for DC in DCs.subtree if not DC.is_empty]
        DC_array = xr.concat(DC_list, dim="DC_no")
        stack_dim = "DC_no"
    else:
        raise ValueError(
            "DCs must be a list of xarray.DataArray's, a single xarray.DataArray containing the DCs to "
            "plot, or a xarray.DataTree of DCs to plot."
        )

    DC_array = DC_array.pint.dequantify()

    # Check correct dimension supplied
    if len(DC_array.shape) != 2:
        raise Exception(
            "Incorrect dimension of data supplied. Please supply a single 2D DataArray or list of 1D DataArrays."
        )

    # Get non-stacking dimension
    other_dim = [dim for dim in DC_array.dims if dim != stack_dim][0]

    # Normalise if required
    if norm:
        for coord in DC_array[stack_dim]:
            DC_array.loc[{stack_dim: coord}] = (
                DC_array.loc[{stack_dim: coord}] / DC_array.loc[{stack_dim: coord}].max()
            )

    # Get absolute offset from fractional offset
    absolute_offset = offset * float(DC_array.max())

    # Define linear offset wave in steps of the supplied or guessed offset
    offset_data = xr.DataArray(
        data=np.arange(len(DC_array[stack_dim])) * absolute_offset,
        dims=stack_dim,
        coords={stack_dim: DC_array[stack_dim]},
    )

    # Add offset to DC data
    DC_array += offset_data

    # If no tiles are provided, no legend will be displayed
    if not titles:
        plotting_kwargs["add_legend"] = False

    # If x and y are not defined, make a guess for which axes to plot the dimensions on.
    if "y" not in plotting_kwargs and "x" not in plotting_kwargs:
        # If it seems to be an EDC plot, set vertical by default
        if other_dim == "eV":
            plotting_kwargs["y"] = "eV"
        # For anything else, set horizontal
        else:
            plotting_kwargs["x"] = other_dim

    # Set up plot
    if ax:
        plotting_kwargs["ax"] = ax
    else:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # If the user has requested to plot lines of a single color
    if color:
        DC_array.plot.line(
            hue=stack_dim, linewidth=linewidth, color=color, **plotting_kwargs
        )

    # Else use a colormap
    else:
        # Define the colour scheme
        cols = getattr(cm, cmap)(np.linspace(0, 1, len(DC_array[stack_dim])))
        cmap = colors.ListedColormap(cols)
        custom_cycler = cycler(color=cmap.colors)
        ax.set_prop_cycle(custom_cycler)
        DC_array.plot.line(hue=stack_dim, linewidth=linewidth, **plotting_kwargs)

    # Add axis titles
    if "y" in plotting_kwargs:
        plt.xlabel("Intensity [arb. units]")
    else:
        plt.ylabel("Intensity [arb. units]")

    # Add legend if user has inputted titles
    if titles:
        if len(titles) == len(DC_array[stack_dim]):
            for ct, i in enumerate(ax.lines):
                i.set_label(titles[ct])
            ax.legend()
        else:
            analysis_warning(
                "Length of supplied titles list does not match number of DCs. Supplied titles have not been used.",
                title="Plotting info",
                warn_type="danger",
            )

    # Tidy up the layout
    plt.tight_layout()


def plot_3d_stack(
    da,
    downsample=2,
    stack_dim="eV",
    vmax=None,
    cmap="Purples",
    figsize=(8, 12),
    aspect=None,
    elev=10.0,
    azim=-60.0,
    **kwargs,
):
    """Plot a stack of surface plots from a data cube

    Parameters
    ------------
    da : xarray.DataArray
        The data to plot.
    downsample : int, optional
        The downsample factor for the plot. Defaults to 2.
    stack_dim : str, optional
        The dimension to stack the data along. Defaults to 'eV'.
    vmax : float, optional
        The maximum value for the color scale. Defaults to None (i.e. the maximum value
        of the data).
    cmap : str, optional
        The colormap to use. Defaults to 'Purples'.
    figsize : tuple, optional
        The size of the figure. Defaults to (8, 12).
    aspect : tuple, optional
        The aspect ratio for the plot (x, y, z), in units of the data, e.g. (1,1,200)
        Defaults to None.
    elev : float, optional
        The elevation angle for the 3D plot. Defaults to 10.0.
    azim : float, optional
        The azimuthal angle for the 3D plot. Defaults to -60.0.
    **kwargs : optional
        Additional standard matplotlib calls arguments to pass to the plot.
        Must be suitable for passing in ax.set(**kwargs) format.

    Notes
    ------------
    `xlim` and `ylim` don't clip the range for this type of plot. Pass data over the
    range you want to plot. Similarly, apply any normalisation to the data before
    passing it to this function.
    """

    # Parse the dimensions
    if stack_dim not in da.dims:
        raise ValueError(
            f"stack_dim '{stack_dim}' is not a dimension of the data array."
        )
    non_stack_dims = [d for d in da.dims if d != stack_dim]

    # Normalise the data
    if not vmax:  # no max colour limit specified
        vmax = da.max().pint.dequantify().data
    C = (da / vmax).copy().pint.dequantify()
    C = np.clip(C, 0, 1)

    # Set up the plot
    plt.interactive(True)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    cmap_sel = plt.get_cmap(cmap)

    # Define the grid
    X = da[non_stack_dims[1]].data
    Y = da[non_stack_dims[0]].data
    XX, YY = np.meshgrid(X, Y)
    Z = np.zeros_like(XX)

    # Set the aspect ratio
    if aspect:
        stack_dim_range = np.ptp(da[stack_dim].data)
        ax.set_box_aspect(
            (np.ptp(X) * aspect[0], np.ptp(Y) * aspect[1], stack_dim_range * aspect[2])
        )

    # Make the plot surfaces (downsample 2x2)
    for i in da["eV"].data:
        ax.plot_surface(
            XX,
            YY,
            Z + i,
            rstride=downsample,
            cstride=downsample,
            facecolors=cmap_sel(C.sel(eV=i).data),
            shade=False,
        )

    # Colour scale
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(np.linspace(0, vmax, 100))
    cbar = plt.colorbar(m, ax=ax, shrink=0.2, aspect=8, pad=0.15, location="right")
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")

    # Some plot display settings
    # Label the axes
    ax.set_xlabel(non_stack_dims[1])
    ax.set_ylabel(non_stack_dims[0])
    ax.set_zlabel(stack_dim)

    # Set overall view (no grids etc.)
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Sort out axis label locations
    [t.set_va("center") for t in ax.get_yticklabels()]
    [t.set_ha("left") for t in ax.get_yticklabels()]
    [t.set_va("center") for t in ax.get_xticklabels()]
    [t.set_ha("right") for t in ax.get_xticklabels()]
    [t.set_va("center") for t in ax.get_zticklabels()]
    [t.set_ha("left") for t in ax.get_zticklabels()]
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.labelpad = 20

    # Set z-axis ticks to match the slices
    ax.zaxis.set_major_locator(plt.FixedLocator(np.round(C["eV"].data, 3)))

    # Set default elevation view
    ax.view_init(elev=elev, azim=azim)

    ax.set(**kwargs)

    # Show the plot
    plt.show()


def plot_fit(fit_results_ds, show_components=True, figsize=None, **kwargs):
    def _plot_single_fit(fit_results, show_components, figsize, **kwargs):
        fig = plt.figure(figsize=figsize)
        for dim in fit_results.dims:
            if dim in kwargs:
                fit_results = fit_results.isel({dim: kwargs.pop(dim)})
        # Add the independent variable to the kwargs for plotting
        kwargs.setdefault("xlabel", fit_results_ds.attrs.get("independent_var", "x"))

        fit_model = fit_results["fit_model"].compute().item()
        fit_model.plot(fig=fig, **kwargs)
        if show_components and len(fit_model.components) > 1:
            ax = plt.gca()
            components = fit_model.eval_components()
            for component_name, component_data in components.items():
                if component_name != "_gauss_conv":
                    ax.plot(
                        fit_model.userkws["x"],
                        component_data,
                        label=component_name,
                        linestyle="--",
                    )
            ax.legend()
        plt.close(fig)
        return fig

    # Check the data array contains fit results
    if "fit_model" not in fit_results_ds:
        raise ValueError(
            "The passed data does not appear to be a DataSet containing fit results. Generate the relevant fit results "
            "by calling the `fit` method on a suitable DataArray, e.g. "
            "`fit_results = disp1.fit(model, 'eV', params)`"
        )

    if len(fit_results_ds["fit_model"].shape) == 0:
        fig = _plot_single_fit(fit_results_ds, show_components, figsize, **kwargs)
        display(fig)
    else:
        # Initialize Panel extension
        pn.extension()

        # Create sliders for dims
        sliders = {}
        for dim in fit_results_ds.dims:
            sliders[dim] = pn.widgets.IntSlider(
                name=dim, start=0, end=len(fit_results_ds[dim]) - 1, step=1, value=0
            )

        # Bind the plot function to the sliders, updating the plot dynamically
        interactive_plot = pn.bind(
            _plot_single_fit,
            fit_results=fit_results_ds,
            show_components=show_components,
            figsize=figsize,
            **sliders,
            **kwargs,
        )

        # Display the sliders and the plot in the notebook
        dashboard = pn.Column(
            pn.Row(*sliders.values()), pn.pane.Matplotlib(interactive_plot)
        )

        dashboard.servable()
        return dashboard


def plot_fit_test(data, model, params, show_components=True, **kwargs):
    """Compare a fit model evaluated for some fit parameters to a 1D data array.

    Parameters
    ------------
    data : xarray.DataArray
        The data to compare the model to.
    model : lmfit.Model
        The model to evaluate.
    params : lmfit.Parameters
        The parameters to evaluate the model with.
    show_components : bool, optional
        Whether to show the individual components of the model. Defaults to True.
    **kwargs : optional
        Additional standard matplotlib calls arguments to pass to the plot.
    """

    data = data.pint.dequantify()

    if len(data.dims) != 1:
        raise ValueError(
            "Data must be 1D with the dimension corresponding to the indpependent variable."
        )

    # Evaluate the model
    model_result = model.eval(params=params, x=data[data.dims[0]].data)
    if show_components:
        components = model.eval_components(params=params, x=data[data.dims[0]].data)

    # Plot the data and the model
    data.plot(label="Data", marker="o", **kwargs)
    plt.plot(data[data.dims[0]].data, model_result, label="Model", **kwargs)
    if show_components:
        for component_name, component_data in components.items():
            if component_name != "_gauss_conv":
                plt.plot(
                    data[data.dims[0]].data,
                    component_data,
                    label=component_name,
                    linestyle="--",
                    **kwargs,
                )
    plt.legend()
    plt.show()


def plot_ROI(
    ROI,
    color="black",
    x=None,
    y=None,
    label=None,
    loc="best",
    ax=None,
    **plotting_kwargs,
):
    """This function plots a region of interest (ROI).

    Parameters
    ------------
    ROI : dict
        A dictionary of two lists which contains the vertices of the polygon for the ROI definition, in the form
        {'dim1': [pt1, pt2, pt3, ...], 'dim2'=[pt1', pt2', pt3', ...]}. As many points can be specified as required,
        but this should be given with the same number of points for each dimension.

    color : str, optional
        Matplotlib color to use for ROI plot. Defaults to 'black'.

    x : str, optional
        Specifies which coordinate to plot along the x-axis. Takes precedence over y (only one of x and y need to be
        specified, the other is set automatically). Defaults to the first coordinate listed in the ROI dictionary.

    y : str, optional
        Specifies which coordinate to plot along the y-axis. Defaults to the second coordinate listed in the ROI
        dictionary.

    label : str, optional
        ROI label to pass to a legend. Defaults to None (where no legend is plotted).

    loc : str, int, optional
        Standard matplotlib call to specify legend location. Defaults to 'best'.

    ax : numpy.ndarray, optional
        Specific matplotlib axis call. Defaults to None.

    **plotting_kwargs
        Additional standard matplotlib calls arguments to pass to the plot.

    Examples
    ------------
    Example usage is as follows::

    from peaks import *

        disp = load('disp1.ibw')

        ROI1 = {'theta_par': [-8, -5.5, -3.1, -5.6], 'eV': [95.45, 95.45, 95.77, 95.77]}
        ROI2 = {'theta_par': [-9, -4, -2.5, -5.9], 'eV': [95.45, 95.45, 95.77, 95.77]}

        # Plot ROI
        plot_ROI(ROI1)
        plt.show()

        # Plot ROI on top of a dispersion with eV plotted along the x-axis.
        disp1.plot()
        plot_ROI(ROI1, x='eV')
        plt.show()

        # Plot ROIs on top of dispersions on specific axes, axes[0] and axes[1], with labels
        fig, axes= plt.subplots(ncols=2)
        disp.T.plot(ax=axes[0])
        plot_ROI(ROI1, label='ROI 1', ax=axes[0])
        disp.T.plot(ax=axes[1])
        plot_ROI(ROI2, label='ROI 2', ax=axes[1])

    """

    # Determine relevant dimensions for ROI
    dims = list(ROI)

    # Determine lists of vertices of ROI
    verts0 = ROI[dims[0]]
    verts1 = ROI[dims[1]]

    # Append initial value to end to close polygon
    verts0.append(verts0[0])
    verts1.append(verts1[0])

    # By default, plot the second dimension as the y-axis
    yax = 1

    # Check if definition of x- or y-axis is given
    if x:
        yax = 1 - dims.index(x)
    elif y:
        yax = dims.index(y)

    # Plot the ROI
    if ax:  # If particular axis specified
        if yax == 0:
            ax.plot(verts1, verts0, color=color, label=label, **plotting_kwargs)
        else:
            ax.plot(verts0, verts1, color=color, label=label, **plotting_kwargs)
    else:  # Otherwise just call with plt
        if yax == 0:
            plt.plot(verts1, verts0, color=color, label=label, **plotting_kwargs)
        else:
            plt.plot(verts0, verts1, color=color, label=label, **plotting_kwargs)

    # If called with label, plot the legend
    if label:
        if ax:  # If particular axis specified
            ax.legend(loc=loc)
        else:  # Otherwise just call with plt
            plt.legend(loc=loc)

    # Tidy up the layout
    plt.tight_layout()


def plot_kpar_cut(
    hv=21.2,
    Eb=0,
    theta_par_range=(-15, 15),
    polar=0,
    tilt=0,
    defl_perp=0,
    ana_type="I",
    ax=None,
    label_cut=True,
    flip=False,
    **kwargs,
):
    """Plot the k_par cut along the analyser slit for a given set of angle parameters.

    Parameters
    ----------
    hv : float, optional
        Photon energy (eV) (default=21.2 eV).
    Eb : float, optional
        Binding energy (eV) (positive for below the Fermi level, default=0).
    theta_par_range : tuple, optional
        Range of theta_par values to calculate over in the form (start, stop);
        default=(-15, 15)).
    polar : float, optional
        Polar angle (deg) (default=0).
    tilt : float, optional
        Tilt angle (deg) (default=0).
    defl_perp : float, optional
        Deflector angle perpendicular to the slit (deg) (default=0).
    ana_type : str, optional
        Analyser type, one of I, II, Ip, IIp (default='I').
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on (default=None).
    label_cut : bool, optional
        Whether to label the cut with the angles used (default=True).
    flip : bool, optional
        Whether to flip the kx and ky axes (default=False).
    **kwargs
        Additional keyword arguments to pass to the plot.
    """

    from peaks.core.process.k_conversion import get_kpar_cut

    kx, ky = get_kpar_cut(
        hv=hv,
        Eb=Eb,
        theta_par_range=theta_par_range,
        polar=polar,
        tilt=tilt,
        defl_perp=defl_perp,
        ana_type=ana_type,
    )

    if not ax:
        fig, ax = plt.subplots()
    if not flip:
        ax.plot(kx, ky, **kwargs)
    else:
        ax.plot(ky, kx, **kwargs)

    if label_cut:
        # Add label to the line showing the tilt and polar angles
        angle_label = f"Pol.: {polar}°, Tilt: {tilt}°"
        label_pos = [kx[-1], ky[-1]] if not flip else [ky[-1], kx[-1]]
        if defl_perp:
            angle_label += f", Defl. perp.: {defl_perp}°"
        ax.text(
            label_pos[0],
            label_pos[1],
            angle_label,
            fontsize=8,
            ha="center",
            va="bottom",
            rotation=0,
            color="black",
        )


def plot_kz_cut(
    hv=21.2,
    Eb=0,
    polar_or_tilt=0,
    theta_par_range=(-15, 15),
    V0=12,
    ax=None,
    label_cut=True,
    **kwargs,
):
    """Plot the kz-dep of the cut along the analyser slit for given parameters.

    Parameters
    ----------
    hv : float, optional
        Photon energy (eV) (default=21.2 eV).
    Eb : float, optional
        Binding energy (eV) (positive for below the Fermi level, default=0).
    theta_par_range : tuple, optional
        Range of theta_par values to calculate over in the form (start, stop);
        default=(-15, 15)).
    polar_or_tilt : float, optional
        Angle offset in direction along the slit (deg) (default=0).
    V0 : float, optional
        Inner potential (eV) (default=12).
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on (default=None).
    label_cut : bool, optional
        Whether to label the cut with the photon energy used (default=True).
    **kwargs
        Additional keyword arguments to pass to the plot.
    """

    from peaks.core.process.k_conversion import get_kz_cut

    kx, kz = get_kz_cut(
        hv=hv,
        Eb=Eb,
        theta_par_range=theta_par_range,
        polar_or_tilt=polar_or_tilt,
        V0=V0,
    )

    if not ax:
        fig, ax = plt.subplots()

    ax.plot(kx, kz, **kwargs)

    if label_cut:
        # Add label to the line showing the photon energy
        label = f"hv: {hv} eV  "
        ax.text(
            kx[0],
            kz[0],
            label,
            fontsize=10,
            ha="right",
            va="bottom",
            rotation=0,
            color="black",
        )


def plot_nanofocus(data, focus="defocus"):
    """Function to determine the focus of a scan obtained at the nano branch of the I05 beamline at Diamond Light
    Source, and plot the results. The function works by determining the focal position at which at scanned feature
    becomes sharpest.

    Parameters
    ------------
    data : xarray.DataArray
        The 2D (or 4D) focussing scan data with some spatial direction dimension and some focussing direction
        dimension (and additionally theta_par and eV dimensions for 4D data).

    focus : str, optional
        The dimension of the data that represents the focussing direction.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        focus_scan = load('i05-1-12345.nxs')

        # Plot the results of how the sharpness of a feature varies with defocus, determining the focal position
        focus_scan.plot_nanofocus()

    """

    # Ensure the data is a 2D DataArray by integrating in energy and angle space if required
    if len(data.dims) == 4:
        data = data.mean(["theta_par", "eV"], keep_attrs=True)

    # Extract the spatial dimension
    data_dims = list(data.dims)
    data_dims.remove(focus)
    spatial = data_dims[0]

    # Determine the step of the focus dimension
    focus_step = data[focus].data[1] - data[focus].data[0]

    # Set up a subplot to plot the focussing analysis
    fig, axes = plt.subplots(figsize=(12, 14), ncols=2, nrows=3)

    # Plot the intensity map
    data.plot(x=focus, ax=axes[0, 0], add_colorbar=False)

    # Plot focus-dependent cuts through the spatial dimension (using the plot_DCs function)
    try:
        plot_DCs(data, color="black", offset=0.1, norm=False, ax=axes[0, 1], y=spatial)
    except ValueError:
        plot_DCs(
            data,
            color="black",
            offset=0.1,
            norm=False,
            ax=axes[0, 1],
            stack_dim=spatial,
        )
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_title("Line cuts", fontsize=16)

    # Calculate and plot the focus-dependent means of the absolute values of the derivatives along the spatial
    # dimension
    mean_abs_deriv = abs(data.diff(spatial)).mean(spatial)
    mean_abs_deriv.plot(ax=axes[1, 0], c="black")

    # Estimate and plot the focal point from the focus-dependent means of the absolute values of the derivatives
    # along the spatial dimension
    mean_abs_deriv_smoothed = mean_abs_deriv.smooth(**{focus: 2 * focus_step})
    mean_abs_deriv_smoothed.plot(ax=axes[1, 0], c="black", alpha=0.2)
    max_mean_abs_deriv = mean_abs_deriv_smoothed.argmax()
    focal_point_mean_abs_deriv = mean_abs_deriv[focus].data[max_mean_abs_deriv]
    axes[1, 0].set_title(
        "Mean of the abs(deriv) estimate: {focal_point}".format(
            focal_point=focal_point_mean_abs_deriv
        ),
        fontsize=16,
    )
    axes[1, 0].axvline(focal_point_mean_abs_deriv, c="black", linestyle="--")
    axes[1, 0].set_ylabel("Intensity (arb. units)")
    axes[1, 0].set_yticks([])

    # Calculate and plot the focus-dependent maximums of the absolute values of the derivatives along the spatial
    # dimension
    max_abs_deriv = abs(data.diff(spatial)).max(spatial)
    max_abs_deriv.plot(ax=axes[1, 1], c="black")

    # Estimate and plot the focal point from the focus-dependent maximums of the absolute values of the derivatives
    # along the spatial dimension
    max_abs_deriv_smoothed = max_abs_deriv.smooth(**{focus: 2 * focus_step})
    max_abs_deriv_smoothed.plot(ax=axes[1, 1], c="black", alpha=0.2)
    max_max_abs_deriv = max_abs_deriv_smoothed.argmax()
    focal_point_max_abs_deriv = max_abs_deriv[focus].data[max_max_abs_deriv]
    axes[1, 1].set_title(
        "Max of the abs(deriv) estimate: {focal_point}".format(
            focal_point=focal_point_max_abs_deriv
        ),
        fontsize=16,
    )
    axes[1, 1].axvline(focal_point_max_abs_deriv, c="black", linestyle="--")
    axes[1, 1].set_ylabel("Intensity (arb. units)")
    axes[1, 1].set_yticks([])

    # Calculate and plot the focus-dependent variance along the spatial dimension
    variance = data.var(spatial)
    variance.plot(ax=axes[2, 0], c="black")

    # Estimate and plot the focal point from the focus-dependent variance along the spatial dimension
    variance_smoothed = variance.smooth(**{focus: 2 * focus_step})
    variance_smoothed.plot(ax=axes[2, 0], c="black", alpha=0.2)
    max_variance = variance_smoothed.argmax()
    focal_point_max_variance = variance[focus].data[max_variance]
    axes[2, 0].set_title(
        "Max of the variance estimate: {focal_point}".format(
            focal_point=focal_point_max_variance
        ),
        fontsize=16,
    )
    axes[2, 0].axvline(focal_point_max_variance, c="black", linestyle="--")
    axes[2, 0].set_ylabel("Intensity (arb. units)")
    axes[2, 0].set_yticks([])

    # Perform a fast Fourier transform (FFT) analysis of the data to look for increased higher frequencies
    # indicative of a sharper step, by transforming the spatial dimension of the data to frequency space
    FFT_data = data.copy(deep="True")
    if focus != data.dims[0]:  # Ensure data shape is as expected
        FFT_data = FFT_data.T
        FFT_data.data = abs(fft(data.pint.dequantify().T.data))  # Perform FFT
    else:
        FFT_data.data = abs(fft(data.pint.dequantify().data))  # Perform FFT
    FFT_data = FFT_data.pint.quantify(data.pint.units)
    FFT_data_out = FFT_data.isel({spatial: FFT_data.argmax(spatial).data[0] + 1})

    # Plot the FFT analysis results
    FFT_data_out.plot(ax=axes[2, 1], c="black")

    # Estimate and plot the focal point from the FFT analysis
    FFT_data_out_smoothed = FFT_data_out.smooth(**{focus: 2 * focus_step})
    FFT_data_out_smoothed.plot(ax=axes[2, 1], c="black", alpha=0.2)
    max_FFT = FFT_data_out_smoothed.argmax()
    focal_point_max_FFT = FFT_data_out[focus].data[max_FFT]
    axes[2, 1].set_title(
        "Fast Fourier transform estimate: {focal_point}".format(
            focal_point=focal_point_max_FFT
        ),
        fontsize=16,
    )
    axes[2, 1].axvline(focal_point_max_FFT, c="black", linestyle="--")
    axes[2, 1].set_ylabel("Intensity (arb. units)")
    axes[2, 1].set_yticks([])

    # Determine the average focal position estimate (removing any estimate outliers that are more than 2 std away
    # from the mean of the estimates)
    estimates = np.array(
        [
            focal_point_mean_abs_deriv,
            focal_point_max_abs_deriv,
            focal_point_max_variance,
            focal_point_max_FFT,
        ]
    )
    estimates_no_outliers = []
    for i, item in enumerate(abs(abs(estimates) - abs(estimates.mean()))):
        if item < (2 * estimates.std()):
            estimates_no_outliers.append(estimates[i])
    avg_estimate = round(np.mean(estimates_no_outliers), 2)

    # Plot the average focal position
    axes[0, 0].set_title(
        "Intensity map ({focus} estimate: {avg_estimate})".format(
            focus=focus, avg_estimate=avg_estimate
        ),
        fontsize=16,
    )
    axes[0, 0].axvline(avg_estimate, c="#ECDE00", linestyle="dotted")
    axes[1, 0].axvline(avg_estimate, c="#ECDE00", linestyle="dotted")
    axes[1, 1].axvline(avg_estimate, c="#ECDE00", linestyle="dotted")
    axes[2, 0].axvline(avg_estimate, c="#ECDE00", linestyle="dotted")
    axes[2, 1].axvline(avg_estimate, c="#ECDE00", linestyle="dotted")

    plt.tight_layout()
