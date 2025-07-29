import os
import pickle
import re

import pandas as pd
import xarray as xr


def _load_core_level_BEs():
    """Load core levels from the stored data."""
    path = os.path.join(os.path.dirname(__file__), "core_level_BEs.pkl")
    with open(path, "rb") as f:
        core_level_BEs = pickle.load(f)
    return core_level_BEs  # Return nested dictionary


def _load_core_level_xcs():
    """Load core level cross sections from the stored data."""
    # Load raw data
    root_path = os.path.dirname(__file__)
    with open(os.path.join(root_path, "xc_name.pkl"), "rb") as f:
        xc_name = pickle.load(f)
    with open(os.path.join(root_path, "xc.pkl"), "rb") as f:
        xc_data = pickle.load(f)

    # Sort the data in sensible order
    orbital_order = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}

    def core_level_key(name):
        element, core_level = name.split("_")
        principal_quantum_number = int(
            core_level[0]
        )  # Extract the principal quantum number
        orbital = core_level[1]  # Extract the orbital character
        return (element, principal_quantum_number, orbital_order.get(orbital, 6))

    xc_name_sorted = sorted(xc_name, key=core_level_key)

    # Iterate through xc_name, xc_data, and En_entry to create DataArrays and store them
    # in a DataTree
    data_dict = {}
    for name in xc_name_sorted:
        try:
            cross_section = xc_data[f"xc_{name}"]
            energy = xc_data[f"En_{name}"]
            element, core_level = name.split("_")

            # Create a DataArray
            data_array = xr.DataArray(
                data=cross_section, dims=["eV"], coords={"eV": energy}, name="data"
            )

            # Add to a dictionary of Datasets
            dataset = xr.Dataset({"data": data_array})
            data_dict[f"{element}/{core_level}"] = dataset
        except KeyError:
            pass

    # Create and return DataTree from the dictionary
    return xr.DataTree.from_dict(data_dict)


class CoreLevels:
    """Helper tools for working with core levels in XPS.

    Class Attributes
    ----------------
    BE : dict
        Nested dictionary of core level binding energies.
    xc : xarray.DataTree
        DataTree of core level cross sections.

    Notes
    -----
    Binding energies are taken from J. A. Bearden and A. F. Burr, "Reevaluation of X-Ray
    Atomic Energy Levels," Rev. Mod. Phys. 39, (1967) p.125; M. Cardona and L. Ley, Eds.,
    Photoemission in Solids I: General Principles (Springer-Verlag, Berlin, 1978) with
    additional corrections, and from J. C. Fuggle and N. Mårtensson, "Core-Level Binding
    Energies in Metals," J. Electron Spectrosc. Relat. Phenom. 21, (1980) p.275.

    Cross sections from https://vuo.elettra.eu/services/elements/WebElements.html.
    """

    BE = _load_core_level_BEs()
    xc = _load_core_level_xcs()

    @staticmethod
    def _BE_to_KE(df, hv, order):
        """Convert data frame with core level data in BE to KE."""
        return (order * hv) - 4.35 - df

    @classmethod
    def _get_BE_df(cls, elements):
        """Get core level binding energies for specified elements as a dataframe."""
        if isinstance(elements, str):
            elements = [elements]
        BE_data = {element: cls.BE[element] for element in elements}
        return pd.DataFrame.from_dict(BE_data, orient="index")

    @classmethod
    def _get_KE_df(cls, df, hv, max_order):
        """Convert df of core level binding energies into kinetic energies."""
        rows = []
        for order in range(1, max_order + 1):
            new_idx = f"KE@hv={hv}eV"
            if order != 1:
                new_idx += (
                    f", {order}nd order"
                    if order == 2
                    else f", {order}rd order"
                    if order == 3
                    else f", {order}th order"
                )
            KE_df = cls._BE_to_KE(df, hv, order)
            KE_df.index = [f"{idx} ({new_idx})" for idx in KE_df.index]
            rows.append(KE_df)

        KE_df = pd.concat(rows)
        return KE_df.map(lambda x: x if x >= 0 else float("nan")).sort_index()

    @staticmethod
    def _clean_df(df):
        """Clean up a dataframe for display."""
        return df.round(2).fillna("").astype(str).sort_index()

    @classmethod
    def by_element(cls, elements, hv=None, max_order=1):
        """Get core level binding and optionally kinetic energies for specified elements.

        Parameters
        ----------
        elements : str | list
            Element symbol(s) to get core level binding energies for.
        hv : float, optional
            Photon energy in eV. If not specified, only binding energy returned.
        max_order : int, optional
            Maximum order of the light to consider. Default is 1.
        """
        df = cls._get_BE_df(elements)

        if hv is not None:
            KE_df = cls._get_KE_df(df, hv, max_order)
            df = pd.concat([df, KE_df])

        return cls._clean_df(df)

    @classmethod
    def by_energy(cls, energy, hv=None, max_order=1, tol=3):
        """Get core level binding and optionally kinetic energies for specified photon
        energy.

        Parameters
        ----------
        energy : float
            Core level energy in eV.
            If `hv=None`, search is performed on binding energy.
            If `hv` is specified, search is performed on kinetic energy from the
            specified photon energy, and optionally higher harmonics up to `max_order`.
        hv : float, optional
            Photon energy in eV.
        max_order : int, optional
            Maximum order of the light to consider. Default is 1.
        """

        df = pd.DataFrame.from_dict(cls.BE, orient="index")
        if hv is not None:
            df = cls._get_KE_df(df, hv, max_order)

        filtered_df = (
            df[(df >= energy - tol) & (df <= energy + tol)]
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )

        return cls._clean_df(filtered_df)

    @classmethod
    def plot(
        cls,
        elements,
        eV=None,
        hv=None,
        max_order=1,
        ax=None,
        show_binding_as_negative=True,
        **kwargs,
    ):
        """Plot markers at core level binding or kinetic energies for specified elements.

        Parameters
        ----------
        elements : str | list
            Element symbol(s) to consider.
        eV: slice, optional
            Energy range in eV to plot.
        hv : float, optional
            Photon energy in eV, for plotting kinetic energy.
            If not specified, binding energy is plotted.
        max_order : int, optional
            Maximum order of the light to consider. Default is 1.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If not specified, a new figure is created
        show_binding_as_negative : bool, optional
            If `True`, binding energies are shown as negative values to match the E-E_F
            convention for photoemission plotting.
        kwargs
            Additional keyword arguments passed to `matplotlib.pyplot.plot`.

        """

        import matplotlib.pyplot as plt

        if eV is None:
            eV = slice(None, None)

        def extract_element_and_order(s):
            # Define the regular expression pattern
            pattern = r"^(\w+)(?:\s+\(([^)]*order[^)]*)\))?"

            # Search for the pattern in the string
            match = re.search(pattern, s)

            if match:
                element = match.group(1)
                order = None
                if match.group(2):
                    order_match = re.search(r"(\d+\w* order)", match.group(2))
                    if order_match:
                        order = order_match.group(1)
                return element, order
            else:
                return None, None

        # Get the relevant data
        df = cls._get_BE_df(elements)
        if hv is not None:
            df = cls._get_KE_df(df, hv, max_order)
            ax_label = "Kinetic Energy (eV)"
        else:
            if show_binding_as_negative:
                df = -df
                ax_label = "E-E_F (eV)"
            else:
                ax_label = "Binding Energy (eV)"

        # Plot the data
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel(ax_label)

        # Set default plot options if not overwritten by kwargs
        color_cycle = ["k", "b", "g", "r", "c", "m", "y", "k"]
        color = kwargs.pop("color", None)
        linestyle = kwargs.pop("linestyle", None)

        elements = [extract_element_and_order(idx)[0] for idx in df.index]
        color_map = {
            element: color_cycle[i % len(color_cycle)]
            for i, element in enumerate(set(elements))
        }

        for idx, row in df.iterrows():
            element, order = extract_element_and_order(idx)
            row_vals = row.dropna()
            for cl, en in row_vals.items():
                # Plot core level if in range
                if en > (eV.start or -float("inf")) and en < (eV.stop or float("inf")):
                    ax.axvline(
                        en,
                        color=(color or color_map[element]),
                        linestyle=(linestyle or ("-" if order is None else "--")),
                        **kwargs,
                    )
                    # Label the core level
                    label = f"{element} {cl}"
                    if order:
                        label += f" ({order})"
                    ax.text(
                        en,
                        1.02,
                        label,
                        rotation=70,
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        transform=plt.gca().get_xaxis_transform(),
                        color=(color or color_map[element]),
                    )

    @classmethod
    def _add_core_levels_to_ptab(cls, df, hv=None):
        # Add in core level binding energies to the df in the ptab function
        df["core levels"] = ""
        cl_df = cls._get_BE_df(cls.BE.keys())
        if hv is not None:
            cl_df = cls._get_KE_df(cl_df, hv, 1)
            cl_df.index = cl_df.index.str.extract(r"^(\w+)")[0]
        cl_df = cl_df.round(2)
        for element, element_df in cl_df.iterrows():
            element_cl_str = ""
            for level, value in element_df.dropna().items():
                element_cl_str += f"{level}: {value} eV<br>"
            df.loc[df["symbol"] == element, "core levels"] = element_cl_str

        return df

    @classmethod
    def ptab(cls, hv=None):
        """Interactive periodic table display showing core level energies.

        Parameters
        ----------
        hv : float, optional
            Photon energy in eV. If not specified, binding energy is displayed. If
            specified, kinetic energy is displayed.
        """
        from peaks.core.GUI.periodic_table.ptab import ptab

        if hv is None:
            tooltip = "Binding energy:<br>@{core levels}{safe}"
        else:
            tooltip = f"Kinetic energy (hv={hv} eV):<br>" + "@{core levels}{safe}"

        ptab(
            lambda df: cls._add_core_levels_to_ptab(df, hv=hv),
            ("Core levels", tooltip),
        )
