try:
    import ase
except ImportError as e:
    raise ImportError(
        "The 'ase' module is required to use this module. Please install it using \
'pip install ase'."
    ) from e

import numpy as np
import pandas as pd


def _get_atoms_cell_lattice(structure_or_lattice):
    """Get atoms, cell, and lattice from the given structure or lattice.

    Parameters
    ----------
    structure_or_lattice : ase.Atoms or ase.lattice.BravaisLattice
        Input structure or lattice object.

    Returns
    -------
    tuple(ase.Atoms, ase.cell.Cell, ase.lattice.BravaisLattice)
        Tuple containing the atoms, cell, and lattice.
    """
    if isinstance(structure_or_lattice, ase.Atoms):
        atoms = structure_or_lattice
        lattice = structure_or_lattice.cell.get_bravais_lattice()
        cell = structure_or_lattice.cell
    elif isinstance(structure_or_lattice, ase.lattice.BravaisLattice):
        lattice = structure_or_lattice
        cell = lattice.tocell()
        atoms = ase.Atoms(cell=cell, pbc=True)
    else:
        raise ValueError(
            "Invalid input. Please provide an ase.Atoms or an ase.lattice.BravaisLattice\
 object."
        )

    return atoms, cell, lattice


def _get_special_points_inv_Ang(lattice):
    """
    Get the symmetry points for the Brillouin zone of the given lattice.

    Parameters
    ----------
    lattice : ase.lattice.BravaisLattice
        Input lattice object.

    Returns
    -------
    sym_points : dict
        Dictionary containing the symmetry points, with values in inverse Angstrom.
    """
    sym_points = lattice.get_special_points()
    sym_points_inv_Ang = {
        k: ase.dft.kpoints.kpoint_convert(lattice.tocell(), skpts_kc=v)
        for k, v in sym_points.items()
    }  # Convert to k in inv Angstrom

    return sym_points_inv_Ang


def sym_points(structure_or_lattice, surface=None, hv=None):
    """
    Get the symmetry points for the Brillouin zone of the given structure or lattice.

    Parameters
    ----------
    structure_or_lattice : ase.Atoms or ase.lattice.BravaisLattice
        Input structure or lattice object.
    surface : tuple | list | np.ndarray | None
        Surface orientation of the structure, optional (default=None --> bulk)
    hv : float | None
        Photon energy in eV, to be used to generate angle along slit for k-points
        Optional (default=None, no angle calculation)

    Returns
    -------
    sym_points : dict
        Dictionary containing the symmetry points
    """
    atoms, cell, lattice = _get_atoms_cell_lattice(structure_or_lattice)

    if surface is not None:
        if not isinstance(surface, (tuple, list, np.ndarray)) and len(surface) != 3:
            raise ValueError(
                "Invalid input for surface orientation. Must be a tuple, \
list, or np.ndarray of length 3."
            )
        surface_atoms = ase.build.surface(atoms, surface, 1)
        lattice = surface_atoms.cell.get_bravais_lattice()

    df = pd.DataFrame.from_dict(
        _get_special_points_inv_Ang(lattice),
        orient="index",
        columns=["k_x", "k_y", "k_z"],
    )
    df["|k|"] = df.apply(np.linalg.norm, axis=1)
    if hv is not None:
        df[f"angle_along_slit @ {hv} eV"] = np.round(
            np.degrees(np.arcsin(df["|k|"] / (0.5123 * np.sqrt(hv - 4.5)))), 1
        )

    return df
