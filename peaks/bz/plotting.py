try:
    import ase
except ImportError as e:
    raise ImportError(
        "The 'ase' module is required to use this module. Please install it using \
'pip install ase'."
    ) from e
import ase.dft.bz
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from ase.dft.bz import bz_vertices
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from peaks.bz.utils import _get_atoms_cell_lattice


class LinearScalingTransform:
    """
    Linear scaling transformation for scaling coordinates along each axis."""

    def __init__(self, scaling_factors):
        """
        Initialize the linear scaling transform.

        Parameters
        ----------
        scaling_factors : tuple of float
            Scaling factors for each axis.
        """
        self.scaling_factors = scaling_factors

    def apply(self, coords):
        """
        Apply the linear scaling transformation.

        Parameters
        ----------
        coords : np.ndarray
            Input coordinates to be transformed.

        Returns
        -------
        transformed_coords : np.ndarray
            Transformed coordinates.
        """
        return coords * self.scaling_factors


def plot_bz(
    structure_or_lattice,
    surface=None,
    path="",
    special_points=None,
    vectors=False,
    azim=36,
    elev=30,
    scale=1,
    rotate=0,
    repeat=(1, 1, 1),
    ax=None,
    show=False,
    show_axes=None,
    **kwargs,
):
    """Plot the brillouin zone. Wrapper around ase.cell.BravaisLattice.plot_bz.

    Parameters
    ----------
    structure_or_lattice : ase.Atoms | ase.lattice.BravaisLattice
        The crystal structure or Bravais lattice representation of the material.
    surface : Tuple[int, int, int] | None, optional
        Surface orientation. Default is None, in which case the bulk BZ is plotted.
        To plot BZ corresponding to a specific surface, provide the miller indices as a
        tuple, e.g. `surface=(1, 1, 0)`.
    path : str | None, optional
        Path to show on the plot. Default is an empty string, in which case no path is
        shown. To show a default path for a DFT calculation, set `path=None`. To show a
        custom path, provide a string with the k-points, where different paths can be
        separated by a comma, e.g. `path=GZR,MX`. If no special_points are provided, the
        path must use only standard high-symmetry points in the BZ. If special_points
        are provided, the path can only use the special points specified.
    special_points : dict | None, optional
        Dictionary of special points to label on the BZ. Default is None, in which case
        standard special points are plotted. To plot specific points, pass a dictionary
        of the form `{"point_label": [kx, ky, kz]}`. To plot no special points, pass an
        empty dictionary `{}`.
    vectors : bool, optional
        if True, show the vector.
    azim : float | None, optional
        Azimuthal angle in degrees for viewing 3D BZ.
    elev : float | None, optional
        Elevation angle in degrees for viewing 3D BZ.
    scale : float | int, optional
        Scale the BZ. Defaults to 1, which gives the size in inv. Angstrom.
    rotate : float | int | list[float, float, float], optional
        Rotate the BZ by angle (in degrees). Defaults to 0. If a float or int is passed,
        the BZ is rotated around the z-axis by the specified angle. If a list of three
        floats is passed, the BZ is rotated by that rotation vector.
        For the vector rotation, see also :class:`scipy.spatial.transform.Rotation`.
    repeat: Tuple[int, int] | Tuple[int, int, int], optional
        Set the repeating draw of BZ. default is (1, 1, 1), no repeat.
    ax : Axes | Axes3D, optional
        matplolib Axes (Axes3D in 3D) object
    show : bool, optional
        If true, show the figure.
    show_axes : bool, optional
        If True, show the axes, if False, hide the axes, if None default to behaviour of
        existing plot, or hide the axes if no plot exists.
    **kwargs
        Additional keyword arguments to pass to ax.plot




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

    # Determine the rotation and scaling transformations
    # Rotation
    if isinstance(rotate, (float, int)):
        rotate = [0, 0, rotate]
    elif not isinstance(rotate, (list, np.ndarray)):
        raise ValueError("Invalid input for rotate. Must be a float, int, or list.")
    r = Rotation.from_rotvec(rotate, degrees=True)

    # Scaling
    if not isinstance(scale, (float, int)):
        raise ValueError("Invalid input for scale. Must be a float or int.")
    # Default from ase is BZ in units of 2pi, so we need to scale it by 2pi
    scale *= 2 * np.pi
    scaling_transform = LinearScalingTransform([scale, scale, scale])

    transforms = [r, scaling_transform]
    lattice.plot_bz(
        path=path,
        special_points=special_points,
        vectors=vectors,
        azim=np.radians(azim),
        elev=np.radians(elev),
        transforms=transforms,
        repeat=repeat,
        ax=ax,
        show=show,
        **kwargs,
    )
    if ax:
        show_axes = show_axes if show_axes is not None else True
    ax = ax or plt.gca()
    if show_axes is not None:
        ax.axis("on" if show_axes else "off")


def plot_bz_section(
    structure_or_lattice,
    plane_origin=None,
    plane_normal=None,
    repeat=1,
    ax=None,
    show=False,
    show_axes=None,
    **kwargs,
):
    """Plot a cross section through the bulk Brillouin zone.

    Parameters
    ----------
    structure_or_lattice : ase.Atoms | ase.lattice.BravaisLattice
        The crystal structure or Bravais lattice representation of the material.
    plane_origin : list | np.ndarray, optional
        Origin of the plane. Default is [0, 0, 0].
    plane_normal : list | np.ndarray, optional
        Normal vector to the plane. Default is [0, 0, 1].
    repeat : int, optional
        Number of times to repeat the Brillouin zone in each direction.
    ax : Axes, optional
        Matplotlib Axes object.
    show : bool, optional
        If True, show the figure.
    show_axes : bool, optional
        If True, show the axes, if False, hide the axes, if None default to behaviour of
        existing plot, or hide the axes if no plot exists.
    **kwargs
        Additional keyword arguments to pass to ax.plot
    """

    if plane_origin is None:
        plane_origin = [0, 0, 0]
    if plane_normal is None:
        plane_normal = [0, 0, 1]

    atoms, cell, lattice = _get_atoms_cell_lattice(structure_or_lattice)
    reciprocal_cell = atoms.cell.reciprocal() * 2 * np.pi  # in inv. Angstrom

    # Define the offsets
    offsets = np.array(
        [
            [i, j, k]
            for i in range(-repeat, repeat + 1)
            for j in range(-repeat, repeat + 1)
            for k in range(-repeat, repeat + 1)
        ]
    )
    offset_vectors = np.dot(offsets, reciprocal_cell)

    # Get the original BZ vertices
    bz_faces = bz_vertices(reciprocal_cell)
    vertices_original = np.vstack([face[0] for face in bz_faces])

    # Set up plot
    if not ax:
        fig, ax = plt.subplots()

    for offset in offset_vectors:
        vertices = vertices_original + offset

        # Compute convex hull for the 3D Brillouin zone
        hull = ConvexHull(vertices)
        mesh = trimesh.Trimesh(vertices=vertices, faces=hull.simplices)

        # Compute intersection
        slice_ = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

        if slice_ is not None:
            # Extract absolute intersection vertices
            slice_vertices = slice_.vertices

            # Find two orthonormal basis vectors orthogonal to the plane normal
            plane_normal = np.array(plane_normal)
            u, _, vh = np.linalg.svd(plane_normal.reshape(1, 3))
            basis = vh[1:3]  # Two orthonormal vectors perpendicular to plane_normal

            # Project slice vertices onto the 2D basis
            xy = slice_vertices @ basis.T

            # Robust 2D Convex Hull calculation
            hull_2D = ConvexHull(xy)
            hull_xy = xy[np.append(hull_2D.vertices, hull_2D.vertices[0])]

            # Plot the hull
            ax.plot(hull_xy[:, 0], hull_xy[:, 1], color="k", **kwargs)

    if show_axes is not None:
        ax.axis("on" if show_axes else "off")

    if show:
        plt.show()
