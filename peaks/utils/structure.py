# Functions used for crystal structure display, simple analysis, and Brillouin zone display, including ARPES cut planning
# Phil King 31/10/21

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from IPython.display import display, clear_output, HTML
from ipywidgets import widgets, interactive_output, VBox, HBox
from ipywidgets.widgets import Layout
import warnings
import ase.atoms
import ase.build
import ase.dft
import ase.dft.bz
import ase.spacegroup
import ase.visualize.plot
import ase.visualize
import pymatgen.core
import pymatgen.io.ase
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.electronic_structure.plotter import BSDOSPlotter
import gemmi
import trimesh
import os

from peaks.utils.OOP_method import add_methods
from peaks.core.fileIO import load_structure
from peaks.core.fileIO import BL_angles
from peaks.utils.misc import warning_simple, warning_standard
from peaks.utils.consts import const


def crystal(struc, **kwargs):
    '''Accepts or generates a crystal structure file in the appropriate ase.atoms.Atoms format for passing to the
    peaks structure class

        Input:
            struc - crystal structure in one of several formats:
                - existing ase.atoms.Atoms object, e.g. as loaded using load_structure()
                - existing pymatgen.core.structure.Structure object
                - string giving file path to relevant .cif file, which is then parsed.
                    Default parser is ase, can also specify `parser=pymatgen` to use pymatgen
                - string giving general label of lattice type, of the following type:
                    sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond,
                    zincblende, rocksalt, cesiumchloride, fluorite, wurtzite, or mx2.
                        for most of these, the following parameters (see ase.build.bulk) can be optionally
                        specified as kwargs:
                            name (string) - compound name, e.g. Fe2O3
                            a - a-axis lattice constant in Angstroms (float)
                            b - b-axis lattice constant in Angstroms (float)
                            c - c-axis lattice constant in Angstroms (float)
                            alpha - angle in degrees for rhombohedral lattice (float)
                            covera (default = sqrt(8/3) - c/a ratio for hcp
                            u - internal coordinate for Wurtzite structure (float)
                            orthorhomic (default = False) -  Construct orthorhombic unit cell instead of
                                primitive cell (bool)
                            cubic (default = False) - construct cubic unit cell if possible (bool)
                        for mx2 type, note only monolayers are supported (see ase.build.mx2) and the following
                        can be specified as kwargs:
                            formula (string) - compound name, e.g. MoS2
                            a (default=3.18) - a-axis lattice constant in Angstroms (float)
                            thickness (default=3.19) - slab thickness
                            kind (default '2H') - 2H or 1T (string)
                            size (default=(1,1,1)) - increased first two numbers for in-plane supercell
                            vacuum (default=None)) - vacuum gap

        Returns:
            cryst - instance of peaks crystal structure class, containing:
                .structure - conventional lattice (ase.atoms.Atoms object)
                .primitive - primitive lattice (ase.atoms.Atoms object)
                .spacegroup - information on the space group
                .symmetries - crystal symmetries of the primitive structure
                .vectors - real and reciprocal lattice vectors
                 '''

    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    if type(struc) == ase.atoms.Atoms:  # Appropriate ase.atoms.Atoms object already passed
        cryst = structure(struc)  # Pass that straight to the class
    elif type(struc) == pymatgen.core.structure.Structure:  # pymatgen structure passed
        cryst = structure(pymatgen.io.ase.AseAtomsAdaptor.get_atoms(struc))
    elif type(struc) == str:
        if '.cif' in struc:  # File path to .cif file has been passed
            if 'parser' in kwargs:
                if kwargs['parser'] == 'pymatgen':  # use pymatgen
                    try:
                        cryst = structure(pymatgen.io.ase.AseAtomsAdaptor.get_atoms(pymatgen.core.Structure.from_file(struc)))
                    except:
                        cryst = structure(load_structure(struc))
                        warn_str = 'Could not be loaded using pymatgen, parsed with ase instead. Ensure you have pymatgen installed.'
                        warnings.warn(warn_str)
                else:
                    cryst = structure(load_structure(struc))
            else:
                cryst = structure(load_structure(struc))
        elif 'mx2' in struc:  # Sepcification of MX2 structure
            cryst = structure(ase.build.mx2(**kwargs))
        else:
            cryst = structure(ase.build.bulk(crystalstructure=struc, **kwargs))

    else:
        err_str = 'Supplied structure format not supported. See help(structure) for approrpriate ways to specify the crystal structures.'
        raise Exception(err_str)

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

    return cryst



class structure():
    '''Stores conventional and primitive versions of a crystal structure and enables subsequent analysis
     and display (plotting, extracting symmetry, Brillouin Zone analysis etc.).'''

    def __init__(self, structure):
        # Structure as loaded - assumed to be a conventional unit cell
        self.structure = structure

        # Generate primitive structure
        try:  # Try and extract new primitive cell using pymatgen
             prim_pymatgen_str = pymatgen.io.ase.AseAtomsAdaptor.get_structure(self.structure.copy()).get_primitive_structure()
             self.primitive = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(prim_pymatgen_str)
        except:  # Otherwise assume that it is the same as the passed conventional one
            self.primitive = self.structure

        # Try and work out some useful properties of the spacegroup from built in functions of ASE and gemmi
        try:
            self.spacegroup = {}
            ase_sg = ase.spacegroup.get_spacegroup(self.structure)  # Get spacegroup directly from ASE Atoms object
            gemmi_sg = gemmi.find_spacegroup_by_number(ase_sg.no)  # Find s.g. by number from gemmi to access more classifications
            self.spacegroup['number'] = ase_sg.no
            self.spacegroup['symbol'] = ase_sg.symbol
            self.spacegroup['crystal_system'] = gemmi_sg.crystal_system_str()
            self.spacegroup['centrosymmetric'] = ase_sg.centrosymmetric
            self.spacegroup['symmorphic'] = gemmi_sg.is_symmorphic()
            self.spacegroup['enantiomorphic'] = gemmi_sg.is_enantiomorphic()
            self.spacegroup['sohncke'] = gemmi_sg.is_sohncke()

            # Extract symmetries of primitive cell
            ops_prim = gemmi_sg.operations()
            ops_prim.change_basis_backward(gemmi_sg.centred_to_primitive())
            self.symmetries = []
            for op in ops_prim:
                self.symmetries.append(op.triplet())
        except:
            self.spacegroup = 'unknown'
            self.symmetries = 'unknown'

        # Work out real and reciprocal space lattice vectors
        self.vectors = {}
        self.vectors['conventional'] = {}
        conv_vectors = self.structure.cell
        conv_vectors[np.abs(conv_vectors) < 1e-10] = 0  # Chop very small values (numerical noise)
        conv_recip_vectors = self.structure.cell.reciprocal()*2*np.pi
        conv_recip_vectors[np.abs(conv_recip_vectors) < 1e-10] = 0  # Chop very small values (numerical noise)
        self.vectors['conventional']['a1'] = str(conv_vectors[0,0]) + 'x + ' + str(conv_vectors[0,1]) + 'y + ' + str(conv_vectors[0,2]) + 'z '
        self.vectors['conventional']['a2'] = str(conv_vectors[1,0]) + 'x + ' + str(conv_vectors[1,1]) + 'y + ' + str(conv_vectors[1,2]) + 'z '
        self.vectors['conventional']['a3'] = str(conv_vectors[2,0]) + 'x + ' + str(conv_vectors[2,1]) + 'y + ' + str(conv_vectors[2,2]) + 'z '
        self.vectors['conventional']['a units'] = "Angstrom"
        self.vectors['conventional']['b1'] = str(conv_recip_vectors[0,0]) + 'x + ' + str(conv_recip_vectors[0,1]) + 'y + ' + str(conv_recip_vectors[0,2]) + 'z '
        self.vectors['conventional']['b2'] = str(conv_recip_vectors[1,0]) + 'x + ' + str(conv_recip_vectors[1,1]) + 'y + ' + str(conv_recip_vectors[1,2]) + 'z '
        self.vectors['conventional']['b3'] = str(conv_recip_vectors[2,0]) + 'x + ' + str(conv_recip_vectors[2,1]) + 'y + ' + str(conv_recip_vectors[2,2]) + 'z '
        self.vectors['conventional']['b units'] = "1/Angstrom"

        self.vectors['primitive'] = {}
        prim_vectors = self.primitive.cell
        prim_vectors[np.abs(prim_vectors) < 1e-10] = 0  # Chop very small values (numerical noise)
        prim_recip_vectors = self.primitive.cell.reciprocal()*2*np.pi
        prim_recip_vectors[np.abs(prim_recip_vectors) < 1e-10] = 0  # Chop very small values (numerical noise)
        self.vectors['primitive']['a1'] = str(prim_vectors[0, 0]) + 'x + ' + str(prim_vectors[0, 1]) + 'y + ' + str(prim_vectors[0, 2]) + 'z '
        self.vectors['primitive']['a2'] = str(prim_vectors[1, 0]) + 'x + ' + str(prim_vectors[1, 1]) + 'y + ' + str(prim_vectors[1, 2]) + 'z '
        self.vectors['primitive']['a3'] = str(prim_vectors[2, 0]) + 'x + ' + str(prim_vectors[2, 1]) + 'y + ' + str(prim_vectors[2, 2]) + 'z '
        self.vectors['primitive']['a units'] = "Angstrom"
        self.vectors['primitive']['b1'] = str(prim_recip_vectors[0, 0]) + 'x + ' + str(prim_recip_vectors[0, 1]) + 'y + ' + str(prim_recip_vectors[0, 2]) + 'z '
        self.vectors['primitive']['b2'] = str(prim_recip_vectors[1, 0]) + 'x + ' + str(prim_recip_vectors[1, 1]) + 'y + ' + str(prim_recip_vectors[1, 2]) + 'z '
        self.vectors['primitive']['b3'] = str(prim_recip_vectors[2, 0]) + 'x + ' + str(prim_recip_vectors[2, 1]) + 'y + ' + str(prim_recip_vectors[2, 2]) + 'z '
        self.vectors['primitive']['b units'] = "1/Angstrom"

        # Set up store for the materials project ID
        self.mp_id = None

        # Set up store for band structure and DOS
        self.band_str = None
        self.DOS = None

    def set_mp_id(self, mp_id):
        self.mp_id = mp_id

    def sites(self, primitive=False):
        '''Display the sites of the strcutrue in a nice form (pandas dataframe), with automatic
          estimations of the oxidation states.

            Input parameters:
                primitive (optional, bool) - display the primative or conventional cell (default=False)

            Output:
                pandas dataframe listing atomic positions and oxidation states'''

        # Convert to pymatgen structure
        if primitive:
            struct = pymatgen.io.ase.AseAtomsAdaptor.get_structure(self.primitive.copy())
        else:
            struct = pymatgen.io.ase.AseAtomsAdaptor.get_structure(self.structure.copy())

        # Make estimates for oxidation states, using pymatgen.core.composition.Composition
        struct.add_oxidation_state_by_guess()

        # Pass this data to a pandas dataframe
        df = struct.as_dataframe()

        # Chop the very small values to zero
        num = df._get_numeric_data()
        num[num < 1e-10] = 0

        # Return the dataframe
        return df


    def surf(self, miller, primitive=False):
        '''Make a given index surface.

                Input parameters:
                    miller (tuple) - miller indicies in format (h, k, l)
                    primitive (optional, bool) - display the primative or conventional cell (default=False)

                Output:
                    new instance of structure class, now for the surface structure
                '''
        if primitive:
            return structure(ase.build.surface(self.primitive, miller, 1))
        else:
            return structure(ase.build.surface(self.structure, miller, 1))


    def plot(self, primitive=False, rotation='90x, 0y, 0z', repeats=(2, 2, 1), radii=0.5, **kwargs):
        '''Plot the crystal structure; thin wrapper for ase.visualize.plot.plot_atoms.

        Input parameters:
            primitive (optional, bool) - display the primative or conventional cell (default=False)
            rotation (optional, str) - In degrees. In the form '90x, 0y, 0z' (which is the defualt)
            repeats (optional, tuple) - number of unit cells to repeat in each direction
            radii (optional, float) - the radii of the atoms (default = 0.5)
            **kwargs (additional optional arguments to pass to ase.visualize.plot.plot_atoms):
                ax : Matplotlib subplot object
                show_unit_cell : int, optional, default 2
                    Draw the unit cell as dashed lines depending on value:
                    0: Don't
                    1: Do
                    2: Do, making sure cell is visible
           colors : list of strings, optional
                Color of the atoms, must be the same length as
                the number of atoms in the atoms object.
            scale : float, optional
                Scaling of the plotted atoms and lines.
            offset : tuple (float, float), optional
                Offset of the plotted atoms and lines.

        '''

        multiplier = np.identity(3) * repeats
        if primitive == False:
            supercell = ase.build.supercells.make_supercell(self.structure, multiplier)
        else:
            supercell = ase.build.supercells.make_supercell(self.primitive, multiplier)

        ase.visualize.plot.plot_atoms(supercell, rotation=rotation, radii=0.5, **kwargs)


    def view(self, primitive=False):
        '''Launch interactive view of the crystal structure; thin wrapper for ase.visualize.view.

                Input parameters:
                    primitive (optional, bool) - display the primative or conventional cell (default=False)

        '''

        if primitive:
            ase.visualize.view(self.primitive)
        else:
            ase.visualize.view(self.structure)


    def plot_bz(self, primitive=True, surf=None, bzs=None, vectors=True, path=None, sym_points=True, sym_labels=True,
                rot_angle=None, rot_axis=(0,0,1), elev=None, azim=None, pointstyle=None, ax=None, hide_axes=True, show=False):
        '''Plot the Brillouin zone; Taken largely from ase.dft.bz.bz_plot with some minor modifications

        Input parameters:
            primitive (optional, bool) - display the primative or conventional cell (default=True)
            surf (optional) - surf=(h,k,l) displays the surface Brillouin zone corresponding
              to the (h,k,l) surface; defaults to displaying the bulk zone
            bzs (optional) - which Brillouin zones to display:
                - None: defaults to (0,0) or (0,0,0)
                - int: plots int zones
                - list of tuples: plots specified bzs,
                    e.g. [(0,0),(0,1)] for a 2D case or [(0,0,0),(0,0,1),(0,0,2)] for a 3D case
            vectors (optional, bool) - display the lattice vectors (default = True)
            path (optional, list of str) - Supply list of special k-points to draw path between,
                e.g. ['G','X','M','G']
            sym_points (optional, bool) - show the high symmetry points (default = True)
            sym_labels (optional, bool) - label the high symmetry points (default = True)
            rot_angle (optional, float) - rotates the BZ in the x-y plane by the specified angle (in degrees),
              for 2D brillouin zones only
            rot_axis (optional, float) - axis to rotate the crystal about for BZ rotation, default = (0,0,1)
            elev (optional, float) - elevation view angle (in degrees)
            azim (optional, float) - azimuthal view angle (in degrees)
            pointstyle (optional) - definition of points shown in format {'c': 'k'}
            ax (optional) - matplotlib axis to plot on
            hide_axes (optional, bool) - hide the plot axes, default=True
            show (optional, bool) - call plt.show() at end of script (default = False)

        Returns:
            plot                '''

        if primitive:
            struct = self.primitive.copy()
        else:
            struct = self.structure.copy()

        if surf is not None:
            struct = self.surf(surf, primitive=primitive).structure.copy()

        if rot_angle is not None:  # Apply a cell rotation
            struct.rotate(a=rot_angle, v=rot_axis, rotate_cell=True)

        cell = struct.cell.copy()
        if sym_points:  # If high symmetry points are to be shown
            kpoints_dict = struct.special_kpoints(in_k=True)  # Dictionary of k-points in inv. Angst

        if ax is None:
            fig = plt.gcf()

        dimensions = cell.rank

        if dimensions == 3:

            Axes3D  # silence pyflakes

            class Arrow3D(FancyArrowPatch):
                def __init__(self, xs, ys, zs, *args, **kwargs):
                    FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                    self._verts3d = xs, ys, zs

                def do_3d_projection(self, renderer=None):
                    xs3d, ys3d, zs3d = self._verts3d
                    xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
                    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

                    return np.min(zs)

                def draw(self, renderer):
                    xs3d, ys3d, zs3d = self._verts3d
                    xs, ys, zs = proj3d.proj_transform(xs3d, ys3d,
                                                       zs3d, ax.axes.M)
                    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                    FancyArrowPatch.draw(self, renderer)

            azim = azim or np.degrees(np.pi / 5)
            elev = elev or np.degrees(np.pi / 6)
            azim = np.radians(azim)
            elev = np.radians(elev)
            x = np.sin(azim)
            y = np.cos(azim)
            view = [x * np.cos(elev), y * np.cos(elev), np.sin(elev)]

            if ax is None:
                ax = fig.add_subplot(projection='3d')
        elif dimensions == 2:
            # 2d in xy
            assert all(abs(cell[2][0:2]) < 1e-6) and all(abs(cell.T[2]
                                                             [0:2]) < 1e-6)
            ax = plt.gca()
            cell = cell.copy()
        else:
            # 1d in x
            assert (all(abs(cell[2][0:2]) < 1e-6) and
                    all(abs(cell.T[2][0:2]) < 1e-6) and
                    abs(cell[0][1]) < 1e-6 and abs(cell[1][0]) < 1e-6)
            ax = plt.gca()
            cell = cell.copy()

        icell = cell.reciprocal()

        bz1 = ase.dft.bz.bz_vertices(icell, dim=dimensions)

        maxp = 0.0
        minp = 0.0
        gmax = [0,0,0]
        gmin = [0,0,0]

        for points, normal in bz1:
            x, y, z = np.concatenate([points, points[:1]]).T
            x *= 2 * np.pi
            y *= 2 * np.pi
            z *= 2 * np.pi

            if dimensions == 3:
                if np.dot(normal, view) < 0:
                    ls = ':'
                else:
                    ls = '-'
                if bzs is not None: # Call to plot multiple of specific BZs
                    if type(bzs) is list:  # Specific bzs requested
                        for i in bzs:
                            g = (i[0] * icell[0] + i[1] * icell[1] + i[2] * icell[2]) * 2 *np.pi
                            ax.plot(x + g[0], y + g[1], z + g[2], c='k', ls=ls)
                            gmax = [max(g[0], gmax[0]), max(g[1], gmax[1]), max(g[2], gmax[2])]
                            gmin = [min(g[0], gmin[0]), min(g[1], gmin[1]), min(g[2], gmin[2])]
                    else:  # Show all bzs in a range
                        for i in range(-bzs+1, bzs):
                            for j in range(-bzs+1, bzs):
                                for k in range(-bzs+1, bzs):
                                    g = (i * icell[0] + j * icell[1] + k * icell[2]) * 2 *np.pi
                                    ax.plot(x + g[0], y + g[1], z + g[2], c='k', ls=ls)
                                    gmax = [max(g[0], gmax[0]), max(g[1], gmax[1]), max(g[2], gmax[2])]
                                    gmin = [min(g[0], gmin[0]), min(g[1], gmin[1]), min(g[2], gmin[2])]
                else: # just first bz
                    ax.plot(x, y, z, c='k', ls=ls)

            elif dimensions == 2:
                if bzs is not None: # Call to plot multiple of specific BZs
                    if type(bzs) is list:  # Specific bzs requested
                        for i in bzs:
                            g = (i[0] * icell[0] + i[1] * icell[1]) * 2 * np.pi
                            ax.plot(x + g[0], y + g[1], c='k', ls='-')
                    else:  # Show all bzs in a range
                        for i in range(-bzs+1, bzs):
                            for j in range(-bzs+1, bzs):
                                g = (i * icell[0] + j * icell[1]) * 2 * np.pi
                                ax.plot(x + g[0], y + g[1], c='k', ls='-')
                else: # just first bz
                    ax.plot(x, y, c='k', ls='-')

            maxp = max(maxp, points.max())
            minp = min(minp, points.min())

        def draw_axis3d(ax, vector):
            ax.add_artist(Arrow3D(
                [0, vector[0]],
                [0, vector[1]],
                [0, vector[2]],
                mutation_scale=20,
                arrowstyle='-|>',
                color='k',
            ))

        def draw_axis2d(ax, x, y):
            ax.arrow(0, 0, x, y,
                     lw=1, color='k',
                     length_includes_head=True,
                     head_width=0.15,
                     head_length=0.1)

        if vectors:
            if dimensions == 3:
                for i in range(3):
                    draw_axis3d(ax, vector=icell[i]*2*np.pi)
                maxp = max(maxp, 0.6 * icell.max())
            elif dimensions == 2:
                for i in range(2):
                    draw_axis2d(ax, icell[i, 0] * 2 * np.pi, icell[i, 1] * 2 * np.pi)
                maxp = max(maxp, icell.max())
            else:
                draw_axis2d(ax, icell[0, 0], 0)
                maxp = max(maxp, icell.max())

        if path is not None:
            x = []
            y = []
            z = []
            for i in path:
                x.append(kpoints_dict[i][0])
                y.append(kpoints_dict[i][1])
                z.append(kpoints_dict[i][2])
            if dimensions == 3:
                ax.plot(x, y, z, c='r', ls='-', marker='.')
            elif dimensions == 2:
                ax.plot(x, y, c='r', ls='-')

        if sym_points:  # If symmetry points to be shown
            # Extract relevant k-points and labels from special points dictionary
            kpoints = []
            kpoints_label = []
            for i in kpoints_dict:
                if type(kpoints_dict[i]) is not str:
                    kpts_array = kpoints_dict[i]
                    if dimensions == 2 and rot_angle is not None:  # For 2D rotations, the special k-points aren't properly updated; correct these manually
                        theta = np.radians(rot_angle)
                        rot_mat = np.array(( (np.cos(theta), -np.sin(theta), 0),
                                             (np.sin(theta),  np.cos(theta), 0),
                                             (0, 0, 1)))
                        kpts_array = rot_mat.dot(kpts_array.T)
                    kpoints.append(kpts_array)
                    kpoints_label.append(i)
            kw = {'c': 'r'}
            if pointstyle is not None:
                kw.update(pointstyle)
            for count, p in enumerate(kpoints):
                if dimensions == 3:
                    ax.scatter(p[0], p[1], p[2], **kw)
                    if sym_labels:
                        ax.text(p[0], p[1], p[2], '%s' % (str(kpoints_label[count])), size=20, zorder=1,
                                color='k')
                elif dimensions == 2:
                    ax.scatter(p[0], p[1], zorder=4, **kw)
                    if sym_labels:
                        ax.annotate(kpoints_label[count], xy=(p[0],p[1]), size=20, zorder=1)

        if hide_axes:
            ax.set_axis_off()
        else:
            ax.set_xlabel('k_x')
            ax.set_ylabel('k_y')
            if dimensions == 3:
                ax.set_zlabel('k_z')

        if dimensions in [1, 2]:
            ax.autoscale_view(tight=True)
            ax.set_aspect('equal')

        if dimensions == 3:
            # ax.set_aspect('equal') <-- won't work anymore in 3.1.0
            ax.view_init(azim=azim / np.pi * 180, elev=elev / np.pi * 180)
            # We want aspect 'equal', but apparently there was a bug in
            # matplotlib causing wrong behaviour.  Matplotlib raises
            # NotImplementedError as of v3.1.0.  This is a bit unfortunate
            # because the workarounds known to StackOverflow and elsewhere
            # all involve using set_aspect('equal') and then doing
            # something more.
            #
            # We try to get square axes here by setting a square figure,
            # but this is probably rather inexact.
            fig = ax.get_figure()
            xx = plt.figaspect(1.0)
            fig.set_figheight(xx[1])
            fig.set_figwidth(xx[0])

            # oldlibs tests with matplotlib 2.0.0 say we have no set_proj_type:
            if hasattr(ax, 'set_proj_type'):
                ax.set_proj_type('ortho')

            minp0 = 0.9 * minp  # Here we cheat a bit to trim spacings
            maxp0 = 0.9 * maxp
            if bzs is not None:
                ax.set_xlim3d(minp0 * 2 * np.pi + gmin[0], maxp0 * 2 * np.pi + gmax[0])
                ax.set_ylim3d(minp0 * 2 * np.pi + gmin[1], maxp0 * 2 * np.pi + gmax[1])
                ax.set_zlim3d(minp0 * 2 * np.pi + gmin[2], maxp0 * 2 * np.pi + gmax[2])
            else:
                ax.set_xlim3d(minp0 * 2 * np.pi, maxp0 * 2 * np.pi)
                ax.set_ylim3d(minp0 * 2 * np.pi, maxp0 * 2 * np.pi)
                ax.set_zlim3d(minp0 * 2 * np.pi, maxp0 * 2 * np.pi)

            if hasattr(ax, 'set_box_aspect'):
                ax.set_box_aspect([1, 1, 1])
            else:
                msg = ('Matplotlib axes have no set_box_aspect() method.  '
                       'Aspect ratio will likely be wrong.  '
                       'Consider updating to Matplotlib >= 3.3.')
                warnings.warn(msg)

        if show:
            plt.show()

        return ax

    def plot_bz_section(self, primitive=True, bzs=None, plane_normal=[0,0,1], plane_origin=[0,0,0],
                    rot_angle=None, ax=None, hide_axes=True, show=False, **kwargs):
        '''Plots a section through one or repreated 3D Brillouin zones for a given plane and origin

        Input parameters:
            primitive (optional, bool) - display the primative or conventional cell (default=True)
            bzs (optional) - which Brillouin zones to display:
                - None: defaults to (0,0) or (0,0,0)
                - int: plots int zones
                - list of tuples: plots specified bzs,
                    e.g. [(0,0),(0,1)] for a 2D case or [(0,0,0),(0,0,1),(0,0,2)] for a 3D case
            plane_normal (optional) - normal of plane to cut the BZ in, NB in cartesian coordinates
                default = [0,0,1]
            plane_origin (optional) - origin of plane to cut BZ in, NB in cartesian coordinates:
                - default = [0,0,0]
                - a specified value [x,y,z] in cartesian coordinates
                - a labelled high symmetry point, e.g. "G"
            rot_angle (optional) - in-plane rotation angle of the BZ section
            ax (optional) - matplotlib axis to plot on
            hide_axes (optional, bool) - hide the plot axes, default=True
            show (optional, bool) - call plt.show() at end of script (default = False)
            **kwargs - additional arguments to pass to the plotting function (standard matplotlib calls)

        Returns:
            plot                '''

        if primitive:
            struct = self.primitive.copy()
        else:
            struct = self.structure.copy()

        cell = struct.cell.copy()


        # Get reciprocal cell
        icell = cell.reciprocal()

        # Get BZ vertices
        bz1 = ase.dft.bz.bz_vertices(icell)

        # To convert ase BZ notation (vertices and normals) to a face-type polyhedron notation
        def to_face_not(bz, shift):
            vertices = []
            faces = []
            count0 = 0

            for i, j in bz:
                count1 = count0 + 1
                nvert = len(i)
                for k in i:  # Add vertices to list
                    k[np.abs(k) < 1e-10] = 0
                    vertices.append((k + shift) * 2 * np.pi)  # Append vertices (and convert to inv. Angstrom)
                for k in range(nvert - 2):
                    faces.append([count0, count1, count1 + 1])  # Define faces and connectivity
                    count1 += 1
                count0 = count1 + 1

            return faces, vertices

        # For in-plane rotation of 2D mesh
        def rot2D(coords_in, theta):
            alpha = np.radians(theta)
            rot_mat = [[np.cos(alpha), -np.sin(alpha)],
                       [np.sin(alpha), np.cos(alpha)]]

            output = np.dot(rot_mat, coords_in.T)
            return output[0], output[1]

        # Define a trimesh for the BZ polyhedron and take a cross section
        def cross_section(bz, plane_origin, plane_normal, rot_angle, shift=[0,0,0]):
            faces, vertices = to_face_not(bz, shift)  # Shift shifts the whole BZ, to plot neighbouring BZs

            # Make mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            slice_ = mesh.section(plane_origin=plane_origin,
                                  plane_normal=plane_normal)

            # transformation matrix for to_planar
            to_2D = trimesh.geometry.align_vectors(plane_normal, [0, 0, -1])

            slice_2D, to_3D = slice_.to_planar(to_2D=to_2D)

            if rot_angle is not None:
                for i in slice_2D.vertices:
                    i[0], i[1] = rot2D(np.asarray(i), rot_angle)

            return slice_2D

        # Set up the plot
        if ax is None:
            fig = plt.gcf()
            ax = plt.gca()

        # Set some default options for the plot
        if 'color' not in kwargs:
            kwargs['color'] = 'k'  # Set black default

        # Determine plane origin - if a string is passed, then the high symmetry label has been passed
        # to define the plane origin, otherwise the normal has been defined directly
        if type(plane_origin) is str:
            plane_origin = struct.special_kpoints()[plane_origin]

        if bzs is not None:  # Call to plot multiple of specific BZs
            if type(bzs) is list:  # Specific bzs requested
                for i in bzs:
                    try:
                        shift = i[0] * icell[0] + i[1] * icell[1] + i[2] * icell[2]
                        slice = cross_section(bz1, plane_origin=plane_origin, plane_normal=plane_normal, rot_angle=rot_angle, shift=shift)
                        for points in slice.discrete:
                            ax.plot(*points.T, **kwargs)
                    except:
                        pass
            else:  # Show all bzs in a range
                for i in range(-bzs + 1, bzs):
                    for j in range(-bzs + 1, bzs):
                        for k in range(-bzs + 1, bzs):
                            try:
                                shift = (i * icell[0]) + (j * icell[1]) + (k * icell[2])
                                slice = cross_section(bz1, plane_origin=plane_origin, plane_normal=plane_normal, rot_angle=rot_angle, shift=shift)
                                for points in slice.discrete:
                                    ax.plot(*points.T, **kwargs)
                            except:
                                pass

        else:  # just first bz
            # Plot the section
            slice = cross_section(bz1, plane_origin=plane_origin, plane_normal=plane_normal, rot_angle=rot_angle)
            for i, points in enumerate(slice.discrete):
                ax.plot(*points.T, **kwargs)


        # Finish configuring the plot
        ax.autoscale_view(tight=True)
        ax.set_aspect('equal')

        if hide_axes:
            ax.set_axis_off()
        if show:
            plt.show()

        return ax


    def cut_planner(self, primitive=True, surf=None, bzs=None, vectors=False, path=None,
                    sym_points=True, sym_labels=True, rot_angle=None, rot_axis=(0,0,1), elev=None, azim=None, hide_axes=False):
        '''Cut planned for aiding ARPES alignment.

        Input parameters:
            primitive (optional, bool) - display the primative or conventional cell (default=True)
            surf (optional) - surf=(h,k,l) displays the surface Brillouin zone corresponding
              to the (h,k,l) surface; defaults to displaying the bulk zone
            bzs (optional) - which Brillouin zones to display:
                - None: defaults to (0,0) or (0,0,0)
                - int: plots int zones
                - list of tuples: plots specified bzs,
                    e.g. [(0,0),(0,1)] for a 2D case or [(0,0,0),(0,0,1),(0,0,2)] for a 3D case
            vectors (optional, bool) - display the lattice vectors (default = True)
            path (optional, list of str) - Supply list of special k-points to draw path between,
                e.g. ['G','X','M','G']
            sym_points (optional, bool) - show the high symmetry points (default = True)
            sym_labels (optional, bool) - label the high symmetry points (default = True)
            rot_angle (optional, float) - rotates the BZ in the x-y plane by the specified angle (in degrees),
              for 2D brillouin zones only
            rot_axis (optional, float) - axis to rotate the crystal about for BZ rotation, default = (0,0,1)
            elev (optional, float) - elevation view angle (in degrees)
            azim (optional, float) - azimuthal view angle (in degrees)
            hide_axes (optional, bool) - hide the plot axes, default=False


        Returns:
            cut planner widget                '''

        cut_planner2d(self.structure, primitive, surf, bzs, vectors, path, sym_points, sym_labels, rot_angle, rot_axis, elev, azim, hide_axes)

    def plot_xrd(self, primitive=False, wavelength='CuKa', two_theta_range=(0,90), annotate_peaks='compact',
                 symprec=0, debye_waller_factors=None, ax=None, with_labels=True, fontsize=16, ylog_scale=True, show=False):
        '''Calculate and plot the powder x-ray diffraction pattern for the crystal, using the
            pymatgen.analysis.diffraction.xrd module.

            Input parameters:
                primitive (optional, bool) - calculate for the primative or conventional cell (default=False)
                  wavelength (str/float) – The wavelength can be specified as either a float or a string.
                  If it is a string, it must be one of:
                    'CuKa', 'CuKa2', 'CuKa1', 'CuKb1', 'MoKa', 'MoKa2', 'MoKa1', 'MoKb1',
                    'CrKa', 'CrKa2', 'CrKa1', 'CrKb1', 'FeKa', 'FeKa2', 'FeKa1', 'FeKb1',
                    'CoKa', 'CoKa2', 'CoKa1', 'CoKb1', 'AgKa', 'AgKa2', 'AgKa1', 'AgKb1'
                  If it is a float, it is interpreted as a wavelength in angstroms.
                    Defaults to “CuKa”, i.e, Cu K_alpha radiation.
                two_theta_range ([float of length 2]) – Tuple for range of two_thetas to
                  calculate in degrees. Defaults to (0, 90). Set to None if you want all
                  diffracted beams within the limiting sphere of radius 2 / wavelength.
                annotate_peaks (str or None) – Whether and how to annotate the peaks with hkl indices.
                  Default is ‘compact’, i.e. show short version (oriented vertically), e.g. 100.
                  If ‘full’, show long version, e.g. (1, 0, 0). If None, do not show anything.
                symprec (float) – Symmetry precision for structure refinement.
                  If set to 0 (default), no refinement is done. Otherwise,
                  refinement is performed using spglib with provided precision.
                ({element symbol (debye_waller_factors) – float}): Allows the specification of
                  Debye-Waller factors. Note that these factors are temperature dependent.
                ax – matplotlib Axes or None if a new figure should be created.
                with_labels – True to add xlabels and ylabels to the plot.
                fontsize – (int) fontsize for peak labels.
                ylog_scale (bool) - display with a log scale for the y-axis; defaults to True
                show (optional, bool) - call plt.show() at end of script (default = False)

            Returns:
                ax

              '''

        # Get correct structure in pymatgen format
        if primitive:
            struct = pymatgen.io.ase.AseAtomsAdaptor.get_structure(self.primitive.copy())
        else:
            struct = pymatgen.io.ase.AseAtomsAdaptor.get_structure(self.structure.copy())

        # Set up the calculator
        calc = XRDCalculator(wavelength, symprec, debye_waller_factors)

        # Make the plot
        calc.get_plot(struct, two_theta_range, annotate_peaks, ax, with_labels, fontsize)

        # Get current axes
        ax = plt.gca()

        # Set the y-axis scale
        if ylog_scale:
            ax.set_yscale('log')

        # Do final plt.show() if requested
        if show:
            plt.show()

        return ax

    def get_band_str(self, mp_id=None, API_key=None):
        '''Attempt to automatically get the band structure of a compound from the Materials Project database.
            Requires an API key which can be generated by creating an account at https://materialsproject.org

            Input:
                mp_id (optional, str) - materials project ID of the structure. If None is provided,
                  and it has not already been defined in the structure loading, then it will attempt
                  to determine the relevant ID, and if there are multiple options, list these and ask
                  for a specification to be made
                API_key (str) - API key for accessing the materials project database. Not required
                  if key is stored in mp_API_key.txt in the .utils directory

            Output:
                The band structure and DOS are added to .band_str and .DOS
                 '''

        # Set up warnings display and formatting
        warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
        warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

        if API_key is None:
            API_key = get_API_key()

        # If materials project ID specified in function call
        if mp_id is not None:
            self.mp_id = mp_id

        # If materials project ID is known
        if self.mp_id is not None:
            warn_str = 'Attempting to download band structure and DOS from Materials Project repository. ' \
                       'NB, these are typically performed at a low level of DFT (normally plain LDA) so treat ' \
                       'these only as a very rough guide.'
            warnings.warn(warn_str)
            with MPRester(API_key) as mpr:
                try:
                    self.band_str = mpr.get_bandstructure_by_material_id(self.mp_id)
                except:
                    print('No band structure available')
                try:
                    self.DOS = mpr.get_dos_by_material_id(self.mp_id)
                except:
                    print('No DOS available')
        else:
            # Get Chemical Formula
            compound = self.structure.symbols.get_chemical_formula()
            sg_num = self.spacegroup['number']

            # Try and get the record from materials project with the above information
            print('Attempting to find bandstructure on Materials Project from compound name and space group')
            struct = get_structure(compound=compound, sg_num=sg_num)

            # Try and download band structure
            try:
                self.mp_id = struct.mp_id
                warn_str = 'Attempting to download band structure and DOS from Materials Project repository. ' \
                           'NB, these are typically performed at a low level of DFT (normally plain LDA) so treat ' \
                           'these only as a very rough guide.'
                warnings.warn(warn_str)
                with MPRester(API_key) as mpr:
                    try:
                        self.band_str = mpr.get_bandstructure_by_material_id(self.mp_id)
                    except:
                        print('No band structure available')
                    try:
                        self.DOS = mpr.get_dos_by_material_id(self.mp_id)
                    except:
                        print('No DOS available')
            except:
                pass

        # Reset warnings display and formatting
        warnings.simplefilter('once', UserWarning)  # Give warnings first time only
        warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

    def plot_band_str(self, mp_id = None, API_key = None, **kwargs):
        '''Plots the band structure and DOS obtained from Materials Project. If these are not already loaded
            it will attempt to automatically get the band structure from the Materials Project database, or
            load one as defined in the function call.
            Requires an API key which can be generated by creating an account at https://materialsproject.org

            Input:
                mp_id (optional, str) - materials project ID of the structure. If None is provided,
                  and it has not already been defined in the structure loading, then it will attempt
                  to determine the relevant ID, and if there are multiple options, list these and ask
                  for a specification to be made
                API_key (str) - API key for accessing the materials project database. Not required
                  if key is stored in mp_API_key.txt in the .utils directory
                kwargs (optional arguments to pass to pymatgen.electronic_structure.plotter.BSPlotter:
                    bs_projection (str) – “elements” or None
                    dos_projection (str) – “elements”, “orbitals”, or None
                    vb_energy_range (float) – energy in eV to show of valence bands
                    cb_energy_range (float) – energy in eV to show of conduction bands
                    fixed_cb_energy (bool) – If true, the cb_energy_range will be interpreted
                      as constant (i.e., no gap correction for cb energy)
                    egrid_interval (float) – interval for grid marks
                    font (str) – font family
                    axis_fontsize (float) – font size for axis
                    tick_fontsize (float) – font size for axis tick labels
                    legend_fontsize (float) – font size for legends
                    bs_legend (str) – matplotlib string location for legend or None
                    dos_legend (str) – matplotlib string location for legend or None
                    rgb_legend (bool) – (T/F) whether to draw RGB triangle/bar for element proj.
                    fig_size (tuple) – dimensions of figure size (width, height)

            Output:
                A plot is shown
             '''

        # If no band structure stored, try and get it automatically
        if self.band_str is None:
            self.get_band_str(mp_id, API_key)

        # Chcek if it is still none, in which case can't proceed
        if self.band_str is not None:
            plotter = BSDOSPlotter(**kwargs)
            # Set defaults for projections (can be overridden by corresponding entries in kwargs
            if 'bs_projection' not in kwargs:
                plotter.bs_projection = 'elements'
            if 'dos_projection' not in kwargs:
                plotter.dos_projection = 'elements'
            plotter.get_plot(self.band_str, self.DOS)

    def get_substrates(self, orientation=None, mp_id=None, API_key=None):
        '''Gets suggested substrates from Materials Project database. Requires an API key
           which can be generated by creating an account at https://materialsproject.org

        Input:
            orientation (optional, tuple) - orientation for the film in format (h,k,l).
              If none is provided, all orientations returned.
            mp_id (optional, str) - materials project ID of the structure. If None is provided,
              and it has not already been defined in the structure loading, then it will attempt
              to determine the relevant ID, and if there are multiple options, list these and ask
              for a specification to be made
            API_key (str) - API key for accessing the materials project database. Not required
              if key is stored in mp_API_key.txt in the .utils directory

        Output:
            Pandas dataframe of substrate options
         '''

        # Set up warnings display and formatting
        warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
        warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

        if API_key is None:
            API_key = get_API_key()

        # If materials project ID specified in function call
        if mp_id is not None:
            self.mp_id = mp_id

        # If materials project ID is still not known
        if self.mp_id is None:
            # Get Chemical Formula
            compound = self.structure.symbols.get_chemical_formula()
            sg_num = self.spacegroup['number']

            # Try and get the record from materials project with the above information
            print('Attempting to find matching record on Materials Project from compound name and space group')
            struct = get_structure(compound=compound, sg_num=sg_num)

            try:
                self.mp_id = struct.mp_id
            except:
                return struct

        # Try and download relevant substrate options
        warn_str = 'Downloading substrate options from Materials Project repository. ' \
                   'NB, use these only as an initial guide.'
        warnings.warn(warn_str)
        with MPRester(API_key) as mpr:
            try:
                records = mpr.get_substrates(self.mp_id)
            except:
                print('No substrates available')
                return

        if orientation is not None:
            desired_orientation = str(orientation[0])+' '+str(orientation[1])+' '+str(orientation[2])
            substrate = []
            sub_orient = []
            sub_id = []
            mcia = []
            elast_energy = []

            for i in records:
                if i['film_orient'] == desired_orientation:
                    substrate.append((i['sub_form']))
                    sub_orient.append(i['orient'])
                    sub_id.append(i['sub_id'])
                    try:
                        mcia.append(i['area'])
                    except:
                        pass
                    try:
                        elast_energy.append(i['energy']*1000)
                    except:
                        pass

            substrates_dict ={'Substrate': substrate,
                              'Orientation': sub_orient,
                              'Materials project ID': sub_id
                              }
            if len(mcia) != 0:
                substrates_dict['Min. coincident interface area (Ang^2)'] = mcia
            if len(elast_energy) != 0:
                substrates_dict['Elastic energy (meV)'] = elast_energy

            substrates = pd.DataFrame(substrates_dict)

        else:
            substrates = pd.DataFrame(records)
            # Tidy up the names
            cols = substrates.columns  # Get current column names
            if 'area' in cols and 'energy' in cols:
                new_cols = {'sub_form': 'Substrate',
                            'sub_id': 'Materials project ID',
                            'film_orient': 'Film orientation',
                            'energy': 'Elastic energy (meV)',
                            'area': 'Min. coincident interface area (Ang^2)',
                            'orient': 'Substrate orientation'}
                substrates['energy'] = substrates['energy'].apply(lambda x: x*1000)  # Convert energy to meV
            elif 'area' in cols:
                new_cols = {'sub_form': 'Substrate',
                            'sub_id': 'Materials project ID',
                            'film_orient': 'Film orientation',
                            'area': 'Min. coincident interface area (Ang^2)',
                            'orient': 'Substrate orientation'}
            else:
                new_cols = {'sub_form': 'Substrate',
                            'sub_id': 'Materials project ID',
                            'film_orient': 'Film orientation',
                            'energy': 'Elastic energy (meV)',
                            'orient': 'Substrate orientation'}

            substrates = substrates.rename(columns=new_cols)

        display(substrates)

        # Reset warnings display and formatting
        warnings.simplefilter('once', UserWarning)  # Give warnings first time only
        warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

        return substrates

class cut_planner2d(structure):

    def __init__(self, structure, primitive, surf, bzs, vectors, path, sym_points, sym_labels, rot_angle, rot_axis, elev, azim, hide_axes):
        super().__init__(structure)
        self.primitive_init = primitive
        self.surf_init = surf
        self.bzs_init = bzs
        self.vectors_init = vectors
        self.path_init = path
        self.sym_points_init = sym_points
        self.sym_labels_init = sym_labels
        self.rot_angle_init = rot_angle
        self.rot_axis_init = rot_axis
        self.elev_init = elev
        self.azim_init = azim
        self.hide_axes_init = hide_axes

        # Get reciprocal lattice vectors
        if primitive:
            self.recip_vec = self.primitive.cell.reciprocal()*2*np.pi
        else:
            self.recip_vec = self.structure.cell.reciprocal() * 2 * np.pi


        self.hv_init = 21.2
        self.wf = 4.35
        self.inner_pot_init = 12

        self.ana_type = BL_angles.angles['default']['ana_type']

        self._create_widgets()

        # Initialise for 3D plot
        if self.surf_init is None:
            self.w['beamline'].disabled = True
            self.w['polar'].disabled = True
            self.w['tilt'].disabled = True
            self.w['defl_perp'].disabled = True
            self.w['defl_par'].disabled = True
        else:
            self.w['inner_pot'].disabled = True
            self.w['bz_shift_h'].disabled = True
            self.w['bz_shift_k'].disabled = True
            self.w['bz_shift_l'].disabled = True


        # Define callback options
        def BL_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                # Set new beamline name
                BL_name = self.w['beamline'].value
                BL_angle_dict = BL_angles.angles[BL_name]
                self.ana_type = BL_angle_dict['ana_type']

                # Change labels
                if BL_name == 'default':
                    self.w['rotation'].description = 'Azimuth'
                    self.w['polar'].description = 'Polar'
                    self.w['tilt'].description = 'Tilt'
                    self.w['defl_perp'].description = 'Defl_perp'
                    self.w['defl_par'].description = 'Defl_par'
                else:
                    self.w['rotation'].description = 'Azimuth ('+BL_angle_dict['azi_name']+')'
                    self.w['polar'].description = 'Polar (' + BL_angle_dict['polar_name'] + ')'
                    self.w['tilt'].description = 'Tilt (' + BL_angle_dict['tilt_name'] + ')'
                    self.w['defl_perp'].description = 'Defl_perp (' + BL_angle_dict['defl_perp_name'] + ')'
                    self.w['defl_par'].description = 'Defl_par (' + BL_angle_dict['defl_par_name'] + ')'

                # Disable or enable outputs
                if BL_angle_dict['polar_name'] == 'None':
                    self.w['polar'].disabled = True
                else:
                    self.w['polar'].disabled = False
                if BL_angle_dict['tilt_name'] == 'None':
                    self.w['tilt'].disabled = True
                else:
                    self.w['tilt'].disabled = False
                if BL_angle_dict['defl_perp_name'] == 'None':
                    self.w['defl_perp'].disabled = True
                else:
                    self.w['defl_perp'].disabled = False
                if BL_angle_dict['defl_par_name'] == 'None':
                    self.w['defl_par'].disabled = True
                else:
                    self.w['defl_par'].disabled = False

                # Refresh the display
                clear_output(wait=True)
                self._displayUI()
                self._set_widget_layout()


        self.w['beamline'].observe(BL_change)

        # Display the UI
        self._displayUI()
        self._set_widget_layout()



    def _plot(self, beamline, hv, ang_mode, binding_energy, rotation, polar, tilt, defl_par, defl_perp, inner_pot, bz_shift_h, bz_shift_k, bz_shift_l):
        # Initalise the axes
        fig = plt.figure(figsize=(5,5))
        if self.surf_init is not None:
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        # Plot the BZ
        self.plot_bz(primitive=self.primitive_init, surf=self.surf_init, bzs=self.bzs_init,
                     vectors=self.vectors_init, path=self.path_init, sym_points=self.sym_points_init,
                     sym_labels=self.sym_labels_init, rot_angle=rotation, rot_axis=self.rot_axis_init,
                     elev=self.elev_init, azim=self.azim_init, hide_axes=self.hide_axes_init, ax=ax)
        # Plot the cut
        if self.surf_init is not None:  # 2D cut
            #TODO Implement the sign conventions
            self.cut_angles = np.linspace(-ang_mode, ang_mode, 101, endpoint=True)  # 2D case
            from peaks.core.process import fI, fII, fIp, fIIp
            if self.ana_type == 'I':
                self.cut = fI(self.cut_angles, polar, 0, tilt, hv - binding_energy - self.wf)
                self.cut_cen = fI(0, polar, 0, tilt, hv - binding_energy - self.wf)
            elif self.ana_type == 'Ip':
                self.cut = fIp(self.cut_angles, defl_perp, 0, tilt, polar, hv - binding_energy - self.wf)
                self.cut_cen = fIp(defl_par, defl_perp, 0, tilt, polar, hv - binding_energy - self.wf)
            elif self.ana_type == 'II':
                self.cut = fII(self.cut_angles, tilt, 0, polar, hv - binding_energy - self.wf)
                self.cut_cen = fII(0, tilt, 0, polar, hv - binding_energy - self.wf)
            elif self.ana_type == 'IIp':
                self.cut = fIIp(self.cut_angles, defl_perp, 0, tilt, polar, hv - binding_energy - self.wf)
                self.cut_cen = fIIp(defl_par, defl_perp, 0, tilt, polar, hv - binding_energy - self.wf)

            ax.plot(self.cut[0],self.cut[1],'r')
            ax.plot(self.cut_cen[0],self.cut_cen[1],'o',c='k')

        else:  #3D cut
            cut_angles = np.linspace(-ang_mode, ang_mode, 51, endpoint=True)
            (cut_anglesx, cut_anglesy) = np.meshgrid(cut_angles,cut_angles)
            from peaks.core.process import fI, fII, fIp, fIIp
            for i,j in zip(cut_anglesx,cut_anglesy):
                cut_kx, cut_ky = fI(cut_anglesx,cut_anglesy,0,0,hv - binding_energy - self.wf)

            kpar = np.sqrt(np.square(cut_kx) + np.square(cut_ky))
            cut_kz = const.kvac_const*np.sqrt(inner_pot + hv - self.wf - binding_energy - ((kpar/const.kvac_const)**2))
            # shift by specified recip lattice vectors
            (g1x,g1y,g1z) = self.recip_vec[0]
            (g2x, g2y, g2z) = self.recip_vec[1]
            (g3x, g3y, g3z) = self.recip_vec[2]
            cut_kx -= bz_shift_h * g1x + bz_shift_k * g2x + bz_shift_l * g3x
            cut_ky -= bz_shift_h * g1y + bz_shift_k * g2y + bz_shift_l * g3y
            cut_kz -= bz_shift_h * g1z + bz_shift_k * g2z + bz_shift_l * g3z
            ax.plot_surface(cut_kx, cut_ky, cut_kz, linewidth=0, antialiased=True,zorder=-10)

    def _create_widgets(self):
        display(HTML('''<style>
            .widget-label { min-width: 20ex !important; }
        </style>'''))
        self.w = dict(
            beamline=widgets.Dropdown(
                options=list(BL_angles.angles),
                value='default',
                description='Beamline:',
                disabled=False,
            ),
            hv=widgets.FloatSlider(
                min=5,
                max=200,
                value=self.hv_init,
                step=0.1,
                description='Photon energy (eV)',
                disabled=False,
                continuous_update=False,
                readout=True,
                indent=True,
                orientation='horizontal',
            ),
            ang_mode=widgets.BoundedFloatText(
                value=15.,
                min=0,
                max=90.0,
                step=1,
                description='Ang. Mode (+/-)',
                disabled=False
            ),
            binding_energy=widgets.FloatText(
                value=0.,
                description='E-E_F (eV)',
                disabled=False
            ),
            rotation=widgets.FloatSlider(
                min=-90,
                max=90,
                value=self.rot_angle_init,
                step=1,
                description='Azimuth',
                disabled=False,
                continuous_update=False,
                readout=True,
                indent=True,
                orientation='horizontal',
            ),
            polar=widgets.FloatSlider(
                min=-90,
                max=90,
                value=0,
                step=0.1,
                description='Polar',
                disabled=False,
                continuous_update=False,
                readout=True,
                indent=True,
                orientation='horizontal',
            ),
            tilt=widgets.FloatSlider(
                min=-50,
                max=50,
                value=0,
                step=0.1,
                description='Tilt',
                disabled=False,
                continuous_update=False,
                readout=True,
                indent=True,
                orientation='horizontal',
            ),
            defl_perp=widgets.FloatSlider(
                min=-20,
                max=20,
                value=0,
                step=0.1,
                description='Defl_perp',
                disabled=False,
                continuous_update=False,
                readout=True,
                indent=True,
                orientation='horizontal',
            ),
            defl_par=widgets.FloatSlider(
                min=-20,
                max=20,
                value=0,
                step=0.1,
                description='Defl_par',
                disabled=False,
                continuous_update=False,
                readout=True,
                indent=True,
                orientation='horizontal',
            ),
            inner_pot=widgets.FloatText(
                value=self.inner_pot_init,
                description='V_0 (eV)',
                disabled=False
            ),
            bz_shift_h=widgets.FloatText(
                value=0,
                description='BZ shift h',
                disabled=False
            ),
            bz_shift_k=widgets.FloatText(
                value=0,
                description='k',
                disabled=False
            ),
            bz_shift_l = widgets.FloatText(
                value=0,
                description='l',
                disabled=False
        )
        )

    def _set_widget_layout(self):
        display(HTML('''<style>
                    .widget-label { min-width: 20ex !important; }
                </style>'''))
        self.w['hv'].layout = Layout(width='400px')
        self.w['ang_mode'].layout = Layout(width='300px')
        self.w['binding_energy'].layout = Layout(width='250px')
        self.w['rotation'].layout = Layout(width='350px')
        self.w['polar'].layout = Layout(width='350px')
        self.w['tilt'].layout = Layout(width='350px')
        self.w['defl_perp'].layout = Layout(width='350px')
        self.w['defl_par'].layout = Layout(width='350px')
        self.w['inner_pot'].layout = Layout(width='250px')
        self.w['bz_shift_h'].layout = Layout(width='200px')
        self.w['bz_shift_k'].layout = Layout(width='200px')
        self.w['bz_shift_l'].layout = Layout(width='200px')


    def _displayUI(self):
        output = interactive_output(self._plot, self.w)
        box = HBox([
            HBox([output]),#, layout=Layout(height='350px', width = '420px')),
            VBox([
                HBox([*self.w.values()][0:2]),
                HBox([*self.w.values()][2:4]),
                HBox([
                    VBox([*self.w.values()][4:9]),
                    VBox([*self.w.values()][9:])
                    ]),
            ]),
        ])

        display(box)




@add_methods(ase.atoms.Atoms)
def special_kpoints(struct, in_k = True, hv=None):
    '''Adding a shortcut method to the ase.atoms.Atoms class to access directly the special k-points
    of the Brillouin zone without having to invoke .bandpath().special_points, and for generating these
    in a useful way for ARPES. See `ase.dft.kpoints.BandPath` for definition of the special_points method.
    The k-point definitions are from a paper by Wahyu Setyawana and Stefano Curtarolo::
        http://dx.doi.org/10.1016/j.commatsci.2010.05.010

    Input:
        struct (ase.atoms.Atoms object) - crystal structure
        in_k (optional, bool) - whether to return the list of symmetry points in fractional coordinates
          and with conventional labels, or a list of k-values in cartesian space for these points (default).
        hv (optional, float) - if provided, the corresponding angle of photoemission to reach each k-point
          is determined for the given photon energy, hv, assuming the path is directly from G-sym point
          (i.e. that the analyser slit is aligned along that high symmetry direction), and a work fn of 4.35 eV.

    Output:
        dict or array of k-points depending on in_k choice. '''

    special_kpts = struct.cell.bandpath().special_points  # Special k-points in scaled coordinates
    if in_k:
        for i in special_kpts:
            kpts = ase.dft.kpoints.kpoint_convert(struct.cell, skpts_kc=special_kpts[i])  # Convert k-point to inv. Angstrom
            kpts[np.abs(kpts) < 1e-10] = 0  # Chop very small values (numerical noise)
            if hv is not None:
                from peaks.core.process import fI_inv
                # Calculate magnitude of k-vector to high sym point
                k_mag = np.sqrt(np.sum(np.square(kpts)))
                # Convert this to equivalent angle for ARPES, assuming slit aligned with this direction
                ARPES_ang = np.round(fI_inv(k_mag,0,0,0,hv-4.35,0)[0],2)
                special_kpts[i] = (kpts,ARPES_ang)
            else:
                special_kpts[i] = kpts
        if hv is not None:
            special_kpts['units'] = "(1/Angstrom, degrees for ARPES @ hv=" + str(hv) + ")"  # Set units so it is clear whether scaled or real
        else:
            special_kpts['units'] = "1/Angstrom"  # Set units so it is clear whether scaled or real
    else:
        special_kpts['units'] = 'scaled'  # Set units

    return special_kpts


def get_structure(compound, index=None, sg_num=None, API_key=None):
    '''Attempt to automatically get the structure of a compound from the Materials Project database.
    Requires an API key which can be generated by creating an account at https://materialsproject.org

    Input:
        compound (str) - Formula for desired compound
        index (int or str) - selection parameters when multiple structures are returned. Either:
            - supply row number (int) of desired structure
            - supply materials project ID (string) of desired structure
        sg_num (int) - spacegroup number, used to specify particular space group only
        API_key (str) - API key for accessing the materials project database. Not required
          if key is stored in mp_API_key.txt in the .utils directory

    Output:
        instance of crystal structure class or pandas df with list of available structures
         '''

    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    if API_key is None:
        API_key = get_API_key()

    with MPRester(API_key) as mpr:
        records = mpr.get_data(compound)
        mat_ids = []
        for i in records:
            mat_ids.append(mpr.get_materials_id_from_task_id(i['task_ids'][0]))

    if len(records) == 1:
        struct = pymatgen.core.Structure.from_str(records[0]['cif'], 'cif')
        print('Single structure found, materials project ID: ' + str(mat_ids[0]))
        structures = crystal(struct)
        structures.set_mp_id(mat_ids[0])
        if sg_num is not None:
            if records[0]['spacegroup']['number'] != sg_num:
                warn_str = 'Only available record does not match specified spacegroup number. Please double check.'
                warnings.warn(warn_str)

    elif index is not None:
        if type(index) is int:
            struct = pymatgen.core.Structure.from_str(records[index]['cif'], 'cif')
            mp_id = mat_ids[index]
        else:
            for count, i in enumerate(records):
                if index in i['task_ids']:
                    struct = pymatgen.core.Structure.from_str(records[count]['cif'], 'cif')
                    break
        structures = crystal(struct)
        structures.set_mp_id(index)

    else:  # Iterate through the list and put into a pandas dataframe
        sg = []
        form = []
        mat_ids_out = []
        for count, i in enumerate(records):
            if sg_num is not None:
                if i['spacegroup']['number'] == sg_num:
                    sg.append(i['spacegroup']['symbol'])
                    form.append(i['unit_cell_formula'])
                    mat_ids_out.append(mat_ids[count])
            else:
                sg.append(i['spacegroup']['symbol'])
                form.append(i['unit_cell_formula'])
                mat_ids_out.append(mat_ids[count])

        if len(mat_ids_out) == 1:  # After filtering by space group, there can be only one left
            index = mat_ids_out[0]
            for count, i in enumerate(records):
                if index in i['task_ids']:
                    struct = pymatgen.core.Structure.from_str(records[count]['cif'], 'cif')
                    break
            structures = crystal(struct)
            structures.set_mp_id(index)
            print('Single structure found, materials project ID: ' + str(index))

        else:  # Still multiple remaining; return a list of these
            structures = pd.DataFrame(
                {'Spacegroup': sg,
                 'u.c. formula': form,
                 'Materials project ID': mat_ids_out
                 })

            print('Multiple structures found. Please call function with materials project ID number of desired structure from following list:')
            display(structures)

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

    return structures


def get_API_key():
    try:
        path = os.path.dirname(__file__)
        with open(os.path.join(path, 'mp_API_key.txt'), 'r') as f:
            lines = f.readlines()
        API_key = lines[0]
    except:
        err_str = 'Please supply a valid API key for materials project.'
        raise Exception(err_str)

    return API_key
