"""Functions to load data from the spin-ARPES system (MBS analyser) at St Andrews.

"""

# Brendan Edwards 01/03/2021
# Phil King 04/06/2021
# Brendan Edwards 22/02/2024

import os
import numpy as np
from peaks.core.options import _BaseARPESConventions, _register_location


def _load_StA_MBS_data(fname):
    """This function loads data that was obtained using the spin-ARPES system (MBS analyser) at St Andrews.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    Returns
    ------------
    data : dict
        Dictionary containing the file scan type, spectrum, and coordinates.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.StA_MBS import _load_StA_MBS_data

        fname = 'C:/User/Documents/Research/disp1.krx'

        # Extract data from file obtained using the spin-ARPES system (MBS analyser) at St Andrews
        data = _load_StA_MBS_data(fname)

    """

    # Get file_extension to determine the file type
    filename, file_extension = os.path.splitext(fname)

    # If the file is a txt file
    if file_extension == ".txt":
        # Open the file and load lines
        with open(fname) as f:
            lines = f.readlines()

        # Fermi maps are only permitted as .krx files, so ensure that the file is a dispersion by checking if MapStartX
        # (associated with Fermi maps) is a valid attribute (Fermi maps are only permitted as .krx files)
        try:
            # If MapStartX is a valid attribute, raise an error
            _StA_MBS_find(lines, "MapStartX")
            raise Exception(
                "Fermi maps in the txt format are not supported. Please load a krx format Fermi map "
                "instead."
            )
        except UnboundLocalError:
            # If MapStartX is not a valid attribute, we should have a dispersion
            scan_type = "dispersion"

        # Determine when scan data starts
        num_lines_to_skip = lines.index("DATA:\t\n") + 1

        # Load file data
        file_data = np.loadtxt(fname, skiprows=num_lines_to_skip)
        KE_values = file_data[:, 0]
        spectrum = file_data[:, 1:].T

        # Extract the theta_par values
        theta_par_min = float(_StA_MBS_find(lines, "ScaleMin"))
        theta_par_max = float(_StA_MBS_find(lines, "ScaleMax"))
        theta_par_step = float(_StA_MBS_find(lines, "ScaleMult"))
        num_theta_par = int(((theta_par_max - theta_par_min) / theta_par_step) + 1)
        theta_par_values = np.linspace(theta_par_min, theta_par_max, num_theta_par)

        # Define data
        data = {
            "scan_type": scan_type,
            "spectrum": spectrum,
            "theta_par": theta_par_values,
            "eV": KE_values,
        }

    # If the file is a binary krx file
    elif file_extension == ".krx":
        # Open the file in read mode
        with open(fname, "rb") as f:
            # Determine whether the file is 32-bit or 64-bit. The data type is little endian, so read initially as
            # 32 bit, but if either of the first 2 32-bit words are 0, then the file is 64-bit.
            dtype_identifier = np.fromfile(f, dtype="<i4", count=2)
            if 0 in dtype_identifier:  # File is 64 bit
                dtype = "<i8"  # 8-byte signed integer (little endian)
            else:  # File is 32 bit
                dtype = "<i4"  # 4-byte signed integer (little endian)

            # Set reading to the start of the file
            f.seek(0)

            # Read the pointer array size, which is the first word of the array
            pointer_array_size = np.fromfile(f, dtype=dtype, count=1)

            # Determine the number of images in the file
            num_images = int(pointer_array_size[0] / 3)

            # Sequentially read the image positions and sizes
            image_pos = []
            Y_size = []
            X_size = []
            for i in range(num_images):
                # Extract position in array of the images in 32-bit integers
                image_pos.append(np.fromfile(f, dtype=dtype, count=1)[0])

                # Extract Y size of array (angular direction)
                Y_size.append(np.fromfile(f, dtype=dtype, count=1)[0])

                # Extract X size of array (energy direction)
                X_size.append(np.fromfile(f, dtype=dtype, count=1)[0])

            # Read more file information to identify the scan type
            scan_identifier = np.fromfile(f, dtype=dtype, count=1)[
                0
            ]  # 5 for spin, 4 for ARPES

            # Determine the scan type
            if scan_identifier == 5:
                scan_type = "spin scan"
            elif num_images == 1:
                scan_type = "dispersion"
            else:
                scan_type = "FS map"

            # Calculate the array size
            array_size = X_size[0] * Y_size[0]

            # Set file position to the first header
            f.seek((image_pos[0] + array_size + 1) * 4)

            # Read the header (allowing up to 1800 bytes) containing metadata and convert into ascii format
            header = f.read(1800).decode("ascii")

            # Shorten header to required length (i.e. up to where scan data starts)
            header = header.split("\r\nDATA:")[0]

            # Convert the header to a metadata dictionary
            metadata = dict(i.split("\t", 1) for i in header.split("\r\n"))

            # If there is a single image, load 2D spectrum
            if num_images == 1:
                # Set read position in the file to the image location
                f.seek(image_pos[0] * 4)

                # Read the image spectrum (images written as 32-bit words even in 64-bit format .krx file)
                spectrum = np.fromfile(f, dtype="<i4", count=array_size)

                # Reshape spectrum into the desired data structure
                spectrum = np.reshape(spectrum, [Y_size[0], X_size[0]])

            # If there are multiple images, load the spectrum as a data cube
            else:
                # Define the spectrum in the order [mapping_dim, theta_par, eV]
                spectrum = np.zeros((num_images, Y_size[0], X_size[0]))

                # Loop through the images and fill in spectrum
                for i, pos in enumerate(image_pos):
                    # Set the read position in the file to the current image location
                    f.seek(pos * 4)

                    # Read the current image data (images written as 32-bit words even in 64-bit format .krx file)
                    current_image_data = np.fromfile(f, dtype="<i4", count=array_size)

                    # Reshape the current image data into the desired data structure, and fill entries in spectrum
                    spectrum[i, :, :] = np.reshape(
                        current_image_data, [Y_size[0], X_size[0]]
                    )

        # Extract the kinetic energy values
        KE_values = np.linspace(
            float(metadata["Start K.E."]), float(metadata["End K.E."]), X_size[0]
        )

        # Extract the theta_par values
        if (scan_type == "dispersion") or (
            scan_type == "FS map"
        ):  # ARPES scans, extract analyser MCP angular scale
            theta_par_values = np.linspace(
                float(metadata["ScaleMin"]), float(metadata["ScaleMax"]), Y_size[0]
            )
        elif (
            scan_type == "spin scan"
        ):  # Spin ARPES scans, extract spin MCP angular scale
            theta_par_values = np.linspace(
                float(metadata["S0ScaleMin"]), float(metadata["S0ScaleMax"]), Y_size[0]
            )

        # If there is more than one image, extract the mapping variable (defl_perp for an FS map, or spin_rot_angle for
        # a spin scan)
        if num_images > 1:
            # If scan type is an FS map, extract the deflector angular values
            if scan_type == "FS map":  # ARPES Deflector scan
                mapping_values = np.linspace(
                    float(metadata["MapStartX"]), float(metadata["MapEndX"]), num_images
                )
                mapping_label = "defl_perp"
            # If scan type is a spin scan, extract the spin rotation angle values
            elif scan_type == "spin scan":  # Spin scan
                mapping_values = []
                for i in range(num_images):
                    current_spin_rot_angle = float(
                        metadata["SpinComp#" + str(i)].split(",", 1)[1].split(">", 1)[0]
                    )
                    mapping_values.append(current_spin_rot_angle)
                mapping_label = "spin_rot_angle"

        # Define data
        if scan_type == "dispersion":
            data = {
                "scan_type": scan_type,
                "spectrum": spectrum,
                "theta_par": theta_par_values,
                "eV": KE_values,
            }
        else:
            data = {
                "scan_type": scan_type,
                "spectrum": spectrum,
                mapping_label: mapping_values,
                "theta_par": theta_par_values,
                "eV": KE_values,
            }

    return data


def _load_StA_MBS_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained using the spin-ARPES system (MBS analyser) at
    St Andrews.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    scan_type : str
        The scan type of the data.

    Returns
    ------------
    metadata : dict
        Dictionary containing the relevant metadata.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.StA_MBS import _load_StA_MBS_metadata

        fname = 'C:/User/Documents/Research/disp1.krx'

        # Extract metadata from file obtained using the spin-ARPES system (MBS analyser) at St Andrews
        metadata = _load_StA_MBS_metadata(fname)

    """

    from peaks.core.fileIO.data_loading import _extract_mapping_metadata

    # Define dictionary to store metadata
    metadata = {}

    # Extract the scan name from the file full path
    fname_split = fname.split("/")
    metadata["scan_name"] = fname_split[len(fname_split) - 1].split(".")[0]

    # Define initial attributes
    metadata["scan_type"] = scan_type
    metadata["sample_description"] = None
    metadata["eV_type"] = "kinetic"
    metadata["beamline"] = _StAMBSConventions.loc_name
    metadata["analysis_history"] = []
    metadata["EF_correction"] = None

    # Get file_extension to determine the file type, to allow other metadata to be determined
    filename, file_extension = os.path.splitext(fname)

    # Extract metadata present in a .txt file
    if file_extension == ".txt":
        # Open the file and load lines
        with open(fname) as f:
            lines = f.readlines()

        # Define attributes, using the _StA_MBS_find function to obtain metadata where possible
        metadata["PE"] = float(_StA_MBS_find(lines, "Pass Energy").replace("PE0", ""))
        metadata["hv"] = None
        metadata["pol"] = None
        metadata["frames"] = int(_StA_MBS_find(lines, "Frames Per Step"))
        metadata["sweeps"] = int(_StA_MBS_find(lines, "No Scans"))
        metadata["steps"] = int(_StA_MBS_find(lines, "No. Steps"))
        metadata["ana_mode"] = _StA_MBS_find(lines, "Lens Mode")
        metadata["ana_slit"] = None
        metadata["ana_slit_angle"] = 0
        metadata["exit_slit"] = None
        metadata["defl_par"] = round(float(_StA_MBS_find(lines, "DeflY")), 2)
        metadata["defl_perp"] = round(float(_StA_MBS_find(lines, "DeflX")), 2)

    # Extract metadata present in a .krx file
    elif file_extension == ".krx":
        # Open the file in read mode and extract metalines
        with open(fname, "rb") as f:
            # Determine whether the file is 32-bit or 64-bit. The data type is little endian, so read initially as
            # 32 bit, but if either of the first 2 32-bit words are 0, then the file is 64-bit.
            dtype_identifier = np.fromfile(f, dtype="<i4", count=2)
            if 0 in dtype_identifier:  # File is 64 bit
                dtype = "<i8"  # 8-byte signed integer (little endian)
            else:  # File is 32 bit
                dtype = "<i4"  # 4-byte signed integer (little endian)

            # Set reading to the start of the file
            f.seek(0)

            # Read the pointer array size, which is the first word of the array. Pointer array size is not used here,
            # but is read so that the next variable read is image position
            pointer_array_size = np.fromfile(f, dtype=dtype, count=1)

            # Extract first image position and array size to know where in the file to find metadata
            image_pos = np.fromfile(f, dtype=dtype, count=1)[0]
            Y_size = np.fromfile(f, dtype=dtype, count=1)[0]
            X_size = np.fromfile(f, dtype=dtype, count=1)[0]
            array_size = X_size * Y_size

            # Set file position to the first header
            f.seek((image_pos + array_size + 1) * 4)

            # Read the header (allowing up to 1800 bytes) containing metadata and convert into ascii format
            header = f.read(1800).decode("ascii")

            # Shorten header to required length (i.e. up to where scan data starts)
            header = header.split("\r\nDATA:")[0]

            # Convert the header to a metadata dictionary
            metalines = dict(i.split("\t", 1) for i in header.split("\r\n"))

        metadata["PE"] = float(metalines["Pass Energy"].replace("PE0", ""))
        metadata["hv"] = None
        metadata["pol"] = None
        metadata["frames"] = int(metalines["Frames Per Step"])
        metadata["sweeps"] = int(metalines["No Scans"])
        metadata["steps"] = int(metalines["No. Steps"])
        metadata["ana_mode"] = metalines["Lens Mode"]
        metadata["ana_slit"] = None
        metadata["ana_slit_angle"] = 0
        metadata["exit_slit"] = None
        metadata["defl_par"] = round(float(metalines["DeflY"]), 2)

        # Extract defl_perp attribute
        if scan_type == "FS map":
            # For a Fermi map, extract range of defl_perp values
            defl_perp_min = float(metalines["MapStartX"])
            defl_perp_max = float(metalines["MapEndX"])
            num_defl_perp = int(metalines["MapNoXSteps"])
            defl_perp_values = np.linspace(defl_perp_min, defl_perp_max, num_defl_perp)

            # Represent the defl_perp using a 'min:max (step)' format (rounding to 3 decimal places)
            metadata["defl_perp"] = _extract_mapping_metadata(
                defl_perp_values, num_dp=3
            )
        else:
            # Otherwise, defl_perp is static
            metadata["defl_perp"] = round(float(metalines["DeflX"]), 2)

    # Define other attributes that are not saved in the metadata
    metadata["x1"] = None
    metadata["x2"] = None
    metadata["x3"] = None
    metadata["polar"] = None
    metadata["tilt"] = None
    metadata["azi"] = None
    metadata["norm_polar"] = None
    metadata["norm_tilt"] = None
    metadata["norm_azi"] = None
    metadata["temp_sample"] = None
    metadata["temp_cryo"] = None

    return metadata


def _StA_MBS_find(lines, item):
    """This function will loop over the lines in a .txt file obtained using the spin-ARPES system (MBS analyser) at
    St Andrews, and then pick out the line starting with the desired keyword.

    Parameters
    ------------
    lines : list
        A list where each entry is a line from the .txt file.

    item : str
        The keyword that is being searched for.

    Returns
    ------------
    line_contents : str
        The line contents following the desired keyword.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.StA_MBS import _StA_MBS_find

        fname = 'C:/User/Documents/Research/disp1.xy'

        # Open the file and load lines
        with open(fname) as f:
            lines = f.readlines()

        # Extract the minimum theta_par value
        theta_par_min = _StA_MBS_find(lines, 'ScaleMin')

    """

    # Loop over lines to extract the line starting with the desired keyword.
    for line in lines:
        if line.startswith(item):
            line_contents = line.replace(item, "").strip()
            break

    return line_contents


@_register_location
class _StAMBSConventions(_BaseARPESConventions):
    loc_name = "StA-MBS"
    loader = _load_StA_MBS_data
    metadata_loader = _load_StA_MBS_metadata
    ana_type = "IIp"
    azi = 1
    polar_name = "polar"
    tilt_name = "tilt"
    azi_name = "azi"
    x1_name = "y"
    x2_name = "z"
    x3_name = "x"
    defl_par_name = "defl_y"
    defl_perp_name = "defl_x"
