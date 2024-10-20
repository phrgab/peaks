"""Functions to load data from the ARPES system (Phoibos analyser) at St Andrews.

"""

# Lewis Hart 11/01/2021
# Brendan Edwards 01/04/2021
# Brendan Edwards 09/02/2024

import numpy as np
from peaks.core.options import _BaseARPESConventions, _register_location


def _load_StA_Phoibos_data(fname):
    """This function loads data that was obtained using the ARPES system (Phoibos analyser) at St Andrews.

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

        from peaks.core.fileIO.loaders.StA_Phoibos import _load_StA_Phoibos_data

        fname = 'C:/User/Documents/Research/disp1.xy'

        # Extract data from file obtained using the ARPES system (Phoibos analyser) at St Andrews
        data = _load_StA_Phoibos_data(fname)

    """

    # Open the file and load lines
    with open(fname) as f:
        lines = f.readlines()

    # Loop through file to determine the scan type
    scan_type = None
    for line in lines:
        # If the file contains phi (tilt) values, it must be a Fermi map
        if line.startswith('# Parameter: "Phi [deg]" = '):
            scan_type = "FS map"
            break
        # Otherwise it must be a dispersion
        elif line.startswith("# Number of Scans:"):
            scan_type = "dispersion"
            break

    # Load file data
    file_data = np.loadtxt(fname)
    energies = file_data[:, 0]
    counts = file_data[:, 1]

    # Extract the non-repeating KE values
    start_KEs = np.where(energies == energies[0])
    number_of_KEs = start_KEs[0][1]
    KE_values = energies[0:number_of_KEs]

    # Extract the theta_par values, which are defined in file lines file that start with '# NonEnergyOrdinate: '
    theta_par_values = []
    for line in lines:
        if line.startswith("# NonEnergyOrdinate: "):
            theta_par_values.append(
                float(line.replace("# NonEnergyOrdinate: ", "").strip())
            )
        elif line.startswith("# Cycle: 1"):
            # Line seen in Fermi maps when the second tilt value is measured . Break here to avoid repeating theta_par.
            break

    # If the file is a dispersion
    if scan_type == "dispersion":
        # Convert counts into a 2D spectrum
        spectrum = np.zeros((len(theta_par_values), number_of_KEs))
        for i in range(len(theta_par_values)):
            for j in range(number_of_KEs):
                spectrum[i, j] = counts[(i * number_of_KEs) + j]

        # Define data
        data = {
            "scan_type": scan_type,
            "spectrum": spectrum,
            "theta_par": theta_par_values,
            "eV": KE_values,
        }

    # If the file is a Fermi map
    elif scan_type == "FS map":
        # Extract the phi (tilt) values, which are defined in file lines that start with '# Parameter: "Phi [deg]"'
        phi_values = []
        for line in lines:
            if line.startswith('# Parameter: "Phi [deg]" = '):
                phi_values.append(
                    float(line.replace('# Parameter: "Phi [deg]" = ', "").strip())
                )

        spectrum = np.zeros((len(phi_values), len(theta_par_values), number_of_KEs))
        for i in range(len(phi_values)):
            for j in range(len(theta_par_values)):
                for k in range(number_of_KEs):
                    spectrum[i, j, k] = counts[
                        (i * len(theta_par_values) * number_of_KEs)
                        + (j * number_of_KEs)
                        + k
                    ]

        # Define data
        data = {
            "scan_type": scan_type,
            "spectrum": spectrum,
            "tilt": phi_values,
            "theta_par": theta_par_values,
            "eV": KE_values,
        }

    # Unable to identify scan, so raise error
    else:
        raise Exception("Scan type not supported.")

    return data


def _load_StA_Phoibos_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained using the ARPES system (Phoibos analyser) at St Andrews.

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

        from peaks.core.fileIO.loaders.StA_Phoibos import _load_StA_Phoibos_metadata

        fname = 'C:/User/Documents/Research/disp1.xy'

        # Extract metadata from file obtained using the ARPES system (Phoibos analyser) at St Andrews
        metadata = _load_StA_Phoibos_metadata(fname)

    """

    from peaks.core.fileIO.data_loading import _extract_mapping_metadata

    # Open the file and load lines
    with open(fname) as f:
        lines = f.readlines()

    # Define dictionary to store metadata
    metadata = {}

    # Extract the scan name from the file full path
    fname_split = fname.split("/")
    metadata["scan_name"] = fname_split[len(fname_split) - 1].split(".")[0]

    # Define initial attributes
    metadata["scan_type"] = scan_type
    metadata["sample_description"] = None
    metadata["eV_type"] = "kinetic"
    metadata["beamline"] = _StAPhoibosConventions.loc_name
    metadata["analysis_history"] = []
    metadata["EF_correction"] = None

    # Define attributes, using the _StA_Phoibos_find function to obtain metadata where possible
    metadata["PE"] = float(_StA_Phoibos_find(lines, "Pass Energy"))
    metadata["hv"] = float(_StA_Phoibos_find(lines, "Excitation Energy"))
    metadata["pol"] = None
    metadata["sweeps"] = int(_StA_Phoibos_find(lines, "Number of Scans"))
    metadata["dwell"] = float(_StA_Phoibos_find(lines, "Dwell Time"))
    metadata["ana_mode"] = _StA_Phoibos_find(lines, "Analyzer Lens")
    metadata["ana_slit"] = _StA_Phoibos_find(lines, "Analyzer Slit")
    metadata["ana_slit_angle"] = 0
    metadata["exit_slit"] = None
    metadata["x1"] = None
    metadata["x2"] = None
    metadata["x3"] = None
    metadata["polar"] = None

    # If the scan type is a Fermi map, we can extract the tilt and azi values
    if scan_type == "FS map":
        # Define lists to store phi and azi values, then loop through the file to extract the metadata
        phi_values = []
        azi_values = []
        for line in lines:
            if line.startswith('# Parameter: "Azi [deg]" = '):
                azi_values.append(
                    float(line.replace('# Parameter: "Azi [deg]" = ', "").strip())
                )
            elif line.startswith('# Parameter: "Phi [deg]" = '):
                phi_values.append(
                    float(line.replace('# Parameter: "Phi [deg]" = ', "").strip())
                )

        # Represent the tilt and azi metadata using a 'min:max (step)' format (rounding to 3 decimal places)
        metadata["tilt"] = _extract_mapping_metadata(phi_values, num_dp=3)
        metadata["azi"] = _extract_mapping_metadata(azi_values, num_dp=3)

    # No access to tilt and azi values if data is not a Fermi map
    else:
        metadata["tilt"] = None
        metadata["azi"] = None

    # Define other attributes that are not saved in the metadata
    metadata["norm_polar"] = None
    metadata["norm_tilt"] = None
    metadata["norm_azi"] = None
    metadata["temp_sample"] = None
    metadata["temp_cryo"] = None

    return metadata


def _StA_Phoibos_find(lines, item):
    """This function will loop over the lines in a .xy text file obtained using the ARPES system (Phoibos analyser) at
    St Andrews, and then pick out the line starting with the desired keyword.

    Parameters
    ------------
    lines : list
        A list where each entry is a line from the .xy text file.

    item : str
        The keyword that is being searched for.

    Returns
    ------------
    line_contents : str
        The line contents following '# ' + the desired keyword.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.StA_Phoibos import _StA_Phoibos_find

        fname = 'C:/User/Documents/Research/disp1.xy'

        # Open the file and load lines
        with open(fname) as f:
            lines = f.readlines()

        # Extract the analyser mode
        ana_mode = _StA_Phoibos_find(lines, 'Analyzer Lens')

    """

    # Loop over lines to extract the line starting with '# ' + the desired keyword.
    for line in lines:
        if line.startswith("# " + item):
            line_contents = line.replace("# " + item + ": ", "").strip()
            break

    return line_contents


@_register_location
class _StAPhoibosConventions(_BaseARPESConventions):
    loc_name = "StA-Phoibos"
    loader = _load_StA_Phoibos_data
    metadata_loader = _load_StA_Phoibos_metadata
    ana_type = "II"
    azi = 1
    polar_name = "Theta"
    tilt_name = "Phi"
    azi_name = "Azi"
    x1_name = "y"
    x2_name = "z"
    x3_name = "x"
