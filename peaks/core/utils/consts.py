"""Common physical constants."""

# Phil King 15/05/2021
# Brendan Edwards 16/10/2023


class consts(object):
    """Class used to store constants that may be used within the peaks package.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        # Get the value of the Boltzmann constant
        kB_value = consts.kB

    """

    kvac_const = (
        0.5123167243227325  # sqrt(2m_e/hbar^2) in units for energy in eV and k in 1/AA
    )
    Cu_Ka_lambda = 1.5406  # Cu K-alpha wavelength in units of Angstroms
    kB = 1.380649 * 10**-23  # Boltzmann constant in units of J/K
    electron_volt = 1.602176634 * 10**-19  # 1 electronvolt in units of J
    electron_mass = 9.1093837 * 10**-31  # Electron mass in units of kg
    Planck_const = 6.62607015 * 10**-34  # Planck constant in units of Js
    reduced_Planck_const = (
        1.054571817 * 10**-34
    )  # Reduced Planck constant in units of Js
