#Sum two dataarrays
#Phil King 15/05/2021

import warnings
from peaks.core.utils.metadata import update_hist
from peaks.core.utils.misc import warning_simple, warning_standard

# Sum two dataarrays together while keeping metadata in tact
def sum_spec(*args):
    ''' Sum two or more dataarrays together, maintaining the metadata

        Input:
            *args - any number of data arrays to sum together (xarray)

        Returns:
            data_out - a single summed dataarray '''

    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    nspec = len(args)

    # Copy the first array
    data_out = args[0].copy(deep=True)
    name_str = args[0].attrs['scan_name']  # Build a name string to update
    temp_attrs0 = data_out.attrs.copy()
    temp_name0 = temp_attrs0.pop('scan_name')  # Remove scan name from the attributes

    # Some warning flags for mismatch of dimensions/attributes
    warn_flag1 = 0
    warn_flag2 = 0

    # Iterate through the list and add in the new data
    for i in range(1,nspec):
        # Check attributes match (except scan name)
        attrs_mismatch = []
        temp_attrs = args[i].attrs.copy()  # Temp attributes
        temp_name = temp_attrs.pop('scan_name')  # Pop out scan name
        if temp_attrs != temp_attrs0:  # Check if all the attributes except scan name are the same
            for j in temp_attrs:  # Check which attributes differ
                if temp_attrs[j] != temp_attrs0[j]:
                    attrs_mismatch.append(j)
        if len(attrs_mismatch) != 0:
            warn_flag1 = 1
            warning_str = 'The following attributes do not match for scan ' + str(temp_name) + ': ' + str(attrs_mismatch) + ', attributes of scan ' + str(temp_name0) + ' kept'
            warnings.warn(warning_str)

        # Check dimension arrays match
        dims_mismatch = []
        for k in args[i].dims:
            if not (data_out[k].data==args[i][k].data).all():
                dims_mismatch.append(k)
        if len(dims_mismatch) != 0:
            warn_flag2 = 1
            warning_str = 'The following dimensions do not match for scan ' + str(temp_name) + ': ' + str(dims_mismatch) + ', dimensions of scan ' + str(temp_name0) + ' kept'
            warnings.warn(warning_str)


        # Add the data together
        data_out += args[i].data
        # Add the new name into the concatanated name string
        name_str += '+'+args[0].attrs['scan_name']

    # Append new name
    data_out.attrs['scan_name'] = name_str
    data_out.name = name_str

    # Update the history
    hist_str = str(nspec) + ' scans summed together.'
    if warn_flag1 != 0:
        hist_str += ' CAUTION: mismatch of some attributes - those of first scan kept.'
    if warn_flag2 != 0:
        hist_str += ' CAUTION: mismatch of some dimensions - those of first scan kept.'
    update_hist(data_out, hist_str)

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

    return data_out