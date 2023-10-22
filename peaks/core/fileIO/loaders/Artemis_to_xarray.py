# Functions to load data from the Artemis TR-ARPES facility into an xarray
# Phil King 14/10/2022
# Functions for data transformation from raw MCP to transformed data based on Artemis's loader

import numpy as np
import xarray as xr
import natsort
import os

def load_Artemis_data(fname, **kwargs):
    '''Loads ARPES data from I05-HR beamline

    Parameters
    ------------
    fname : str
        Path to folder containing time series data
        In Artemis structure, should contain Info.tsv,
        Stat Log.tsv and N=xxx folders with the data

    kwargs : optional
        Additional Artemis keyword arguments:
        for image transformation:
            Ang_offset_px
            Edge_pos
            Edge_slope
        for partial data loading:
            num_scans

    Returns
    ------------
    data : xr.DataArray
        xarray DataArray with loaded data <br>
    '''
    if 'Ang_Offset_px' in kwargs:
        Ang_Offset_px = kwargs.pop('Ang_Offset_px')
    else:
        Ang_Offset_px = 0

    if 'Edge_pos' in kwargs:
        Edge_pos = kwargs.pop('Edge_pos')
    else:
        Edge_pos = 10

    if 'Edge_slope' in kwargs:
        Edge_slope = kwargs.pop('Edge_slope')
    else:
        Edge_slope = -1

    if 'N' in kwargs:
        num_scans = kwargs.pop('N')
    else:
        num_scans = None


    #############################
    #   Read folder structure   #
    #############################

    # Folder contents
    file_list = natsort.natsorted(os.listdir(fname))
    scan_folders = [item for item in file_list if 'N=' in item]  # Only ones containing scan data

    # Get an example dispersion
    test_scan_path = os.path.join(fname,scan_folders[0],natsort.natsorted(os.listdir(os.path.join(fname,scan_folders[0])))[0])

    # Load this
    test_scan = np.loadtxt(test_scan_path)

    # Get image size
    nx_pixel, ny_pixel = test_scan.shape

    # Determine binning
    binning = int(1920/nx_pixel)

    # Set some default parameters
    PixelSize=0.00645
    magnification=4.41
    WF=4.215

    # Load metadata from info file
    with open(os.path.join(fname,'Info.tsv'), 'r') as f:
        info_file = f.readlines()
    temp_meta = info_file[0].split(';')

    meta = {}
    for i in temp_meta:
        temp = i.split(":")
        try:
            meta[temp[0]] = float(temp[1])
        except:
            try:
                meta[temp[0]] = temp[1]
            except:
                pass

    # Calculate retarding ratio
    rr = (meta['KE']-WF)/meta['PE']


    #############################
    #    E/ang mapping info     #
    #############################

    # Extract relevant info from the calib2d file for the data transformation
    phoibos_path = os.path.dirname(__file__)

    with open(os.path.join(phoibos_path, 'Artemis_phoibos100.calib2d'), 'r') as f:
        calib2d_in = f.readlines()

    # Constant parameters
    calib2d = {}
    lens_rr = []
    aInner = []
    Da1_1 = []
    Da1_2 = []
    Da1_3 = []
    Da3_1 = []
    Da3_2 = []
    Da3_3 = []
    Da5_1 = []
    Da5_2 = []
    Da5_3 = []
    Da7_1 = []
    Da7_2 = []
    Da7_3 = []

    i0 = None
    i1 = None

    for ct, i in enumerate(calib2d_in):
        # Pull out some constants
        if 'eRange' in i:
            calib2d['eRange'] = [float(j) for j in i.split(" ")[2:4]]
        if 'De1' in i:
            calib2d['De1'] = float(i.split("= ")[1])

        # Find the relevant block
        if meta['LensMode'] in i and '#' in i:
            i0 = ct+1
        elif '# ===' in i and i0 and ct-i0 > 2:
            i1 = ct
            break

    # Pull out the relevant block for the required lens mode
    if i1:
        calib_lens = calib2d_in[i0:i1]
    else:
        calib_lens = calib2d_in[i0:]

    for i in calib_lens:
        # Static parts from this section
        if 'aUnit' in i:
            calib2d['aUnit'] = i.split('\"')[1]
        if 'aRange' in i:
            calib2d['aRange'] = [float(j) for j in i.split(" ")[2:4]]
        if 'eShift' in i:
            calib2d['eShift'] = [float(j) for j in i.split(" ")[2:5]]

        # Now extract the parts that depend on retarding ratio
        if meta['LensMode'] in i and '@' in i:
            lens_rr.append(i.split('@')[1].split(']')[0])
        if 'aInner' in i:
            aInner.append(float(i.split('= ')[1].split('#')[0]))
        if 'Da1' in i:
            temp = [float(j) for j in i.split(" ")[2:5]]
            Da1_1.append(temp[0])
            Da1_2.append(temp[1])
            Da1_3.append(temp[2])
        if 'Da3' in i:
            temp = [float(j) for j in i.split(" ")[2:5]]
            Da3_1.append(temp[0])
            Da3_2.append(temp[1])
            Da3_3.append(temp[2])
        if 'Da5' in i:
            temp = [float(j) for j in i.split(" ")[2:5]]
            Da5_1.append(temp[0])
            Da5_2.append(temp[1])
            Da5_3.append(temp[2])
        if 'Da7' in i:
            temp = [float(j) for j in i.split(" ")[2:5]]
            Da7_1.append(temp[0])
            Da7_2.append(temp[1])
            Da7_3.append(temp[2])

    # Interpolate to get correct values for current retarding ratio
    calib2d['aInner'] = np.interp(rr,lens_rr,aInner)
    calib2d['Da1'] = xr.DataArray(data=np.asarray([np.interp(rr,lens_rr,Da1_1),np.interp(rr,lens_rr,Da1_2),np.interp(rr,lens_rr,Da1_3)]), dims='eV', coords={'eV': np.asarray(calib2d['eShift'])*meta['PE']+meta['KE']})
    calib2d['Da3'] = xr.DataArray(data=np.asarray([np.interp(rr,lens_rr,Da3_1),np.interp(rr,lens_rr,Da3_2),np.interp(rr,lens_rr,Da3_3)]), dims='eV', coords={'eV': np.asarray(calib2d['eShift'])*meta['PE']+meta['KE']})
    calib2d['Da5'] = xr.DataArray(data=np.asarray([np.interp(rr,lens_rr,Da5_1),np.interp(rr,lens_rr,Da5_2),np.interp(rr,lens_rr,Da5_3)]), dims='eV', coords={'eV': np.asarray(calib2d['eShift'])*meta['PE']+meta['KE']})
    calib2d['Da7'] = xr.DataArray(data=np.asarray([np.interp(rr,lens_rr,Da7_1),np.interp(rr,lens_rr,Da7_2),np.interp(rr,lens_rr,Da7_3)]), dims='eV', coords={'eV': np.asarray(calib2d['eShift'])*meta['PE']+meta['KE']})

    # Work out extremes of detector range
    Ek_low = meta['KE']+calib2d['eRange'][0]*meta['PE']
    Ek_high = meta['KE']+calib2d['eRange'][1]*meta['PE']
    theta_par_low = calib2d['aRange'][0]
    theta_par_high = calib2d['aRange'][1]

    # Get relevant poly fits to these
    calib2d['Da_value'] = np.asarray(calib2d['eShift'])*meta['PE']+meta['KE']
    calib2d['Da1_coeff'] = np.polyfit(calib2d['Da_value'], calib2d['Da1'], 2)
    calib2d['Da3_coeff'] = np.polyfit(calib2d['Da_value'], calib2d['Da3'], 2)
    calib2d['Da5_coeff'] = np.polyfit(calib2d['Da_value'], calib2d['Da5'], 2)
    calib2d['Da7_coeff'] = np.polyfit(calib2d['Da_value'], calib2d['Da7'], 2)

    #############################
    # Calculate transformations #
    #############################
    eV_xarray = xr.DataArray(data=np.linspace(Ek_low,Ek_high,nx_pixel),
                          dims='eV',
                          coords={'eV': np.linspace(Ek_low,Ek_high,nx_pixel)})

    theta_par_xarray = xr.DataArray(data=np.linspace(theta_par_low,theta_par_high,ny_pixel),
                          dims='theta_par',
                          coords={'theta_par': np.linspace(theta_par_low,theta_par_high,ny_pixel)})

    # Work out correction matrices
    E_correction = np.round((eV_xarray-meta['KE'])/meta['PE']/calib2d['De1']/magnification/(PixelSize*binning) + nx_pixel/2)
    # Set up the interpolation xarrays
    eV_i = xr.DataArray(data=eV_xarray.data, dims='eV_i', coords={'eV_i': np.linspace(0, nx_pixel-1, nx_pixel)})
    E_correction_interp = xr.DataArray(data=E_correction, dims='eV_i', coords={'eV_i': np.linspace(0, nx_pixel-1, nx_pixel)})
    E_correction_xarray = xr.DataArray(data=eV_i.interp(eV_i=E_correction_interp).data,
                                       dims='eV',
                                       coords={'eV': eV_xarray.eV})

    zInner_data = calib2d['Da1'].interp(eV=eV_xarray,method='quadratic', kwargs={"fill_value": "extrapolate"}) * theta_par_xarray \
                  + 1e-2 * calib2d['Da3'].interp(eV=eV_xarray,method='quadratic', kwargs={"fill_value": "extrapolate"}) * (theta_par_xarray**3) \
                  + 1e-4 * calib2d['Da5'].interp(eV=eV_xarray,method='quadratic', kwargs={"fill_value": "extrapolate"}) * (theta_par_xarray**5) \
                  + 1e-6 * calib2d['Da7'].interp(eV=eV_xarray,method='quadratic', kwargs={"fill_value": "extrapolate"}) * (theta_par_xarray**7)

    zInner_Diff_data = calib2d['Da1'].interp(eV=eV_xarray,method='quadratic', kwargs={"fill_value": "extrapolate"}) * theta_par_xarray \
                  + 3e-2 * calib2d['Da3'].interp(eV=eV_xarray,method='quadratic', kwargs={"fill_value": "extrapolate"}) * (theta_par_xarray**2) \
                  + 5e-4 * calib2d['Da5'].interp(eV=eV_xarray,method='quadratic', kwargs={"fill_value": "extrapolate"}) * (theta_par_xarray**4) \
                  + 7e-6 * calib2d['Da7'].interp(eV=eV_xarray,method='quadratic', kwargs={"fill_value": "extrapolate"}) * (theta_par_xarray**6)

    MCP_Position_mm_1 = (zInner_data.where(abs(zInner_data.theta_par) <= calib2d['aInner'])).fillna(0)
    MCP_Position_mm_2 = (np.sign(theta_par_xarray)*(zInner_data + (abs(theta_par_xarray)-calib2d['aInner'])*zInner_Diff_data)).where(abs(zInner_data.theta_par) > calib2d['aInner']).fillna(0)
    MCP_Position_mm = MCP_Position_mm_1 + MCP_Position_mm_2

    # Edge correction - an addtional linear term correction of the warping.
    Edge_Coef = np.tan(np.radians(Edge_slope))/MCP_Position_mm.interp({'eV': meta['KE'], 'theta_par': Edge_pos}).data/meta['PE']/calib2d['De1']
    w_LinearCorrection=1/(1+(Edge_Coef*(eV_xarray-meta['KE'])))

    Angular_correction = (w_LinearCorrection*MCP_Position_mm)/magnification/(PixelSize*binning) + ny_pixel/2 + Ang_Offset_px
    # Set up the interpolation xarrays
    theta_par_i = xr.DataArray(data=theta_par_xarray.data, dims='theta_par_i', coords={'theta_par_i': np.linspace(0, ny_pixel-1, ny_pixel)})
    Angular_correction_interp = xr.DataArray(data=Angular_correction,
                                             dims=['eV_i','theta_par_i'],
                                             coords={'eV_i': np.linspace(0, nx_pixel-1, nx_pixel),
                                                     'theta_par_i': np.linspace(0, ny_pixel-1, ny_pixel)})
    Angular_correction_xarray = xr.DataArray(data=theta_par_i.interp(theta_par_i=Angular_correction_interp).data,
                                       dims=['eV','theta_par'],
                                       coords={'eV': eV_xarray.eV,
                                               'theta_par': theta_par_xarray.theta_par})


    #############################
    #  LOAD AND TRANSFORM DATA  #
    #############################
    # Get relevant folder for scans
    if num_scans:  # Scan number input - only load up to that point
        scan_folder = os.path.join(fname,'N='+str(num_scans))
    else:
        scan_folder = os.path.join(fname,scan_folders[-1])
        num_scans = int(scan_folders[-1].split('=')[1])

    # Get scan names
    scan_names =  [i for i in natsort.natsorted(os.listdir(scan_folder)) if '.tsv' in i]
    num_delays = len(scan_names)

    # Make dummy arrays to store data
    delay_pos = np.empty(num_delays)  # Delay stage position in mm
    data = np.empty((nx_pixel,ny_pixel,num_delays))  # ARPES data

    # Load the data
    for i in range(num_delays):
        delay_pos[i] = float(scan_names[i].split('.tsv')[0])
        # Read in the raw data
        data[:,:,i] = np.loadtxt(os.path.join(scan_folder,scan_names[i]))

    # Work out delay time
    delay_time = (delay_pos - meta['Time Zero']) * 2/299704645*1e12  # Convert to time in fs

    # Convert it to an xarray
    data_xarray = xr.DataArray(data=data,
                               dims=['eV','theta_par','t'],
                               coords={'eV': eV_xarray.eV,
                                       'theta_par': theta_par_xarray.theta_par,
                                       't': delay_time})

    # Do the transformation
    spectrum = data_xarray.interp(eV=E_correction_xarray,theta_par=Angular_correction_xarray)

    # Set some units
    spectrum.coords['theta_par'].attrs = {'units' : 'deg'}
    spectrum.coords['t'].attrs = {'units' : 'fs'}

    # Add the delay stage position
    spectrum = spectrum.assign_coords({'delay_pos': ("t", delay_pos)})

    spectrum['eV'] = spectrum.eV.data - WF

    #############################
    #   Get relevant metadata   #
    #############################
    #Todo move this to a dedicated logbook function in due course...
    meta_list = {}
    meta_list['scan_name'] = fname.split('/')[-1]
    meta_list['scan_type'] = 'dispersion t-series'
    meta_list['sample_description'] = None
    meta_list['eV_type'] = 'kinetic'
    meta_list['beamline'] = 'Artemis'
    meta_list['analysis_history'] = []
    meta_list['EF_correction'] = None
    meta_list['PE'] = meta['PE']
    meta_list['hv'] = None
    meta_list['pol'] = None
    meta_list['sweeps'] = num_scans
    meta_list['dwell'] = None
    meta_list['ana_mode'] = meta['LensMode']
    meta_list['ana_slit'] = None
    meta_list['ana_slit_angle'] = 90
    meta_list['exit_slit'] = None
    meta_list['defl_par'] = None
    meta_list['defl_perp'] = None
    meta_list['x1'] = None
    meta_list['x2'] = None
    meta_list['x3'] = None
    meta_list['polar'] = None
    meta_list['tilt'] = None
    meta_list['azi'] = None
    meta_list['norm_polar'] = None
    meta_list['norm_tilt'] = None
    meta_list['norm_azi'] = None

    # Get Temp data
    with open(os.path.join(fname,'Stat Log.tsv'), 'r') as f:
        log_file = f.readlines()
    time_wave = []
    temp_wave = []
    stats_wave = []
    for i in log_file:
        time_wave.append(float(i.split('\t')[0]))
        stats_wave.append(float(i.split('\t')[1]))
        temp_wave.append(float(i.split('\t')[2]))

    time_wave = time_wave[:num_scans]
    temp_wave = temp_wave[:num_scans]
    stats_wave = stats_wave[:num_scans]

    meta_list['temp_sample'] = np.round(np.mean(np.asarray(temp_wave)),2)
    meta_list['temp_variation'] = np.asarray(temp_wave)
    meta_list['temp_cryo'] = None
    meta_list['counts'] = np.asarray(stats_wave)
    meta_list['t0'] = meta['Time Zero']

    # Attach metadata
    for i in meta_list:
        spectrum.attrs[i] = meta_list[i]
    spectrum.name = meta_list['scan_name']

    return spectrum