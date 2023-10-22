#Functions to load data from the MAX IV Bloch beamline into an xarray, or for the logbook maker
#Liam Trzaska 15/04/2020
#Phil King 25/05/22
#Brendan Edwards 30/05/2022

import os
import zipfile
import tempfile
import numpy as np
import xarray as xr
from PyPhoto.process.spin import calc_spin_pol

def load_bloch_spin_data(file, logbook, **kwargs):
    '''This function loads ARPES data from MAX IV Bloch beamline spin branchline
    
    Input:
        fname - Path to the file to be loaded (string)
        logbook - Whether the data should be returned for use by the logbook maker (True) or xarray (False) (Boolean)
        scan_type - Type of scan so the loader knows how to load the data (string)
        **kwargs - Additional slicing options for partial loading of data or for binning on load (strings)
    
    Returns:
        if logbook == False:
            data - DataArray or DataSet with loaded data (xarray)
        else:
            sample_upload - List of relevant metadata (list)
            scan_timestamp - Timestamp of the scan (string)'''
    
    try:
        scan_type = kwargs['scan_type']
    except:
        raise Exception("Scan type must be one of: 'FS map', 'dispersion', 'spin FS map',  'spin disp', 'angled spin disp', 'spin EDC' or 'spin MDC''")
    if scan_type not in ['FS map','dispersion','spin FS map', 'spin disp', 'angled spin disp', 'spin EDC', 'spin MDC']:
        raise Exception("Scan type must be one of: 'FS map', 'dispersion', 'spin FS map',  'spin disp', 'angled spin disp', 'spin EDC' or 'spin MDC''")
    
    #determine the scan type
    filename, file_extension = os.path.splitext(file)
    if file_extension == '.xy':
        #Based on Max-IV spin loader from Craig Polley
        with open(file) as f:
            #scan_type = None
            for i in range(50):
                line = f.readline()

    
        with open(file, 'r') as fi:
            f = fi.readlines()

            ############################################################
            ######################### spin EDC #########################
            ############################################################            
            if scan_type == 'spin EDC':

                num_pnt = 0
                cyc_num = 0
                neg_pol = 0
                intensities = []
                negative_polarity = []
                energy = []
                step = []
                meta_lines=[]
                start = 1e9

                for i, line in enumerate(f):
                    if 'Values/Curve:' in line: 
                        num_pnt = int(line.split(":")[1])
                    if 'Cycle:' in line: 
                        cyc_num = int(line.split(":")[1].split(",")[0])
                    if 'Parameter: "Step" = ' in line:
                        step.append(int(line.split("=")[1]))
                    if 'Parameter: "NegativePolarity" =' in line:
                        OnOff = line.split("=")[1]
                        if 'OFF' in OnOff: 
                            neg_pol = 0
                        else: 
                            neg_pol = 1
                    if 'energy counts/s' in line: # start loading data
                        start = i+2
                        X = []; Y = []
                        for p in range(num_pnt):
                            row = f[start + p].rstrip("\n").split('  ')
                            if cyc_num == 0:
                                X.append(float(row[0]))
                            Y.append(float(row[1]))
                        if cyc_num == 0:
                            energy = np.array(X)
                        intensities.append(np.array(Y))
                        negative_polarity.append(neg_pol)
                    if i<start:
                        meta_lines.append(line)


                raw = xr.DataArray(
                    data=intensities,
                    dims=["step", "eV"],
                    coords=dict(
                        eV = energy,
                        step = step,)
                )
                raw.coords['neg_pol'] = ('step', negative_polarity)

                data = xr.Dataset(
                    data_vars={
                        'raw': raw,
                    }
                        )
                if 'sherman' not in kwargs:
                    data.attrs['sherman'] = 0.29

                calc_spin_pol(data, **kwargs)
                
            #############################################################
            ################### spin disp or spin MDC ###################
            #############################################################  
            elif scan_type == 'spin disp' or scan_type == 'spin MDC':
                defl_values = []
                defl_values_all = []
                i=0
                for line in f:
                    if line.startswith('# Parameter: "SAL Y [deg]" = '):
                        defl_values_all.append(float(line.replace('# Parameter: "SAL Y [deg]" = ', "").strip()))
                        if '# Parameter: "Step" = 1' in f[i+1]:
                            defl_values.append(float(line.replace('# Parameter: "SAL Y [deg]" = ', "").strip()))
                    if line.startswith('# Parameter: "SAL X [deg]" = '):
                        defl_values_all.append(float(line.replace('# Parameter: "SAL X [deg]" = ', "").strip()))
                        if '# Parameter: "Step" = 1' in f[i+1]:
                            defl_values.append(float(line.replace('# Parameter: "SAL X [deg]" = ', "").strip()))
                    i=i+1
                
                
                num_pnt = 0
                cyc_num = 0
                neg_pol = 0
                intensities = []
                negative_polarity_steps = []
                energy = []
                step = []
                steps = []
                meta_lines=[]
                start = 1e9

                for i, line in enumerate(f):
                    if 'Values/Curve:' in line: 
                        num_pnt = int(line.split(":")[1])
                    if 'Cycle:' in line: 
                        cyc_num = int(line.split(":")[1].split(",")[0])
                    if 'Parameter: "Step" = ' in line:
                        steps.append(int(line.split("=")[1]))
                        if int(line.split("=")[1]) not in step:
                            step.append(int(line.split("=")[1]))
                    if 'Parameter: "NegativePolarity" =' in line:
                        OnOff = line.split("=")[1]
                        if 'OFF' in OnOff: 
                            neg_pol = 0
                        else: 
                            neg_pol = 1
                    if 'energy counts/s' in line: # start loading data
                        start = i+2
                        X = []; Y = []
                        for p in range(num_pnt):
                            row = f[start + p].rstrip("\n").split('  ')
                            if cyc_num == 0:
                                X.append(float(row[0]))
                            Y.append(float(row[1]))
                        if cyc_num == 0:
                            energy = np.array(X)
                        intensities.append(np.array(Y))
                        negative_polarity_steps.append(neg_pol)
                    if i<start:
                        meta_lines.append(line)

                data_to_be_loaded = np.loadtxt(file)
                counts = data_to_be_loaded[:, 1]
                #put measured spectra into a 3D array
                     
                spectrum=np.zeros((len(step),len(defl_values),len(energy)))
                                  
                raw = xr.DataArray(
                    data=spectrum,
                    dims=["step", "defl","eV"],
                    coords=dict(
                        eV = energy,
                        step = step,
                        defl = defl_values)
                )
                
                for ct, i in enumerate(intensities):
                    raw.loc[dict(step=steps[ct], defl=defl_values_all[ct])] = i
                           
                # Determine negative polarity index
                negative_polarity = np.zeros(len(step))
                for ct, i in enumerate(step):
                    negative_polarity[ct] = negative_polarity_steps[steps.index(i)]
                raw.coords['neg_pol'] = ('step', negative_polarity)

                data = xr.Dataset(
                    data_vars={
                        'raw': raw,
                    }
                        )
                if 'sherman' not in kwargs:
                    data.attrs['sherman'] = 0.29

                calc_spin_pol(data, **kwargs)
            
            ############################################################
            ##################### angled spin disp #####################
            ############################################################
            
            elif scan_type == 'angled spin disp':
                deflX_values = []
                deflX_values_all = []
                deflY_values = []
                deflY_values_all = []
                i=0
                for line in f:
                    if line.startswith('# Parameter: "SAL Y [deg]" = '):
                        deflY_values_all.append(float(line.replace('# Parameter: "SAL Y [deg]" = ', "").strip()))
                        if '# Parameter: "Step" = 1' in f[i+1]:
                            deflY_values.append(float(line.replace('# Parameter: "SAL Y [deg]" = ', "").strip()))
                    if line.startswith('# Parameter: "SAL X [deg]" = '):
                        deflX_values_all.append(float(line.replace('# Parameter: "SAL X [deg]" = ', "").strip()))
                        if '# Parameter: "Step" = 1' in f[i+1]:
                            deflX_values.append(float(line.replace('# Parameter: "SAL X [deg]" = ', "").strip()))
                    i=i+1
                
                
                num_pnt = 0
                cyc_num = 0
                neg_pol = 0
                intensities = []
                negative_polarity_steps = []
                energy = []
                step = []
                steps = []
                meta_lines=[]
                start = 1e9

                for i, line in enumerate(f):
                    if 'Values/Curve:' in line: 
                        num_pnt = int(line.split(":")[1])
                    if 'Cycle:' in line: 
                        cyc_num = int(line.split(":")[1].split(",")[0])
                    if 'Parameter: "Step" = ' in line:
                        steps.append(int(line.split("=")[1]))
                        if int(line.split("=")[1]) not in step:
                            step.append(int(line.split("=")[1]))
                    if 'Parameter: "NegativePolarity" =' in line:
                        OnOff = line.split("=")[1]
                        if 'OFF' in OnOff: 
                            neg_pol = 0
                        else: 
                            neg_pol = 1
                    if 'energy counts/s' in line: # start loading data
                        start = i+2
                        X = []; Y = []
                        for p in range(num_pnt):
                            row = f[start + p].rstrip("\n").split('  ')
                            if cyc_num == 0:
                                X.append(float(row[0]))
                            Y.append(float(row[1]))
                        if cyc_num == 0:
                            energy = np.array(X)
                        intensities.append(np.array(Y))
                        negative_polarity_steps.append(neg_pol)
                    if i<start:
                        meta_lines.append(line)

                data_to_be_loaded = np.loadtxt(file)
                counts = data_to_be_loaded[:, 1]
                #put measured spectra into a 3D array
                     
                spectrum=np.zeros((len(step),len(deflY_values),len(energy)))
                                  
                raw = xr.DataArray(
                    data=spectrum,
                    dims=["step", "deflY","eV"],
                    coords=dict(
                        eV = energy,
                        step = step,
                        deflY = deflY_values)
                )
                
                for ct, i in enumerate(intensities):
                    raw.loc[dict(step=steps[ct], deflY=deflY_values_all[ct])] = i
                           
                # Determine negative polarity index
                negative_polarity = np.zeros(len(step))
                for ct, i in enumerate(step):
                    negative_polarity[ct] = negative_polarity_steps[steps.index(i)]
                raw.coords['neg_pol'] = ('step', negative_polarity)

                data = xr.Dataset(
                    data_vars={
                        'raw': raw,
                    }
                        )
                if 'sherman' not in kwargs:
                    data.attrs['sherman'] = 0.29

                calc_spin_pol(data, **kwargs)
                
                
                
            #######################################################
            ##################### spin FS map #####################
            #######################################################
            elif scan_type == 'spin FS map':
                defl_X_values = []
                defl_X_values_all = []
                defl_Y_values = []
                defl_Y_values_all = []
                for line in f:
                    if line.startswith('# Parameter: "SAL Y [deg]" = '):
                        defl_Y_values_all.append(float(line.replace('# Parameter: "SAL Y [deg]" = ', "").strip()))
                    if line.startswith('# Parameter: "SAL X [deg]" = '):
                        defl_X_values_all.append(float(line.replace('# Parameter: "SAL X [deg]" = ', "").strip()))

                
                defl_X_values = list(dict.fromkeys(defl_X_values_all))           
                defl_Y_values = list(dict.fromkeys(defl_Y_values_all))               

                

                num_pnt = 0
                cyc_num = 0
                neg_pol = 0
                intensities = []
                negative_polarity_steps = []
                energy = []
                step = []
                steps = []
                meta_lines=[]
                start = 1e9

                for i, line in enumerate(f):
                    if 'Values/Curve:' in line: 
                        num_pnt = int(line.split(":")[1])
                    if 'Cycle:' in line: 
                        cyc_num = int(line.split(":")[1].split(",")[0])
                    if 'Parameter: "Step" = ' in line:
                        steps.append(int(line.split("=")[1]))
                        if int(line.split("=")[1]) not in step:
                            step.append(int(line.split("=")[1]))
                    if 'Parameter: "NegativePolarity" =' in line:
                        OnOff = line.split("=")[1]
                        if 'OFF' in OnOff: 
                            neg_pol = 0
                        else: 
                            neg_pol = 1
                    if 'energy counts/s' in line: # start loading data
                        start = i+2
                        X = []; Y = []
                        for p in range(num_pnt):
                            row = f[start + p].rstrip("\n").split('  ')
                            if cyc_num == 0:
                                X.append(float(row[0]))
                            Y.append(float(row[1]))
                        if cyc_num == 0:
                            energy = np.array(X)
                        intensities.append(np.array(Y))
                        negative_polarity_steps.append(neg_pol)
                    if i<start:
                        meta_lines.append(line)

                data_to_be_loaded = np.loadtxt(file)
                counts = data_to_be_loaded[:, 1]
                #put measured spectra into a 3D array
                     
                spectrum=np.zeros((len(step), len(defl_X_values), len(defl_Y_values), len(energy)))

                for i in range(len(step)):
                    for j in range(len(defl_X_values)):
                        for k in range(len(defl_Y_values)):
                            for l in range(len(energy)):
                                spectrum[i][j][k][l] = counts[(i*len(defl_X_values)*len(defl_Y_values)*len(energy))+(j*len(energy*defl_Y_values))+k*len(energy)+l]                
                
                raw = xr.DataArray(
                    data=spectrum,
                    dims=["step", "deflX","deflY","eV"],
                    coords=dict(
                        eV = energy,
                        step = step,
                        deflX = defl_X_values,
                        deflY = defl_Y_values)
                )
                
                for ct, i in enumerate(intensities):
                    raw.loc[dict(step=steps[ct], deflX=defl_X_values_all[ct], deflY=defl_Y_values_all[ct])] = i
                           
                # Determine negative polarity index
                negative_polarity = np.zeros(len(step))
                for ct, i in enumerate(step):
                    negative_polarity[ct] = negative_polarity_steps[steps.index(i)]
                raw.coords['neg_pol'] = ('step', negative_polarity)

                data = xr.Dataset(
                    data_vars={
                        'raw': raw,
                    }
                        )
                if 'sherman' not in kwargs:
                    data.attrs['sherman'] = 0.29

                calc_spin_pol(data, **kwargs)
                
            elif scan_type == 'spin disp old':
                defl_values = []
                i=0
                for line in f:
                    if line.startswith('# Parameter: "SAL Y [deg]" = '):
                        if '# Parameter: "Step" = 1' in f[i+1]:
                            defl_values.append(float(line.replace('# Parameter: "SAL Y [deg]" = ', "").strip()))
                    i=i+1
                
                
                num_pnt = 0
                cyc_num = 0
                neg_pol = 0
                intensities = []
                negative_polarity = []
                energy = []
                step = []
                meta_lines=[]
                start = 1e9

                for i, line in enumerate(f):
                    if 'Values/Curve:' in line: 
                        num_pnt = int(line.split(":")[1])
                    if 'Cycle:' in line: 
                        cyc_num = int(line.split(":")[1].split(",")[0])
                    if 'Parameter: "Step" = ' in line:
                        if int(line.split("=")[1]) not in step:
                            step.append(int(line.split("=")[1]))
                    if 'Parameter: "NegativePolarity" =' in line:
                        OnOff = line.split("=")[1]
                        if 'OFF' in OnOff: 
                            neg_pol = 0
                        else: 
                            neg_pol = 1
                    if 'energy counts/s' in line: # start loading data
                        start = i+2
                        X = []; Y = []
                        for p in range(num_pnt):
                            row = f[start + p].rstrip("\n").split('  ')
                            if cyc_num == 0:
                                X.append(float(row[0]))
                            Y.append(float(row[1]))
                        if cyc_num == 0:
                            energy = np.array(X)
                        intensities.append(np.array(Y))
                        negative_polarity.append(neg_pol)
                    if i<start:
                        meta_lines.append(line)

                data_to_be_loaded = np.loadtxt(file)
                counts = data_to_be_loaded[:, 1]
                #put measured spectra into a 3D array
                     
                
                
                spectrum = [[[0 for i in range(len(energy))] for j in range(len(defl_values))] for k in range(len(step))]
                for i in range(len(step)):
                    for j in range(len(defl_values)):
                        for k in range(len(energy)):
                            spectrum[i][j][k] = counts[(i*len(defl_values)*len(energy))+(j*len(energy))+k]
                            

                raw = xr.DataArray(
                    data=spectrum,
                    dims=["step", "defl","eV"],
                    coords=dict(
                        eV = energy,
                        step = step,
                        defl = defl_values)
                )
                #raw.coords['neg_pol'] = ('step', negative_polarity)

                data = xr.Dataset(
                    data_vars={
                        'raw': raw,
                    }
                        )
                if 'sherman' not in kwargs:
                    data.attrs['sherman'] = 0.29

                #calc_spin_pol(data, **kwargs)
    
            ######################################################
            ##################### dispersion #####################
            ######################################################

            if scan_type == 'dispersion':
                with open(file) as f:
                    lines = f.readlines()
                #load data
                data_to_be_loaded = np.loadtxt(file)
                energies = data_to_be_loaded[:, 0]
                counts = data_to_be_loaded[:, 1]

                #get non-repeating KE values
                start_KEs = np.where(energies == energies[0])
                number_of_energies = start_KEs[0][1]       
                energy_values = energies[0:number_of_energies]

                #get theta_par values
                theta_par_values = []
                for line in lines:
                    if line.startswith('# NonEnergyOrdinate: '):
                        theta_par_values.append(float(line.replace('# NonEnergyOrdinate: ', "").strip()))

                #put measured spectra into a 2D array
                spectrum = [[0 for i in range(len(energy_values))] for j in range(len(theta_par_values))]
                for i in range(len(theta_par_values)):
                    for j in range(len(energy_values)):
                        spectrum[i][j] = counts[(i*len(energy_values))+j]

                #create xarray
                data = xr.DataArray(spectrum, dims=("theta_par","eV"), coords={"theta_par": theta_par_values,"eV": energy_values})
                data.coords['theta_par'].attrs = {'units' : 'deg'}
                meta_lines = lines[0:50]
                
            ##################################################
            ##################### FS map #####################
            ##################################################
                
            elif scan_type == 'FS map':
                with open(file) as f:
                    lines = f.readlines()
                #load data - takes 1 min 42 seconds to do a map that Igor takes 3 mins 18 seconds
                data_to_be_loaded = np.loadtxt(file)
                energies = data_to_be_loaded[:, 0]
                counts = data_to_be_loaded[:, 1]

                #get non-repeating KE values
                start_KEs = np.where(energies == energies[0])
                number_of_energies = start_KEs[0][1]       
                energy_values = energies[0:number_of_energies]

                with open(file) as f:
                    lines = f.readlines()
                #get theta_par values
                theta_par_values = []
                for line in lines:
                    if line.startswith('# NonEnergyOrdinate: '):
                        theta_par_values.append(float(line.replace('# NonEnergyOrdinate: ', "").strip()))
                    elif line.startswith('# Cycle: 1'):
                        break

                defl_values = []
                for line in lines:
                    if line.startswith('# Parameter: "SAL X [deg]" = '):
                        defl_values.append(float(line.replace('# Parameter: "SAL X [deg]" = ', "").strip()))

                #get theta_perp values
                theta_perp_values = defl_values

                #put measured spectra into a 3D array
                spectrum = [[[0 for i in range(len(energy_values))] for j in range(len(theta_par_values))] for k in range(len(theta_perp_values))]
                for i in range(len(theta_perp_values)):
                    for j in range(len(theta_par_values)):
                        for k in range(len(energy_values)):
                            spectrum[i][j][k] = counts[(i*len(theta_par_values)*len(energy_values))+(j*len(energy_values))+k]

                # Convert to a faster dtype for future processing
                spectrum = np.asarray(spectrum, dtype=np.float64)
                spectrum = spectrum.newbyteorder().newbyteorder()

                #create xarray
                data = xr.DataArray(spectrum, dims=("defl_perp","theta_par","eV"), coords={"defl_perp": theta_perp_values, "theta_par": theta_par_values,"eV": energy_values})
                data.coords['defl_perp'].attrs = {'units' : 'deg'}
                data.coords['theta_par'].attrs = {'units' : 'deg'}

                # Convert to a faster dtype ordering for future processing
                #data = data.astype('<f8', order='C', casting='equiv')

                meta_lines = lines[0:50]
           
    
    elif file_extension == '.sp2':
        # Based on Max-IV Phiobos loader from Craig Polley
        scan_type = 'dispersion'
        
        meta_lines = []
        
        with open(file,'rb') as f:
    
            start = 1e9
            for i, row_ in enumerate(f):
                row = row_.decode(errors='ignore')
                #f_.write(row + '\n')
                if i < start:
                    if row.startswith("# lensmode"): 
                        lens_mode = row.split(" = ")[1].replace('"','').strip('\n')
        
                    if row.startswith("# Kinetic Energy (Target)"): 
                        kinetic_energy = float(row.split("=")[1])
        
                    if row.startswith("# Pass Energy (Target)"): 
                        pass_energy = float(row.split("=")[1])
        
                    if row.startswith("# ERange"):
                        vals = row.split(' = ')[1].split(' # ')[0].split(' ')
                        energy_start = float(vals[0])
                        energy_stop = float(vals[1])
        
                    if row.startswith("# aRange"):
                        vals = row.split(' = ')[1].split(' # ')[0].split()
                        angle_start = float(vals[0])
                        angle_stop = float(vals[1])
        
                    if row.startswith("# SIZE_X"):
                        energy_steps = int(row.split("=")[1].split("#")[0])
        
                    if row.startswith("# SIZE_Y"): 
                        angle_steps = int(row.split("=")[1].split("#")[0])
        
                    try:
                        if float(row.split(" ")[0])  == energy_steps:
                            if float(row.split(" ")[1]) == angle_steps:
                                start = i+1
                                data = np.zeros(energy_steps * angle_steps)
                    except:
                        pass
                if i >= start:
                    data[i-start] = float(row)
                else:
                    meta_lines.append(row)
            
        spectrum = data.reshape(angle_steps, energy_steps)
        data_min = np.min(data)
        data_max = np.max(data)
        axis_energy = np.linspace(energy_start, energy_stop, energy_steps)
        axis_angle = np.linspace(angle_start, angle_stop, angle_steps)
        
        data = xr.DataArray(
            spectrum, 
            dims=("theta_par","eV"),
            coords={"theta_par": axis_angle,"eV": axis_energy}
        )
        
    
    #get xarray metadata
    if logbook == False:
        meta_list = get_bloch_spin_metadata(file,meta_lines,scan_type,logbook)
    
    #get logbook metadata
    else:
        sample_upload, scan_timestamp = get_bloch_spin_metadata(file,meta_lines,scan_type,logbook, [min(data.eV.data),max(data.eV.data)])
        return sample_upload, scan_timestamp
    
    #attach metadata
    for i in meta_list:
        data.attrs[i] = meta_list[i]
    
    try:
        #if xarray.DataArray (dispersion)
        data.name = meta_list['scan_name']
    except:
        #if xarray.DataSet (Spin EDC)
        data['name'] = meta_list['scan_name']
    
    return data

def get_bloch_spin_metadata(file,lines,scan_type,logbook,*args):
    '''This function will extract the relevant metadata from the data files obtained at the the MAX IV Bloch beamline
    
    Input:
        file - Path to the file being loaded (string)
        lines - A list where each entry is a line from the text file (list)
        scan_type - The type of scan, e.g. FS map (string)
        logbook - Whether the data should be returned for use by the logbook maker (True) or xarray (False) (Boolean)
        args - Min and Max energy, used for logbook (list)
        
    Retuns:
        if logbook == True:
            meta_list - List of relevant metadata (dictionary)
        else:
            sample_upload - List of relevant metadata (list)
            scan_timestamp - Timestamp of the scan (string)
    '''
    
    #split file full path to seperate file name
    fname = file.split('/')
    fname_noext = fname[len(fname)-1].split('.')
    
    #assign metadata
    meta_list = {}
    meta_list['scan_name'] = fname_noext[0]
    meta_list['scan_type'] = scan_type
    meta_list['sample_description'] = None
    meta_list['eV_type'] = 'kinetic'
    meta_list['beamline'] = 'MAX IV Bloch spin'
    meta_list['analysis_history'] = []
    meta_list['EF_correction'] = None
    meta_list['PE'] = float(my_find_bloch(lines, 'Pass Energy'))
    hv = my_find_bloch(lines, 'Excitation Energy')
    try:
        meta_list['hv'] = np.round(float(hv),2)
    except:
        try: 
            hv=hv.replace(',', '.')
            meta_list['hv'] = np.round(float(hv),2)
        except:
            meta_list['hv'] = None
    meta_list['pol'] = None
    meta_list['ana_slit'] = None
    meta_list['ana_slit_angle'] = 90
    meta_list['exit_slit'] = None
    meta_list['defl_par'] = None
    meta_list['defl_perp'] = None
    
    try:
        meta_list['sweeps'] = int(my_find_bloch(lines, 'Curves/Scan'))
        meta_list['dwell'] = float(my_find_bloch(lines, 'Dwell Time').split('#')[0])    
        meta_list['ana_mode'] = my_find_bloch(lines, 'Analyzer Lens').replace('"','')
        
    except:
        meta_list['sweeps'] = int(my_find_bloch(lines, 'numOfAccImages'))
        meta_list['dwell'] = float(my_find_bloch(lines, 'integration_time').split('#')[0])
        meta_list['ana_mode'] = my_find_bloch(lines, 'Lens Mode').replace('"','')
        
    meta_list['x1'] = None
    meta_list['x2'] = None
    meta_list['x3'] = None
    meta_list['polar'] = None
    meta_list['tilt'] = None
    meta_list['azi'] = None
    meta_list['norm_polar'] = None
    meta_list['norm_tilt'] = None
    meta_list['norm_azi'] = None
    meta_list['temp_sample'] = None
    meta_list['temp_cryo'] = None
    
    #if the data is being used for the logbook maker, we need additional data
    if logbook == True:
        scan_ext = fname_noext[1]
        scan_starttime = ''
        for line in lines:
            if line.startswith('# Acquisition Date') and ':' in line:
                scan_starttime = line.split()[4]
                break
            elif line.startswith('# Creation Date') and '=' in line:
                scan_starttime = line.split('"')[1].split()[1]
                break

        #kinetic Energy
        KE_st = args[0][0]
        KE_end = args[0][1]
        analyser_KE = str(round(KE_st,3))+':'+str(round(KE_end,3))
        analyser_step = ''

        #deflector
        if meta_list['defl_perp'] != None:
            analyser_thy = meta_list['defl_perp']
        else:
            analyser_thy = ''
        
        if meta_list['hv'] == None:
            meta_list['hv'] = ''
        
        #data to return
        scan_timestamp = str(os.path.getmtime(file))
        sample_upload = [scan_ext,meta_list['scan_name'],'',scan_type,scan_starttime,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),str(meta_list['ana_mode']),'','','','','',analyser_thy,'','','','','','','','',str(meta_list['hv'])]
        return sample_upload,scan_timestamp
    
    #if the data is being used for an xarray
    else:
        return meta_list
    
    
def my_find_bloch(lines, item):
    '''This function will loop over the lines in the file and then pick out the line
    starting with the desired keyword. 
    
    Input:
        lines - A list where each entry is a line from the text file (list)
        item - The word or group of characters you are searching for (string)
        
    Returns:
        The line starting with item (string)
    '''
    
    for line in lines:
        if line.startswith('# '+item) and '=' in line:
            return (line.split('='))[1].strip()
        if line.startswith('# '+item) and ':' in line:
            return (line.split(':'))[1].strip()