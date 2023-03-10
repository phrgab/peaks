#Relevant definitions for logbook classes etc. with beamline specificity
#PK 22/02/2020
#BE 22/09/2021

#Import packages needed to be loaded within this procedure
import gspread #For interfacing with Google sheets
import gspread_formatting as gsf #For formatting
   
def make_entry_logbook(scan_file,BL):
    '''This function will call the function for the relevant beamline to extract metadata from a file
    
    Input:
        scan_file - The name of the file from which metadata is being extracted.
        BL - Beamline selected by the user
        
    Retuns:
        sample_upload - List of relevant metadata
        scan.timestamp - Timestamp of scan_file (used to check if it has changed since last data write)
    '''
    
    logbook = True
    if 'Diamond I05-HR' in BL:
        from fileIO.loaders.i05HR_to_xarray import load_i05HR_data
        sample_upload,timestamp = load_i05HR_data(scan_file,logbook)
    elif 'Diamond I05-nano' in BL:
        from fileIO.loaders.i05nano_to_xarray import load_i05nano_data
        sample_upload,timestamp = load_i05nano_data(scan_file,logbook)
    elif 'MAX IV Bloch' in BL:
        from fileIO.loaders.bloch_to_xarray import load_bloch_data
        sample_upload,timestamp = load_bloch_data(scan_file,logbook)
    elif 'Elettra APE' in BL:
        from fileIO.loaders.APE_to_xarray import load_APE_data
        sample_upload,timestamp = load_APE_data(scan_file,logbook)
    elif 'SOLEIL CASSIOPEE' in BL:
        from fileIO.loaders.cassiopee_to_xarray import load_cassiopee_data
        sample_upload,timestamp = load_cassiopee_data(scan_file,logbook)
    elif 'St Andrews - Phoibos' in BL:
        from fileIO.loaders.StA_phoibos_to_xarray import load_StA_phoibos_data
        sample_upload,timestamp = load_StA_phoibos_data(scan_file,logbook)
    elif 'St Andrews - MBS' in BL:
        from fileIO.loaders.StA_MBS_to_xarray import load_StA_MBS_data
        sample_upload,timestamp = load_StA_MBS_data(scan_file,logbook)
    else:
        #Select Diamond i05-HR as default
        from fileIO.loaders.i05HR_to_xarray import load_i05HR_data
        sample_upload,timestamp = load_i05HR_data(scan_file,logbook)
    return sample_upload,timestamp

def write_data(ws,index,sample_upload,sname,BL,timestamp):
    '''This function will write the extracted metadata into the sample sheet
    
    Input:
        ws - Active worksheet
        index - Row index where entry exists
        sample_upload - List of relevant metadata
        sname - Sample name
        BL - Beamline selected by the user
        timestamp - Timestamp of file being scanned (used to check if it has changed since last data write)
    '''
    
    if 'Diamond I05-HR' in BL:
        ws.update_acell(gspread.utils.rowcol_to_a1(index,1), sample_upload[0])
        cell_list = ws.range('C'+str(index)+':'+gspread.utils.rowcol_to_a1(index,len(sample_upload)))
        x = 2
        for cell in cell_list:
            cell.value = sample_upload[x]
            x+=1
        ws.update_cells(cell_list)       
        ws.update_acell(gspread.utils.rowcol_to_a1(index,ws.col_count), timestamp)
    elif 'Diamond I05-nano' in BL:
        ws.update_acell(gspread.utils.rowcol_to_a1(index,1), sample_upload[0])
        cell_list = ws.range('C'+str(index)+':'+gspread.utils.rowcol_to_a1(index,len(sample_upload)))
        x = 2
        for cell in cell_list:
            cell.value = sample_upload[x]
            x+=1
        ws.update_cells(cell_list)       
        ws.update_acell(gspread.utils.rowcol_to_a1(index,ws.col_count), timestamp)
    elif 'MAX IV Bloch' in BL:
        if 'txt' in sample_upload[0] or 'zip' in sample_upload[0] or 'ibw' in sample_upload[0]:  #if data loaded from .txt file .zip file .ibw file
            ws.update_acell(gspread.utils.rowcol_to_a1(index,1), sample_upload[1])
            cell_list = ws.range('C'+str(index)+':'+'J'+str(index))
            x = 3
            for cell in cell_list:
                cell.value = sample_upload[x]
                x+=1
            ws.update_cells(cell_list)
            if sample_upload[13] != '':
                ws.update_acell('M'+str(index), sample_upload[13])
            cell_list = ws.range('N'+str(index)+':'+'S'+str(index))
            x = 14
            for cell in cell_list:
                cell.value = sample_upload[x]
                x+=1
            ws.update_cells(cell_list) 
            if sample_upload[22] != '':
                ws.update_acell('V'+str(index), sample_upload[22])
            ws.update_acell(gspread.utils.rowcol_to_a1(index,ws.col_count), timestamp)
        else:
            print('File format not yet supported')
    elif 'Elettra APE' in BL:
        if 'txt' in sample_upload[0] or 'zip' in sample_upload[0] or 'ibw' in sample_upload[0]:  #if data loaded from .txt file .zip file .ibw file
            ws.update_acell(gspread.utils.rowcol_to_a1(index,1), sample_upload[1])
            cell_list = ws.range('C'+str(index)+':'+'J'+str(index))
            x = 3
            for cell in cell_list:
                cell.value = sample_upload[x]
                x+=1
            ws.update_cells(cell_list)
            if sample_upload[13] != '':
                ws.update_acell('M'+str(index), sample_upload[13])
            if sample_upload[22] != '':
                ws.update_acell('V'+str(index), sample_upload[22])
            ws.update_acell(gspread.utils.rowcol_to_a1(index,ws.col_count), timestamp)   
        else:
            print('File format not yet supported')    
    elif 'SOLEIL CASSIOPEE' in BL:
        if 'txt' in sample_upload[0] or 'folder' in sample_upload[0] or 'ibw' in sample_upload[0]:  #if data loaded from .txt file .ibw file or folder
            ws.update_acell(gspread.utils.rowcol_to_a1(index,1), sample_upload[1])
            cell_list = ws.range('C'+str(index)+':'+'J'+str(index))
            x = 3
            for cell in cell_list:
                cell.value = sample_upload[x]
                x+=1
            ws.update_cells(cell_list)
            if sample_upload[12] != '':
                cell_list = ws.range('L'+str(index)+':'+'Q'+str(index))
                x = 12
                for cell in cell_list:
                    cell.value = sample_upload[x]
                    x+=1
                ws.update_cells(cell_list) 
            ws.update_acell('T'+str(index), sample_upload[20])        
            if sample_upload[22] != '':
                cell_list = ws.range('V'+str(index)+':'+'W'+str(index))
                x = 22
                for cell in cell_list:
                    cell.value = sample_upload[x]
                    x+=1
                ws.update_cells(cell_list) 
            ws.update_acell(gspread.utils.rowcol_to_a1(index,ws.col_count), timestamp)
        else:
            print('File format not yet supported')
    elif 'St Andrews - Phoibos' in BL:
        if 'xy' in sample_upload[0]:  #if data loaded from .xy
            ws.update_acell(gspread.utils.rowcol_to_a1(index,1), sample_upload[1])
            cell_list = ws.range('C'+str(index)+':'+'J'+str(index))
            x = 3
            for cell in cell_list:
                cell.value = sample_upload[x]
                x+=1
            ws.update_cells(cell_list)
            cell_list = ws.range('M'+str(index)+':'+'N'+str(index))
            x = 13
            for cell in cell_list:
                cell.value = sample_upload[x]
                x+=1
            ws.update_cells(cell_list)           
            if sample_upload[16] != '':
                cell_list = ws.range('P'+str(index)+':'+'Q'+str(index))
                x = 16
                for cell in cell_list:
                    cell.value = sample_upload[x]
                    x+=1
                ws.update_cells(cell_list)
            if sample_upload[23] != '':
                cell_list = ws.range('W'+str(index)+':'+'X'+str(index))
                x = 23
                for cell in cell_list:
                    cell.value = sample_upload[x]
                    x+=1
                ws.update_cells(cell_list)
            else:
                ws.update_acell('X'+str(index), sample_upload[24])
            ws.update_acell(gspread.utils.rowcol_to_a1(index,ws.col_count), timestamp)   
        else:
            print('File format not yet supported')
    elif 'St Andrews - MBS' in BL:
        if 'txt' in sample_upload[0] or 'krx' in sample_upload[0]:  #if data loaded from .txt file or folder
            ws.update_acell(gspread.utils.rowcol_to_a1(index,1), sample_upload[1])   
            cell_list = ws.range('C'+str(index)+':'+'L'+str(index))
            x = 3
            for cell in cell_list:
                cell.value = sample_upload[x]
                x+=1
            ws.update_cells(cell_list)    
            cell_list = ws.range('N'+str(index)+':'+'O'+str(index))
            x = 14
            for cell in cell_list:
                cell.value = sample_upload[x]
                x+=1
            ws.update_cells(cell_list) 
            ws.update_acell(gspread.utils.rowcol_to_a1(index,ws.col_count), timestamp)   
        else:
            print('File format not yet supported')
    else:
        print('Beamline not supported')

def get_extension(BL):
    '''This function selects the relevant file extensions for the beamline scans
    
    Input:
        BL - Beamline selected by the user
        
    Returns:
        scan_extension - List containing the relevant file extensions for the beamline scans
    '''
    
    if 'Diamond I05-HR' in BL or 'Diamond I05-nano' in BL:
        scan_extension = ['.nxs']
    elif 'MAX IV Bloch' in BL or 'Elettra APE' in BL:
        scan_extension = ['.txt','.zip','.ibw']
    elif 'SOLEIL CASSIOPEE' in BL:
        scan_extension = ['.txt','.ibw']
    elif 'St Andrews - Phoibos' in BL:
        scan_extension = ['.xy']
    elif 'St Andrews - MBS' in BL:
        scan_extension = ['.txt', '.krx']
    else:
        print('Selected beamline not supported. Setting nexus format by default')
        scan_extension = ['.nxs']
    return scan_extension

def get_headers(BL):
    '''This function selects the headers and columns information for the sample sheet, for the relevant beamline
    
    Input:
        BL - Beamline selected by the user
    
    Returns:
        master_boundaries - List containing the boundaries of the merged columns on the top row
        master_headers - List containing the entries in the top row
        master_headers_loc - List containing the column positions for the master headers
        headers - List containing the entries in the second row
        col_width - List containing the column widths
    '''
    
    if 'Diamond I05-HR' in BL:
        master_boundaries = [0,5,13,19,21,24,25]  #Where the grouped columns are
        master_headers = ['Scan','Analyser','Manipulator','Temperature','Beamline','Other']
        master_headers_loc = ['A','F','N','T','V','Y'] #Column labels for master headers
        headers = ['File Name','Notes','Type','Start time','Duration','KE range\n(eV)','Step\n(meV)','PE\n(eV)','No. of\nsweeps','Dwell\n(s)','Mode','Slit','Defl X\n(deg)','Polar\n(deg)','Tilt\n(deg)','Azi\n(deg)','x\n(mm)','y\n(mm)','z\n(mm)','Sample\n(K)','Cryostat\n(K)','hv\n(eV)','Exit slit\n(um)','Polarisation','File timestamp']
        col_width = [140,250,80,80,80,60,50,50,50,50,50,50,50,50,50,50,50,50,50,60,60,50,50,80,90]
    elif 'Diamond I05-nano' in BL:
        master_boundaries = [0,5,13,19,21,24,30,31]  #Where the grouped columns are
        master_headers = ['Scan','Analyser','Manipulator','Temperature','Beamline','Beamline focussing','Other']
        master_headers_loc = ['A','F','N','T','V','Y','AE'] #Column labels for master headers
        headers = ['File Name','Notes','Type','Start time','Duration','KE range\n(eV)','Step\n(meV)','PE\n(eV)','No. of\nsweeps','Dwell\n(s)','Mode','Slit','Ana. polar\n(deg)','Polar\n(deg)','Azi\n(deg)','x\n(um)','y\n(um)','z\n(um)','Defocus\n(um)','Sample\n(K)','Cryostat\n(K)','hv\n(eV)','Exit slit\n(um)','Polarisation','ZPx\n(um)','ZPy\n(um)','ZPz\n(um)','OSAx\n(um)','OSAy\n(um)','OSAz\n(um)','File timestamp']
        col_width = [140,250,80,80,80,60,50,50,50,50,50,50,65,50,50,50,50,50,60,60,60,50,50,80,50,50,50,50,50,50,90]
    elif 'MAX IV Bloch' in BL:
        master_boundaries = [0,4,13,19,21,24,25]  #Where the grouped columns are
        master_headers = ['Scan','Analyser','Manipulator','Temperature','Beamline','Other']
        master_headers_loc = ['A','E','N','T','V','Y'] #Column labels for master headers
        headers = ['File Name','Notes','Type','Time\nacquired','KE range\n(eV)','Step\n(meV)','PE\n(eV)','No. of\nsweeps','Dwell\n(s)','Mode','Slit','Theta_x\n(deg)','Theta_y\n(deg)','Polar\n(deg)','Tilt\n(deg)','Azi\n(deg)','x\n(mm)','y\n(mm)','z\n(mm)','Sample\n(K)','Cryostat\n(K)','hv\n(eV)','Exit slit\n(um)','Polarisation','File timestamp']
        col_width = [140,250,80,80,60,50,50,50,50,50,50,60,60,50,50,50,50,50,50,60,60,50,50,80,90]   
    elif 'Elettra APE' in BL:
        master_boundaries = [0,4,13,19,21,24,25]  #Where the grouped columns are
        master_headers = ['Scan','Analyser','Manipulator','Temperature','Beamline','Other']
        master_headers_loc = ['A','E','N','T','V','Y'] #Column labels for master headers
        headers = ['File Name','Notes','Type','Time\nacquired','KE range\n(eV)','Step\n(meV)','PE\n(eV)','No. of\nsweeps','Dwell\n(s)','Mode','Slit','Theta_x\n(deg)','Theta_y\n(deg)','Polar\n(deg)','Tilt\n(deg)','Azi\n(deg)','x\n(mm)','y\n(mm)','z\n(mm)','Sample\n(K)','Cryostat\n(K)','hv\n(eV)','Exit slit\n(um)','Polarisation','File timestamp']
        col_width = [140,250,80,80,60,50,50,50,50,50,50,60,60,50,50,50,50,50,50,60,60,50,50,80,90] 
    elif 'SOLEIL CASSIOPEE' in BL:
        master_boundaries = [0,4,11,17,19,23,24]  #Where the grouped columns are
        master_headers = ['Scan','Analyser','Manipulator','Temperature','Beamline','Other']
        master_headers_loc = ['A','E','L','R','T','X'] #Column labels for master headers
        headers = ['File Name','Notes','Type','Time\nacquired','KE range\n(eV)','Step\n(meV)','PE\n(eV)','No. of\nsweeps','Dwell\n(s)','Mode','Slit','Polar\n(deg)','Tilt\n(deg)','Azi\n(deg)','x\n(mm)','y\n(mm)','z\n(mm)','Sample\n(K)','Cryostat\n(K)','hv\n(eV)','Exit slit\n(um)','Resolution\n(meV)','Polarisation','File timestamp']
        col_width = [140,250,80,80,60,50,50,50,50,50,50,50,50,50,50,50,50,60,60,50,50,80,80,90]
    elif 'St Andrews - Phoibos' in BL:
        master_boundaries = [0,4,14,20,22,25,26]  #Where the grouped columns are
        master_headers = ['Scan','Analyser','Manipulator','Temperature','Excitation source','Other']
        master_headers_loc = ['A','E','O','U','W','Z'] #Column labels for master headers
        headers = ['File Name','Notes','Type','Time\nacquired','KE range\n(eV)','Step\n(meV)','PE\n(eV)','No. of\nsweeps','Dwell\n(s)','Bias\n(V)','L3\n(V)','Iris\n(mm)','Lens\nmode','Slit','Theta\n(deg)','Phi\n(deg)','Azi\n(deg)','x\n(mm)','y\n(mm)','z\n(mm)','Sample\n(K)','Cryostat\n(K)','Source','hv\n(eV)','Polarisation','File timestamp']
        col_width = [140,250,80,80,60,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,60,60,60,60,80,90]
    elif 'St Andrews - MBS' in BL:
        master_boundaries = [0,5,15,21,23,26,29]  #Where the grouped columns are
        master_headers = ['Scan','Analyser','Manipulator','Temperature','Excitation source','Other']
        master_headers_loc = ['A','F','P','V','X','AA'] #Column labels for master headers
        headers = ['File Name','Notes','Region name','Type','Time\nacquired','KE range\n(eV)','Step\n(meV)','PE\n(eV)','No. of\nframes','No. of\nscans','No. of\nsteps','Lens\nmode','Slit','DefX','DefY','Theta\n(deg)','Phi\n(deg)','Azi\n(deg)','x\n(mm)','y\n(mm)','z\n(mm)','Sample\n(K)','Cryostat\n(K)','Source','hv\n(eV)','Polarisation','Mono Z\n(mm)','Mono T\n(deg)','File timestamp']
        col_width = [140,250,90,80,80,60,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,60,60,60,60,80,50,50,90]        
    else:
        print('Selected beamline not supported. Setting Diamond I05-HR type by default')
        master_boundaries = [0,4,11,17,19,22,23]  #Where the grouped columns are
        master_headers = ['Scan','Analyser','Manipulator','Temperature','Beamline','Other']
        master_headers_loc = ['A','E','L','R','T','W'] #Column labels for master headers
        headers = ['File Name','Notes','Start time','Duration','KE range\n(eV)','Step\n(meV)','PE\n(eV)','No. of\nsweeps','Dwell\n(s)','Mode','Slit','Polar\n(deg)','Tilt\n(deg)','Azi\n(deg)','x\n(mm)','y\n(mm)','z\n(mm)','Sample\n(K)','Cryostat\n(K)','hv\n(eV)','Exit slit\n(um)','Polarisation','File timestamp']
        col_width = [140,250,80,80,60,50,50,50,50,50,50,50,50,50,50,50,50,60,60,50,50,80,90]
    
    return master_boundaries, master_headers, master_headers_loc, headers, col_width

def format_main_block(ws, BL, row_start, row_end, master_boundaries, fmtColumns, fmtColumnsToFill):
    '''This function defines and sets the formatting of the main block columns in the sample sheet, for the relevant beamline
    
    Input:
        ws - Active worksheet
        BL - Beamline selected by the user
        row_start - row to begin formatting
        row_end - row to finish formatting
        master_boundaries - List containing the boundaries of the merged columns on the top row
        fmtColumns - Cell formatting to give a very pale blue colour
        fmtColumnsToFill -Cell formatting to give a pale green colour
    '''
    
    if 'Diamond I05-HR' in BL:
        gsf.format_cell_ranges(ws, [(gspread.utils.rowcol_to_a1(row_start,master_boundaries[1]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[2]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[3]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[4]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[5]+1)+':'+gspread.utils.rowcol_to_a1(row_end,ws.col_count), fmtColumns)])
    elif 'Diamond I05-nano' in BL:
        gsf.format_cell_ranges(ws, [(gspread.utils.rowcol_to_a1(row_start,master_boundaries[1]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[2]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[3]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[4]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[5]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[6]), fmtColumns)])
    elif 'MAX IV Bloch' in BL:
        gsf.format_cell_ranges(ws, [(gspread.utils.rowcol_to_a1(row_start,master_boundaries[1]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[2]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[3]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[4]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[5]+1)+':'+gspread.utils.rowcol_to_a1(row_end,ws.col_count), fmtColumns)])
    elif 'Elettra APE' in BL:
        gsf.format_cell_ranges(ws, [(gspread.utils.rowcol_to_a1(row_start,master_boundaries[1]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[2]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[3]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[4]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[5]+1)+':'+gspread.utils.rowcol_to_a1(row_end,ws.col_count), fmtColumns)])
    elif 'SOLEIL CASSIOPEE' in BL:
        gsf.format_cell_ranges(ws, [(gspread.utils.rowcol_to_a1(row_start,master_boundaries[1]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[2]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[3]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[4]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[5]+1)+':'+gspread.utils.rowcol_to_a1(row_end,ws.col_count), fmtColumns)])
    elif 'St Andrews - Phoibos' in BL:
        gsf.format_cell_ranges(ws, [(gspread.utils.rowcol_to_a1(row_start,master_boundaries[1]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[2]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[3]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[4]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[5]+1)+':'+gspread.utils.rowcol_to_a1(row_end,ws.col_count), fmtColumns)])
    elif 'St Andrews - MBS' in BL:
        gsf.format_cell_ranges(ws, [(gspread.utils.rowcol_to_a1(row_start,master_boundaries[1]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[2]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[3]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[4]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[5]+1)+':'+gspread.utils.rowcol_to_a1(row_end,ws.col_count), fmtColumns)])
    else:
        gsf.format_cell_ranges(ws, [(gspread.utils.rowcol_to_a1(row_start,master_boundaries[1]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[2]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[3]+1)+':'+gspread.utils.rowcol_to_a1(row_end,master_boundaries[4]), fmtColumns),(gspread.utils.rowcol_to_a1(row_start,master_boundaries[5]+1)+':'+gspread.utils.rowcol_to_a1(row_end,ws.col_count), fmtColumns)])