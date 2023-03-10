#Helper functions for acting on metadata
#Phil King 20/04/2021
#Brendan Edwards 26/042021
    
def update_hist(array, hist): 
    '''This function updates the analysis history metadata of the supplied xarray
    
    Input:
        array - the array to which the analysis history is to be updated (xarray)
        hist - string, to be written to analysis history (string)
    
    Returns:
        updated_array - the array with updated analysis history (xarray)'''
    
    
    #if analysis history is not in the xarray metadata
    if 'analysis_history' not in array.attrs:
        array.attrs['analysis_history'] = []
    
    #update the xarray analysis history (the following is done so that the original xarray is not overwritten)
    analysis_history = []
    for item in array.attrs['analysis_history']:
        analysis_history.append(item)
    analysis_history.append(hist)
    array.attrs['analysis_history'] = analysis_history
    
    return array