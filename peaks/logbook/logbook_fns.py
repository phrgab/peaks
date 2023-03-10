#Relevant definitions for logging
#BE 13/05/2021

#Import packages needed to be loaded within this procedure
import os  #For reading file lists from folder
from decimal import Decimal #Used to round to required number of decimal places
import random
import time
import sys, signal
from more_itertools import locate
from datetime import datetime, timedelta
import gspread #For interfacing with Google sheets
import gspread_formatting as gsf #For formatting
from oauth2client.service_account import ServiceAccountCredentials #For authentication

def authentication(KeyID):
    '''This function opens a connection to a google spreadheet
    
    Input:
        KeyID - Key ID of the project spreadsheet
    
    Returns:
        sh - Opens the spreadsheet
    '''
    
    #Use creds to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds']
    file = os.path.join(os.path.dirname(__file__), 'client_secret.json')
    creds = ServiceAccountCredentials.from_json_keyfile_name(file, scope)
    client = gspread.authorize(creds)
    
    #Find a the usage log spreadsheet using the keyID and open the first sheet
    sh = client.open_by_key(KeyID)
    return sh

def exponential_backoff(func, info_message):
    '''This function periodically retrys a failed request over an increasing amount of time. It is used to prevent API errors
    where only so many read/write requests are allowed per 100 seconds. 
    
    Input:
        func - Function which is to be attempted e.g. create a logbook front page, or a folder upload
        info_message - Widget used to update the user about errors
        
    Returns:
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            func()
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")
            else:
                #If the error is not an API error
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                return successful_result
    return successful_result

def create_sample_folder(BL, sample_name, data_path):
    '''This function creates a folder where sample scan data will be saved
    
    Input:
        BL - Beamline name
        sample_name - Sample name
        data_path - Path to directory where data folder will be created
    '''
    
    date = datetime.today() 
    year = datetime.strftime(date,'%Y')
    year_short = datetime.strftime(date,'%y')
    month = datetime.strftime(date,'%m')
    month_word = datetime.strftime(date,'%B')
    day = datetime.strftime(date,'%d')
    
    if BL == 'St Andrews - Phoibos':
        #Directory for sample sample_name on the StA ARPES system
        scan_dir = data_path+year+'/'+month+'/'+year_short+month+day+'/'+sample_name+'/'+'Phoibos225/'
    elif BL == 'St Andrews - MBS':
        #Directory for sample sample_name on the StA Spin-ARPES system
        scan_dir = data_path+year+'/'+month+'/'+year_short+month+day+'/'+sample_name+'/'+'MBS_A1/'
    else:
        #Directory for sample sample_name at a beamline
        scan_dir = data_path+'Data/Synchrotron/'+BL+'/'+year+'/'+month_word+'/'+sample_name+'/'
    try:
        os.makedirs(scan_dir)
    except FileExistsError:
        pass   
    
def check_previous_session_ended_correctly(system, info_message):
    '''This function checks if the previous system session log is complete. If it is not,
    it will also check which other relevant logs are incomplete.
    
    Input:
        system - The system being used (ARPES or Spin-ARPES)
        info_message - Widget used to update the user about errors
    
    Returns:
        system_log_state - Whether the previous system session log is complete (true or false)
        helium_lamp_log_state - Whether the previous system helium lamp log is complete (true or false)
        usage_date - Date the system was last used
        usage_start_time - Start time when the system was previously used
        helium_lamp_start_time - Start time when the helium lamp was previously used
        system_users - Users who previously used the system
        project_name - Name of the previous session project
        active_sample - The most recently used sample
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #open google spreadsheet
            sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
            if system == 'ARPES':
                #Set active sheet to ARPES helium lamp sheet
                active_sheet = sh.worksheet('ARPES')
            elif system == 'Spin-ARPES':
                #Set active sheet to Spin-ARPES helium lamp sheet
                active_sheet = sh.worksheet('Spin-ARPES')
            #sixth column in the spreadsheet (start time column)
            sixth_column_entries = active_sheet.col_values(6)
            #Set row index_6 as the row of the last entry in column 6
            index_6 = len(sixth_column_entries)
            #seventh column in the spreadsheet (end time column)
            seventh_column_entries = active_sheet.col_values(7)
            #Set row index_7 as the row of the last entry in column 7
            index_7 = len(seventh_column_entries)
            #if there is a start and end time for the previous entry
            if index_6 == index_7:
                system_log_state = True
                #previous helium lamp entry must also be compelte as session can only be ended if helium lamp is turned off
                helium_lamp_log_state = True
                #these variables don't matter if the log was complete
                usage_date = ''
                usage_start_time = ''
                helium_lamp_start_time = ''
                system_users = ''
                project_name = ''
                active_sample = ''
            #if there is no end time for the previous entry
            else:
                system_log_state = False
                #extract previous session information
                last_entry = active_sheet.row_values(index_6)
                usage_date = last_entry[0]
                usage_start_time = last_entry[5]
                system_users = last_entry[1]
                project_name = last_entry[2]
                samples = last_entry[4]
                #get last sample entry
                samples_split = samples.split(", ")
                active_sample = samples_split[len(samples_split)-1]
                #check if the helium lamp log is complete
                if system == 'ARPES':
                    #Set active sheet to ARPES helium lamp sheet
                    active_sheet = sh.worksheet('ARPES helium lamp')
                elif system == 'Spin-ARPES':
                    #Set active sheet to Spin-ARPES helium lamp sheet
                    active_sheet = sh.worksheet('Spin-ARPES helium lamp')
                #third column in the spreadsheet (start time column)
                third_column_entries = active_sheet.col_values(3)
                #Set row index_3 as the row of the last entry in column 3
                index_3 = len(third_column_entries)
                #fourth column in the spreadsheet (end time column)
                fourth_column_entries = active_sheet.col_values(4)
                #Set row index_ as the row of the last entry in column 4
                index_4 = len(fourth_column_entries)
                #if there is a start and end time for the previous entry
                if index_3 == index_4:
                    helium_lamp_log_state = True
                #if there is no end time for the previous entry
                else:
                    helium_lamp_log_state = False
                helium_lamp_start_time = active_sheet.acell('C'+str(index_3)).value             
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")
                    system_log_state = ''
                    helium_lamp_log_state = ''
                    usage_date = ''
                    usage_start_time = ''
                    helium_lamp_start_time = ''
                    system_users = ''
                    project_name = ''
                    active_sample = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                system_log_state = ''
                helium_lamp_log_state = ''
                usage_date = ''
                usage_start_time = ''
                helium_lamp_start_time = ''
                system_users = ''
                project_name = ''
                active_sample = ''
                return system_log_state, helium_lamp_log_state, usage_date, usage_start_time, helium_lamp_start_time, system_users, project_name, active_sample, successful_result           
    return system_log_state, helium_lamp_log_state, usage_date, usage_start_time, helium_lamp_start_time, system_users, project_name, active_sample, successful_result

def get_project_info(project_name, info_message):
    '''This fuction gets the project infomation from the project log
    
    Input:
        project_name - Name of the project
        info_message - Widget used to update the user about errors
        
    Returns:
        project_lead - Name of the project lead
        project_grant - Project grant code
        project_keyID - Project spreadsheet key ID
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #open google spreadsheet
            sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
            #Set active sheet to projects sheet
            active_sheet = sh.worksheet('Projects')
            #first column in the spreadsheet consists of project lead names
            first_column_entries = active_sheet.col_values(1)    
            #second column in the spreadsheet consists of project names
            second_column_entries = active_sheet.col_values(2)
            #third column in the spreadsheet consists of project grant codes
            third_column_entries = active_sheet.col_values(3)
            #fourth column in the spreadsheet consists of project spreadsheet key IDs
            fourth_column_entries = active_sheet.col_values(4)
            #get corresponding project info
            project_lead = first_column_entries[second_column_entries.index(project_name)]
            project_grant = third_column_entries[second_column_entries.index(project_name)]
            project_keyID = fourth_column_entries[second_column_entries.index(project_name)]            
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")  
                    project_lead = ''
                    project_grant = ''
                    project_keyID = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                project_lead = ''
                project_grant = ''
                project_keyID = ''
                return project_lead, project_grant, project_keyID, successful_result
    return project_lead, project_grant, project_keyID, successful_result

def auto_complete_usage_info(system, info_message):
    '''This function gets the information to automatically complete the usage logs if the previous 
    session was ended incorrectly
    
    Input:
        system - The system being used (ARPES or Spin-ARPES)
        info_message - Widget used to update the user about errors
    
    Returns:
        helium_lamp_log_state - Whether the previous system helium lamp log is complete (true or false)
        system_end_time - The time that logging of system was assumed to finished
        helium_lamp_duration - The duration the helium lamp was assumed to be on
        successful_result - Whether or not the action was performed (True or False)
    '''

    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #open google spreadsheet
            sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
            if system == 'ARPES':
                #Set active sheet to ARPES helium lamp sheet
                active_sheet = sh.worksheet('ARPES')
            elif system == 'Spin-ARPES':
                #Set active sheet to Spin-ARPES helium lamp sheet
                active_sheet = sh.worksheet('Spin-ARPES')
            #sixth column in the spreadsheet (start time column)
            sixth_column_entries = active_sheet.col_values(6)
            #Set row index_6 as the row of the last entry in column 6
            index_6 = len(sixth_column_entries)
            #extract previous session information
            usage_date = active_sheet.acell('A'+str(index_6)).value
            usage_start_time = active_sheet.acell('F'+str(index_6)).value
            system_start_time = datetime.strptime(usage_date + " " + usage_start_time, "%Y-%m-%d %H:%M:%S")
            system_end_time = system_start_time + timedelta(hours=8)
            #check if the helium lamp log is complete
            if system == 'ARPES':
                #Set active sheet to ARPES helium lamp sheet
                active_sheet = sh.worksheet('ARPES helium lamp')
            elif system == 'Spin-ARPES':
                #Set active sheet to Spin-ARPES helium lamp sheet
                active_sheet = sh.worksheet('Spin-ARPES helium lamp')
            #third column in the spreadsheet (start time column)
            third_column_entries = active_sheet.col_values(3)
            #Set row index_3 as the row of the last entry in column 3
            index_3 = len(third_column_entries)
            #fourth column in the spreadsheet (end time column)
            fourth_column_entries = active_sheet.col_values(4)
            #Set row index_ as the row of the last entry in column 4
            index_4 = len(fourth_column_entries)
            #if there is a start and end time for the previous entry
            if index_3 == index_4:
                helium_lamp_log_state = True
                helium_lamp_duration = ''
            #if there is no end time for the previous entry
            else:
                helium_lamp_log_state = False
                helium_lamp_start_time = active_sheet.acell('C'+str(index_3)).value
                helium_lamp_start_time = datetime.strptime(usage_date + " " + helium_lamp_start_time, "%Y-%m-%d %H:%M:%S")
                #Calcuate helium lamp duration
                helium_lamp_duration_full = system_end_time - helium_lamp_start_time
                #Get helium lamp duration in hr:min:sec format
                hours, remainder = divmod(helium_lamp_duration_full.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                helium_lamp_duration = str(hours) + ':' + str(minutes) + ':' + str(seconds)              
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")  
                    helium_lamp_log_state = ''
                    system_end_time = ''
                    helium_lamp_duration = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                helium_lamp_log_state = ''
                system_end_time = ''
                helium_lamp_duration = ''
                return helium_lamp_log_state, system_end_time, helium_lamp_duration, successful_result
    return helium_lamp_log_state, system_end_time, helium_lamp_duration, successful_result

def key_ID_check(keyID, info_message):
    '''This function checks if the key ID used is valid
    
    Input:
        keyID - Key ID of the project spreadsheet
        info_message - Widget used to update the user about errors
    
    Returns:
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    successful_result = False
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            sh = authentication(keyID)
            sh.worksheets()
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                #if API usage limit error
                if n < n_stop-1:
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")
            elif str(e).split(",")[0] == "{'code': 404":
                #if invalid keyID
                with info_message:
                    info_message.clear_output()
                    print("Invalid key ID. Please check that it is valid and that the spreadsheet has been shared")
                    return successful_result
            else:
                #if other error
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                return successful_result
    return successful_result

def get_users_projects_info(main_user, info_message):
    '''This function retrieves the projects that the main user is lead on from the usage spreadsheet,
    along with the corresponding project grants and project spreadsheet key IDs
    
    Input:
        main_user - Name of the project lead
        info_message - Widget used to update the user about errors
    
    Returns:
        project_name_list - List of all the projects that the main user is lead on
        project_grant_list - List of all corresponding project grants
        spreadsheet_keyID_list - List of all corresponding project spreadsheet key IDs
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #open google spreadsheet
            sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
            #Set active sheet to projects sheet
            active_sheet = sh.worksheet('Projects')
            #first column in the spreadsheet consists of the name of the project lead
            first_column_entries = active_sheet.col_values(1)
            rows_of_main_users_projects = [] #rows start at index 1 in google sheets
            i = 1
            #Loops through all the columns in the spreadsheet
            for entry in first_column_entries:
                #If a project lead name in the google spreadsheet matches the main user
                if entry == main_user:
                    rows_of_main_users_projects.append(i)
                i+=1
            project_name_list = []
            project_grant_list = []
            spreadsheet_keyID_list = []
            #Loops through all the rows in the google spreadsheet which have the main user as the project lead
            for row in rows_of_main_users_projects:
                project_name_list.append(active_sheet.acell('B'+str(row)).value)
                project_grant_list.append(active_sheet.acell('C'+str(row)).value)
                spreadsheet_keyID_list.append(active_sheet.acell('D'+str(row)).value)            
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":            
            
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")  
                    project_name_list = ''
                    project_grant_list = ''
                    spreadsheet_keyID_list = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                project_name_list = ''
                project_grant_list = ''
                spreadsheet_keyID_list = ''                
                return project_name_list, project_grant_list, spreadsheet_keyID_list, successful_result
    return project_name_list, project_grant_list, spreadsheet_keyID_list, successful_result

def get_project_samples(keyID, info_message):
    '''This function retrieves the previous samples used during the selected project
    
    Input:
        keyID - Key ID of the project spreadsheet
        info_message - Widget used to update the user about errors
    
    Returns:
        sample_list - List of all the project samples
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #open google spreadsheet
            sh = authentication(keyID)
            #Get list of worksheets
            ws_list = sh.worksheets()
            sample_list = []
            #ws_list is a list of all the worksheets in the spreadsheet
            for worksheet in ws_list:
                #This is done because the worksheet name is saved in between two apostrophes in ws_list
                current_sample = str(worksheet).split("'")
                sample_list.append(current_sample[1])
            #Remove front page to just get a sample list
            if "'Info'" in str(ws_list):
                sample_list.remove('Info')            
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")
                    sample_list = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)  
                sample_list = ''
                return sample_list, successful_result
    return sample_list, successful_result

def new_project_entry(project_lead, project_name, project_grant, project_keyID, info_message):
    '''This function adds a new entry into the project spreadsheet in the usage log
    
    Input:
        project_lead - The name of the project lead
        project_name - The name of the project
        project_grant - The project grant
        project_keyID - The key ID of the project spreadsheet
        info_message - Widget used to update the user about errors
    
    Returns:
        project_exists - Whether the project name has been used already (true or false)
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #open google spreadsheet
            sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
            #Set active sheet to projects sheet
            active_sheet = sh.worksheet('Projects')
            #second column in the spreadsheet consists of the names of the projects
            second_column_entries = active_sheet.col_values(2)
            project_exists = False
            for name in second_column_entries:
                if name == project_name:
                    project_exists = True
                    return project_exists, successful_result
            #Set row index as the row after the last entry in the left column
            index = len(second_column_entries)+1
            #The cells that will be updated
            cell_list = active_sheet.range('A'+str(index)+':'+'D'+str(index))
            #Info to be uploaded
            project_info_upload = [project_lead, project_name, project_grant, project_keyID]
            i = 0
            #Assign cell values to the relevant project info
            for cell in cell_list:
                cell.value = project_info_upload[i]
                i+=1
            #Update spreadsheet cells
            active_sheet.update_cells(cell_list)           
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")
                    project_exists = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                project_exists = ''
                return project_exists, successful_result
    return project_exists, successful_result

def helium_lamp_start_entry(system, main_user, helium_lamp_start_time):
    '''This function is called when the helium lamp is turned on and it updates the relevant 
    entries in the helium lamp spreadsheet in the usage log
    
    Input:
        system - The system being used (ARPES or Spin-ARPES)
        main_user - The name of the project lead
        helium_lamp_start_time - The time that logging of helium lamp usage begun
    '''
    
    #open google spreadsheet
    sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
    if system == 'ARPES':
        #Set active sheet to ARPES helium lamp sheet
        active_sheet = sh.worksheet('ARPES helium lamp')
    elif system == 'Spin-ARPES':
        #Set active sheet to Spin-ARPES helium lamp sheet
        active_sheet = sh.worksheet('Spin-ARPES helium lamp')
    #Extract the date and start time
    date = datetime.strftime(helium_lamp_start_time,'%Y-%m-%d')
    start_time = datetime.strftime(helium_lamp_start_time,"%H:%M:%S")
    #first column in the spreadsheet
    first_column_entries = active_sheet.col_values(1)
    #Set row index as the row after the last entry in the left column
    index = len(first_column_entries)+1
    #The cells that will be updated
    cell_list = active_sheet.range('A'+str(index)+':'+'C'+str(index))
    #Info to be uploaded
    usage_info = [date, main_user, start_time]
    i = 0
    #Assign cell values to the relevant project info
    for cell in cell_list:
        cell.value = usage_info[i]
        i+=1
    #Update spreadsheet cells
    active_sheet.update_cells(cell_list) 

def helium_lamp_end_entry(system, helium_lamp_end_time, helium_lamp_duration, info_message):
    '''This function is called when the helium lamp is turned off and it updates the relevant 
    entries in the helium lamp spreadsheet in the usage log
    
    Input:
        system - The system being used (ARPES or Spin-ARPES)
        helium_lamp_end_time - The time that logging of helium lamp usage finished
        helium_lamp_duration - The amount of time the helium lamp was used for
        info_message - Widget used to update the user about errors
    
    Returns:
        new_usage_since_reset - The total amount of the time the helium lamp has been used for since the previous reset
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #open google spreadsheet
            sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
            if system == 'ARPES':
                #Set active sheet to ARPES helium lamp sheet
                active_sheet = sh.worksheet('ARPES helium lamp')
            elif system == 'Spin-ARPES':
                #Set active sheet to Spin-ARPES helium lamp sheet
                active_sheet = sh.worksheet('Spin-ARPES helium lamp')
            #Extract the end time
            end_time = datetime.strftime(helium_lamp_end_time,"%H:%M:%S")
            #first column in the spreadsheet
            first_column_entries = active_sheet.col_values(1)
            #Set row index as the row of the last entry in the left column
            index = len(first_column_entries)
            #The cells that will be updated
            cell_list = active_sheet.range('D'+str(index)+':'+'E'+str(index))
            #Info to be uploaded
            usage_info = [end_time, helium_lamp_duration]
            i = 0
            #Assign cell values to the relevant info
            for cell in cell_list:
                cell.value = usage_info[i]
                i+=1
            #Update spreadsheet cells
            active_sheet.update_cells(cell_list)
            #Get current total helium lamp usage cells values
            current_total_usage = active_sheet.acell('K3').value
            current_usage_since_reset = active_sheet.acell('K4').value
            #Split values into hr:min:sec
            current_total_usage_split = str(current_total_usage).split(":")
            current_usage_since_reset_split = str(current_usage_since_reset).split(":")
            helium_lamp_duration_split = helium_lamp_duration.split(":")
            #Calculate new total helium lamp usage values
            new_total_usage_minutes_to_add, new_total_usage_seconds = divmod(int(current_total_usage_split[2]) + int(helium_lamp_duration_split[2]), 60)
            new_total_usage_hours_to_add, new_total_usage_minutes  =  divmod(int(current_total_usage_split[1]) + int(helium_lamp_duration_split[1]) + new_total_usage_minutes_to_add, 60)
            new_total_usage_hours = int(current_total_usage_split[0]) + int(helium_lamp_duration_split[0]) + new_total_usage_hours_to_add
            new_total_usage = str(new_total_usage_hours) + ':' + str(new_total_usage_minutes) + ':' + str(new_total_usage_seconds)
            new_usage_since_reset_minutes_to_add, new_usage_since_reset_seconds = divmod(int(current_usage_since_reset_split[2]) + int(helium_lamp_duration_split[2]), 60)
            new_usage_since_reset_hours_to_add, new_usage_since_reset_minutes  =  divmod(int(current_usage_since_reset_split[1]) + int(helium_lamp_duration_split[1]) + new_usage_since_reset_minutes_to_add, 60)
            new_usage_since_reset_hours = int(current_usage_since_reset_split[0]) + int(helium_lamp_duration_split[0]) + new_usage_since_reset_hours_to_add
            new_usage_since_reset = str(new_usage_since_reset_hours) + ':' + str(new_usage_since_reset_minutes) + ':' + str(new_usage_since_reset_seconds)
            #Update current total helium lamp usage cells values
            active_sheet.update_acell('K3', new_total_usage)
            active_sheet.update_acell('K4', new_usage_since_reset)            
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")
                    new_usage_since_reset = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                new_usage_since_reset = ''
                return new_usage_since_reset, successful_result
    return new_usage_since_reset, successful_result

def reset_helium_lamp_usage(system):
    '''This function resets the helium lamp log for the relevant system
    
    Input:
        system - The system being used (ARPES or Spin-ARPES)
    '''
    
    #open google spreadsheet
    sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
    if system == 'ARPES':
        #Set active sheet to ARPES helium lamp sheet
        active_sheet = sh.worksheet('ARPES helium lamp')
    elif system == 'Spin-ARPES':
        #Set active sheet to Spin-ARPES helium lamp sheet
        active_sheet = sh.worksheet('Spin-ARPES helium lamp')
    #Extract the end time
    date = datetime.strftime(datetime.today(),'%Y-%m-%d')
    #Get current total helium lamp usage since previous reset cell value
    current_usage_since_reset = active_sheet.acell('K4').value
    #seventh column in the spreadsheet
    seventh_column_entries = active_sheet.col_values(7)
    #Set row index as the row after the last entry in the seventh column
    index = len(seventh_column_entries) + 1
    #The cells that will be updated
    cell_list = active_sheet.range('G'+str(index)+':'+'H'+str(index))
    #Info to be uploaded
    reset_info = [date, current_usage_since_reset]
    i = 0
    #Assign cell values to the relevant info
    for cell in cell_list:
        cell.value = reset_info[i]
        i+=1
    #Update spreadsheet cells
    active_sheet.update_cells(cell_list)
    active_sheet.update_acell('K4', '0:0:0')
    active_sheet.update_acell('K5', date)
    
def system_usage_start_entry(system, team, project_name, project_grant, sample_name, system_start_time):
    '''This function is called when logging begins and it updates the relevant 
    entries in the system usage spreadsheet (ARPES or Spin-ARPES) in the usage log
    
    Input:
        system - The system being used (ARPES or Spin-ARPES)
        team - Names of the users using the system during the session
        project_name - The name of the project
        project_grant - The project grant
        sample_name - Inital sample name
        system_start_time - The time that logging of the system begun
    '''
    
    #open google spreadsheet
    sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
    if system == 'ARPES':
        #Set active sheet to ARPES helium lamp sheet
        active_sheet = sh.worksheet('ARPES')
    elif system == 'Spin-ARPES':
        #Set active sheet to Spin-ARPES helium lamp sheet
        active_sheet = sh.worksheet('Spin-ARPES')
    #Extract the date and start time
    date = datetime.strftime(system_start_time,'%Y-%m-%d')
    start_time = datetime.strftime(system_start_time,"%H:%M:%S")    
    #first column in the spreadsheet
    first_column_entries = active_sheet.col_values(1)
    #Set row index as the row after the last entry in the left column
    index = len(first_column_entries)+1
    #The cells that will be updated
    cell_list = active_sheet.range('A'+str(index)+':'+'F'+str(index))
    #Info to be uploaded
    session_info = [date, team, project_name, project_grant, sample_name, start_time]
    i = 0
    #Assign cell values to the relevant project info
    for cell in cell_list:
        cell.value = session_info[i]
        i+=1
    #Update spreadsheet cells
    active_sheet.update_cells(cell_list) 
    
def system_usage_end_entry(system, system_end_time, system_usage_duration):
    '''This function is called when logging ends and it updates the relevant 
    entries in the system usage spreadsheet (ARPES or Spin-ARPES) in the usage log
    
    Input:
        system - The system being used (ARPES or Spin-ARPES)
        system_end_time - The time that logging of the system finished  
        system_usage_duration - The amount of time the system was used for
    '''
    
    #open google spreadsheet
    sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
    if system == 'ARPES':
        #Set active sheet to ARPES helium lamp sheet
        active_sheet = sh.worksheet('ARPES')
    elif system == 'Spin-ARPES':
        #Set active sheet to Spin-ARPES helium lamp sheet
        active_sheet = sh.worksheet('Spin-ARPES')
    #Extract the end time
    end_time = datetime.strftime(system_end_time,"%H:%M:%S")
    #Get capped duration (capped at 8 hours)
    system_usage_duration_split = str(system_usage_duration).split(":")
    if int(system_usage_duration_split[0]) >= 8:
        capped_duration = '8:0:0'
    else:
        capped_duration = system_usage_duration
    #first column in the spreadsheet
    first_column_entries = active_sheet.col_values(1)
    #Set row index as the row of the last entry in the left column
    index = len(first_column_entries)
    #The cells that will be updated
    cell_list = active_sheet.range('G'+str(index)+':'+'I'+str(index))
    #Info to be uploaded
    usage_info = [end_time, system_usage_duration, capped_duration]
    i = 0
    #Assign cell values to the relevant info
    for cell in cell_list:
        cell.value = usage_info[i]
        i+=1
    #Update spreadsheet cells
    active_sheet.update_cells(cell_list)

def create_st_andrews_frontpage(keyID, project_name, project_lead, project_grant):
    '''This function creates the frontpage of the project logbook when using the St Andrews lab
    
    Input:
        keyID - Key ID of the project spreadsheet
        project_name - The name of the project
        project_lead - The name of the project lead
        project_grant - The project grant
    '''
    
    #Formatting definitions
    #Title cells format (pale yellow plus bold)
    fmtTitle = gsf.cellFormat(
        backgroundColor=gsf.color(255.0/255, 233.0/255, 171.0/255),
        textFormat=gsf.textFormat(bold=True)
        )
    
    #Sub title cells format (bold)
    fmtSubTitle = gsf.cellFormat(
        textFormat=gsf.textFormat(bold=True)
        )
    
    #open google spreadsheet
    sh = authentication(keyID)
    #returns list of existing worksheets
    ws_list = sh.worksheets()
    
    #Does Info sheet already exist?
    if "'Info'" not in str(ws_list):
        #Create a new sheet with label "Info" for general beamtime info
        sh.add_worksheet(title="Info", rows="100", cols="5")
        
    #Resize columns
    sheetId = sh.worksheet('Info')._properties['sheetId']
    col_width = [100,240,400,400,100]
    x = 0
    for i in col_width:
        body = {"requests": [{"updateDimensionProperties": {"range": {"sheetId": sheetId,"dimension": 
            "COLUMNS","startIndex": x,"endIndex": x+1},
            "properties": {"pixelSize": i},"fields": "pixelSize"}}]}
        sh.batch_update(body)
        x += 1

    if "'Sheet1'" in str(ws_list):
        #Delete old blank sheet
        ws = sh.worksheet('Sheet1')
        sh.del_worksheet(ws)

    #Set active sheet to Info sheet
    active_sheet = sh.worksheet('Info')    
    
    #Write entries to frontsheet
    
    #Project name
    active_sheet.update_acell('A1', 'Project name')
    gsf.format_cell_range(active_sheet, 'A1', fmtTitle)
    active_sheet.update_acell('B1', project_name)

    #Project lead
    active_sheet.update_acell('A3', 'Project lead')
    gsf.format_cell_range(active_sheet, 'A3', fmtTitle)
    active_sheet.update_acell('B3', project_lead)
    
    #Project grant
    active_sheet.update_acell('A5', 'Project grant')
    gsf.format_cell_range(active_sheet, 'A5', fmtTitle)
    active_sheet.update_acell('B5', project_grant)

    #Start date
    active_sheet.update_acell('A7', 'Start date')
    gsf.format_cell_range(active_sheet, 'A7', fmtTitle)
    start_date = datetime.strftime(datetime.today(),'%Y-%m-%d')
    active_sheet.update_acell('B7', start_date)    
    
    #Sample infomation
    cell_list = active_sheet.range('A9:E9')
    #Info to be uploaded
    cell_values = ['Sample history', 'Sample name', 'Sample description', 'Collaborators', 'Sample date']
    i = 0
    #Assign cell values to the relevant info
    for cell in cell_list:
        cell.value = cell_values[i]
        i+=1
    #Update spreadsheet cells
    active_sheet.update_cells(cell_list)
    gsf.format_cell_range(active_sheet, 'A9', fmtTitle)
    gsf.format_cell_ranges(active_sheet, [('B9:E9', fmtSubTitle)])

def create_samplesheet(BL, keyID, sample_name, sample_des, sample_info, team, info_message):
    '''This function creates a new sample sheet in the logbook
    
    Input:
        BL - Beamline name
        keyID - Key ID of the project spreadsheet
        sample_name - Name of the active sample
        sample_des - Description of the active sample
        sample_info - Additional information of the active sample
        team - Names of the users using the system during the session when a new sample is created
        info_message - Widget used to update the user about errors
        
    Returns:
        sample_sheet_already_created - Whether or not a sample sheet had already been created
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #Formatting definitions
    #Title cells (pale yellow plus bold)
    fmtTitle = gsf.cellFormat(
        backgroundColor=gsf.color(255.0/255, 233.0/255, 171.0/255),
        textFormat=gsf.textFormat(bold=True)
        )

    #Comment cells (pale yellow)
    fmtComment = gsf.cellFormat(
        backgroundColor=gsf.color(255.0/255, 233.0/255, 171.0/255),
        )
    
    #Master header cells (pale blue plus bold)
    fmtMasterHeaders = gsf.cellFormat(
        backgroundColor=gsf.color(153.0/255, 204.0/255, 255.0/255),
        textFormat=gsf.textFormat(bold=True),
        horizontalAlignment='CENTER'
        )

    #Header cells (pale blue not bold)
    fmtHeaders = gsf.cellFormat(
        backgroundColor=gsf.color(153.0/255, 204.0/255, 255.0/255),
        textFormat=gsf.textFormat(bold=False),
        horizontalAlignment='CENTER'
        )

    #Cells to give column contrast (very pale blue)
    fmtColumns = gsf.cellFormat(
        backgroundColor=gsf.color(215.0/255, 236.0/255, 245.0/255),
        textFormat=gsf.textFormat(bold=False),
        )
    
    #Cells for user to complete (pale green)
    fmtColumnsToFill = gsf.cellFormat(
        backgroundColor=gsf.color(194.0/255, 227.0/255, 175.0/255),
        textFormat=gsf.textFormat(bold=False),
        )
    
    #Define relevant layout options and file extensions for sample sheets (varies by syncrotron and beamline)
    from .beamline_dfns import get_headers
    master_boundaries, master_headers, master_headers_loc, headers, col_width = get_headers(BL)
    
    date = date = datetime.strftime(datetime.today(),'%Y-%m-%d')
    #Open google spreadsheet
    sh = authentication(keyID)
    
    #Check if sample sheet is already created
    sample_sheet_already_created, successful_result = create_samplesheet_check(sh, sample_name, headers, info_message)
    if sample_sheet_already_created == False and successful_result == True:
    #If sample sheet not already created, format a new one, fill the contents and update the info page
    #Only allow next function to occur if the previous was successful
        successful_result = exponential_backoff(lambda: create_samplesheet_format(sh, sample_name, fmtMasterHeaders, fmtHeaders, master_boundaries, master_headers, headers, col_width), info_message)
        if successful_result == True:
            successful_result = exponential_backoff(lambda: create_samplesheet_write(sh, BL, date, sample_name, sample_des, sample_info, fmtTitle, fmtComment, fmtColumns, fmtColumnsToFill, master_boundaries, master_headers, master_headers_loc, headers), info_message)
            if successful_result == True:
                successful_result = exponential_backoff(lambda: create_samplesheet_update_info_page(sh, BL, date, sample_name, sample_des, team), info_message)
    return sample_sheet_already_created, successful_result

def create_samplesheet_check(sh, sample_name, headers, info_message):
    '''This function check if a sample sheet has already been created, and if it has not it creates a new one
    (exponential backoff algorithm built in since we want sample_sheet_already_created to be returned)
    
    Input:
        sh - Opens the spreadsheet
        sample_name - Name of the active sample
        headers - List containing the entries in the second row
        info_message - Widget used to update the user about errors
        
    Returns:
        sample_sheet_already_created - Whether or not a sample sheet had already been created
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #return list of existing worksheets
            ws_list = sh.worksheets()
            #If sheet already exists
            if ("'"+sample_name+"'") in str(ws_list):
                sample_sheet_already_created = True
            #If sheet does not already exists, create sample sheet
            else:
                sample_sheet_already_created = False 
                #Create a new sheet with label sample_name for sample sample_name logbook
                sh.add_worksheet(title=sample_name, rows="100", cols=len(headers))
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")  
                    sample_sheet_already_created = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)    
                sample_sheet_already_created = ''
                return sample_sheet_already_created, successful_result
    return sample_sheet_already_created, successful_result
    
def create_samplesheet_format(sh, sample_name, fmtMasterHeaders, fmtHeaders, master_boundaries, master_headers, headers, col_width):
    '''This function formats the sample sheet
    
    Input:
        sh - Opens the spreadsheet
        sample_name - Name of the active sample
        fmtMasterHeaders - Master header format definition
        fmtHeaders - Header format definition
        master_boundaries - List containing the boundaries of the merged columns on the top row
        master_headers - List containing the entries in the top row
        headers - List containing the entries in the second row
        col_width - List containing the column widths
    '''
    
    #Sort out layout
    sheetId = sh.worksheet(sample_name)._properties['sheetId']
    #Merge cells for master headers
    for x in range(0,len(master_headers)):
        body = {"requests": [{"mergeCells": {"mergeType": "MERGE_ALL","range": {"sheetId": sheetId,
            "startRowIndex": 0,"endRowIndex": 1,"startColumnIndex": master_boundaries[x],"endColumnIndex": master_boundaries[x+1]}}}]}
        sh.batch_update(body)
    #Resize columns
    x = 0
    for i in col_width:
        body = {"requests": [{"updateDimensionProperties": {"range": {"sheetId": sheetId,"dimension": 
            "COLUMNS","startIndex": x,"endIndex": x+1},
            "properties": {"pixelSize": i},"fields": "pixelSize"}}]}
        sh.batch_update(body)
        x += 1

    #Set active sheet to sample_name sheet
    active_sheet = sh.worksheet(sample_name)    
     
    #Format header cells
    gsf.format_cell_ranges(active_sheet, [('A1:'+gspread.utils.rowcol_to_a1(1,len(headers)), fmtMasterHeaders),('A2:'+gspread.utils.rowcol_to_a1(2,len(headers)), fmtHeaders)])

def create_samplesheet_write(sh, BL, date, sample_name, sample_des, sample_info, fmtTitle, fmtComment, fmtColumns, fmtColumnsToFill, master_boundaries, master_headers, master_headers_loc, headers):
    '''This function writes the conent in the sample sheet
    
    Input:
        sh - Opens the spreadsheet
        date - Date in YYYY-MM-DD format
        sample_name - Name of the active sample
        sample_des - Description of the active sample
        sample_info - Additional information of the active sample
        fmtTitle - Title format definition
        fmtComment - Comment format definition
        fmtColumns  - Column format definition
        fmtColumnsToFill - Column to fill format definition
        master_boundaries - List containing the boundaries of the merged columns on the top row
        master_headers - List containing the entries in the top row
        master_headers_loc - List containing the column positions for the master headers
        headers - List containing the entries in the second row
    '''
    
    active_sheet = sh.worksheet(sample_name)
    #Write master headers
    for x in range(0,len(master_headers)):
        active_sheet.update_acell(master_headers_loc[x]+'1', master_headers[x])

    #Write headers
    cell_list = active_sheet.range('A2:'+gspread.utils.rowcol_to_a1(2,len(headers)))
    x = 0
    for cell in cell_list:
        cell.value = headers[x]
        x+=1
    active_sheet.update_cells(cell_list)

    #Freeze header rows
    gsf.set_frozen(active_sheet, rows=2)
    #Write sample info
    active_sheet = sh.worksheet(sample_name)
    
    cell_list = active_sheet.range('A3:A6')
    #Info to be uploaded
    cell_values = ['Sample name:', 'Sample description:', 'Sample information:', 'Start date:']
    i = 0
    #Assign cell values to the relevant info
    for cell in cell_list:
        cell.value = cell_values[i]
        i+=1
    #Update spreadsheet cells
    active_sheet.update_cells(cell_list)
    
    cell_list = active_sheet.range('B3:B6')
    #Info to be uploaded
    sample_info = [sample_name, sample_des, sample_info, date]
    i = 0
    #Assign cell values to the relevant info
    for cell in cell_list:
        cell.value = sample_info[i]
        i+=1
    #Update spreadsheet cells
    active_sheet.update_cells(cell_list)
    
    #format cells
    gsf.format_cell_range(active_sheet, 'A3:A6', fmtTitle)
    gsf.format_cell_range(active_sheet, 'B3:'+gspread.utils.rowcol_to_a1(6,len(headers)), fmtComment)

    #Call fuction to format columns for useful contrast between blocks
    from .beamline_dfns import format_main_block
    format_main_block(active_sheet, BL, 7, active_sheet.row_count, master_boundaries, fmtColumns, fmtColumnsToFill)  #Format as per BL specific layout

def create_samplesheet_update_info_page(sh, BL, date, sample_name, sample_des, team):
    '''This function updates the info page with new sample info
    
    Input:
        sh - Opens the spreadsheet
        BL - Name of beamline
        date - Date in YYYY-MM-DD format
        sample_name - Name of the active sample
        sample_des - Description of the active sample
        sample_info - Additional information of the active sample
        team - Names of the users using the system during the session when a new sample is created
    '''
    
    #Write sample deatils on front sheet
    #Set active sheet to Info sheet
    active_sheet = sh.worksheet('Info') 
    
    if BL == 'St Andrews - Phoibos' or BL == 'St Andrews - MBS':
        #The end of the second column in the spreadsheet consists of the sample names
        second_column_entries = active_sheet.col_values(2)
        #Set row index as the row after the last entry in the left column
        index = len(second_column_entries)+1
        #Check enough rows exist in the spreadsheet and add more if needed
        if index > active_sheet.row_count:
            active_sheet.add_rows(10)
        #The cells that will be updated
        cell_list = active_sheet.range('B'+str(index)+':'+'E'+str(index))
        #Info to be uploaded
        sample_info = [sample_name, sample_des, team, date]
        i = 0
        #Assign cell values to the relevant project info
        for cell in cell_list:
            cell.value = sample_info[i]
            i+=1
        #Update spreadsheet cells
        active_sheet.update_cells(cell_list)
    
    else:
        #The end of the second column in the spreadsheet consists of the sample names
        second_column_entries = active_sheet.col_values(2)
        #Set row index as the row after the last entry in the second column
        index = len(second_column_entries)+1
        cell_list = active_sheet.range('B'+str(index)+':'+'C'+str(index))
        #Info to be uploaded
        sample_info = [sample_name, sample_des]
        i = 0
        #Assign cell values to the relevant project info
        for cell in cell_list:
            cell.value = sample_info[i]
            i+=1
        #Update spreadsheet cells
        active_sheet.update_cells(cell_list)

def update_system_log_with_new_sample(system, sample_name):
    '''This function updates the system usage log with a new sample entry
    
    Input:
        system - The system being used (ARPES or Spin-ARPES)
        sample_name - Name of the active sample
    '''
    
    #open google spreadsheet
    sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
    if system == 'ARPES':
        #Set active sheet to ARPES helium lamp sheet
        active_sheet = sh.worksheet('ARPES')
    elif system == 'Spin-ARPES':
        #Set active sheet to Spin-ARPES helium lamp sheet
        active_sheet = sh.worksheet('Spin-ARPES')
    #fifth column in the spreadsheet contains samples
    fifth_column_entries = active_sheet.col_values(5)
    #Set row index as the row of the last entry in the fifth column
    index = len(fifth_column_entries)
    #Get current session samples
    current_session_samples = active_sheet.acell('E'+str(index)).value
    current_session_samples_split = current_session_samples.split(", ")
    sample_listed = False
    #Go through all samples listed
    for sample in current_session_samples_split:
        #if new sample already listed
        if sample == sample_name:
            sample_listed = True
    #if sample not listed
    if sample_listed == False:
        updated_session_samples = current_session_samples + ', ' + sample_name
        active_sheet.update_acell('E'+str(index), updated_session_samples)

def automatic_logging(scan_dir, sample_name, project_lead, system, helium_lamp_state, BL, keyID, interval, session_info, info_message):
    '''This function begins ad performs automatic logging of a sample folder to the sample sheet
    
    Input:
        scan_dir - Directory for sample sample_name
        sample_name - Sample name (and name of folder being uploaded)
        project_lead - Name of the project lead
        system - The system being used (ARPES or Spin-ARPES)
        helium_lamp_state - Whether or not the helium lamp is on (true or false)
        BL - Beamline selected by the user
        keyID - Used to identify the relevant spreadsheet saved in google sheets
        interval - Time between succesive updates of the logbook during automatic logging
        session_info - Widget used to update the user about the automatic logging process
        info_message - Widget used to update the user about errors
    '''  

    #Used to exit the program during automatic logging
    def signal_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    #used to determine how long to wait until the next update
    new_interval = interval
    
    #Start automatic logging
    try:
        while (True):
            time_start = time.time()
            #Do the update to the log
            successful_result = exponential_backoff(lambda: upload_sample_folder(scan_dir, sample_name, BL, keyID, info_message), info_message)
            #API error
            if successful_result == False:
                with session_info:
                    session_info.clear_output()
                    print("Project lead: " + project_lead)
                    print("Sample: " + sample_name)
                    print("System: " + system)
                    if helium_lamp_state == True:
                        print("Helium lamp state: On")        
                    elif helium_lamp_state == False:
                        print("Helium lamp state: Off")
                    print("Previous action: Stopped automatic logging from "+sample_name+" folder\n")
                return
            with info_message:
                info_message.clear_output()            
            #Update session_info
            with session_info:
                session_info.clear_output()
                print("Project lead: " + project_lead)
                print("Sample: " + sample_name)
                print("System: " + system)
                if helium_lamp_state == True:
                    print("Helium lamp state: On")        
                elif helium_lamp_state == False:
                    print("Helium lamp state: Off")
                print("Previous action: Started automatic logging from "+sample_name+" folder")
                print('Last checked: '+ str(datetime.strftime(datetime.today(),"%H:%M:%S")))
                print("Note: The UI is not responsive during automatic logging")
                print("Interrupt the kernel to stop automatic logging and resume control of the UI\n")
            #Calculate correct time to wait to get desired refresh interval
            time_end = time.time()
            new_interval = max(0,interval-(time_end-time_start))
            #Wait to get correct interval between refresh
            time.sleep(new_interval)
            
    #When logging is stopped
    except:
        #Update session_info
        with session_info:
            session_info.clear_output()
            print("Project lead: " + project_lead)
            print("Sample: " + sample_name)
            print("System: " + system)
            if helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Stopped automatic logging from "+sample_name+" folder\n")

def upload_sample_folder(scan_dir, sample_name, BL, keyID, info_message):
    '''This function uploads a sample folder to the sample sheet
    
    Input:
        scan_dir - Directory for sample sample_name
        sample_name - Sample name (and name of folder being uploaded)
        BL - Beamline selected by the user
        keyID - Used to identify the relevant spreadsheet saved in google sheets
        info_message - Widget used to update the user about errors
    '''
    
    from .beamline_dfns import get_extension
    scan_extension = get_extension(BL)

    #Read all files from this list and sort into order
    file_list = os.listdir(scan_dir)
    file_list.sort()

    #Make list of files of relevant type only
    file_list_scans = []
    for entry in file_list:
        for scan in scan_extension:
            if entry.endswith(scan):
                file_list_scans.append(entry)
    
    #For Soleil, Fermi maps are saved in folders, this allows them to be uploaded
    if 'SOLEIL CASSIOPEE' in BL:
        for entry in file_list:
            entry_split = entry.split('.')
            #If file has no extension and is therefore a folder
            if len(entry_split) == 1:
                file_list_scans.append(entry)
    
    #open google spreadsheet
    sh = authentication(keyID)
    #Return list of existing worksheets
    ws_list = sh.worksheets()

    #If the sample sheet already exists
    if ("'"+sample_name+"'") in str(ws_list):
        #set as active sheet 
        active_sheet = sh.worksheet(sample_name)   
        #extract first column from the spreadhsheet
        values_list = active_sheet.col_values(1)
        upload_times = active_sheet.col_values(active_sheet.col_count)
        files_to_upload = []
        #Go through all entries and check if they should be uploaded
        for entry in file_list_scans:
            if 'SOLEIL CASSIOPEE' in BL and len(entry.split('.')) == 1:
                scan_no_ext = entry
            else:
                scan_no_ext = entry.split('.')[0]
            #If scan not already uploaded, add to upload list
            if scan_no_ext not in values_list:
                files_to_upload.append(entry)
            #If scan is already uploaded, add to upload list only if the file has been modified
            else:
                timestamp = str(os.path.getmtime(os.path.join(scan_dir, entry)))
                index = values_list.index(scan_no_ext)
                try:
                    upload_time = upload_times[index]
                    try:
                        float(upload_time)
                    except:
                        upload_time = 0
                except:
                    upload_time = 0
                #upload_time +1 because timestamp extracted from google sheet does not include decimal places so it potentially sees timestamp as larger than upload time when it should not
                if float(timestamp) > float(upload_time)+1:
                    files_to_upload.append(entry)
        #Extract file metadata and update logbook
        for entry in files_to_upload:
            from .beamline_dfns import make_entry_logbook
            sample_upload,timestamp = make_entry_logbook(os.path.join(scan_dir, entry),BL)
            upload_to_logbook(sample_upload,sample_name,BL,timestamp,keyID)

def upload_to_logbook(sample_upload, sample_name, BL, timestamp, keyID):
    '''This function uploads sample data to the sample sheet of the logbook
    
    Input:
        sample_upload - List of relevant metadata
        sample_name - Sample name
        BL - Beamline selected by the user
        timestamp - Timestamp of file being scanned (used to check if it has changed since last data write)
        keyID - Used to identify the relevant spreadsheet saved in google sheets
    '''
    
    #open google spreadsheet
    sh = authentication(keyID)
    #Return list of existing worksheets
    ws_list = sh.worksheets()

    #If the sample sheet already exists
    if ("'"+sample_name+"'") in str(ws_list):
        #set as active sheet 
        active_sheet = sh.worksheet(sample_name)   
        #Check to see if an entry already exists
        #extract first column from the spreadhsheet
        values_list = active_sheet.col_values(1)
        upload_time = 0
        
        if 'Diamond I05-HR' in BL or 'Diamond I05-nano' in BL:
            if sample_upload[0] in values_list:
                index = values_list.index(sample_upload[0])+1  #index is row index where entry exists (in spreadsheet row format, i.e. starting from 1)
                #Check if the file has been updated since last entry made
                upload_time = active_sheet.acell(gspread.utils.rowcol_to_a1(index,active_sheet.col_count)).value
            else:
                index = len(values_list)+1  #if it doesn't exist already, set row index as the row after the last entry in the left column
        elif 'MAX IV Bloch' in BL or 'Elettra APE' in BL or 'SOLEIL CASSIOPEE' in BL or 'St Andrews - Phoibos' in BL or 'St Andrews - MBS' in BL:
            if sample_upload[1] in values_list:
                index = values_list.index(sample_upload[1])+1  #index is row index where entry exists (in spreadsheet row format, i.e. starting from 1)
                #Check if the file has been updated since last entry made
                upload_time = active_sheet.acell(gspread.utils.rowcol_to_a1(index,active_sheet.col_count)).value
            else:
                index = len(values_list)+1  #if it doesn't exist already, set row index as the row after the last entry in the left column
        else:
            print('Beamline not supported.')
        try:
            float(upload_time)
        except:
            upload_time = 0
        #Only do the update if the file is a newer version than for the entry already made
        #upload_time +1 because timestamp extracted from google sheet does not include decimal places so it potentially sees timestamp as larger than upload time when it should not
        if float(timestamp) > float(upload_time)+1:
            #Check enough rows exist in the spreadsheet and add more if needed
            if index > active_sheet.row_count:
                active_sheet.add_rows(10)
            #Upload/update data to relevant entry
            from .beamline_dfns import write_data
            write_data(active_sheet,index,sample_upload,sample_name,BL,timestamp)

def clean_up_sample_sheet(sample_name, keyID):
    '''This function is a clean-up utility to hide alignment scans, i.e. it hides scans between rows where the scan name entry are '(-' and '-)'
    
    Input:
        sample_name - Sample name
        keyID - Used to identify the relevant spreadsheet saved in google sheets
    '''
    
    #open google spreadsheet
    sh = authentication(keyID)
    #Return list of existing worksheets
    ws_list = sh.worksheets()
    #If the sample sheet already exists
    if ("'"+sample_name+"'") in str(ws_list):
        #set active sheet 
        active_sheet = sh.worksheet(sample_name)   
        sheetId = sh.worksheet(sample_name)._properties['sheetId']
        #Extract first column from the spreadhsheet
        values_list = active_sheet.col_values(1)
        #Search for hiding characters
        hide_start = list(locate(values_list, lambda x: x == '(-'))
        hide_end = list(locate(values_list, lambda x: x == '-)'))
        N_hs = len(hide_start)
        N_he = len(hide_end)
        if N_hs == N_he or N_hs == N_he+1:
            for x in range(0,N_he):
                body = {"requests": [{"updateDimensionProperties": {"range": {"sheetId": sheetId,"dimension": 
                    "ROWS","startIndex": hide_start[x],"endIndex": hide_end[x]+1},
                    "properties": {"hiddenByUser": True, }, "fields": 'hiddenByUser'}}]}
                sh.batch_update(body)

def usage_stats(start_date, end_date):
    '''This function retreives information about usage for both systems between two dates (used for funding purposes)
    
    Input:
        start_date - day from which usage info is to be retrieved (in the form "YYYY-MM-DD")
        end_date - day up to which usage info is to be retrieved (in the form "YYYY-MM-DD")
    '''
    
    #extract the start and end date
    start_date_timestamp = time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple())
    end_date_timestamp = time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple())
    #open google spreadsheet
    sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
    systems = ['ARPES', 'Spin-ARPES']
    for system in systems:
        #Set active sheet to ARPES usage sheet
        active_sheet = sh.worksheet(system)
        #extract relevant info from usage sheet
        dates = active_sheet.col_values(1)
        grants = active_sheet.col_values(4)
        durations = active_sheet.col_values(8)
        capped_durations = active_sheet.col_values(9)
        no_of_entries = len(dates)
        start_date_found = False
        search_success = False
        i = 1
        while start_date_found == False:
            ith_date_timestamp = time.mktime(datetime.strptime(dates[i], "%Y-%m-%d").timetuple())
            if start_date_timestamp - ith_date_timestamp <= 0:
                start_date_found = True
                search_success = True
            else:
                if i < no_of_entries-1:
                    i = i + 1
                else:
                    print(str(system) + " system search failed - no entries after the entered start date\n")
                    start_date_found = True
        if search_success == True:
            start_date_row = i
            end_date_found = False
            while end_date_found == False:
                ith_date_timestamp = time.mktime(datetime.strptime(dates[i], "%Y-%m-%d").timetuple())
                if end_date_timestamp - ith_date_timestamp < 0:
                    end_date_found = True
                else:
                    if i < no_of_entries-1:
                        i = i + 1
                    else:
                        end_date_found = True
                        i = i + 1
            end_date_row = i - 1
            grants_no_duplicates = list(dict.fromkeys(grants[start_date_row:end_date_row+1]))
            if end_date_row < start_date_row:
                print(str(system) + " system search failed - no entries between the entered dates\n")
            else:
                total_capped_durations_by_grant = [0.0 for x in range(len(grants_no_duplicates))]
                total_durations_by_grant = [0.0 for x in range(len(grants_no_duplicates))]
                for grant in grants_no_duplicates:
                    i = start_date_row
                    for grant_entry in grants[start_date_row:end_date_row+1]:
                        if grant_entry == grant:
                            index = grants_no_duplicates.index(grant)
                            try:
                                capped_duration_split = capped_durations[i].split(":")
                                capped_duration_seconds = float(capped_duration_split[2]) + (float(capped_duration_split[1])*60) + (float(capped_duration_split[0])*60*60)
                                total_capped_durations_by_grant[index] = total_capped_durations_by_grant[index] + capped_duration_seconds
                                duration_split = durations[i].split(":")
                                duration_seconds = float(duration_split[2]) + (float(duration_split[1])*60) + (float(duration_split[0])*60*60)
                                total_durations_by_grant[index] = total_durations_by_grant[index] + duration_seconds
                            except IndexError:
                                #if a measurement is ongoing
                                pass
                        i = i + 1
                print(system + " usage:")
                print("     Total duration:")
                for grant in grants_no_duplicates:
                    current_total_duration = total_durations_by_grant[grants_no_duplicates.index(grant)]
                    hours, remainder  = divmod(current_total_duration, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(str("          ") + grant + str(": ") + str(int(hours)) +" hours and " + str(int(minutes)) + " minutes")
                print("     Percentage of total duration:")
                total = 0
                for duration in total_durations_by_grant:
                    total = total + duration
                for grant in grants_no_duplicates:
                    current_total_duration = total_durations_by_grant[grants_no_duplicates.index(grant)]
                    print(str("          ") + grant + str(": ") + str(round(Decimal(100*current_total_duration/total), 1)) +"%")
                print("     Capped duration:")
                for grant in grants_no_duplicates:
                    current_total_capped_duration = total_capped_durations_by_grant[grants_no_duplicates.index(grant)]
                    hours, remainder  = divmod(current_total_capped_duration, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(str("          ") + grant + str(": ") + str(int(hours)) +" hours and " + str(int(minutes)) + " minutes")
                print("     Percentage of capped duration:")
                total = 0
                for duration in total_capped_durations_by_grant:
                    total = total + duration
                for grant in grants_no_duplicates:
                    current_capped_total_duration = total_capped_durations_by_grant[grants_no_duplicates.index(grant)]
                    print(str("          ") + grant + str(": ") + str(round(Decimal(100*current_capped_total_duration/total), 1)) +"%")
                print("")


#################################
#Beamline functions
#################################

def new_beamline_entry_BL(BL, start_date, sample_name, team, local_contacts, proposal_ID, beamtime_keyID, info_message):
    '''This function adds a new entry into the beamline spreadsheet in the usage log
    
    Input:
        BL - Beamline name
        start_date - Beamtime start date
        sample_name - Initial sample name
        team - Team members names
        local_contacts - Local contacts names
        proposal_ID - Proposal ID
        beamtime_keyID - The key ID of the beamtime spreadsheet
        info_message - Widget used to update the user about errors
    
    Returns:
        beamtime_exists - Whether the beamtime has been created already (true or false)
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #open google spreadsheet
            sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
            #Set active sheet to projects sheet
            active_sheet = sh.worksheet('Beamtimes')
            #second column in the spreadsheet consists of the names of the proposal IDs
            sixth_column_entries = active_sheet.col_values(6)
            beamtime_exists = False
            for ID in sixth_column_entries:
                if ID == proposal_ID:
                    beamtime_exists = True
                    return beamtime_exists, successful_result
            #Set row index as the row after the last entry
            index = len(sixth_column_entries)+1
            #The cells that will be updated
            cell_list = active_sheet.range('A'+str(index)+':'+'G'+str(index))
            #Info to be uploaded
            beamtime_info_upload = [BL, start_date, sample_name, team, local_contacts, proposal_ID, beamtime_keyID]
            i = 0
            #Assign cell values to the relevant project info
            for cell in cell_list:
                cell.value = beamtime_info_upload[i]
                i+=1
            #Update spreadsheet cells
            active_sheet.update_cells(cell_list)           
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")
                    beamtime_exists = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                beamtime_exists = ''
                return beamtime_exists, successful_result
    return beamtime_exists, successful_result

def get_most_recent_beamtime_info_BL(info_message):
    '''This function gets the required information to continue the most recent beamtime session from the beamtimes log spreadsheet
    
    Input:
        info_message - Widget used to update the user about errors
    
    Returns:
        BL - Beamline name
        active_sample - The most recently used sample
        proposal_ID - Proposal ID
        keyID - Used to identify the relevant spreadsheet saved in google sheets
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #open google spreadsheet
            sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
            active_sheet = sh.worksheet('Beamtimes')
            #first column in the spreadsheet
            first_column_entries = active_sheet.col_values(1)            
            #Set row index as the row of the last entry in column 1
            index = len(first_column_entries)
            #extract previous beamtime information
            last_entry = active_sheet.row_values(index)            
            BL = last_entry[0]
            samples = last_entry[2]
            #get last sample entry
            samples_split = samples.split(", ")
            active_sample = samples_split[len(samples_split)-1]
            proposal_ID = last_entry[5]
            keyID = last_entry[6]
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")
                    BL = ''
                    active_sample = ''
                    proposal_ID = ''
                    keyID = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                BL = ''
                active_sample = ''
                proposal_ID = ''
                keyID = ''
                return BL, active_sample, proposal_ID, keyID, successful_result
    return BL, active_sample, proposal_ID, keyID, successful_result

def create_beamtime_frontpage_BL(beamtime_keyID, BL, start_date, team, local_contacts, proposal_ID):
    '''This function creates the frontpage of the beamtime logbook
    
    Input:
        beamtime_keyID - The key ID of the beamtime spreadsheet
        BL - Beamline name
        start_date - Beamtime start date
        team - Team members names
        local_contacts - Local contacts names
        proposal_ID - Proposal ID
    '''
    
    #Formatting definitions
    #Title cells format (pale yellow plus bold)
    fmtTitle = gsf.cellFormat(
        backgroundColor=gsf.color(255.0/255, 233.0/255, 171.0/255),
        textFormat=gsf.textFormat(bold=True)
        )
    
    #Sub title cells format (bold)
    fmtSubTitle = gsf.cellFormat(
        textFormat=gsf.textFormat(bold=True)
        )
    
    #open google spreadsheet
    sh = authentication(beamtime_keyID)
    #returns list of existing worksheets
    ws_list = sh.worksheets()
    
    #Does Info sheet already exist?
    if "'Info'" not in str(ws_list):
        #Create a new sheet with label "Info" for general beamtime info
        sh.add_worksheet(title="Info", rows="100", cols="3")
    
    #Resize columns
    sheetId = sh.worksheet('Info')._properties['sheetId']
    col_width = [120,200,400]
    x = 0
    for i in col_width:
        body = {"requests": [{"updateDimensionProperties": {"range": {"sheetId": sheetId,"dimension": 
            "COLUMNS","startIndex": x,"endIndex": x+1},
            "properties": {"pixelSize": i},"fields": "pixelSize"}}]}
        sh.batch_update(body)
        x += 1

    if "'Sheet1'" in str(ws_list):
        #Delete old blank sheet
        ws = sh.worksheet('Sheet1')
        sh.del_worksheet(ws)

    #Set active sheet to Info sheet
    active_sheet = sh.worksheet('Info')    
    
    #Write entries to frontsheet
    
    #Beamline
    active_sheet.update_acell('A1', 'Beamline')
    gsf.format_cell_range(active_sheet, 'A1', fmtTitle)
    active_sheet.update_acell('B1', BL)

    #Proposal ID
    active_sheet.update_acell('A3', 'Proposal ID')
    gsf.format_cell_range(active_sheet, 'A3', fmtTitle)
    active_sheet.update_acell('B3', proposal_ID)

    #Start date
    active_sheet.update_acell('A5', 'Start date')
    gsf.format_cell_range(active_sheet, 'A5', fmtTitle)
    active_sheet.update_acell('B5', start_date)

    #Team members
    team_list = team.split(', ')
    active_sheet.update_acell('A7', 'Team')
    gsf.format_cell_range(active_sheet, 'A7', fmtTitle)
    team_no = len(team_list)
    cell_list = active_sheet.range('B7:B'+str(team_no+6))
    x = 0
    for cell in cell_list:
        cell.value = team_list[x]
        x+=1
    active_sheet.update_cells(cell_list)

    #Local contacts
    local_contacts_list = local_contacts.split(', ')
    cell = 'A'+str(8+team_no)
    active_sheet.update_acell(cell, 'Local contacts')
    gsf.format_cell_range(active_sheet, cell, fmtTitle)
    local_con_no = len(local_contacts_list)
    cell_list = active_sheet.range('B'+str(8+team_no)+':B'+str(8+team_no+local_con_no-1))
    x = 0
    for cell in cell_list:
        cell.value = local_contacts_list[x]
        x+=1
    active_sheet.update_cells(cell_list)

    #Sample infomation
    cell_list = active_sheet.range('A'+str(9+team_no+local_con_no)+':C'+str(9+team_no+local_con_no))
    #Info to be uploaded
    cell_values = ['Sample history', 'Sample name', 'Sample description']
    i = 0
    #Assign cell values to the relevant info
    for cell in cell_list:
        cell.value = cell_values[i]
        i+=1
    #Update spreadsheet cells
    active_sheet.update_cells(cell_list)
    gsf.format_cell_range(active_sheet, 'A'+str(9+team_no+local_con_no), fmtTitle)
    gsf.format_cell_ranges(active_sheet, [('B'+str(9+team_no+local_con_no)+':C'+str(9+team_no+local_con_no), fmtSubTitle)])

def automatic_logging_BL(scan_dir, sample_name, BL, proposal_ID, keyID, interval, beamtime_info, info_message):
    '''This function begins ad performs automatic logging of a sample folder to the sample sheet
    
    Input:
        scan_dir - Directory for sample sample_name
        sample_name - Sample name (and name of folder being uploaded)
        BL - Beamline selected by the user
        proposal_ID - Proposal ID
        keyID - Used to identify the relevant spreadsheet saved in google sheets
        interval - Time between succesive updates of the logbook during automatic logging
        beamtime_info - Widget used to update the user about the automatic logging process
        info_message - Widget used to update the user about errors
    '''  

    #Used to exit the program during automatic logging
    def signal_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    #used to determine how long to wait until the next update
    new_interval = interval
    
    #Start automatic logging
    try:
        while (True):
            time_start = time.time()
            #Do the update to the log
            successful_result = exponential_backoff(lambda: upload_sample_folder(scan_dir, sample_name, BL, keyID, info_message), info_message)
            #API error
            if successful_result == False:
                with beamtime_info:
                    beamtime_info.clear_output()
                    print("Beamline: " + BL)
                    print("Proposal ID: " + proposal_ID)
                    print("Sample: " + sample_name)
                    print("Previous action: Stopped automatic logging from "+sample_name+" folder\n")
                return
            with info_message:
                info_message.clear_output()            
            #Update beamtime_info
            with beamtime_info:
                beamtime_info.clear_output()
                print("Beamline: " + BL)
                print("Proposal ID: " + proposal_ID)
                print("Sample: " + sample_name)
                print("Previous action: Started automatic logging from "+sample_name+" folder")
                print('Last checked: '+ str(datetime.strftime(datetime.today(),"%H:%M:%S")))
                print("Note: The UI is not responsive during automatic logging")
                print("Interrupt the kernel to stop automatic logging and resume control of the UI\n")
            #Calculate correct time to wait to get desired refresh interval
            time_end = time.time()
            new_interval = max(0,interval-(time_end-time_start))
            #Wait to get correct interval between refresh
            time.sleep(new_interval)
            
    #When logging is stopped
    except:
        #Update beamtime_info
        with beamtime_info:
            beamtime_info.clear_output()
            print("Beamline: " + BL)
            print("Proposal ID: " + proposal_ID)
            print("Sample: " + sample_name)
            print("Previous action: Stopped automatic logging from "+sample_name+" folder\n")

def update_beamtime_log_with_new_sample_BL(proposal_ID,sample_name):
    '''This function updates the beamtime log with a new sample entry
    
    Input:
        proposal_ID - Proposal ID
        sample_name - Name of the active sample
    '''
    
    #open google spreadsheet
    sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
    active_sheet = sh.worksheet('Beamtimes')
    #sixth column in the spreadsheet consists of the proposal IDs
    sixth_column_entries = active_sheet.col_values(6)
    #get beamtime row
    index = sixth_column_entries.index(proposal_ID) + 1
    #Get current session samples
    current_samples = active_sheet.acell('C'+str(index)).value
    current_samples_split = current_samples.split(", ")
    sample_listed = False
    #Go through all samples listed
    for sample in current_samples_split:
        #if new sample already listed
        if sample == sample_name:
            sample_listed = True
    #if sample not listed
    if sample_listed == False:
        updated_samples = current_samples + ', ' + sample_name
        active_sheet.update_acell('C'+str(index), updated_samples)
        
def get_beamtimes_info_BL(info_message):
    '''This function retrieves the previous beamtimes info from the usage spreadsheet
    
    Input:
        info_message - Widget used to update the user about errors
    
    Returns:
        beamtimes_list - List of all the projects that the main user is lead on
        proposalID_list - List of all corresponding proposal IDs
        spreadsheet_keyID_list - List of all corresponding beamtime spreadsheet key IDs
        successful_result - Whether or not the action was performed (True or False)
    '''
    
    #See exponential_backoff function for details of the try/except used here
    successful_result = False
    #n determines time to wait (2^n seconds)
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #open google spreadsheet
            sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
            #Set active sheet to projects sheet
            active_sheet = sh.worksheet('Beamtimes')
            #first column in the spreadsheet consists of the beamlines
            first_column_entries = active_sheet.col_values(1)
            #third column in the spreadsheet consists of the samples
            third_column_entries = active_sheet.col_values(3)
            #sixth column in the spreadsheet consists of the proposal IDs
            sixth_column_entries = active_sheet.col_values(6)
            #seventh column in the spreadsheet consists of the spreadsheet keyIDs
            seventh_column_entries = active_sheet.col_values(7)
            
            beamtimes_list = first_column_entries[1:]
            samples_list = third_column_entries[1:]
            proposalID_list = sixth_column_entries[1:]
            spreadsheet_keyID_list = seventh_column_entries[1:]
                
            successful_result = True
            break
        except Exception as e:
            if str(e).split(",")[0] == "{'code': 429":            
            
                if n < n_stop-1:
                    #random_number_milliseconds is a random number of milliseconds less than or equal to 1000. 
                    #This is necessary to avoid certain lock errors in some concurrent implementations. 
                    #random_number_milliseconds must be redefined after each wait.
                    random_number_milliseconds = random.random()
                    with info_message:
                        info_message.clear_output()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                else:
                    #If no success after the 256 second wait
                    with info_message:
                        info_message.clear_output()
                        print("There has been an error. The request never succeeded")  
                    beamtimes_list = ''
                    samples_list = ''
                    proposalID_list = ''
                    spreadsheet_keyID_list = ''
            else:
                with info_message:
                    info_message.clear_output()
                    print("The following error has occured:")
                    print(e)
                beamtimes_list = ''
                samples_list = ''
                proposalID_list = ''
                spreadsheet_keyID_list = ''                
                return beamtimes_list, samples_list, proposalID_list, spreadsheet_keyID_list, successful_result
    return beamtimes_list, samples_list, proposalID_list, spreadsheet_keyID_list, successful_result
