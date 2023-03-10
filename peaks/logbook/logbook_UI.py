#Class describing the logbook user interface
#BE 22/09/2021

#To update users, make relevant changes to the get_main_user_dropdown widget
#To update grants, make relevant changes to the get_grant_dropdown widget

#Import packages needed to be loaded within this procedure
from __future__ import print_function
import os  #For reading file lists from folder
from datetime import datetime
from IPython.display import display, clear_output
from ipywidgets.widgets import Button, Layout
import ipywidgets as widgets
from .logbook_fns import exponential_backoff #For dealing with API request bottlenecks

class logbook(object):
    
    def __init__(self, data_path):
        '''This function initiates the logbook UI'''
        
        #make sure the path is valid for use in the logbook maker code
        if data_path.endswith('/'):
            self.data_path = data_path
        else:
            self.data_path = data_path + '/'
        try:
            os.listdir(self.data_path)
        except:
            raise Exception("Invalid folder path. Note you must use '/' and not '\\'")
        
        #start UI
        self._create_widgets()
        self.start_logging()
    
    def _create_widgets(self):
        '''This function defines the widgets used in the UI'''
            
        #################################
        #Lab UI widgets
        #################################
        
        self.enter_system_button = Button(
            description = 'Start lab logging',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.start_beamline_button = Button(
            description = 'Start beamline logging',
            button_style = 'warning',
            layout = Layout(width = '352px')
        )
        
        self.continue_session_button = Button(
            description = 'Resume previous session',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.new_session_button = Button(
            description = 'Automatically complete usage logs and start new session',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.existing_project_button = Button(
            description = 'Continue existing project',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.new_project_button = Button(
            description = 'Start new project',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.get_main_user_dropdown = widgets.Dropdown(
            options = ['Please select your name', 'Anđela Živanović (az55)', 'Brendan Edwards (be23)', 'Chiara Bigi (cb407)', 'Daniel Halliday (drh7)', 'Gesa Siemann (grs6)', 'Kaycee Underwood (ku5)', 'Lewis Hart (lsh8)', 'Liam Trzaska (lt80)', 'Phil Murgatroyd (paem1)', 'Sebastian Buchberger (sb423)', 'Tommaso Antonelli (ta50)', 'Yoshiko Nanao (yn25)', 'Phil King', 'Other', 'Maintenance'],
            description = 'Project lead:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False,
        )
        
        self.use_existing_sample_checkbox = widgets.Checkbox(
            value=False,
            description='Continue measuring a previous sample?',
            disabled=False,
            indent=False,
        )
        
        self.enter_project_lead_button = Button(
            description = 'Next',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.enter_project_name_button = Button(
            description = 'Next',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.enter_existing_project_info_button = Button(
            description = 'Enter',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.get_additional_user1_input = widgets.Text(
            placeholder='Please enter name',
            description='Additional user 1:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_additional_user2_input = widgets.Text(
            placeholder='Please enter name',
            description='Additional user 2:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_project_name_input = widgets.Text(
            placeholder='Please enter project name',
            description='Project name:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_grant_dropdown = widgets.Dropdown(
            options = ['Please select the grant', 'SPA0 XERH10 (delafossites, nickelates)', 'SPA0 XCL068 (Leverhulme TMDs)', 'SPA0 YGR032 (Royal Soc.)', 'SPA0 YGR035 (HREELS)', 'SPA0 YEP256 (SARPES setup)', 'SPA0 YEP285 (strain)', 'SPA0 XRR053 (SARRF)', 'SPA0 YMR051 (CuOx)','Maintenance'],
            description = 'Project grant:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False,
        )
        
        self.get_system_dropdown = widgets.Dropdown(
            options = ['ARPES', 'Spin-ARPES'],
            description = 'System:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '174px'),
            disabled=False,
        )
        
        self.keyID_input = widgets.Text(
            placeholder='Please enter the key ID',
            description='Spreadsheet key ID:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.sname_input = widgets.Text(
            placeholder='Please enter the sample name',
            description='Sample name:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.sdes_input = widgets.Textarea(
            placeholder='Please enter the sample description',
            description='Sample description:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.sinfo_input = widgets.Textarea(
            placeholder='Please enter the sample information',
            description='Sample information:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_initial_helium_lamp_state_checkbox = widgets.Checkbox(
            value=False,
            description='Has the helium lamp already been started?',
            disabled=False,
            indent=False,
        )
        
        self.enter_new_project_info_button = Button(
            description = 'Enter',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        
        #Used to give the user feedback e.g. invalid entries or updating log
        self.info_message = widgets.Output()
        
        #Used to display the session information
        self.session_info = widgets.Output()
        
        #Used to display the project information
        self.project_info = widgets.Output()
        
        self.turn_on_helium_lamp_button = Button(
            description = 'Start helium lamp usage',
            button_style = 'success',
            layout = Layout(width = '352px')
        )
         
        self.turn_off_helium_lamp_button = Button(
            description = 'Finish helium lamp usage',
            button_style = 'danger',
            layout = Layout(width = '352px')
        )
        
        self.get_logging_interval = widgets.BoundedIntText(
            value=60,
            min=10,
            step=1,
            description='Logging interval:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '174px'),
            disabled=False
        )
        
        self.start_automatic_logging_button = Button(
            description = 'Start automatic logging',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.file_name_input = widgets.Text(
            placeholder='Enter file name',
            description='File name:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '174px'),
            disabled=False
        )
        
        self.upload_file_button = Button(
            description = 'Upload file',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.upload_folder_button = Button(
            description = 'Upload sample folder',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.start_clean_up_button = Button(
            description = 'Clean-up spreadsheet',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.change_sample_button = Button(
            description = 'New sample',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.additional_options_button = Button(
            description = 'Additional options',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.see_project_info_button = Button(
            description = 'See project information',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.save_spreadsheet_locally_button = Button(
            description = 'Save project spreadsheet locally',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.reset_ARPES_helium_lamp_log_button = Button(
            description = 'Reset ARPES helium lamp log',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.reset_spin_ARPES_helium_lamp_log_button = Button(
            description = 'Reset spin-ARPES helium lamp log',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.confirm_reset_ARPES_helium_lamp_log_button = Button(
            description = 'Confirm ARPES helium lamp log reset',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.confirm_reset_spin_ARPES_helium_lamp_log_button = Button(
            description = 'Confirm spin-ARPES helium lamp log reset',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.end_session_button = Button(
            description = 'End session',
            button_style = 'warning',
            layout = Layout(width = '352px')
        )
        
        self.return_to_main_menu_button = Button(
            description = 'Main menu',
            button_style = 'warning',
            layout = Layout(width = '352px')
        )
        
        self.create_sample_sheet_button = Button(
            description = 'Create new sample sheet',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.back_to_system_select_button = Button(
            description = 'Back to system select',
            button_style = 'warning',
            layout = Layout(width = '352px')
        )
        
        self.back_to_project_select_button = Button(
            description = 'Back to project select',
            button_style = 'warning',
            layout = Layout(width = '174px')
        )
        
        self.confirm_helium_lamp_alert_button = Button(
            description = 'HELIUM LAMP SERVICE REQUIRED! Click to continue',
            button_style = 'danger',
            layout = Layout(width = '352px')
        )
        
        #################################
        #Beamline UI widgets
        #################################
        
        self.most_recent_beamtime_button_BL = Button(
            description = 'Continue most recent beamtime',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.other_beamtime_button_BL = Button(
            description = 'Continue other beamtime',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.new_beamtime_button_BL = Button(
            description = 'Start new beamtime',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.back_to_beamline_select_button_BL = Button(
            description = 'Back to beamline select',
            button_style = 'warning',
            layout = Layout(width = '174px')
        )
        
        self.enter_new_beamline_info_button_BL = Button(
            description = 'Enter',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.enter_proposalID_button_BL = Button(
            description = 'Enter',
            button_style = 'info',
            layout = Layout(width = '174px')
        )  
        
        self.get_beamline_dropdown_BL = widgets.Dropdown(
            options = ['Please select the beamline', 'Diamond I05-HR', 'Diamond I05-nano', 'MAX IV Bloch', 'Elettra APE', 'SOLEIL CASSIOPEE'],
            description = 'Beamline:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False,
        )
        
        self.get_proposal_ID_BL = widgets.Text(
            placeholder='Please enter proposal ID',
            description='Proposal ID:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_team_member_1_BL = widgets.Text(
            placeholder='Please enter name',
            description='Team member 1:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_team_member_2_BL = widgets.Text(
            placeholder='Please enter name',
            description='Team member 2:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_team_member_3_BL = widgets.Text(
            placeholder='Please enter name',
            description='Team member 3:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_team_member_4_BL = widgets.Text(
            placeholder='Please enter name',
            description='Team member 4:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_team_member_5_BL = widgets.Text(
            placeholder='Please enter name',
            description='Team member 5:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_local_contact_1_BL = widgets.Text(
            placeholder='Please enter name',
            description='Local contact 1:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        self.get_local_contact_2_BL = widgets.Text(
            placeholder='Please enter name',
            description='Local contact 2:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '352px'),
            disabled=False
        )
        
        #Used to display the beamtime information
        self.beamtime_info = widgets.Output()
        
        self.get_logging_interval_BL = widgets.BoundedIntText(
            value=60,
            min=10,
            step=1,
            description='Logging interval:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '174px'),
            disabled=False
        )
        
        self.start_automatic_logging_button_BL = Button(
            description = 'Start automatic logging',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.file_name_input_BL = widgets.Text(
            placeholder='Enter file name',
            description='File name:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '174px'),
            disabled=False
        )
        
        self.upload_file_button_BL = Button(
            description = 'Upload file',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.upload_folder_button_BL = Button(
            description = 'Upload sample folder',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.start_clean_up_button_BL = Button(
            description = 'Clean-up spreadsheet',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.change_sample_button_BL = Button(
            description = 'New sample',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.create_sample_sheet_button_BL = Button(
            description = 'Create new sample sheet',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.return_to_main_menu_button_BL = Button(
            description = 'Main menu',
            button_style = 'warning',
            layout = Layout(width = '352px')
        )
        
        self.additional_options_button_BL = Button(
            description = 'Additional options',
            button_style = 'info',
            layout = Layout(width = '174px')
        )
        
        self.save_spreadsheet_locally_button_BL = Button(
            description = 'Save project spreadsheet locally',
            button_style = 'info',
            layout = Layout(width = '352px')
        )
        
        self.end_session_button_BL = Button(
            description = 'End session',
            button_style = 'warning',
            layout = Layout(width = '352px')
        )
        
        #################################
        #Button assignments
        #################################
        
        #Assigns what functions are called when a button is pressed
        self.enter_system_button.on_click(self.check_previous_session_state)
        self.continue_session_button.on_click(self.continue_session)
        self.new_session_button.on_click(self.auto_complete_and_start_new_session)
        self.enter_new_project_info_button.on_click(self.new_project_check_info)
        self.turn_on_helium_lamp_button.on_click(self.helium_lamp_on)
        self.turn_off_helium_lamp_button.on_click(self.helium_lamp_off)
        self.change_sample_button.on_click(self.new_sample_interface)
        self.create_sample_sheet_button.on_click(self.create_new_sample)
        self.additional_options_button.on_click(self.additional_options_interface)
        self.return_to_main_menu_button.on_click(self.main_interface)
        self.existing_project_button.on_click(self.existing_project_get_name_interface)
        self.new_project_button.on_click(self.new_project_interface)
        self.enter_project_lead_button.on_click(self.existing_project_get_project_interface)
        self.enter_project_name_button.on_click(self.existing_project_get_session_info_interface)
        self.enter_existing_project_info_button.on_click(self.existing_project_check_info)
        self.start_automatic_logging_button.on_click(self.start_automatic_logging)
        self.upload_file_button.on_click(self.start_upload_file)
        self.upload_folder_button.on_click(self.start_upload_sample_folder)
        self.start_clean_up_button.on_click(self.start_clean_up_spreadsheet)
        self.save_spreadsheet_locally_button.on_click(self.save_spreadsheet_locally)
        self.reset_ARPES_helium_lamp_log_button.on_click(self.confirm_reset_ARPES_helium_lamp_interface)
        self.reset_spin_ARPES_helium_lamp_log_button.on_click(self.confirm_reset_spin_ARPES_helium_lamp_interface)
        self.see_project_info_button.on_click(self.see_project_info_interface)
        self.confirm_reset_ARPES_helium_lamp_log_button.on_click(self.reset_ARPES_helium_lamp)
        self.confirm_reset_spin_ARPES_helium_lamp_log_button.on_click(self.reset_spin_ARPES_helium_lamp)
        self.back_to_system_select_button.on_click(self.start_logging_via_button)
        self.back_to_project_select_button.on_click(self.start_session_interface_via_button)
        self.confirm_helium_lamp_alert_button.on_click(self.main_interface)
        self.end_session_button.on_click(self.end_session)
        self.start_beamline_button.on_click(self.start_session_interface_BL)
        self.most_recent_beamtime_button_BL.on_click(self.continue_most_recent_beamtime_BL)
        self.other_beamtime_button_BL.on_click(self.continue_other_beamtime_interface_BL)
        self.enter_proposalID_button_BL.on_click(self.continue_other_beamtime_BL)
        self.new_beamtime_button_BL.on_click(self.new_beamtime_interface_BL)
        self.back_to_beamline_select_button_BL.on_click(self.start_session_interface_BL)
        self.enter_new_beamline_info_button_BL.on_click(self.new_beamtime_check_info_BL)
        self.start_automatic_logging_button_BL.on_click(self.start_automatic_logging_BL)
        self.upload_file_button_BL.on_click(self.start_upload_file_BL)
        self.upload_folder_button_BL.on_click(self.start_upload_sample_folder_BL)
        self.start_clean_up_button_BL.on_click(self.start_clean_up_spreadsheet_BL)
        self.change_sample_button_BL.on_click(self.new_sample_interface_BL)
        self.create_sample_sheet_button_BL.on_click(self.create_new_sample_BL)
        self.return_to_main_menu_button_BL.on_click(self.main_interface_BL)
        self.additional_options_button_BL.on_click(self.additional_options_interface_BL)
        self.save_spreadsheet_locally_button_BL.on_click(self.save_spreadsheet_locally_BL)
        self.end_session_button_BL.on_click(self.end_session_BL)
    
    
    #################################
    #UI functions
    #################################
    
    def start_logging(self):
        '''This function asks the user to select which system they will be using'''
        
        print("Please select which system you will be using")
        display(widgets.VBox([widgets.HBox([self.get_system_dropdown, self.enter_system_button]), self.start_beamline_button, self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
            
    def start_logging_via_button(self, button):
        '''This function calls start_logging and is callled via a button'''
        
        clear_output(wait=True)
        self.start_logging()
    
    def check_previous_session_state(self, button):
        '''This function checks if the previous system session ended correctly'''
    
        with self.info_message:
            self.info_message.clear_output()
            print("Checking that the previous " + self.get_system_dropdown.value + " session finished correctly")
        #Check previous session ended correctly
        from .logbook_fns import check_previous_session_ended_correctly
        system_log_state, helium_lamp_log_state, usage_date, usage_start_time, helium_lamp_start_time, system_users, project_name, active_sample, successful_result = check_previous_session_ended_correctly(self.get_system_dropdown.value, self.info_message)
        if system_log_state == True:
            clear_output(wait=True)
            self.start_session_interface()
        elif system_log_state == False:
            clear_output(wait=True)
            print("The previous session did not end correctly\n")
            print("Session details")
            print("Date: " + usage_date)
            print("Start time: " + usage_start_time)
            print("System users: " + system_users)
            print("Project name: " + project_name)
            print("Active sample: " + active_sample)
            if helium_lamp_log_state == False:
                print("Helium lamp state: On\n")
            elif helium_lamp_log_state == True:
                print("Helium lamp state: Off\n")
            print("Please select an option")
            display(widgets.VBox([self.continue_session_button, self.new_session_button, self.back_to_system_select_button, self.info_message]))
            with self.info_message:
                self.info_message.clear_output()
        #API error
        else:
            return
    
    def continue_session(self, button):
        '''This function resumes the previous session if it did not end correctly'''
        
        with self.info_message:
            self.info_message.clear_output()
            print("Resuming previous session")
        from .logbook_fns import check_previous_session_ended_correctly, get_project_info
        system_log_state, helium_lamp_log_state, usage_date, usage_start_time, self.helium_lamp_start_time, system_users, project_name, active_sample, successful_result = check_previous_session_ended_correctly(self.get_system_dropdown.value, self.info_message)
        #API error
        if successful_result == False:
            return
        #get project info
        project_lead, project_grant, project_keyID, successful_result = get_project_info(project_name, self.info_message)
        #API error
        if successful_result == False:
            return
        #assign session/project information
        self.get_project_name_input.value = project_name
        self.get_main_user_dropdown.value = project_lead
        self.sname_input.value = active_sample
        self.keyID_input.value = project_keyID
        self.get_grant_dropdown.value = project_grant
        self.system_start_time = datetime.strptime(usage_date + " " + usage_start_time, "%Y-%m-%d %H:%M:%S")
        if helium_lamp_log_state == True:
            self.helium_lamp_state = False
        else:
            self.helium_lamp_state = True
            self.helium_lamp_start_time = datetime.strptime(usage_date + " " + self.helium_lamp_start_time, "%Y-%m-%d %H:%M:%S")
        self.main_interface(button)
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Entered session information\n")
    
    def auto_complete_and_start_new_session(self, button):
        '''This function auto-completes the previous session log if did not end correctly and goes to new session interface'''
        
        with self.info_message:
            self.info_message.clear_output()
            print("Automatically completing the previous session usage logs")
        from .logbook_fns import auto_complete_usage_info
        #get information to complete usage logs
        helium_lamp_log_state, system_end_time, helium_lamp_duration, successful_result = auto_complete_usage_info(self.get_system_dropdown.value, self.info_message)
        #API error
        if successful_result == False:
            return
        #if helium lamp was left on
        if helium_lamp_log_state == False:
            from .logbook_fns import helium_lamp_end_entry
            #Update helium lamp usage log
            new_usage_since_reset, successful_result = helium_lamp_end_entry(self.get_system_dropdown.value, system_end_time, helium_lamp_duration, self.info_message)
            #API error
            if successful_result == False:
                return        
            new_usage_since_reset_split = new_usage_since_reset.split(":")
        system_usage_duration = str(8) + ':' + str(0) + ':' + str(0)
        from .logbook_fns import system_usage_end_entry
        #Update system usage log
        successful_result = exponential_backoff(lambda: system_usage_end_entry(self.get_system_dropdown.value, system_end_time, system_usage_duration), self.info_message)
        #API error
        if successful_result == False:
            return
        clear_output(wait=True)
        if helium_lamp_log_state == False:
            print("Automatically completed usage logs")
            print("Total helium lamp usage since previous reset: " + new_usage_since_reset_split[0] + " hours\n")
        else:
            print("Automatically completed usage logs\n")
        self.start_session_interface()
    
    def start_session_interface(self):
        '''This function lets a user select whether they want to continue with an existing project or start a new one'''
        
        print("Please select an option")
        display(widgets.VBox([self.existing_project_button, self.new_project_button, self.back_to_system_select_button]))
    
    def start_session_interface_via_button(self, button):
        '''This function calls start_session_interface and is callled via a button'''
        
        clear_output(wait=True)
        self.start_session_interface()
    
    def existing_project_get_name_interface(self, button):
        '''This function lets the user select an existing project'''
        
        clear_output(wait=True)
        print("Please select the name of the project lead")
        display(widgets.VBox([self.get_main_user_dropdown, widgets.HBox([self.back_to_project_select_button, self.enter_project_lead_button]), self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
            
    def existing_project_get_project_interface(self, button):
        '''This function lets the user select an existing project'''
        
        #If they have not selected a user
        if self.get_main_user_dropdown.value == 'Please select your name':
            with self.info_message:
                self.info_message.clear_output()
                print("Please select the project lead")
            return
        with self.info_message:
            self.info_message.clear_output()
            print("Getting the project lead's project list")
        #Get project lead's projects and corresponding information from log spreadsheet
        from .logbook_fns import get_users_projects_info
        self.project_name_list, self.project_grant_list, self.spreadsheet_keyID_list, successful_result = get_users_projects_info(self.get_main_user_dropdown.value, self.info_message)
        #API error
        if successful_result == False:
            return
        #If user has no existing projects
        if self.project_name_list == []:
            clear_output(wait=True)
            print("User has no existing projects. Please create a new project")
            display(widgets.VBox([self.new_project_button]))
        #If user has existing projects
        else:
            self.project_name_list.reverse() #Done to get the latest project listed first
            self.get_project_dropdown = widgets.Dropdown(
                options = self.project_name_list,
                description = 'Project name:',
                style = {'description_width': 'initial'},
                layout = Layout(width = '350px'),
                disabled=False,
            )
            self.project_name_list.reverse() #Return project list order to chronological
            clear_output(wait=True)
            print("Please select the name of the project")
            display(widgets.VBox([self.get_project_dropdown, self.use_existing_sample_checkbox, widgets.HBox([self.back_to_project_select_button, self.enter_project_name_button]), self.info_message]))
            with self.info_message:
                self.info_message.clear_output()
    
    def existing_project_get_session_info_interface(self, button):
        '''This function displays the inputs for a user to enter the session information for an existing project'''
        
        self.get_project_name_input.value = self.get_project_dropdown.value
        #Get project grant
        self.get_grant_dropdown.value = self.project_grant_list[self.project_name_list.index(self.get_project_dropdown.value)]
        #Get project spreadsheet key ID
        self.keyID_input.value = self.spreadsheet_keyID_list[self.project_name_list.index(self.get_project_dropdown.value)]
        #Check that the key ID is valid
        from .logbook_fns import key_ID_check
        successful_result = key_ID_check(self.keyID_input.value, self.info_message)
        #If there is an error (request limit or invalid key ID)
        if successful_result == False:
            return
        #If a previous sample is to be used
        if self.use_existing_sample_checkbox.value == False:
            clear_output(wait=True)
            print("Please enter the session information")
            display(widgets.VBox([self.get_additional_user1_input, self.get_additional_user2_input, self.sname_input, self.sdes_input, self.sinfo_input, self.get_initial_helium_lamp_state_checkbox, widgets.HBox([self.back_to_project_select_button, self.enter_existing_project_info_button]), self.info_message]))
        #If a new sample is to be used
        elif self.use_existing_sample_checkbox.value == True:
            with self.info_message:
                self.info_message.clear_output()
                print("Getting project sample list")
            #Get existing samples from spreadsheet
            from .logbook_fns import get_project_samples
            sample_list, successful_result = get_project_samples(self.keyID_input.value, self.info_message)
            #API error
            if successful_result == False:
                return
            sample_list.reverse() #Done to get the latest sample listed first
            self.get_existing_sample_dropdown = widgets.Dropdown(
                options = sample_list,
                description = 'Sample name:',
                style = {'description_width': 'initial'},
                layout = Layout(width = '350px'),
                disabled=False,
            )
            sample_list.reverse() #Return sample list order to chronological
            clear_output(wait=True)
            print("Please enter the session information")
            display(widgets.VBox([self.get_additional_user1_input, self.get_additional_user2_input, self.get_existing_sample_dropdown, self.get_initial_helium_lamp_state_checkbox, widgets.HBox([self.back_to_project_select_button, self.enter_existing_project_info_button]), self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def existing_project_check_info(self, button):
        '''This function checks the user has entered all the required information for an existing project.
        If they have, the main interface will be displayed'''
        
        if self.use_existing_sample_checkbox.value == False:
            if self.sname_input.value == '':
                with self.info_message:
                    self.info_message.clear_output()
                    print("Please fill out all the required entries")
                return
        elif self.use_existing_sample_checkbox.value == True:
            self.sname_input.value = self.get_existing_sample_dropdown.value
        with self.info_message:
            self.info_message.clear_output()
            print("Updating system log")
        #Get team member names
        team = self.get_main_user_dropdown.value
        if self.get_additional_user1_input.value != '':
            team = team + ', ' + self.get_additional_user1_input.value
        if self.get_additional_user2_input.value != '':
            team = team + ', ' + self.get_additional_user2_input.value
        #Get system start time
        self.system_start_time = datetime.today()
        from .logbook_fns import system_usage_start_entry
        #Update system usage log
        successful_result = exponential_backoff(lambda: system_usage_start_entry(self.get_system_dropdown.value, team, self.get_project_name_input.value, self.get_grant_dropdown.value, self.sname_input.value, self.system_start_time), self.info_message)
        #API error
        if successful_result == False:
            return    
        self.helium_lamp_state = self.get_initial_helium_lamp_state_checkbox.value
        #If helium lamp is on
        if self.helium_lamp_state == True:
            with self.info_message:
                self.info_message.clear_output()
                print("Updating helium lamp log")
            self.helium_lamp_start_time = self.system_start_time
            from .logbook_fns import helium_lamp_start_entry
            #Update helium lamp usage log
            successful_result = exponential_backoff(lambda: helium_lamp_start_entry(self.get_system_dropdown.value, self.get_main_user_dropdown.value, self.helium_lamp_start_time), self.info_message)
            #API error
            if successful_result == False:
                return
        #If the user is measuring a previous sample
        if self.use_existing_sample_checkbox.value == False:
            from .logbook_fns import create_samplesheet
            with self.info_message:
                self.info_message.clear_output()
                print("Creating " + self.sname_input.value + " sample sheet")
            #Create sample sheet
            if self.get_system_dropdown.value == 'ARPES':
                sample_sheet_already_created, successful_result = create_samplesheet('St Andrews - Phoibos', self.keyID_input.value, self.sname_input.value, self.sdes_input.value, self.sinfo_input.value, team, self.info_message)
            elif self.get_system_dropdown.value == 'Spin-ARPES':
                sample_sheet_already_created, successful_result = create_samplesheet('St Andrews - MBS', self.keyID_input.value, self.sname_input.value, self.sdes_input.value, self.sinfo_input.value, team, self.info_message)
            #API error
            if successful_result == False:
                return
        #Create sample folder
        from .logbook_fns import create_sample_folder
        if self.get_system_dropdown.value == 'ARPES':
            create_sample_folder('St Andrews - Phoibos', self.sname_input.value, self.data_path)
        elif self.get_system_dropdown.value == 'Spin-ARPES':
            create_sample_folder('St Andrews - MBS', self.sname_input.value, self.data_path)
        self.main_interface(button)
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Entered session information\n")
       
    def new_project_interface(self, button):
        '''This function displays the inputs for a user to enter the project information and session information for a new project'''
        
        clear_output(wait=True)
        #reset widget values
        self._create_widgets()
        print("Please enter the project information")
        display(widgets.VBox([self.get_main_user_dropdown, self.get_project_name_input, self.get_grant_dropdown, self.keyID_input]))
        print("\nPlease enter the session information")
        display(widgets.VBox([self.get_additional_user1_input, self.get_additional_user2_input, self.sname_input, self.sdes_input, self.sinfo_input, self.get_initial_helium_lamp_state_checkbox, widgets.HBox([self.back_to_project_select_button, self.enter_new_project_info_button]), self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def new_project_check_info(self, button):
        '''This function checks the user has entered all the required information for a new project.
        If they have, the new project will be logged and the main interface will be displayed'''
        
        if self.get_main_user_dropdown.value == 'Please select your name' or self.get_project_name_input.value == '' or self.get_grant_dropdown.value == 'Please select the grant' or self.sname_input.value == '' or self.keyID_input.value == '':
            with self.info_message:
                self.info_message.clear_output()
                print("Please fill out all the required entries")
            return
        #Check that the key ID is valid
        from .logbook_fns import key_ID_check
        successful_result = key_ID_check(self.keyID_input.value, self.info_message)
        #If there is an error (request limit or invalid key ID)
        if successful_result == False:
            return
        with self.info_message:
            self.info_message.clear_output()
            print("Updating project log")
        from .logbook_fns import new_project_entry
        #Update project log
        project_exists, successful_result = new_project_entry(self.get_main_user_dropdown.value, self.get_project_name_input.value, self.get_grant_dropdown.value, self.keyID_input.value, self.info_message)
        if project_exists == True:
            with self.info_message:
                self.info_message.clear_output()
                print("Project name already exists. Please use another project name")
            return    
        #API error
        if successful_result == False:
            return
        with self.info_message:
            self.info_message.clear_output()
            print("Updating system log")
        #Get team members names
        team = self.get_main_user_dropdown.value
        if self.get_additional_user1_input.value != '':
            team = team + ', ' + self.get_additional_user1_input.value
        if self.get_additional_user2_input.value != '':
            team = team + ', ' + self.get_additional_user2_input.value
        #Get system start time
        self.system_start_time = datetime.today()
        from .logbook_fns import system_usage_start_entry
        #Update system usage log
        successful_result = exponential_backoff(lambda: system_usage_start_entry(self.get_system_dropdown.value, team, self.get_project_name_input.value, self.get_grant_dropdown.value, self.sname_input.value, self.system_start_time), self.info_message)
        #API error
        if successful_result == False:
            return   
        self.helium_lamp_state = self.get_initial_helium_lamp_state_checkbox.value
        if self.helium_lamp_state == True:
            with self.info_message:
                self.info_message.clear_output()
                print("Updating helium lamp log")
            self.helium_lamp_start_time = self.system_start_time
            from .logbook_fns import helium_lamp_start_entry
            #Update helium lamp usage log
            successful_result = exponential_backoff(lambda: helium_lamp_start_entry(self.get_system_dropdown.value, self.get_main_user_dropdown.value, self.helium_lamp_start_time), self.info_message)
            #API error
            if successful_result == False:
                return
        with self.info_message:
            self.info_message.clear_output()
            print("Creating project logbook front page")
        from .logbook_fns import create_st_andrews_frontpage, create_samplesheet
        #Create project logbook frontpage
        successful_result = exponential_backoff(lambda: create_st_andrews_frontpage(self.keyID_input.value, self.get_project_name_input.value, self.get_main_user_dropdown.value, self.get_grant_dropdown.value), self.info_message)
        #API error
        if successful_result == False:
            return
        with self.info_message:
            self.info_message.clear_output()
            print("Creating " + self.sname_input.value + " sample sheet")
        #Create sample sheet
        if self.get_system_dropdown.value == 'ARPES':
            sample_sheet_already_created, successful_result = create_samplesheet('St Andrews - Phoibos', self.keyID_input.value, self.sname_input.value, self.sdes_input.value, self.sinfo_input.value, team, self.info_message)
        elif self.get_system_dropdown.value == 'Spin-ARPES':
            sample_sheet_already_created, successful_result = create_samplesheet('St Andrews - MBS', self.keyID_input.value, self.sname_input.value, self.sdes_input.value, self.sinfo_input.value, team, self.info_message)
        #API error
        if successful_result == False:
            return
        #Create sample folder
        from .logbook_fns import create_sample_folder
        if self.get_system_dropdown.value == 'ARPES':
            create_sample_folder('St Andrews - Phoibos', self.sname_input.value, self.data_path)
        elif  self.get_system_dropdown.value == 'Spin-ARPES':
            create_sample_folder('St Andrews - MBS', self.sname_input.value, self.data_path)
        self.main_interface(button)
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Entered session information\n")
    
    def main_interface(self, button):
        '''This function displays the main interface'''
        
        clear_output(wait=True)
        if self.helium_lamp_state == False:
            display(widgets.VBox([self.session_info, self.turn_on_helium_lamp_button, widgets.HBox([self.get_logging_interval, self.start_automatic_logging_button]), widgets.HBox([self.file_name_input, self.upload_file_button]), widgets.HBox([self.upload_folder_button, self.start_clean_up_button]), widgets.HBox([self.change_sample_button, self.additional_options_button]), self.end_session_button, self.info_message]))
        elif self.helium_lamp_state == True:
            display(widgets.VBox([self.session_info, self.turn_off_helium_lamp_button, widgets.HBox([self.get_logging_interval, self.start_automatic_logging_button]), widgets.HBox([self.file_name_input, self.upload_file_button]), widgets.HBox([self.upload_folder_button, self.start_clean_up_button]), widgets.HBox([self.change_sample_button, self.additional_options_button]), self.end_session_button, self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def helium_lamp_on(self, button):
        '''This function changes the helium lamp state to on, updates the interface and calls the relelvant functions'''
        
        self.helium_lamp_start_time = datetime.today()
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Starting helium lamp usage\n")
        from .logbook_fns import helium_lamp_start_entry
        #Update helium lamp usage log
        successful_result = exponential_backoff(lambda: helium_lamp_start_entry(self.get_system_dropdown.value, self.get_main_user_dropdown.value, self.helium_lamp_start_time), self.info_message)
        self.helium_lamp_state = True
        #API error
        if successful_result == False:
            return
        self.main_interface(button)
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Started helium lamp usage\n")
            
    def helium_lamp_off(self, button):
        '''This function changes the helium lamp state to off, updates the interface and calls the relelvant functions'''
        
        self.helium_lamp_end_time = datetime.today()
        #Calcuate helium lamp duration
        helium_lamp_duration_full = self.helium_lamp_end_time - self.helium_lamp_start_time
        #Get helium lamp duration in hr:min:sec format
        hours, remainder = divmod(helium_lamp_duration_full.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        helium_lamp_duration_days = str(helium_lamp_duration_full).split(" day, ")
        if len(helium_lamp_duration_days) == 2:
            hours = hours + 24*int(helium_lamp_duration_days[0])
        helium_lamp_duration_days = str(helium_lamp_duration_full).split(" days, ")
        if len(helium_lamp_duration_days) == 2:
            hours = hours + 24*int(helium_lamp_duration_days[0])
        helium_lamp_duration = str(hours) + ':' + str(minutes) + ':' + str(seconds) 
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Finishing helium lamp usage\n")
        from .logbook_fns import helium_lamp_end_entry
        #Update helium lamp usage log
        new_usage_since_reset, successful_result = helium_lamp_end_entry(self.get_system_dropdown.value, self.helium_lamp_end_time, helium_lamp_duration, self.info_message)
        #API error
        if successful_result == False:
            return
        new_usage_since_reset_split = new_usage_since_reset.split(":")
        self.helium_lamp_state = False
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Finished helium lamp usage (" + new_usage_since_reset_split[0] + " hours total usage since previous reset)\n")
        if int(new_usage_since_reset_split[0]) > 180:
            self.confirm_helium_lamp_alert_interface()
        else:
            self.main_interface(button)
    
    def confirm_helium_lamp_alert_interface(self):
        '''This function gets the user to confirm they have seen the helum lamp alert'''
        
        clear_output(wait=True)
        display(widgets.VBox([self.session_info, self.confirm_helium_lamp_alert_button, self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def start_automatic_logging(self, button):
        '''This function calls the relevant functions in logbook_fns to begin automatic logging'''
        
        #Checks if the sample folder exists
        try:
            date = datetime.today() 
            year = datetime.strftime(date,'%Y')
            year_short = datetime.strftime(date,'%y')
            month = datetime.strftime(date,'%m')
            day = datetime.strftime(date,'%d')
            #Directory for sample sample_name
            if self.get_system_dropdown.value == 'ARPES':
                scan_dir = self.data_path+year+'/'+month+'/'+year_short+month+day+'/'+self.sname_input.value+'/'+'Phoibos225/'
            elif self.get_system_dropdown.value == 'Spin-ARPES':
                scan_dir = self.data_path+year+'/'+month+'/'+year_short+month+day+'/'+self.sname_input.value+'/'+'MBS_A1/'        
            os.listdir(scan_dir)
            valid_folder = True
        except:
            valid_folder = False
            with self.session_info:
                self.session_info.clear_output()
                print("Project lead: " + self.get_main_user_dropdown.value)
                print("Sample: " + self.sname_input.value)
                print("System: " + self.get_system_dropdown.value)
                if self.helium_lamp_state == True:
                    print("Helium lamp state: On")        
                elif self.helium_lamp_state == False:
                    print("Helium lamp state: Off")
                print("Previous action: Automatic logging was not started because " + self.sname_input.value + " folder path is invalid\n")
        #If sample folder exists, upload folder (not in try in case other errors occur in upload_sample_folder)
        if valid_folder == True:
            with self.session_info:
                self.session_info.clear_output()
                print("Project lead: " + self.get_main_user_dropdown.value)
                print("Sample: " + self.sname_input.value)
                print("System: " + self.get_system_dropdown.value)
                if self.helium_lamp_state == True:
                    print("Helium lamp state: On")        
                elif self.helium_lamp_state == False:
                    print("Helium lamp state: Off")
                print("Previous action: Started automatic logging from "+self.sname_input.value+" folder\n")
            #Start automatic logging
            from .logbook_fns import automatic_logging
            if self.get_system_dropdown.value == 'ARPES':
                automatic_logging(scan_dir, self.sname_input.value, self.get_main_user_dropdown.value, self.get_system_dropdown.value, self.helium_lamp_state, 'St Andrews - Phoibos', self.keyID_input.value, self.get_logging_interval.value, self.session_info, self.info_message)
            elif self.get_system_dropdown.value == 'Spin-ARPES':
                automatic_logging(scan_dir, self.sname_input.value, self.get_main_user_dropdown.value, self.get_system_dropdown.value, self.helium_lamp_state, 'St Andrews - MBS', self.keyID_input.value, self.get_logging_interval.value, self.session_info, self.info_message)
                    
    def start_upload_file(self, button):
        '''This function calls the relevant functions in logbook_fns to begin a file upload'''
        
        #Checks if file path is valid
        try:
            date = datetime.today() 
            year = datetime.strftime(date,'%Y')
            year_short = datetime.strftime(date,'%y')
            month = datetime.strftime(date,'%m')
            day = datetime.strftime(date,'%d')
            #Directory for sample sample_name
            if self.get_system_dropdown.value == 'ARPES':
                scan_dir = self.data_path+year+'/'+month+'/'+year_short+month+day+'/'+self.sname_input.value+'/'+'Phoibos225/'
            elif self.get_system_dropdown.value == 'Spin-ARPES':
                scan_dir = self.data_path+year+'/'+month+'/'+year_short+month+day+'/'+self.sname_input.value+'/'+'MBS_A1/'
            file_list = os.listdir(scan_dir)
            valid_file = False
            for file in file_list:
                if file == self.file_name_input.value:
                    valid_file = True
            valid_folder = True
        except:
            valid_file = False
            valid_folder = False
        #If file exists, upload file (not in try in case other errors occur in upload_to_logbook)
        if valid_file == True: 
            with self.session_info:
                self.session_info.clear_output()
                print("Project lead: " + self.get_main_user_dropdown.value)
                print("Sample: " + self.sname_input.value)
                print("System: " + self.get_system_dropdown.value)
                if self.helium_lamp_state == True:
                    print("Helium lamp state: On")        
                elif self.helium_lamp_state == False:
                    print("Helium lamp state: Off")
                print("Previous action: Uploading " + self.file_name_input.value + " to " + self.sname_input.value + " sample sheet\n")
            #Extract relevant data from file and write it to spreadsheet
            from .beamline_dfns import make_entry_logbook
            from .logbook_fns import upload_to_logbook
            if self.get_system_dropdown.value == 'ARPES':
                scan_file = self.data_path+year+'/'+month+'/'+year_short+month+day+'/'+self.sname_input.value+'/'+'Phoibos225/'+self.file_name_input.value
                sample_upload,timestamp = make_entry_logbook(scan_file,'St Andrews - Phoibos')
                successful_result = exponential_backoff(lambda: upload_to_logbook(sample_upload, self.sname_input.value, 'St Andrews - Phoibos', timestamp, self.keyID_input.value), self.info_message)
            elif self.get_system_dropdown.value == 'Spin-ARPES':
                scan_file = self.data_path+year+'/'+month+'/'+year_short+month+day+'/'+self.sname_input.value+'/'+'MBS_A1/'+self.file_name_input.value
                sample_upload,timestamp = make_entry_logbook(scan_file,'St Andrews - MBS')
                successful_result = exponential_backoff(lambda: upload_to_logbook(sample_upload, self.sname_input.value, 'St Andrews - MBS', timestamp, self.keyID_input.value), self.info_message)
            #API error
            if successful_result == False:
                return
            with self.session_info:
                self.session_info.clear_output()
                print("Project lead: " + self.get_main_user_dropdown.value)
                print("Sample: " + self.sname_input.value)
                print("System: " + self.get_system_dropdown.value)
                if self.helium_lamp_state == True:
                    print("Helium lamp state: On")        
                elif self.helium_lamp_state == False:
                    print("Helium lamp state: Off")
                print("Previous action: Uploaded " + self.file_name_input.value + " to " + self.sname_input.value + " sample sheet\n")
            with self.info_message:
                self.info_message.clear_output()
        #If file does not exist
        else:
            #If sample folder exists
            if valid_folder == True:
                with self.session_info:
                    self.session_info.clear_output()
                    print("Project lead: " + self.get_main_user_dropdown.value)
                    print("Sample: " + self.sname_input.value)
                    print("System: " + self.get_system_dropdown.value)
                    if self.helium_lamp_state == True:
                        print("Helium lamp state: On")        
                    elif self.helium_lamp_state == False:
                        print("Helium lamp state: Off")
                    print("Previous action: File upload failed because " + self.file_name_input.value + " is not in the " + self.sname_input.value + " folder (remember the file extension)\n")
            #If sample folder does not exist
            else:
                with self.session_info:
                    self.session_info.clear_output()
                    print("Project lead: " + self.get_main_user_dropdown.value)
                    print("Sample: " + self.sname_input.value)
                    print("System: " + self.get_system_dropdown.value)
                    if self.helium_lamp_state == True:
                        print("Helium lamp state: On")        
                    elif self.helium_lamp_state == False:
                        print("Helium lamp state: Off")
                    print("Previous action: File upload failed because " + self.sname_input.value + " folder path is invalid\n")
    
    def start_upload_sample_folder(self, button):
        '''This function calls the relevant functions in logbook_fns to begin a folder upload'''
        
        #Checks if the sample folder exists
        try:
            date = datetime.today() 
            year = datetime.strftime(date,'%Y')
            year_short = datetime.strftime(date,'%y')
            month = datetime.strftime(date,'%m')
            day = datetime.strftime(date,'%d')
            #Directory for sample sample_name
            if self.get_system_dropdown.value == 'ARPES':
                scan_dir = self.data_path+year+'/'+month+'/'+year_short+month+day+'/'+self.sname_input.value+'/'+'Phoibos225/'
            elif self.get_system_dropdown.value == 'Spin-ARPES':
                scan_dir = self.data_path+year+'/'+month+'/'+year_short+month+day+'/'+self.sname_input.value+'/'+'MBS_A1/'
            os.listdir(scan_dir)
            valid_folder = True
        except:
            valid_folder = False
            with self.session_info:
                self.session_info.clear_output()
                print("Project lead: " + self.get_main_user_dropdown.value)
                print("Sample: " + self.sname_input.value)
                print("System: " + self.get_system_dropdown.value)
                if self.helium_lamp_state == True:
                    print("Helium lamp state: On")        
                elif self.helium_lamp_state == False:
                    print("Helium lamp state: Off")
                print("Previous action: Sample folder upload failed because " + self.sname_input.value + " folder path is invalid\n")
        #If sample folder exists, upload folder (not in try in case other errors occur in upload_sample_folder)
        if valid_folder == True:
            with self.session_info:
                self.session_info.clear_output()
                print("Project lead: " + self.get_main_user_dropdown.value)
                print("Sample: " + self.sname_input.value)
                print("System: " + self.get_system_dropdown.value)
                if self.helium_lamp_state == True:
                    print("Helium lamp state: On")        
                elif self.helium_lamp_state == False:
                    print("Helium lamp state: Off")
                print("Previous action: Uploading "+self.sname_input.value+" sample folder\n")
            from .logbook_fns import upload_sample_folder
            #Start folder upload
            if self.get_system_dropdown.value == 'ARPES':
                successful_result = exponential_backoff(lambda: upload_sample_folder(scan_dir, self.sname_input.value, 'St Andrews - Phoibos', self.keyID_input.value, self.info_message), self.info_message)
            elif self.get_system_dropdown.value == 'Spin-ARPES':
                successful_result = exponential_backoff(lambda: upload_sample_folder(scan_dir, self.sname_input.value, 'St Andrews - MBS', self.keyID_input.value, self.info_message), self.info_message)
            #API error
            if successful_result == False:
                return
            with self.session_info:
                self.session_info.clear_output()
                print("Project lead: " + self.get_main_user_dropdown.value)
                print("Sample: " + self.sname_input.value)
                print("System: " + self.get_system_dropdown.value)
                if self.helium_lamp_state == True:
                    print("Helium lamp state: On")        
                elif self.helium_lamp_state == False:
                    print("Helium lamp state: Off")
                print("Previous action: Uploaded "+self.sname_input.value+" sample folder\n")
            with self.info_message:
                self.info_message.clear_output()
            
    def start_clean_up_spreadsheet(self, button):
        '''This function calls the relevant functions in logbook_fns to begin a clean-up of the sample sheet'''
        
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Cleaning-up sample sheet\n")
        from .logbook_fns import clean_up_sample_sheet
        successful_result = exponential_backoff(lambda: clean_up_sample_sheet(self.sname_input.value, self.keyID_input.value), self.info_message)
        #API error
        if successful_result == False:
            return
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Cleaned-up sample sheet\n")
        with self.info_message:
            self.info_message.clear_output()
            
    def new_sample_interface(self, button):
        '''This function opens the new sample menu'''
        
        clear_output(wait=True)
        display(widgets.VBox([self.session_info, self.sname_input, self.sdes_input, self.sinfo_input, self.create_sample_sheet_button, self.return_to_main_menu_button, self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def create_new_sample(self, button):
        '''This function updates the active sample and calls a function to create a new sample sheet in the project logbook'''
        
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Creating " + self.sname_input.value + " sample sheet\n")  
        #Get team members names
        team = self.get_main_user_dropdown.value
        if self.get_additional_user1_input.value != '':
            team = team + ', ' + self.get_additional_user1_input.value
        if self.get_additional_user2_input.value != '':
            team = team + ', ' + self.get_additional_user2_input.value
        from .logbook_fns import create_samplesheet
        #Create sample sheet
        if self.get_system_dropdown.value == 'ARPES':
            sample_sheet_already_created, successful_result = create_samplesheet('St Andrews - Phoibos', self.keyID_input.value, self.sname_input.value, self.sdes_input.value, self.sinfo_input.value, team, self.info_message)
        elif self.get_system_dropdown.value == 'Spin-ARPES':
            sample_sheet_already_created, successful_result = create_samplesheet('St Andrews - MBS', self.keyID_input.value, self.sname_input.value, self.sdes_input.value, self.sinfo_input.value, team, self.info_message)
        #API error
        if successful_result == False:
            return    
        from .logbook_fns import update_system_log_with_new_sample
        #Update system log
        successful_result = exponential_backoff(lambda: update_system_log_with_new_sample(self.get_system_dropdown.value, self.sname_input.value), self.info_message)
        #API error
        if successful_result == False:
            return
        #Create sample folder
        from .logbook_fns import create_sample_folder
        if self.get_system_dropdown.value == 'ARPES':
            create_sample_folder('St Andrews - Phoibos', self.sname_input.value, self.data_path)
        elif  self.get_system_dropdown.value == 'Spin-ARPES':
            create_sample_folder('St Andrews - MBS', self.sname_input.value, self.data_path)
        clear_output(wait=True)
        self.main_interface(button)
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            if sample_sheet_already_created == True:
                print("Previous action: " + self.sname_input.value + " sample sheet already created. " + self.sname_input.value + " set to active sample\n")
            elif sample_sheet_already_created == False:
                print("Previous action: Created " + self.sname_input.value + " sample sheet\n")
    
    def additional_options_interface(self, button):
        '''This function opens the additional options menu'''
        
        clear_output(wait=True)
        display(widgets.VBox([self.session_info, self.see_project_info_button, self.save_spreadsheet_locally_button, self.reset_ARPES_helium_lamp_log_button, self.reset_spin_ARPES_helium_lamp_log_button, self.return_to_main_menu_button, self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def see_project_info_interface(self, button):
        '''This function displays the project info (used mainly to get project spreadsheet key ID)'''
        
        clear_output(wait=True)
        display(widgets.VBox([self.session_info, self.project_info, self.return_to_main_menu_button, self.info_message]))
        with self.project_info:
            self.project_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Project name: " + self.get_project_name_input.value)
            print("Project grant: " + self.get_grant_dropdown.value)
            print("Project spreadsheet key ID: " + self.keyID_input.value)
        with self.info_message:
            self.info_message.clear_output()
    
    def save_spreadsheet_locally(self, button):
        '''This function calls the relevant functions to save the project spreadsheet locally'''
        
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Saving project spreadsheet locally (feature not yet implemented)\n")
    
    def confirm_reset_ARPES_helium_lamp_interface(self, button):
        '''This function asks the user to confirm they want to reset the ARPES helium lamp usage'''
        
        clear_output(wait=True)
        display(widgets.VBox([self.session_info, self.confirm_reset_ARPES_helium_lamp_log_button, self.return_to_main_menu_button, self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
        
    def confirm_reset_spin_ARPES_helium_lamp_interface(self, button):
        '''This function asks the user to confirm they want to reset the spin-ARPES helium lamp usage'''
        
        clear_output(wait=True)
        display(widgets.VBox([self.session_info, self.confirm_reset_spin_ARPES_helium_lamp_log_button, self.return_to_main_menu_button, self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def reset_ARPES_helium_lamp(self, button):
        '''This function resets the ARPES helium lamp usage'''
        
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Reseting ARPES helium lamp usage log\n")
        from .logbook_fns import reset_helium_lamp_usage
        successful_result = exponential_backoff(lambda: reset_helium_lamp_usage('ARPES'), self.info_message)
        #API error
        if successful_result == False:
            return
        clear_output(wait=True)
        self.main_interface(button)
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Reset ARPES helium lamp usage log\n")
        
    def reset_spin_ARPES_helium_lamp(self, button):    
        '''This function resets the spin-ARPES helium lamp usage'''
        
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Reseting spin-ARPES helium lamp usage log\n")
        from .logbook_fns import reset_helium_lamp_usage
        successful_result = exponential_backoff(lambda: reset_helium_lamp_usage('Spin-ARPES'), self.info_message)
        #API error
        if successful_result == False:
            return
        clear_output(wait=True)
        self.main_interface(button)
        with self.session_info:
            self.session_info.clear_output()
            print("Project lead: " + self.get_main_user_dropdown.value)
            print("Sample: " + self.sname_input.value)
            print("System: " + self.get_system_dropdown.value)
            if self.helium_lamp_state == True:
                print("Helium lamp state: On")        
            elif self.helium_lamp_state == False:
                print("Helium lamp state: Off")
            print("Previous action: Reset spin-ARPES helium lamp usage log\n")
        
    def end_session(self, button):
        '''This function completes the system usage log and ends the session'''
        
        #Only end the session if the helium lamp is off
        if self.helium_lamp_state == True:
            with self.session_info:
                self.session_info.clear_output()
                print("Project lead: " + self.get_main_user_dropdown.value)
                print("Sample: " + self.sname_input.value)
                print("System: " + self.get_system_dropdown.value)
                print("Helium lamp state: On")
                print("Previous action: End session failed. Please finish helium lamp usage first\n")
        elif self.helium_lamp_state == False:
            clear_output(wait=True)
            with self.session_info:
                self.session_info.clear_output()
                print("Project lead: " + self.get_main_user_dropdown.value)
                print("Sample: " + self.sname_input.value)
                print("System: " + self.get_system_dropdown.value)
                print("Helium lamp state: Off")
                print("Previous action: Ending session\n")
            system_end_time = datetime.today()
            #Calcuate system usage duration
            system_usage_duration_full = system_end_time - self.system_start_time
            #Get system usage duration in hr:min:sec format
            hours, remainder = divmod(system_usage_duration_full.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            system_usage_duration_days = str(system_usage_duration_full).split(" day, ")
            if len(system_usage_duration_days) == 2:
                hours = hours + 24*int(system_usage_duration_days[0])
            system_usage_duration_days = str(system_usage_duration_full).split(" days, ")
            if len(system_usage_duration_days) == 2:
                hours = hours + 24*int(system_usage_duration_days[0])       
            system_usage_duration = str(hours) + ':' + str(minutes) + ':' + str(seconds)
            from .logbook_fns import system_usage_end_entry
            #Update system usage log
            successful_result = exponential_backoff(lambda: system_usage_end_entry(self.get_system_dropdown.value, system_end_time, system_usage_duration), self.info_message)
            #API error
            if successful_result == False:
                return
            clear_output(wait=True)
            print("Session ended\n")
            self.start_logging()
    
    
    #################################
    #Beamline UI functions
    #################################
    
    def start_session_interface_BL(self, button):
        '''This function lets a user select whether they want to continue with an existing beamtime or start a new one'''
        
        clear_output(wait=True)
        print("Please select an option")
        display(widgets.VBox([self.most_recent_beamtime_button_BL, self.other_beamtime_button_BL, self.new_beamtime_button_BL, self.back_to_system_select_button, self.info_message]))
    
    def continue_most_recent_beamtime_BL(self, button):
        '''This function continues the most recent beamtime session on the beamtimes log spreadsheet'''    
    
        with self.info_message:
            self.info_message.clear_output()
            print("Resuming most recent beamtime")
        from .logbook_fns import get_most_recent_beamtime_info_BL
        BL, active_sample, proposal_ID, keyID, successful_result = get_most_recent_beamtime_info_BL(self.info_message)
        #API error
        if successful_result == False:
            return
        #assign beamtime information
        self.get_beamline_dropdown_BL.value = BL
        self.sname_input.value = active_sample
        self.get_proposal_ID_BL.value = proposal_ID
        self.keyID_input.value = keyID
        #Check that the key ID is valid
        from .logbook_fns import key_ID_check
        successful_result = key_ID_check(self.keyID_input.value, self.info_message)
        #If there is an error (request limit or invalid key ID)
        if successful_result == False:
            return
        #Create sample folder
        from .logbook_fns import create_sample_folder
        create_sample_folder(self.get_beamline_dropdown_BL.value, self.sname_input.value, self.data_path)
        self.main_interface_BL(button)
        with self.beamtime_info:
            self.beamtime_info.clear_output()
            print("Beamline: " + self.get_beamline_dropdown_BL.value)
            print("Proposal ID: " + self.get_proposal_ID_BL.value)
            print("Sample: " + self.sname_input.value)
            print("Previous action: Entered session information\n")
    
    def continue_other_beamtime_interface_BL(self, button):
        '''This function allows the user to continue a beamtime session of their choice'''
        
        with self.info_message:
            self.info_message.clear_output()
            print("Getting the previous beamtimes list")
        #Get previous beamtimes and corresponding information from log spreadsheet
        from .logbook_fns import get_beamtimes_info_BL
        self.beamtimes_list, self.samples_list, self.proposalID_list, self.spreadsheet_keyID_list, successful_result = get_beamtimes_info_BL(self.info_message)
        #API error
        if successful_result == False:
            return
        self.proposalID_list.reverse() #Done to get the latest beamtime listed first
        self.get_previous_beamtime_proposalID_dropdown = widgets.Dropdown(
            options = self.proposalID_list,
            description = 'Proposal ID:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '350px'),
            disabled=False,
        )
        self.proposalID_list.reverse() #return to chronological order
        clear_output(wait=True)
        print("Please select the beamtime")
        display(widgets.VBox([self.get_previous_beamtime_proposalID_dropdown, widgets.HBox([self.back_to_beamline_select_button_BL, self.enter_proposalID_button_BL]), self.info_message]))
        with self.info_message:
            self.info_message.clear_output()

    def continue_other_beamtime_BL(self, button):
        '''This function continues the a chosen beamtime session on the beamtimes log spreadsheet'''    
    
        with self.info_message:
            self.info_message.clear_output()
            print("Resuming selected beamtime")
        index = self.proposalID_list.index(self.get_previous_beamtime_proposalID_dropdown.value)
        #assign beamtime information
        BL = self.beamtimes_list[index]
        active_sample = self.samples_list[index].split(',')[-1].split()[0]
        proposal_ID = self.proposalID_list[index]
        keyID = self.spreadsheet_keyID_list[index]

        self.get_beamline_dropdown_BL.value = BL
        self.sname_input.value = active_sample
        self.get_proposal_ID_BL.value = proposal_ID
        self.keyID_input.value = keyID
        #Check that the key ID is valid
        from .logbook_fns import key_ID_check
        successful_result = key_ID_check(self.keyID_input.value, self.info_message)
        #If there is an error (request limit or invalid key ID)
        if successful_result == False:
            return
        #Create sample folder
        from .logbook_fns import create_sample_folder
        create_sample_folder(self.get_beamline_dropdown_BL.value, self.sname_input.value, self.data_path)
        self.main_interface_BL(button)
        with self.beamtime_info:
            self.beamtime_info.clear_output()
            print("Beamline: " + self.get_beamline_dropdown_BL.value)
            print("Proposal ID: " + self.get_proposal_ID_BL.value)
            print("Sample: " + self.sname_input.value)
            print("Previous action: Entered session information\n")
    
    def new_beamtime_interface_BL(self, button):
        '''This function displays the inputs for a user to enter the beamline information when a new beamtime is created'''
        
        clear_output(wait=True)
        #reset widget values
        self._create_widgets()
        print("Please enter the beamtime information")
        display(widgets.VBox([self.get_beamline_dropdown_BL, self.get_proposal_ID_BL, self.get_team_member_1_BL, self.get_team_member_2_BL, self.get_team_member_3_BL, self.get_team_member_4_BL, self.get_team_member_5_BL, self.get_local_contact_1_BL, self.get_local_contact_2_BL, self.keyID_input]))
        print("\nPlease enter the initial sample information")
        display(widgets.VBox([self.sname_input, self.sdes_input, self.sinfo_input, widgets.HBox([self.back_to_beamline_select_button_BL, self.enter_new_beamline_info_button_BL]), self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def new_beamtime_check_info_BL(self, button):
        '''This function checks the user has entered all the required information for a new beamtime.
        If they have, the new beamtime will be logged and the main interface will be displayed'''
        
        if self.get_beamline_dropdown_BL.value == 'Please select the beamline' or self.get_proposal_ID_BL.value == '' or self.sname_input.value == '' or self.keyID_input.value == '' or self.get_team_member_1_BL.value == '' or self.get_local_contact_1_BL.value == '':
            with self.info_message:
                self.info_message.clear_output()
                print("Please fill out all the required entries")
            return
        #Check that the key ID is valid
        from .logbook_fns import key_ID_check
        successful_result = key_ID_check(self.keyID_input.value, self.info_message)
        #If there is an error (request limit or invalid key ID)
        if successful_result == False:
            return
        #Get team members
        team = self.get_team_member_1_BL.value
        if self.get_team_member_2_BL.value != '':
            team = team + ', ' + self.get_team_member_2_BL.value
        if self.get_team_member_3_BL.value != '':
            team = team + ', ' + self.get_team_member_3_BL.value
        if self.get_team_member_4_BL.value != '':
            team = team + ', ' + self.get_team_member_4_BL.value
        if self.get_team_member_5_BL.value != '':
            team = team + ', ' + self.get_team_member_5_BL.value
        #Get local contacts
        local_contacts = self.get_local_contact_1_BL.value
        if self.get_local_contact_2_BL.value != '':
            local_contacts = local_contacts + ', ' + self.get_local_contact_2_BL.value
        #Get beamline start date
        start_date_BL = datetime.strftime(datetime.today(),'%Y-%m-%d')
        with self.info_message:
            self.info_message.clear_output()
            print("Updating beamline log")
        from .logbook_fns import new_beamline_entry_BL
        #Update beamline log
        beamtime_exists, successful_result = new_beamline_entry_BL(self.get_beamline_dropdown_BL.value, start_date_BL, self.sname_input.value, team, local_contacts, self.get_proposal_ID_BL.value, self.keyID_input.value, self.info_message)
        if beamtime_exists == True:
            with self.info_message:
                self.info_message.clear_output()
                print("Beamtime already exists. Please use the continue existing beamtime option")
            return    
        #API error
        if successful_result == False:
            return
        with self.info_message:
            self.info_message.clear_output()
            print("Creating beamtime logbook front page")
        from .logbook_fns import create_beamtime_frontpage_BL, create_samplesheet
        #Create project logbook frontpage
        successful_result = exponential_backoff(lambda: create_beamtime_frontpage_BL(self.keyID_input.value, self.get_beamline_dropdown_BL.value, start_date_BL, team, local_contacts, self.get_proposal_ID_BL.value), self.info_message)
        #API error
        if successful_result == False:
            return
        with self.info_message:
            self.info_message.clear_output()
            print("Creating " + self.sname_input.value + " sample sheet")
        #Create sample sheet
        sample_sheet_already_created, successful_result = create_samplesheet(self.get_beamline_dropdown_BL.value, self.keyID_input.value, self.sname_input.value, self.sdes_input.value, self.sinfo_input.value, team, self.info_message)
        #API error
        if successful_result == False:
            return
        #Create sample folder
        from .logbook_fns import create_sample_folder
        create_sample_folder(self.get_beamline_dropdown_BL.value, self.sname_input.value, self.data_path)
        self.main_interface_BL(button)
        with self.beamtime_info:
            self.beamtime_info.clear_output()
            print("Beamline: " + self.get_beamline_dropdown_BL.value)
            print("Proposal ID: " + self.get_proposal_ID_BL.value)
            print("Sample: " + self.sname_input.value)
            print("Previous action: Entered session information\n")
            
    def main_interface_BL(self, button):
        '''This function displays the main interface for beamline logging'''
        
        clear_output(wait=True)
        display(widgets.VBox([self.beamtime_info, widgets.HBox([self.get_logging_interval_BL, self.start_automatic_logging_button_BL]), widgets.HBox([self.file_name_input_BL, self.upload_file_button_BL]), widgets.HBox([self.upload_folder_button_BL, self.start_clean_up_button_BL]), widgets.HBox([self.change_sample_button_BL, self.additional_options_button_BL]), self.end_session_button_BL, self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def start_automatic_logging_BL(self, button):
        '''This function calls the relevant functions in logbook_fns to begin automatic logging for beamtimes'''
        
        #Checks if the sample folder exists
        #Directory for sample sample_name
        date = datetime.today() 
        year = datetime.strftime(date,'%Y')
        month_word = datetime.strftime(date,'%B')
        scan_dir = self.data_path+'Data/Synchrotron/'+self.get_beamline_dropdown_BL.value+'/'+year+'/'+month_word+'/'+self.sname_input.value+'/'
        try:
            os.listdir(scan_dir)
            valid_folder = True
        except:
            valid_folder = False
            with self.beamtime_info:
                self.beamtime_info.clear_output()
                print("Beamline: " + self.get_beamline_dropdown_BL.value)
                print("Proposal ID: " + self.get_proposal_ID_BL.value)
                print("Sample: " + self.sname_input.value)
                print("Previous action: Automatic logging was not started because " + self.sname_input.value + " folder path is invalid\n")
        #If sample folder exists, upload folder (not in try in case other errors occur in upload_sample_folder)
        if valid_folder == True:
            with self.beamtime_info:
                self.beamtime_info.clear_output()
                print("Beamline: " + self.get_beamline_dropdown_BL.value)
                print("Proposal ID: " + self.get_proposal_ID_BL.value)
                print("Sample: " + self.sname_input.value)
                print("Previous action: Started automatic logging from "+self.sname_input.value+" folder\n")
            #Start automatic logging
            from .logbook_fns import automatic_logging_BL
            automatic_logging_BL(scan_dir, self.sname_input.value, self.get_beamline_dropdown_BL.value, self.get_proposal_ID_BL.value, self.keyID_input.value, self.get_logging_interval_BL.value, self.beamtime_info, self.info_message)
            
    def start_upload_file_BL(self, button):
        '''This function calls the relevant functions in logbook_fns to begin a file upload for beamtimes'''
        
        #Checks if file path is valid
        #Directory for sample sample_name
        date = datetime.today() 
        year = datetime.strftime(date,'%Y')
        month_word = datetime.strftime(date,'%B')
        scan_dir = self.data_path+'Data/Synchrotron/'+self.get_beamline_dropdown_BL.value+'/'+year+'/'+month_word+'/'+self.sname_input.value+'/'
        try:
            file_list = os.listdir(scan_dir)
            valid_file = False
            for file in file_list:
                if file == self.file_name_input_BL.value:
                    valid_file = True
            valid_folder = True
        except:
            valid_file = False
            valid_folder = False
        #If file exists, upload file (not in try in case other errors occur in upload_to_logbook)
        if valid_file == True: 
            with self.beamtime_info:
                self.beamtime_info.clear_output()
                print("Beamline: " + self.get_beamline_dropdown_BL.value)
                print("Proposal ID: " + self.get_proposal_ID_BL.value)
                print("Sample: " + self.sname_input.value)
                print("Previous action: Uploading " + self.file_name_input_BL.value + " to " + self.sname_input.value + " sample sheet\n")
            #Extract relevant data from file
            from .beamline_dfns import make_entry_logbook
            scan_file = self.data_path+'Data/Synchrotron/'+self.get_beamline_dropdown_BL.value+'/'+year+'/'+month_word+'/'+self.sname_input.value+'/'+self.file_name_input_BL.value
            sample_upload,timestamp = make_entry_logbook(scan_file, self.get_beamline_dropdown_BL.value)
            #Write it to spreadsheet
            from .logbook_fns import upload_to_logbook
            successful_result = exponential_backoff(lambda: upload_to_logbook(sample_upload, self.sname_input.value, self.get_beamline_dropdown_BL.value, timestamp, self.keyID_input.value), self.info_message)
            #API error
            if successful_result == False:
                return
            with self.beamtime_info:
                self.beamtime_info.clear_output()
                print("Beamline: " + self.get_beamline_dropdown_BL.value)
                print("Proposal ID: " + self.get_proposal_ID_BL.value)
                print("Sample: " + self.sname_input.value)
                print("Previous action: Uploaded " + self.file_name_input_BL.value + " to " + self.sname_input.value + " sample sheet\n")
            with self.info_message:
                self.info_message.clear_output()
        #If file does not exist
        else:
            #If sample folder exists
            if valid_folder == True:
                with self.beamtime_info:
                    self.beamtime_info.clear_output()
                    print("Beamline: " + self.get_beamline_dropdown_BL.value)
                    print("Proposal ID: " + self.get_proposal_ID_BL.value)
                    print("Sample: " + self.sname_input.value)
                    print("Previous action: File upload failed because " + self.file_name_input_BL.value + " is not in the " + self.sname_input.value + " folder (remember the file extension)\n")
            #If sample folder does not exist
            else:
                with self.beamtime_info:
                    self.beamtime_info.clear_output()
                    print("Beamline: " + self.get_beamline_dropdown_BL.value)
                    print("Proposal ID: " + self.get_proposal_ID_BL.value)
                    print("Sample: " + self.sname_input.value)
                    print("Previous action: File upload failed because " + self.sname_input.value + " folder path is invalid\n")
    
    def start_upload_sample_folder_BL(self, button):
        '''This function calls the relevant functions in logbook_fns to begin a folder upload for beamtimes'''
        
        #Checks if the sample folder exists
        #Directory for sample sample_name
        date = datetime.today() 
        year = datetime.strftime(date,'%Y')
        month_word = datetime.strftime(date,'%B')
        scan_dir = self.data_path+'Data/Synchrotron/'+self.get_beamline_dropdown_BL.value+'/'+year+'/'+month_word+'/'+self.sname_input.value+'/'
        try:
            os.listdir(scan_dir)
            valid_folder = True
        except:
            valid_folder = False
            with self.beamtime_info:
                self.beamtime_info.clear_output()
                print("Beamline: " + self.get_beamline_dropdown_BL.value)
                print("Proposal ID: " + self.get_proposal_ID_BL.value)
                print("Sample: " + self.sname_input.value)
                print("Previous action: Sample folder upload failed because " + self.sname_input.value + " folder path is invalid\n")
        #If sample folder exists, upload folder (not in try in case other errors occur in upload_sample_folder)
        if valid_folder == True:
            with self.beamtime_info:
                self.beamtime_info.clear_output()
                print("Beamline: " + self.get_beamline_dropdown_BL.value)
                print("Proposal ID: " + self.get_proposal_ID_BL.value)
                print("Sample: " + self.sname_input.value)
                print("Previous action: Uploading "+self.sname_input.value+" sample folder\n")
            from .logbook_fns import upload_sample_folder
            #Start folder upload
            successful_result = exponential_backoff(lambda: upload_sample_folder(scan_dir, self.sname_input.value, self.get_beamline_dropdown_BL.value, self.keyID_input.value, self.info_message), self.info_message)
            #API error
            if successful_result == False:
                return
            with self.beamtime_info:
                self.beamtime_info.clear_output()
                print("Beamline: " + self.get_beamline_dropdown_BL.value)
                print("Proposal ID: " + self.get_proposal_ID_BL.value)
                print("Sample: " + self.sname_input.value)
                print("Previous action: Uploaded "+self.sname_input.value+" sample folder\n")
            with self.info_message:
                self.info_message.clear_output()
            
    def start_clean_up_spreadsheet_BL(self, button):
        '''This function calls the relevant functions in logbook_fns to begin a clean-up of the sample sheet for beamtimes'''
        
        with self.beamtime_info:
            self.beamtime_info.clear_output()
            print("Beamline: " + self.get_beamline_dropdown_BL.value)
            print("Proposal ID: " + self.get_proposal_ID_BL.value)
            print("Sample: " + self.sname_input.value)
            print("Previous action: Cleaning-up sample sheet\n")
        from .logbook_fns import clean_up_sample_sheet
        successful_result = exponential_backoff(lambda: clean_up_sample_sheet(self.sname_input.value, self.keyID_input.value), self.info_message)
        #API error
        if successful_result == False:
            return
        with self.beamtime_info:
            self.beamtime_info.clear_output()
            print("Beamline: " + self.get_beamline_dropdown_BL.value)
            print("Proposal ID: " + self.get_proposal_ID_BL.value)
            print("Sample: " + self.sname_input.value)
            print("Previous action: Cleaned-up sample sheet\n")
        with self.info_message:
            self.info_message.clear_output()
    
    def new_sample_interface_BL(self, button):
        '''This function opens the new sample menu for beamtimes'''
        
        clear_output(wait=True)
        display(widgets.VBox([self.beamtime_info, self.sname_input, self.sdes_input, self.sinfo_input, self.create_sample_sheet_button_BL, self.return_to_main_menu_button_BL, self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def create_new_sample_BL(self, button):
        '''This function updates the active sample and calls a function to create a new sample sheet in the beamtime logbook'''
        
        with self.beamtime_info:
            self.beamtime_info.clear_output()
            print("Beamline: " + self.get_beamline_dropdown_BL.value)
            print("Proposal ID: " + self.get_proposal_ID_BL.value)
            print("Sample: " + self.sname_input.value)
            print("Previous action: Creating " + self.sname_input.value + " sample sheet\n")  
        from .logbook_fns import create_samplesheet
        #Create sample sheet
        sample_sheet_already_created, successful_result = create_samplesheet(self.get_beamline_dropdown_BL.value, self.keyID_input.value, self.sname_input.value, self.sdes_input.value, self.sinfo_input.value, '', self.info_message)
        #API error
        if successful_result == False:
            return    
        from .logbook_fns import update_beamtime_log_with_new_sample_BL
        #Update system log
        successful_result = exponential_backoff(lambda: update_beamtime_log_with_new_sample_BL(self.get_proposal_ID_BL.value, self.sname_input.value), self.info_message)
        #API error
        if successful_result == False:
            return
         #Create sample folder
        from .logbook_fns import create_sample_folder
        create_sample_folder(self.get_beamline_dropdown_BL.value, self.sname_input.value, self.data_path)
        clear_output(wait=True)
        self.main_interface_BL(button)
        with self.beamtime_info:
            self.beamtime_info.clear_output()
            print("Beamline: " + self.get_beamline_dropdown_BL.value)
            print("Proposal ID: " + self.get_proposal_ID_BL.value)
            print("Sample: " + self.sname_input.value)
            if sample_sheet_already_created == True:
                print("Previous action: " + self.sname_input.value + " sample sheet already created. " + self.sname_input.value + " set to active sample\n")
            elif sample_sheet_already_created == False:
                print("Previous action: Created " + self.sname_input.value + " sample sheet\n")
    
    def additional_options_interface_BL(self, button):
        '''This function opens the additional options menu for beamtimes'''
        
        clear_output(wait=True)
        display(widgets.VBox([self.beamtime_info, self.save_spreadsheet_locally_button_BL, self.return_to_main_menu_button_BL, self.info_message]))
        with self.info_message:
            self.info_message.clear_output()
    
    def save_spreadsheet_locally_BL(self, button):
        '''This function calls the relevant functions to save the beamtime spreadsheet locally'''
        
        with self.beamtime_info:
            self.beamtime_info.clear_output()
            print("Beamline: " + self.get_beamline_dropdown_BL.value)
            print("Proposal ID: " + self.get_proposal_ID_BL.value)
            print("Sample: " + self.sname_input.value)
            print("Previous action: Saving project spreadsheet locally (feature not yet implemented)\n")
    
    def end_session_BL(self, button):
        '''This function completes the system usage log and ends the session'''
        
        clear_output(wait=True)
        print("Session ended\n")
        self.start_logging()
