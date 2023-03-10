import concurrent.futures
import numpy as np
import socket
import struct
import threading
import logging
import time

'''
Module to interface with MBS A1 Soft external scan routine.

Running mode:

First make sure that the server on the motor side is running...

Run python code (run this file, and then run the function run(file) where file is the script to be run
 - load script to be run
 - calculate number of steps
 - start python server, listening for messages from A1 soft
 - when that is up and running write message that A1 soft can be started via 'ExternalStart'

ExternalStart pressed in A1 soft (can we automate that?):
 - OpenUser1 flag set=T
 - Labview connects to server and says 'INIT'
 - Python replies with the calculated number of steps
 - NoImages global set in labview
 - StartMapFlag set=T in labview
 - StartRegion executes in labview

Start Region starts:
 - CROF set=T when analyser ready
 - When CROF set=T, Labview connects to server and says 'MOVE'
 - Python starts the scan code, goes through first move and gets to first staticscan()
 - Python replies on server to say ready to measure and sends back metadata
 - Labview CROF set=F and metadata passed to CustomerHeadStr

First scan:
 - Scan occurs, labview CROF set=T
 - Labview connects to server and says 'MOVE'
 - Python gets out of staticscan code and goes through the next step
 - Goes through code to next staticscan(), replies with metadata and labview CROF set=T
 - etc.

After last scan:
 - RDRF flag set=F in labview
 - Labview sends 'DONE' to python
 - ready_to_move flag set in python to allow getting out of the final staticscan() loop 
    and to complete any final commands from script
 - Scan complete printed to screen

'''

class scans:

    # create shared ready flags as threading Events
    ready_to_move = threading.Event()
    ready_to_measure = threading.Event()
    scan_start = threading.Event()
    scan_done = threading.Event()

    # Create some class variables
    num_scans = 0
    meta = b'test_meta'
    scan_no = 0
    start_time = None

    def __init__(self):
        pass

    def pos(self, id, val=None):
        '''
        Interface to motor server

        Parameters
        ----------
        id : str
            identifier of axis or other hardware parameter to set

        val : float, optional
            value to set
            Deafult is None -- readback only

        Returns
        -------
        float

        '''

        logging.debug('%s', id)
        logging.debug('%s', val)
        # Set send string in correct format
        if val is not None:
            to_send = bytes('MOTOR_' + id + '_' + str(val), 'utf-8')
        else:
            to_send = bytes('MOTOR_' + id, 'utf-8')

        logging.debug('To send: %s', to_send)

        # Set up client and connect
        port = 9001
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', port))

            s.sendall(to_send)
            data = s.recv(1024)
            logging.debug(f"Received {data!r}")

            temp = str(data).split('_')

            if 'set' in temp[0]:
                logging.info('Set position, axis %s: %s', temp[1], temp[2])
            else:
                logging.info('Current position, axis %s: %s', temp[1], temp[2])


    def scan(self, ax1, ax1_start, ax1_stop, ax1_step, ax2=None, ax2_start=None, ax2_stop=None, ax2_step=None, anchors=None, dummy=False):
        '''
        Define an axis scan with a secondary following axis scan or full nested loop

        Parameters
        ----------
        ax1 : str
            identifier of primary axis or slow-axis for a nested loop

        ax1_start : float
            starting value of ax1

        ax1_stop : float
            ending value of ax1

        ax1_step : float
            step size for ax1

        ax2 : str
            identifier of secondary axis or fast axis for a nested loop

        ax2_start : float
            starting value of ax2

        ax2_stop : float
            ending value of ax2

        ax2_step : float
            step size for ax2

        anchors : dict
            anchor points in the form of nested dictionaries
            e.g. {'ax2': {-10: 0, 10:1}, {'ax3: {-10: 0.2, 10: 3}}
            Only works for calls without an ax2 spec

        dummy : bool
            set True to return number of scans that will be run
            rather than actually running them

        Returns
        -------

        '''

        following_ax = []

        # Define primary loop
        ax1_npts = int(np.floor((ax1_stop-ax1_start)/ax1_step))+1
        ax1_stop_floor = ax1_start + ((ax1_npts-1) * ax1_step)
        ax1_vals = np.linspace(start=ax1_start, stop=ax1_stop_floor, num=ax1_npts)

        # Check for secondary loop specified via ax2 call
        if ax2:
            if ax2_step:  # Nested loop
                ax2_npts = int(np.floor((ax2_stop - ax2_start) / ax2_step)) + 1
                ax2_stop_floor = ax2_start + ((ax2_npts - 1) * ax2_step)
                ax2_vals = np.linspace(start=ax2_start, stop=ax2_stop_floor, num=ax2_npts)
            else:  # Then a following axis
                following_ax.append(ax2)
                following = np.zeros((ax1_npts, 1))
                following[:,0] = np.linspace(start=ax2_start, stop=ax2_stop, num=ax1_npts)

        # Check for additional following axis calls (ignore if a nested loop)
        if not ax2:
            if anchors:
                n_following = len(anchors)  # Number of following axes
                following = np.zeros((ax1_npts,n_following))
                for ct, i in enumerate(anchors):
                    following_ax.append(i)
                    # Work out parameters of anchor line
                    x_pts = []
                    y_pts = []
                    for j in anchors[i]:
                        x_pts.append(j)
                        y_pts.append(anchors[i][j])
                    m = (y_pts[1]-y_pts[0])/(x_pts[1]-x_pts[0])
                    c = y_pts[0] - (m*x_pts[0])
                    following[:,ct] = (m*ax1_vals + c).T

        # Run the scan
        if not dummy:
            if ax2_step:  # Nested loop
                for i in ax1_vals:
                    self.pos(ax1, i)  # Set the slow axis position
                    for j in ax2_vals:
                        self.pos(ax2, j)  # Set the fast axis position

                        self.staticscan()  # Run the scan

            elif following_ax:  # Single loop with following axes
                for ct, i in enumerate(ax1_vals):
                    self.pos(ax1, i)  # Set the primary axis position
                    for j in range(len(following_ax)):
                        self.pos(following_ax[j], following[ct,j])  # Set the following axis position

                    self.staticscan()  # Run the scan

            else:  # Single loop, no following axis
                for i in ax1_vals:
                    self.pos(ax1, i)  # Set the axis position
                    self.staticscan()  # Run the scan
        else:
            if ax2_step:
                return ax1_npts * ax2_npts
            else:
                return ax1_npts

    def staticscan(self):
        logging.debug('Staticscan triggered')
        self.ready_to_move.clear()  # No longer ready to move

        # Get/set metadata
        self.meta = 'scan number: '+str(self.scan_no)
        logging.debug('Meta set: %s', self.meta)

        logging.info('Ready to measure')
        self.ready_to_measure.set()  # Set ready_to_measure flag true

        # Wait for scan to be completed:
        self.ready_to_move.wait()

    def run_scan(self, file):
        logging.debug('Scan file waiting for start command')
        # Wait for scan start flag
        self.scan_start.wait()
        logging.debug('Scan file has start command')

        # Start the scan script
        execfile(file)
        scan_setup(self)

        # Indicate scan has finished
        self.scan_done.set()

    # Python server
    def server(self):
        logging.info('Starting server...')
        port = 9000

        # Start server from python side
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.server:
            # Allow reusing socket, seems to be needed to cope with port not being feed up properly
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(('localhost', port))
            self.server.listen(1)

            # Now server set up scan can be started
            print('Start scan in A1 soft via \'ExternalStart\'')

            # Start listening
            while True:
                conn, addr = self.server.accept()
                cmnd = conn.recv(4)  # The default size of the command packet is 4 bytes
                logging.debug('Message recieved on server: %s', cmnd)

                if 'INIT' in str(cmnd):
                    # Return number of scans
                    val = struct.pack('!i', self.num_scans)
                    conn.sendall(val)
                elif 'MOVE' in str(cmnd):
                    logging.debug('Ready to move')
                    if self.scan_no == 0:  # First scan
                        logging.info('------Scan started------')
                        self.start_time = time.time()
                        self.scan_start.set()  # Trigger first scan to run
                    else:
                        self.ready_to_move.set()  # Set ready to move flag

                    # Wait for measure ready flag
                    self.ready_to_measure.wait()

                    # Scan number
                    logging.debug('Scan Number: %s', self.scan_no)

                    # Pack up meta data to send back
                    meta_send = bytes(self.meta, 'utf-8')

                    # Send back metadata
                    conn.sendall(meta_send)

                    # Increment scan number
                    self.scan_no += 1

                    # Lower ready to measure flag
                    self.ready_to_measure.clear()

                elif 'DONE' in str(cmnd):
                    # Scanning finished
                    self.ready_to_move.set()  # Set this to allow final staticscan function  to complete loop

                    # Wait for any last parts of the script (e.g. final moves) to complete
                    self.scan_done.wait()

                    # Reply DONE
                    conn.sendall(b'DONE')

                    logging.info('------Scan complete------')
                    logging.info('Scan duration: %s seconds', round(time.time()-self.start_time, 1))
                    break



def run(file):

    # Determine number of scans in file to be run
    with open(file, 'r') as f:
        scan_contents = f.readlines()

    # Create instance of scans class
    scan = scans()

    # Ensure flags are all cleared at start
    scan.ready_to_move.clear()
    scan.ready_to_measure.clear()
    scan.scan_start.clear()
    scan.scan_done.clear()

    for i in scan_contents:
        if not i.startswith('#') and 'staticscan' in i:
            scan.num_scans += 1
        elif not i.startswith('#') and 'scan(' in i:
            scan_input = 'scan.' + i.split('.', 1)[1].split(')')[0] + ', dummy=True)'
            scan.num_scans += eval(scan_input)

    logging.info('%s scans to be acquired', scan.num_scans)

    # Run processes on different threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(scan.server)  # Launch server in listening mode on one thread
        executor.submit(scan.run_scan, file)  # Run the scan file


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt = "%H:%M:%S")





