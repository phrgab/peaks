import numpy as np
import socket
import struct
import logging
from time import sleep

# Start logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.DEBUG, datefmt = "%H:%M:%S")

logging.info('Starting server...')
port = 9001

# Start server from python side
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    # Allow reusing socket, seems to be needed to cope with port not being feed up properly
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', port))
    server.listen(1)

    # Start listening
    while True:
        conn, addr = server.accept()
        cmnd = conn.recv(1024)
        logging.debug('Message recieved on server: %s', cmnd)

        if 'MOTOR' in str(cmnd):
            # Protocol is MOTOR_axis_value for setting or MOTOR_axis for query
            temp = str(cmnd).split('_')
            ax = temp[1]
            if len(temp) == 2:  # Query for position only
                # simulate fetching position
                val = np.random.rand()

                # Set return string
                ret = bytes('current_' + ax + '_' + str(val), 'utf-8')
            else:
                # simulate setting position
                sleep(0.1)

                # simulate fetching position
                val = temp[2]

                # Set return string
                ret = bytes('set_' + ax + '_' + str(val), 'utf-8')

            # Return
            conn.sendall(ret)