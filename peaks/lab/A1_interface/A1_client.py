import socket
import struct
import logging
from time import sleep

port = 9000
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.DEBUG, datefmt = "%H:%M:%S")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(('localhost', port))


    s.sendall(b"INIT")
    data = s.recv(1024)
    logging.debug(f"Received {data!r}")

num_scans = struct.unpack('!i', data)[0]
logging.debug('%s scans', num_scans)

for i in range(num_scans):
    logging.debug('Scan No. %s', str(i+1))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', port))
        s.sendall(b"MOVE")
        data = s.recv(1024)
        logging.debug(f"Received {data!r}")
        sleep(0.1)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(('localhost', port))
    s.sendall(b"DONE")
    data = s.recv(1024)
    logging.debug(f"Received {data!r}")

