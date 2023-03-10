# Always use scan_setup(scan) as function name with scan. syntax
import logging
import time

def scan_setup(scan):
    st = time.time()

    # Run a scan
    scan.scan('x',0, 10, 1, 'y', 0, 10, 1)

    # Readback current position of 'x'
    scan.pos('x')

    # Set current position of 'x'
    scan.pos('x', 1)

    # Run a scan
    scan.staticscan()

    # Readback current position of y
    scan.pos('y')

