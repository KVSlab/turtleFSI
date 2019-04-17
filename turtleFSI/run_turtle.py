# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Entry point for the setup.py. This small wrapper function makes it possible to run
turtleFSI from any location. The idea for this file came from github.com/mikaem/Oasis
"""

import sys, os

sys.path.append(os.getcwd())

def main():
    from turtleFSI import monolithic

if __name__ == '__main__':
    main()
