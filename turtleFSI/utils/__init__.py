from .argpar import *
from .Womersley import *

try:
    from .Probe import Probes
except ModuleNotFoundError:
    msg = 'WARNING: Could not import module "{0}" - it is required for using\n Probes, but not installed on your system. Install with "pip install {0}"'
    try:
        import cppimport
    except ModuleNotFoundError:
        print(msg.format("cppimport"))
    try:
        import mpi4py
    except ModuleNotFoundError:
        print(msg.format("mpi4py"))
