"""
FFTX
====

This Python package provides NumPy/CuPy-compatible access to the 
high performance multi-platform code generation capabilities of 
the FFTX Project (https://github.com/spiral-software/fftx).  Both
FFTX and this Python package originated under the Exascale Computing
Project, with a focus on perforance-portable FFTs and FFT-using
methods for the rapidly evolving high performance computing landscape.

Modules
-------

fft
    Compatible subset of NumPy and CuPy fft subpackages

convo
    Convolutions and convolution-like methods
"""

from . import fft
from . import convo
from . import utils

__version__ = '1.0.1'
