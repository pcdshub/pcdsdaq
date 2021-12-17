"""
Module for controlling all of the LCLS photon-side data acquisition systems.

The actual raw control API for these systems is supplied and maintained by the
DAQ group, this library provides consistent interfacing for user operations
and scanning that are beyond the scope of the DAQ itself, which is focused
on providing high-performance data acquisition and related tooling.

This top-level __init__ for pcdsdaq.daq contains the user Daq classes that
correspond to each of the existing Daq versions, as well as the ``get_daq``
helper for locating the Daq singleton.

``DaqLCLS1`` is aliased here as ``Daq`` for backcompatibility with previous
versions of the library, e.g. the following should still work:

.. code-block:: python

   # Old code is expecting this to be LCLS1
   from pcdsdaq.daq import Daq
"""
# flake8: noqa
from .interface import DaqError, DaqTimeoutError, StateTransitionError, get_daq
from .lcls1 import DaqLCLS1
from .lcls2 import DaqLCLS2
from .original import Daq  # Backcompat, will be deprecated
