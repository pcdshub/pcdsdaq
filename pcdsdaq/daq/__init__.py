"""
Module for controlling all of the LCLS photon-side data acquisition systems.

The actual raw control API for these systems is supplied and maintained by the
DAQ group, this library provides consistent interfacing for user operations
and scanning that are beyond the scope of the DAQ itself, which is focused
on providing high-performance data acquisition and related tooling.

This top-level __init__ for pcdsdaq.daq contains the user Daq classes that
correspond to each of the existing Daq versions, as well as the ``get_daq``
helper for locating the Daq singleton.

Here are the available classes:
- ``Daq`` is the original lcls1-compatible interface.
- ``DaqLCLS2`` is the latest lcls2-compatible interface.
- ``DaqLCLS1`` is not yet ready, but it is intended to be a new
  lcls1-compatible interface for maximum parity between lcls1 and lcls2.
  It will be importable once it is ready for testing.
"""
from .interface import get_daq
# from .lcls1 import DaqLCLS1  # DaqLCLS1 is not yet ready
from .lcls2 import DaqLCLS2
from .original import Daq  # Backcompat, will be deprecated

__all__ = [
    "get_daq",
    "Daq",
    # "DaqLCLS1",
    "DaqLCLS2",
]
