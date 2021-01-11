Scan PVs
========
.. currentmodule:: pcdsdaq.scan_vars

When using the DAQ, it is convenient to keep track of the state of the scan
in EPICS to set up nice run tables. This is implemented as the `ScanVars`
class.

In this document I will use "scan vars" and "scan pvs" interchangably.

Usage
#####
Once instantiated, you can use `ScanVars.enable` and
`ScanVars.disable` to control whether or not we will update the PVs
during a scan.

The class is instantiated as
``scan_vars = ScanVars('XPP:SCAN', name='scan_vars', RE=RE)``,
replacing ``XPP`` with the correct hutch.

PVs
###
=================== ===========================================
PV Suffix           Function
ISTEP               Current scan step, 0-indexed by default
ISSCAN              1 if we are doing a scan, 0 otherwise
SCANVAR0{0,1,2}     The name of our {1st, 2nd, 3rd} positioner
MAX0{0,1,2}         The max scan position of each positioner
MIN0{0,1,2}         The min scan position of each positioner
NSTEPS              The total number of steps in the scan
NSHOTS              Number of events per point in the DAQ
=================== ===========================================

API
###
The `ScanVars` class is an ``ophyd`` ``Device`` with special ``RunEngine``
subscription mechanisms for updating the scan PVs. It reads plan metadata
and daq configuration to update the PVs.

.. autoclass:: ScanVars
   :members:
   :member-order: bysource
