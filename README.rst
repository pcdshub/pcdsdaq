=======
pcdsdaq
=======
.. image:: https://travis-ci.org/pcdshub/pcdsdaq.svg?branch=master
   :target: https://travis-ci.org/pcdshub/pcdsdaq
   :alt: Build Status

The ``pcdsdaq`` module provides a higher-level interface for user control of the
LCLS1 DAQ through the LCLS1 DAQ's Python C libraries, ``pydaq``, ``pycdb``,
and ``pyami``. It provides utilities for controlling DAQ runs and incorporating
the DAQ into ``bluesky`` plans.

Requirements
------------

- ``python`` >= 3.6
- ``bluesky`` >= 1.2.0
- ``ophyd`` >= 1.1.0
- ``pydaq``
- ``pycdb``
- ``pyami``
