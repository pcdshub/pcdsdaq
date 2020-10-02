"""
This module provides a simulated pydaq module for offline testing and training.
It does not assume that the real pydaq module is available.
"""
import logging
from importlib import import_module

import pcdsdaq.ami as ami
import pcdsdaq.daq as daq
import pcdsdaq.ext_scripts as ext
from . import pyami
from . import pydaq

logger = logging.getLogger(__name__)


def set_sim_mode(sim_mode):
    """
    Parameters
    ----------
    sim_mode: bool
        If True, we'll set the Daq class to run in simulated mode. If False,
        we'll attempt to import pydaq to run the Daq in real mode.
    """
    if sim_mode:
        ami.pyami = pyami
        if ami.ami_proxy is None:
            ami.set_pyami_proxy('tstproxy')
            ami.set_l3t_file('tstfile')
        daq.pydaq = pydaq
        ext.get_hutch_name = pydaq.sim_get_hutch_name
        ext.get_run_number = pydaq.sim_get_run_number
    else:
        if ami.ami_proxy == 'tstproxy':
            ami._reset_globals()
        try:
            real_pyami = import_module('pyami')
            ami.pyami = real_pyami
        except ImportError:
            logger.error('pyami not available in this session')
        try:
            real_pydaq = import_module('pydaq')
            daq.pydaq = real_pydaq
        except ImportError:
            logger.error('pydaq not available in this session')
        ext.get_hutch_name = pydaq.get_hutch_name
        ext.get_run_number = pydaq.get_run_number
