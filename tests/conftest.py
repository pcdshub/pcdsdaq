import functools
import os
import sys

import pytest
from bluesky import RunEngine
from ophyd.sim import SynSignal, motor1

import pcdsdaq.daq.original as daq_module
import pcdsdaq.sim.pyami as sim_pyami
import pcdsdaq.sim.pydaq as sim_pydaq
from pcdsdaq.ami import AmiDet
from pcdsdaq.ami import _reset_globals as ami_reset_globals
from pcdsdaq.daq.original import Daq
from pcdsdaq.sim import set_sim_mode
from pcdsdaq.sim.pydaq import SimNoDaq


@pytest.fixture(scope='function')
def reset():
    ami_reset_globals()


@pytest.fixture(scope='function')
def sim(reset):
    set_sim_mode(True)


@pytest.fixture(scope='function')
def nosim(reset):
    set_sim_mode(False)


@pytest.fixture(scope='function')
def daq(RE, sim):
    if sys.platform == 'win32':
        pytest.skip('Cannot make DAQ on windows')
    sim_pydaq.conn_err = None
    daq_module.BEGIN_THROTTLE = 0
    daq = Daq(RE=RE)
    yield daq
    try:
        # Sim daq can freeze pytest's exit if we don't end the run
        daq.end_run()
    except Exception:
        pass


@pytest.fixture(scope='function')
def nodaq(RE):
    return SimNoDaq(RE=RE)


@pytest.fixture(scope='function')
def ami_det(sim):
    sim_pyami.connect_success = True
    sim_pyami.set_l3t_count = 0
    sim_pyami.clear_l3t_count = 0
    return AmiDet('TST', name='test')


@pytest.fixture(scope='function')
def ami_det_2():
    return AmiDet('TST2', name='test2')


@pytest.fixture(scope='function')
def RE():
    RE = RunEngine({})
    RE.verbose = True
    return RE


@pytest.fixture(scope='function')
def sig():
    sig = SynSignal(name='test')
    sig.put(0)
    return sig


@pytest.fixture(scope='function')
def mot():
    motor1.set(0)
    return motor1


@pytest.fixture(scope='session', autouse=True)
def windows_compat():
    if sys.platform == 'win32':
        os.uname = lambda: 'localhost'


skip_windows = functools.partial(pytest.mark.skipif, sys.platform == 'win32')
