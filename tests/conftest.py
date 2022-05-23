import sys

from bluesky import RunEngine
from ophyd.sim import SynSignal, motor1

import pcdsdaq.daq as daq_module
import pcdsdaq.sim.pyami as sim_pyami
import pcdsdaq.sim.pydaq as sim_pydaq
from pcdsdaq.ami import (AmiDet, _reset_globals as ami_reset_globals)
from pcdsdaq.daq import Daq
from pcdsdaq.sim import set_sim_mode
from pcdsdaq.sim.pydaq import SimNoDaq

import pytest


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
