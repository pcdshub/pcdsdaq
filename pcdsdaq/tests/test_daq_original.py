import logging
import os
import os.path
import signal
import time
from threading import Thread

import pytest
from bluesky.plan_stubs import trigger_and_read
from bluesky.plans import count
from bluesky.preprocessors import run_wrapper, stage_wrapper
from ophyd.status import wait as status_wait

from .. import ext_scripts as ext
from ..daq import original as daq_module
from ..daq.original import BEGIN_TIMEOUT, StateTransitionError
from ..exceptions import DaqTimeoutError
from ..sim import pydaq as sim_pydaq
from .conftest import skip_windows

logger = logging.getLogger(__name__)


def test_connect(daq):
    """
    We expect connect to bring the daq from a disconnected state to a connected
    state.
    """
    logger.debug('test_connect')
    assert not daq.connected
    daq.connect()
    assert daq.connected
    daq.connect()  # Coverage
    # If something goes wrong...
    daq._control = None
    daq_module.pydaq = None
    daq.connect()
    assert daq._control is None


def test_connect_errors(daq):
    # Make sure we cover these so the log statements don't fail
    sim_pydaq.conn_err = 'Initial query failed'
    daq.connect()
    sim_pydaq.conn_err = 'Connect failed'
    daq.connect()
    with pytest.raises(RuntimeError):
        daq.begin()


def test_disconnect(daq):
    """
    We expect disconnect to bring the daq from a connected state to a
    disconnected state.
    """
    logger.debug('test_disconnect')
    assert not daq.connected
    daq.connect()
    assert daq.connected
    daq.disconnect()
    assert not daq.connected


class Dummy:
    position = 4


def test_configure(daq, sig):
    """
    We expect the configured attribute to be correct.
    We expect a disconnected daq to connect and then configure.
    We expect a configure with no args to pick the defaults.
    We expect to be able to disconnect after a configure.
    We expect a connected daq to be able to configure.
    We expect configure to return both the old and new configurations.
    We expect read_configure to give us the current configuration, including
    default args.
    """
    logger.debug('test_configure')
    assert not daq.connected
    assert not daq.configured
    daq.configure()
    assert daq.config == daq.default_config
    assert daq.connected
    assert daq.configured
    daq.disconnect()
    assert not daq.connected
    assert not daq.configured
    daq.connect()
    assert daq.connected
    assert not daq.configured
    configs = [
        dict(events=1000, use_l3t=True),
        dict(events=1000, use_l3t=True, record=True),
        dict(duration=10, controls=dict(test=Dummy())),
    ]
    prev_config = daq.read_configuration()
    for config in configs:
        old, new = daq.configure(**config)
        assert old == prev_config
        assert daq.read_configuration() == new
        for key, value in config.items():
            assert daq.config[key] == value
        prev_config = daq.read_configuration()


def test_disconnect_config(daq):
    logger.debug('test_disconnect_config')

    daq.configure(events=120)
    daq.disconnect()
    assert daq.next_config['events'] == 120


def test_record(daq):
    """
    Make sure the record convenience property works.
    """
    logger.debug('test_record')

    daq.configure(record=True)
    assert daq.record
    daq.configure(record=False)
    assert not daq.record

    daq.record = True
    assert daq.record
    daq.configure()
    assert daq.config['record']
    assert daq.record

    daq.record = False
    assert not daq.record
    daq.begin()
    assert not daq.config['record']
    assert not daq.record


@pytest.mark.timeout(10)
def test_basic_run(daq, sig):
    """
    We expect a begin without a configure to automatically configure
    We expect the daq to run for the time passed into begin
    We expect that we close the run upon calling end_run
    """
    logger.debug('test_basic_run')
    assert daq.state == 'Disconnected'
    daq.begin(duration=1, controls=[sig])
    assert daq.state == 'Running'
    time.sleep(1.3)
    assert daq.state == 'Open'
    daq.end_run()
    assert daq.state == 'Configured'
    daq.begin(events=1, wait=True, use_l3t=True)
    daq.end_run()
    # Configure for 1 event, if takes less than 1s then inf broken
    daq.configure(events=1)
    daq.begin_infinite()
    time.sleep(1)
    assert daq.state == 'Running'
    daq.end_run()
    daq.begin(events=1, wait=True, end_run=True)
    assert daq.state == 'Configured'
    daq.begin(events=1, wait=False, end_run=True)
    time.sleep(0.2)
    assert daq.state == 'Configured'

    # now we force the kickoff to time out
    daq._control._state = 'Disconnected'
    start = time.time()
    status = daq.kickoff(duration=BEGIN_TIMEOUT+3)
    with pytest.raises(RuntimeError):
        status_wait(status, timeout=BEGIN_TIMEOUT+1)
    dt = time.time() - start
    assert dt < BEGIN_TIMEOUT + 1


@pytest.mark.timeout(10)
def test_begin_record_arg(daq):
    """
    We expect that the record argument in begin overrides the daq's record
    configuration for the run.
    """
    logger.debug('test_begin_record_arg')
    # Sanity checks
    daq.configure(record=False)
    assert not daq.record
    daq.begin(events=1, wait=True)
    daq.end_run()
    assert not daq.config['record']
    assert not daq._desired_config
    # Did we record?
    daq.begin(events=1, wait=True, record=True)
    daq.end_run()
    assert daq.config['record']
    assert not daq._desired_config['record']
    # 2 in a row: did we record?
    daq.begin(events=1, wait=True, record=True)
    daq.end_run()
    assert daq.config['record']
    assert not daq._desired_config['record']
    # Remove record arg: did we not record?
    daq.begin(events=1, wait=True)
    daq.end_run()
    assert not daq.config['record']
    assert not daq._desired_config
    # Configure for record=True, then also pass to begin
    daq.record = True
    daq.begin(events=1, wait=True, record=True)
    daq.end_run()
    assert daq.config['record']
    assert not daq._desired_config

    # Same tests, but swap all the booleans
    daq.configure(record=True)
    assert daq.record
    daq.begin(events=1, wait=True)
    daq.end_run()
    assert daq.config['record']
    assert not daq._desired_config
    daq.begin(events=1, wait=True, record=False)
    daq.end_run()
    assert not daq.config['record']
    assert daq._desired_config['record']
    daq.begin(events=1, wait=True, record=False)
    daq.end_run()
    assert not daq.config['record']
    assert daq._desired_config['record']
    daq.begin(events=1, wait=True)
    daq.end_run()
    assert daq.config['record']
    assert not daq._desired_config
    daq.record = False
    daq.begin(events=1, wait=True, record=False)
    daq.end_run()
    assert not daq.config['record']
    assert not daq._desired_config


@pytest.mark.timeout(3)
def test_stop_run(daq):
    """
    We expect the daq to run indefinitely if no time is passed to begin
    We expect that the running stops early if we call stop
    """
    logger.debug('test_stop_run')
    t0 = time.time()
    daq.begin()
    assert daq.state == 'Running'
    time.sleep(1.3)
    assert daq.state == 'Running'
    daq.stop()
    assert daq.state == 'Open'
    less_than_2 = time.time() - t0
    assert less_than_2 < 2


@pytest.mark.timeout(3)
def test_wait_run(daq):
    """
    We expect that wait will block the thread until the daq is no longer
    running
    We expect that a wait when nothing is happening will do nothing
    """
    logger.debug('test_wait_run')
    t0 = time.time()
    daq.wait()
    short_time = time.time() - t0
    assert short_time < 1
    t1 = time.time()
    daq.begin(duration=1)
    daq.wait()
    just_over_1 = time.time() - t1
    assert 1 < just_over_1 < 2
    t3 = time.time()
    daq.wait()
    short_time = time.time() - t3
    assert short_time < 1


@pytest.mark.timeout(3)
def test_configured_run(daq, sig):
    """
    We expect begin() to run for the configured time, should a time be
    configured
    """
    logger.debug('test_configured_run')
    daq.configure(duration=1, controls=dict(sig=sig))
    t0 = time.time()
    daq.begin()
    daq.wait()
    just_over_1 = time.time() - t0
    assert 1 < just_over_1 < 2


@pytest.mark.timeout(3)
def test_pause_resume(daq):
    """
    We expect pause and resume to work.
    """
    logger.debug('test_pause_resume')
    daq.begin(duration=5)
    assert daq.state == 'Running'
    daq.pause()
    assert daq.state == 'Open'
    daq.resume()
    assert daq.state == 'Running'
    daq.stop()
    assert daq.state == 'Open'
    daq.end_run()
    assert daq.state == 'Configured'


def test_check_connect(nodaq):
    """
    If the daq can't connect for any reason, we should get an error on any
    miscellaneous method that has the check_connect wrapper.
    """
    logger.debug('test_check_connect')
    with pytest.raises(RuntimeError):
        nodaq.wait()


@pytest.mark.timeout(10)
def test_basic_plans(daq, RE):
    logger.debug('test_basic_plans')
    daq.configure(events=12)

    start = time.time()
    RE(stage_wrapper(run_wrapper(trigger_and_read([daq])), [daq]))
    dt = time.time() - start
    assert 0.1 < dt < 0.2
    assert daq.state == 'Configured'

    start = time.time()
    RE(count([daq], num=10))
    dt = time.time() - start
    assert 1 < dt < 2
    assert daq.state == 'Configured'

    def n_runs(det, n):
        for i in range(n):
            yield from run_wrapper(trigger_and_read([det]))

    RE(stage_wrapper(n_runs(daq, 10), [daq]))
    assert daq.state == 'Configured'


def test_trigger_error(daq, RE):
    logger.debug('test_trigger_error')
    daq.configure(events=None, duration=None)
    with pytest.raises(RuntimeError):
        daq.trigger()


def test_preconfig(daq):
    logger.debug('test_preconfig')

    daq.preconfig(events=120, use_l3t=True)
    assert daq.state == 'Disconnected'
    daq.configure()
    assert daq.config['events'] == 120
    assert daq.config['use_l3t']

    daq.preconfig(events=240, use_l3t=True)
    daq.preconfig(duration=1)
    daq.configure(use_l3t=False)
    assert daq.config['events'] is None
    assert daq.config['duration'] == 1
    assert not daq.config['use_l3t']


def test_restore_state(daq, RE):
    logger.debug('test_restore_state')

    assert daq.state == 'Disconnected'
    daq.preconfig(events=1)
    RE(count([daq]))
    assert daq.state == 'Configured'

    daq.begin_infinite()
    assert daq.state == 'Running'
    RE(count([daq]))
    assert daq.state == 'Running'


def test_bad_stuff(daq, RE):
    """
    Miscellaneous exception raises
    """
    logger.debug('test_bad_stuff')

    daq.connect()

    # Daq internal error
    configure = daq._control.configure
    daq._control.configure = None
    with pytest.raises(RuntimeError):
        daq.configure()
    daq._control.configure = configure

    # Run is too short
    with pytest.raises(RuntimeError):
        daq._check_duration(0.1)

    # Configure during a run
    daq.begin(duration=1)
    with pytest.raises(StateTransitionError):
        daq.configure()

    with pytest.raises(StateTransitionError):
        daq.begin(record=True)


@pytest.mark.timeout(3)
def test_call_bluesky(daq):
    """
    These are things that bluesky uses. Let's check them.
    """
    logger.debug('test_call_bluesky')
    daq.describe()
    daq.describe_configuration()
    daq.stage()
    daq.begin(duration=10)
    # unstage should end the run and we don't time out
    daq.unstage()


@pytest.mark.timeout(3)
def test_misc(daq, sig):
    """
    Blatant coverage-grab
    """
    logger.debug('test_misc')
    daq.configure(controls=dict(sig=sig))
    daq_module.pydaq = None
    with pytest.raises(ImportError):
        daq_module.Daq()
    end_status = daq._get_end_status()
    status_wait(end_status)
    daq.begin(duration=10)
    # Interrupt run with new run and we don't time out
    daq.begin(events=12)
    daq.wait()


@skip_windows(reason='os.kill tears down the whole suite on windows')
def test_begin_sigint(daq):
    logger.debug('test_begin_sigint')
    pid = os.getpid()

    def interrupt():
        time.sleep(0.1)
        os.kill(pid, signal.SIGINT)

    start = time.time()
    Thread(target=interrupt, args=()).start()
    daq.begin(duration=1, wait=True)
    assert time.time() - start < 0.5


def test_run_number(daq, monkeypatch):
    logger.debug('test_run_number')
    # Make sure simulated run count works and code is covered
    start_num = daq.run_number()
    daq.begin(record=True, events=1, wait=True, end_run=True)
    assert daq.run_number() == start_num + 1

    # Get during a run to text other branch
    daq.begin(record=True, events=1000)
    assert daq.run_number() == start_num + 2
    daq.end_run()

    # Make sure correct exceptions are raised
    with pytest.raises(ValueError):
        assert daq.run_number('not_a_hutch')

    def no_file(*args, **kwargs):
        raise FileNotFoundError('test')

    monkeypatch.setattr(ext, 'get_hutch_name', no_file)

    with pytest.raises(RuntimeError):
        daq.run_number()

    # We shouldn't have an exception in begin if run_number fails!
    # Not important enough to hold up the show
    daq.begin(events=100, record=True)


def test_infinite_trigger_status(daq):
    logger.debug('test_infinite_trigger_status')
    daq.configure(events=0)
    status = daq.trigger()
    status.wait(timeout=1)
    assert status.done
    assert status.success


def test_wait_error(daq):
    logger.debug('test_wait_error')
    daq.begin_infinite()
    with pytest.raises(RuntimeError):
        daq.wait()


def test_read_stops(daq):
    logger.debug('test_read_stops')
    daq.begin_infinite()
    assert daq.state == 'Running'
    daq.read()
    assert daq.state == 'Open'


def test_complete_no_error(daq):
    logger.debug('test_complete_no_error')
    # complete shouldn't error if we call it when the daq isn't running
    daq.configure(events=120)
    daq.complete()


def test_begin_throttle(daq):
    logger.debug('test_begin_throttle')
    daq_module.BEGIN_THROTTLE = 1
    start = time.time()
    daq.stop()
    daq.begin(duration=1)
    assert 1 < time.time() - start < 3


def test_timeouts(daq, monkeypatch):
    logger.debug('test_timeouts')
    daq.begin(duration=1)
    with pytest.raises(DaqTimeoutError):
        daq.wait(timeout=0.1)
    daq.stop()
    with pytest.raises(DaqTimeoutError):
        monkeypatch.setattr(daq_module, 'BEGIN_TIMEOUT', 0.1)
        daq._control._begin_delay = 3
        daq.begin(duration=1)
    daq.stop()


def test_pass_hutch(sim):
    logger.debug('test_pass_hutch')
    # Mainly for coverage, make sure no typos in the chain here
    daq = daq_module.Daq(hutch_name='tst')
    assert daq.hutch_name == 'tst'
