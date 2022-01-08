import logging
from threading import Event

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine
from ophyd.positioner import SoftPositioner
from ophyd.signal import Signal
from ophyd.utils.errors import WaitTimeoutError
from psdaq.control.ControlDef import ControlDef

from pcdsdaq.daq import DaqLCLS2

logger = logging.getLogger(__name__)


@pytest.fixture(scope='function')
def daq_lcls2(RE: RunEngine) -> DaqLCLS2:
    return DaqLCLS2(
        platform=0,
        host='tst',
        timeout=1000,
        RE=RE,
        hutch_name='tst',
        sim=True,
    )


def sig_wait_value(sig, goal, timeout=1, assert_success=True):
    ev = Event()

    def cb(value, **kwargs):
        if value == goal:
            ev.set()

    cbid = sig.subscribe(cb)
    ev.wait(timeout)
    sig.unsubscribe(cbid)
    if assert_success:
        assert sig.get() == goal


def test_state(daq_lcls2: DaqLCLS2):
    """Check that the state attribute reflects the DAQ state."""
    # TODO investigate why this fails sometimes
    logger.debug('test_state')
    for state in ControlDef.states:
        if daq_lcls2.state_sig.get().name != state:
            daq_lcls2._control.sim_set_states(1, state)
            sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum[state])
        assert daq_lcls2.state == state


def test_preconfig(daq_lcls2: DaqLCLS2):
    """
    Check that preconfig has the following behavior:
    - Writes to cfg signals
    - Checks types, raising TypeError if needed
    - Has no change for unpassed kwargs
    - Reverts "None" kwargs to the default
    """
    logger.debug('test_preconfig')

    def test_one(keyword, good_values, bad_values):
        if keyword == 'motors':
            # Backcompat alias
            sig = daq_lcls2.controls_cfg
        else:
            sig = getattr(daq_lcls2, keyword + '_cfg')
        orig = sig.get()
        for value in good_values:
            daq_lcls2.preconfig(**{keyword: value})
            assert sig.get() == value
        some_value = sig.get()
        daq_lcls2.preconfig(show_queued_cfg=False)
        assert sig.get() == some_value
        for value in bad_values:
            with pytest.raises(TypeError):
                daq_lcls2.preconfig(**{keyword: value})
        daq_lcls2.preconfig(**{keyword: None})
        assert sig.get() == orig

    test_one('events', (1, 10, 100), (45.3, 'peanuts', object()))
    test_one('duration', (1, 2.3, 100), ('walnuts', object()))
    test_one('record', (True, False), (1, 0, object()))

    good_controls = dict(
        sig=Signal(name='sig'),
        mot=SoftPositioner(name='mot'),
    )
    for kw in ('controls', 'motors'):
        test_one(
            kw,
            (good_controls, list(good_controls.values())),
            ('text', 0, Signal(name='sig')),
        )
        # NOTE: we don't check that the contents of the controls are OK
        # This is a bit obnoxious to do generically

    test_one('begin_timeout', (1, 2.3, 100), ('cashews', object()))
    test_one('begin_sleep', (1, 2.3, 100), ('pistachios', object()))
    test_one('group_mask', (1, 10, 100), (23.4, 'coconuts', object()))
    test_one('detname', ('epix', 'wave8'), (1, 2.3, object()))
    test_one('scantype', ('cool', 'data'), (1, 2.4, object()))
    test_one('serial_number', ('213', 'AV34'), (1, 2.5, object()))
    test_one('alg_name', ('math', 'calc'), (1, 2.6, object()))
    test_one('alg_version', ([1, 2, 3], [2, 3, 4]), (1, 2.6, 'n', object()))


def test_record(daq_lcls2: DaqLCLS2):
    """
    Tests on record, record_cfg, recording_sig:

    Check that the recording_sig gets the correct value as reported by
    the monitorStatus calls.

    Then, check that the record property has the following behavior:
    - When setattr, record_cfg is put to as appropriate (True/False/None)
    - When getattr, we see the correct value:
      - True if the last set was True
      - False if the last set was False
      - Match the control's record state otherwise
    """
    logger.debug('test_record')

    # Establish that recording_sig is a reliable proxy for _control state
    for record in (True, False, True, False):
        daq_lcls2._control.setRecord(record)
        sig_wait_value(daq_lcls2.recording_sig, record)

    # Establish that record setattr works
    for record in (True, False, True, False):
        daq_lcls2.record = record
        assert daq_lcls2.record_cfg.get() == record

    # Establish that the getattr logic table is correct
    daq_cfg = (True, False, None)
    daq_status = (True, False)

    def assert_expected(daq: DaqLCLS2):
        if daq.record_cfg.get() is None:
            assert daq.record == daq.recording_sig.get()
        else:
            assert daq.record == daq.record_cfg.get()

    for cfg in daq_cfg:
        daq_lcls2.record_cfg.put(cfg)
        for status in daq_status:
            daq_lcls2.recording_sig.put(status)
            assert_expected(daq_lcls2)


def test_run_number(daq_lcls2: DaqLCLS2):
    """
    Test that the values from monitorStatus can be returned via run_number.
    """
    logger.debug('test_run_number')

    for run_num in range(10):
        daq_lcls2._control._run_number = run_num
        daq_lcls2._control.sim_new_status(
            daq_lcls2._control._headers['status'],
        )
        sig_wait_value(daq_lcls2.run_number_sig, run_num)
        assert daq_lcls2.run_number() == run_num


@pytest.mark.timeout(60)
def test_stage_unstage(daq_lcls2: DaqLCLS2, RE: RunEngine):
    """
    Test the following behavior on stage:
    - RE subscription to end run on stop, if not already subscribed
    - end run if one is already going
    Test the following behavior on unstage:
    - RE subscription cleaned up if still active
    - end run if the run hasn't ended yet
    - infinite run if we were running before
    These tests imply that daq can call stage/unstage multiple times
    with no errors, but this isn't a requirement.
    """
    logger.debug('test_stage_unstage')

    def empty_run():
        yield from bps.open_run()
        yield from bps.close_run()

    def do_empty_run():
        logger.debug('do_empty_run')
        RE(empty_run())

    def set_running():
        logger.debug('set_running')
        if daq_lcls2._control._state == 'running':
            return
        status = running_status()
        daq_lcls2._control.sim_set_states('enable', 'running')
        status.wait(timeout=1)

    def running_status():
        logger.debug('running_status')
        return daq_lcls2.get_status_for(
            state=['running'],
            check_now=False,
        )

    def end_run_status():
        logger.debug('end_run_status')
        return daq_lcls2.get_status_for(
            transition=['endrun'],
            check_now=False,
        )

    # Nothing special happens if no stage
    logger.debug('nothing special')
    set_running()
    status = end_run_status()
    do_empty_run()
    with pytest.raises(WaitTimeoutError):
        status.wait(timeout=1)
    status.set_finished()

    # If we stage, the previous run should end
    logger.debug('stage ends run')
    set_running()
    status = end_run_status()
    daq_lcls2.stage()
    status.wait(timeout=1)
    daq_lcls2.unstage()

    # If we stage, the run should end itself in the plan
    logger.debug('plan ends staged run')
    daq_lcls2.stage()
    set_running()
    status = end_run_status()
    do_empty_run()
    status.wait(timeout=1)
    daq_lcls2.unstage()

    # Redo first test after an unstage
    logger.debug('nothing special, reprise')
    set_running()
    status = end_run_status()
    do_empty_run()
    with pytest.raises(WaitTimeoutError):
        status.wait(timeout=1)
    status.set_finished()

    # Unstage should end the run if it hasn't already ended
    logger.debug('unstage ends run')
    daq_lcls2.stage()
    set_running()
    status = end_run_status()
    daq_lcls2.unstage()
    status.wait(timeout=1)

    # Running -> Staged (not running) -> Unstaged (running)
    logger.debug('unstage resumes run')
    set_running()
    status = end_run_status()
    daq_lcls2.stage()
    status.wait(timeout=1)
    status = running_status()
    daq_lcls2.unstage()
    status.wait(timeout=1)
    daq_lcls2.end_run()


def test_configure(daq_lcls2: DaqLCLS2):
    """
    Configure must have the following behavior:
    - kwargs end up in cfg signals (spot check 1 or 2)
    - Returns (old_cfg, new_cfg)
    - Configure transition caused if needed from conn/conf states
    - Conf needed if recording gui clicked, or critical kwargs changed,
      or if never done.
    - From the conf state, we unconf before confing
    - Configure transition not caused if not needed
    - Error if we're not in conn/conf states and a transition is needed
    """
    logger.debug('test_configure')
    # The first configure should cause a transition
    # Let's start in connected and check the basic stuff
    daq_lcls2._control.sim_set_states(
        transition='connect',
        state='connected',
    )
    daq_lcls2.get_status_for(state=['connected']).wait(timeout=1)
    prev_tst = daq_lcls2.read_configuration()
    prev_cfg, post_cfg = daq_lcls2.configure(events=100, detname='dat')
    assert daq_lcls2.events_cfg.get() == 100
    assert daq_lcls2.detname_cfg.get() == 'dat'
    post_tst = daq_lcls2.read_configuration()
    assert (prev_cfg, post_cfg) == (prev_tst, post_tst)

    # Changing record should make us reconfigure
    st_conn = daq_lcls2.get_status_for(state=['connected'], check_now=False)
    st_conf = daq_lcls2.get_status_for(state=['configured'], check_now=False)
    daq_lcls2.configure(record=not daq_lcls2.record)
    st_conn.wait(timeout=1)
    st_conf.wait(timeout=1)

    # Changing events should not make us reconfigure
    st_any = daq_lcls2.get_status_for(check_now=False)
    daq_lcls2.configure(events=1000)
    with pytest.raises(WaitTimeoutError):
        st_any.wait(1)
    st_any.set_finished()

    # Configure should error if transition needed from most of the states
    bad_states = daq_lcls2.state_enum.exclude(['connected', 'configured'])
    prev_record = daq_lcls2.record
    for state in bad_states:
        logger.debug('testing %s', state)
        daq_lcls2.state_sig.put(state)
        with pytest.raises(RuntimeError):
            daq_lcls2.configure(record=not prev_record)

    # If we get desynced from the recording state, we should reconfigure
    # Get us into a normal state, regardless of previous testing
    daq_lcls2._control.setState('connected', {})
    daq_lcls2.get_status_for(state=['connected']).wait(timeout=1)
    daq_lcls2.configure(record=False)
    daq_lcls2.configure(record=True)
    sig_wait_value(daq_lcls2.recording_sig, True)
    # Simulate someone changing the recording state
    daq_lcls2._control.setRecord(False)
    sig_wait_value(daq_lcls2.recording_sig, False)
    # Configure something else and check for transitions
    st_conn = daq_lcls2.get_status_for(state=['connected'], check_now=False)
    st_conf = daq_lcls2.get_status_for(state=['configured'], check_now=False)
    daq_lcls2.configure(events=999)
    st_conn.wait(timeout=1)
    st_conf.wait(timeout=1)


def test_config_info(daq_lcls2: DaqLCLS2):
    """Simply test that config_info can run without errors."""
    daq_lcls2.config_info()


def test_config(daq_lcls2: DaqLCLS2):
    """
    Test the following:
    - daq.config matches the values put into configure
    - mutating daq.config doesn't change daq.config
    """
    assert daq_lcls2.config is not daq_lcls2.config
    daq_lcls2._control.setState('connected', {})
    daq_lcls2.get_status_for(state=['connected']).wait(timeout=1)
    conf = dict(events=100, record=True)
    daq_lcls2.configure(**conf)
    for key, value in conf.items():
        assert daq_lcls2.config[key] == value
    full_conf = daq_lcls2.config
    full_conf['events'] = 10000000
    assert daq_lcls2.config['events'] != full_conf['events']
    assert daq_lcls2.config is not daq_lcls2.config


def test_default_config(daq_lcls2: DaqLCLS2):
    """
    Test the following:
    - default config exists
    - is unchanged by configure
    - matches config at start
    - immutable
    """
    daq_lcls2._control.setState('connected', {})
    default = daq_lcls2.default_config
    assert daq_lcls2.config == default
    daq_lcls2.get_status_for(state=['connected']).wait(timeout=1)
    daq_lcls2.configure(events=1000, record=False, begin_timeout=12)
    assert daq_lcls2.default_config == default
    default_events = daq_lcls2.default_config['events']
    daq_lcls2.default_config['events'] = 1000000
    assert daq_lcls2.default_config['events'] == default_events
    assert daq_lcls2.default_config is not daq_lcls2.default_config


def test_configured(daq_lcls2: DaqLCLS2):
    """
    Configured means we're in the "configured" state or higher.
    """
    def transition_wait_assert(state, expected_configured):
        daq_lcls2._control.setState(state, {})
        daq_lcls2.get_status_for(state=[state]).wait(timeout=1)
        sig_wait_value(daq_lcls2.configured_sig, expected_configured)
        assert daq_lcls2.configured == expected_configured

    transition_wait_assert('reset', False)
    transition_wait_assert('unallocated', False)
    transition_wait_assert('allocated', False)
    transition_wait_assert('connected', False)
    transition_wait_assert('configured', True)
    transition_wait_assert('starting', True)
    transition_wait_assert('paused', True)
    transition_wait_assert('running', True)


def test_kickoff(daq_lcls2: DaqLCLS2):
    """
    kickoff must have the following behavior:
    - starts or resumes the run (goes to running)
    - configures if needed
    - errors if not connected, or if already running
    - errors if a configure is needed and cannot be done
    - config params can be passed, and are reverted after the run
    """
    # Errors if not connected or already running
    for state in ('reset', 'unallocated', 'allocated', 'running'):
        daq_lcls2.state_transition(state, timeout=1, wait=True)
        with pytest.raises(RuntimeError):
            daq_lcls2.kickoff()

    # Starts from normal states
    for state in ('connected', 'configured', 'starting', 'paused'):
        daq_lcls2.state_transition(state, timeout=1, wait=True)
        daq_lcls2.kickoff()
        daq_lcls2.get_status_for(state=['running']).wait(timeout=1)

    # Configures if needed, reverts parameters
    # Start in configured state, wait for unconfig/config/enable/endstep
    daq_lcls2.state_transition('configured', timeout=1, wait=True)
    unconf_st = daq_lcls2.get_status_for(
        transition=['unconfigure'],
        check_now=False,
    )
    conf_st = daq_lcls2.get_status_for(
        transition=['configure'],
        check_now=False,
    )
    run_st = daq_lcls2.get_status_for(
        transition=['enable'],
        check_now=False,
    )
    end_st = daq_lcls2.get_status_for(
        transition=['endstep'],
        check_now=False,
    )
    daq_lcls2.kickoff(events=10, record=not daq_lcls2.recording_sig.get())
    unconf_st.wait(timeout=1)
    conf_st.wait(timeout=1)
    run_st.wait(timeout=1)
    end_st.wait(timeout=1)
    # While the run is still open, our config is still set
    assert daq_lcls2.events_cfg.get() == 10
    daq_lcls2.state_transition('configured', timeout=1, wait=True)
    # But now, after end_run, our config should have reverted
    # Need to wait because this is largely asynchronous
    sig_wait_value(daq_lcls2.events_cfg, None)

    # Errors if a configure is needed and cannot be done
    # This case here is start/stop recording during a run,
    # Which must be invalid due to the DAQ architecture
    daq_lcls2.state_transition('paused', timeout=1, wait=True)
    with pytest.raises(RuntimeError):
        daq_lcls2.kickoff(record=not daq_lcls2.recording_sig.get())


@pytest.mark.skip(reason='Test not written yet.')
def test_wait():
    # Test this after kickoff
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_trigger():
    # Test this after kickoff
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_begin():
    # Test this after wait
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_stop():
    # Test this after begin
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_begin_infinite():
    # Test this after stop
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_end_run():
    # Test this after stop
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_read():
    # Test this after begin_infinite
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_pause_resume():
    # Test after begin_infinite
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_collect():
    # Test this after kickoff
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_complete():
    # Test this after collect
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_describe_collect():
    # Test this after collect
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_step_scan():
    # Test at the end
    1/0


@pytest.mark.skip(reason='Test not written yet.')
def test_fly_scan():
    # Test at the end
    1/0
