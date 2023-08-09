import logging
import time
from contextlib import contextmanager
from threading import Event

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
import pytest
from bluesky import RunEngine
from ophyd.positioner import SoftPositioner
from ophyd.signal import Signal
from ophyd.sim import motor
from ophyd.utils.errors import WaitTimeoutError

from ..daq import DaqLCLS2
from ..daq.interface import TernaryBool
from ..exceptions import DaqStateTransitionError, DaqTimeoutError

try:
    from psdaq.control.ControlDef import ControlDef
except ImportError:
    ControlDef = None

logger = logging.getLogger(__name__)


@pytest.fixture(scope='function')
def daq_lcls2(RE: RunEngine) -> DaqLCLS2:
    if ControlDef is None:
        pytest.skip(reason='psdaq is not installed')
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


@contextmanager
def assert_timespan(min, max):
    start = time.monotonic()
    yield
    duration = time.monotonic() - start
    assert min <= duration <= max


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
    test_one(
        'record',
        (True, False, TernaryBool.TRUE, TernaryBool.FALSE, TernaryBool.NONE),
        ('true', 1, 0, -1, object()),
    )

    good_controls = (
        Signal(name='sig'),
        SoftPositioner(name='mot'),
        ('constant', 4),
        ('altname', Signal(name='potato')),
    )
    for kw in ('controls', 'motors'):
        test_one(
            kw,
            (good_controls, list(good_controls)),
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
    - When setattr, record_cfg is put to as an int-compatible type
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
        assert bool(daq_lcls2.record_cfg.get()) == record

    # Establish that the getattr logic table is correct
    daq_cfg = (TernaryBool.TRUE, TernaryBool.FALSE, TernaryBool.NONE)
    daq_status = (True, False)

    def assert_expected(daq: DaqLCLS2):
        if daq.record_cfg.get() is TernaryBool.NONE:
            assert daq.record == daq.recording_sig.get()
        else:
            assert daq.record == bool(daq.record_cfg.get())

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
    - restore previous recording state
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
        return daq_lcls2._get_status_for(
            state=['running'],
            check_now=False,
        )

    def end_run_status():
        logger.debug('end_run_status')
        return daq_lcls2._get_status_for(
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
    daq_lcls2._control.sim_set_states('endrun', 'configured')
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.configured)
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

    # Unstage should revert any changes to the recording state
    logger.debug('unstage reverts record')
    recording_before = daq_lcls2.recording_sig.get()
    daq_lcls2.stage()
    daq_lcls2._control.setRecord(not recording_before)
    sig_wait_value(daq_lcls2.recording_sig, not recording_before)
    daq_lcls2.unstage()
    sig_wait_value(daq_lcls2.recording_sig, recording_before)


@pytest.mark.xfail
def test_configure(daq_lcls2: DaqLCLS2):
    """
    Configure must have the following behavior:
    - kwargs end up in cfg signals (spot check 1 or 2)
    - Returns (old_cfg, new_cfg)
    - Configure transition caused if needed from conn/conf states
    - Conf needed if recording gui clicked, or critical kwargs changed,
      or if never done, or if someone else configured
    - From the conf state, we unconf before confing
    - Configure transition not caused if not needed
    - Error if we're not in conn/conf states and a transition is needed
    - Controls arg processed with no errors (except bad user ValueError)
    - record doesn't require a configure transition, but it does
      require no open run
    """
    logger.debug('test_configure')
    # The first configure should cause a transition
    # Let's start in connected and check the basic stuff
    daq_lcls2._control.sim_set_states(
        transition='connect',
        state='connected',
    )
    daq_lcls2._get_status_for(state=['connected']).wait(timeout=1)
    prev_tst = daq_lcls2.read_configuration()
    prev_cfg, post_cfg = daq_lcls2.configure(events=100, detname='dat')
    assert daq_lcls2.events_cfg.get() == 100
    assert daq_lcls2.detname_cfg.get() == 'dat'
    post_tst = daq_lcls2.read_configuration()
    assert (prev_cfg, post_cfg) == (prev_tst, post_tst)

    # Changing controls should make us reconfigure
    st_conn = daq_lcls2._get_status_for(state=['connected'], check_now=False)
    st_conf = daq_lcls2._get_status_for(state=['configured'], check_now=False)
    daq_lcls2.configure(controls=(Signal(name='sig'),))
    st_conn.wait(timeout=1)
    st_conf.wait(timeout=1)

    # Changing events should not make us reconfigure
    st_any = daq_lcls2._get_status_for(check_now=False)
    daq_lcls2.configure(events=1000)
    with pytest.raises(WaitTimeoutError):
        st_any.wait(1)
    st_any.set_finished()

    # Any out of process configuration should make us reconfigure
    daq_lcls2._control.setState('connected', {})
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.connected)
    daq_lcls2._control.setState('configured', {})
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.configured)
    st_conn = daq_lcls2._get_status_for(state=['connected'], check_now=False)
    st_conf = daq_lcls2._get_status_for(state=['configured'], check_now=False)
    assert (
        daq_lcls2.configures_seen_sig.get()
        > daq_lcls2.configures_requested_sig.get()
    )
    daq_lcls2.configure()
    st_conn.wait(timeout=1)
    st_conf.wait(timeout=1)

    # Configure should error if transition needed from most of the states
    bad_states = daq_lcls2.state_enum.exclude(['connected', 'configured'])
    for state in bad_states:
        logger.debug('testing %s', state)
        daq_lcls2.state_sig.put(state)
        with pytest.raises(RuntimeError):
            daq_lcls2.configure(detname=state.name)

    # Let's set up a controls arg and make sure it at least doesn't error
    # Hard to check the DAQ side without making a very sophisticated sim
    daq_lcls2._control.setState('connected', {})
    daq_lcls2._get_status_for(state=['connected']).wait(timeout=1)
    daq_lcls2._last_config = {}
    daq_lcls2.configure(controls=(
        Signal(name='sig'),
        SoftPositioner(init_pos=0, name='pos'),
        ('const', 3),
    ))
    # Again, but with at least one example of a bad controls arg
    daq_lcls2._control.setState('connected', {})
    daq_lcls2._get_status_for(state=['connected']).wait(timeout=1)
    daq_lcls2._last_config = {}
    with pytest.raises(ValueError):
        daq_lcls2.configure(controls=(
            Signal(name='one_good_thing'),
            'some bad stuff here',
            0.232323232323,
        ))
    daq_lcls2.preconfig(controls=())

    # Changing record during an open run should fail.
    daq_lcls2.configure(record=True)
    daq_lcls2._control.setState('running', {})
    sig_wait_value(daq_lcls2.recording_sig, True)
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.running)
    with pytest.raises(RuntimeError):
        daq_lcls2.configure(record=False)
    # Unless it's to the same recording state we already have
    daq_lcls2.configure(record=True)
    # But changing events should not cause an open run to fail
    daq_lcls2.configure(events=120)


def test_config_info(daq_lcls2: DaqLCLS2):
    """Simply test that config_info can run without errors."""
    logger.debug('test_config_info')
    daq_lcls2.config_info()


def test_config(daq_lcls2: DaqLCLS2):
    """
    Test the following:
    - daq.config matches the values put into configure
    - mutating daq.config doesn't change daq.config
    """
    logger.debug('test_config')

    assert daq_lcls2.config is not daq_lcls2.config
    daq_lcls2._control.setState('connected', {})
    daq_lcls2._get_status_for(state=['connected']).wait(timeout=1)
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
    logger.debug('test_default_config')

    daq_lcls2._control.setState('connected', {})
    default = daq_lcls2.default_config
    assert daq_lcls2.config == default
    daq_lcls2._get_status_for(state=['connected']).wait(timeout=1)
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
    logger.debug('test_configured')

    def transition_wait_assert(state, expected_configured):
        daq_lcls2._control.setState(state, {})
        daq_lcls2._get_status_for(state=[state]).wait(timeout=1)
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
    logger.debug('test_kickoff')

    # Errors if not connected or already running
    for state in ('reset', 'unallocated', 'allocated', 'running'):
        daq_lcls2._state_transition(state, timeout=1, wait=True)
        with pytest.raises(RuntimeError):
            daq_lcls2.kickoff()

    # Starts from normal states
    for state in ('connected', 'configured', 'starting', 'paused'):
        daq_lcls2._state_transition(state, timeout=1, wait=True)
        daq_lcls2.kickoff()
        daq_lcls2._get_status_for(state=['running']).wait(timeout=1)

    # Configures if needed, reverts parameters
    # Start in configured state, wait for unconfig/config/enable/endstep
    daq_lcls2._state_transition('configured', timeout=1, wait=True)
    unconf_st = daq_lcls2._get_status_for(
        transition=['unconfigure'],
        check_now=False,
    )
    conf_st = daq_lcls2._get_status_for(
        transition=['configure'],
        check_now=False,
    )
    run_st = daq_lcls2._get_status_for(
        transition=['enable'],
        check_now=False,
    )
    end_st = daq_lcls2._get_status_for(
        transition=['endstep'],
        check_now=False,
    )
    daq_lcls2.kickoff(events=10, controls=(Signal(name='test'),))
    unconf_st.wait(timeout=1)
    conf_st.wait(timeout=1)
    run_st.wait(timeout=1)
    end_st.wait(timeout=1)
    # Now, after endstep, our config should have reverted
    # Need to wait because this is largely asynchronous
    sig_wait_value(daq_lcls2.events_cfg, 0)

    # Errors if a configure is needed and cannot be done
    # This case here is start/stop recording during a run,
    # Which must be invalid due to the DAQ architecture
    daq_lcls2._state_transition('paused', timeout=1, wait=True)
    with pytest.raises(RuntimeError):
        daq_lcls2.kickoff(record=not daq_lcls2.recording_sig.get())


def test_wait(daq_lcls2: DaqLCLS2):
    """
    wait has the following behavior:
    - If we have an open run, pause the thread until no more events
    - If no open run, don't wait
    - If we time out, raise the daq timeout error
    - If end_run=True, end the run automatically
    """
    logger.debug('test_wait')
    daq_lcls2._state_transition('configured')
    # Normal behavior: wait for the 1 second run
    with assert_timespan(min=1, max=2):
        daq_lcls2.kickoff(events=120).wait(timeout=1)
        daq_lcls2.wait(timeout=2)
    # We should end in starting state, not configured
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.starting)
    # No open events means we should barely wait at all
    # End run should end the run
    with assert_timespan(min=0, max=1):
        daq_lcls2.wait(timeout=2, end_run=True)
    # Now we should have the end run state!
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.configured)
    # If timing out, a wait raises an exception
    daq_lcls2.kickoff().wait(timeout=1)
    with pytest.raises(DaqTimeoutError):
        daq_lcls2.wait(timeout=0.1)


def test_trigger(daq_lcls2: DaqLCLS2):
    """
    Trigger must do the following:
    - Start acquisition, with whatever fixed length we've configured
    - Return a status that is marked done when the acquisition is done
    - Return a status that is marked done immediately if we're configured
      for infinite runs.

    Other elements should be adequately tested in kickoff
    Note that this implicitly relies on it being possible to reconfigure
    events/duration during an open run.
    It also requires configuring on events and duration to work at all.
    """
    logger.debug('test_trigger')
    daq_lcls2._state_transition('connected')
    for config in (
        {'events': 120},
        {'duration': 1},
    ):
        daq_lcls2.configure(**config)
        step_status = daq_lcls2.trigger()
        sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.running)
        assert not step_status.done
        step_status.wait(timeout=2)
        assert step_status.done
        sig_wait_value(
            daq_lcls2.transition_sig,
            daq_lcls2.transition_enum.endstep,
        )
    daq_lcls2.configure(events=0)
    status = daq_lcls2.trigger()
    status.wait(timeout=1)
    assert status.done


@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_begin(daq_lcls2: DaqLCLS2):
    """
    Begin must do the following:
    - Start or resume a run (go from connected or higher to running)
    - wait kwarg = wait until we stop taking events
    - end_run kwarg = end the run after we stop taking events
    - other kwargs = configure for that kwarg before running, revert after
    """
    logger.debug('test_begin')

    def assert_no_cfg_change():
        sig_wait_value(daq_lcls2.duration_cfg, 0)
        assert daq_lcls2.config == daq_lcls2.default_config

    daq_lcls2._state_transition('connected')
    # Simple one second acquisition
    daq_lcls2.begin(duration=1)
    for transition in ('enable', 'endstep'):
        sig_wait_value(
            daq_lcls2.transition_sig,
            daq_lcls2.transition_enum[transition],
            timeout=2,
        )
    assert_no_cfg_change()

    # Same thing, but with wait. We also need to verify we hit running.
    enable_status = daq_lcls2._get_status_for(state=['running'])
    daq_lcls2.begin(duration=1, wait=True)
    sig_wait_value(
        daq_lcls2.transition_sig,
        daq_lcls2.transition_enum.endstep,
    )
    assert enable_status.success
    assert_no_cfg_change()

    # Repeat, but with end_run
    daq_lcls2.begin(duration=1, wait=True, end_run=True)
    sig_wait_value(
        daq_lcls2.transition_sig,
        daq_lcls2.transition_enum.endrun,
    )
    assert_no_cfg_change()

    # Last case: end_run, but no wait. This requires some threading.
    enable_status = daq_lcls2._get_status_for(state=['running'])
    daq_lcls2.begin(duration=1, end_run=True)
    enable_status.wait(timeout=1)
    sig_wait_value(
        daq_lcls2.transition_sig,
        daq_lcls2.transition_enum.endrun,
        timeout=2,
    )
    assert_no_cfg_change()

    # Check the "record" configuration specifically, which may get some use
    # and requires extra handling
    daq_lcls2._state_transition('connected')
    assert not daq_lcls2._control._recording
    daq_lcls2.begin(duration=1, record=True, end_run=True)
    assert daq_lcls2._control._recording
    sig_wait_value(
        daq_lcls2.state_sig,
        daq_lcls2.state_enum.configured,
        timeout=2,
    )
    sig_wait_value(daq_lcls2.record_cfg, TernaryBool.NONE)

    # Check multiple record=True in the same run is ok
    for _ in range(10):
        daq_lcls2.begin(duration=0.1, record=True, wait=True)
    # But switching to record=False should break it
    with pytest.raises(RuntimeError):
        daq_lcls2.begin(duration=0.1, record=False)


def test_stop(daq_lcls2: DaqLCLS2):
    """
    Stop brings us down to the endstep transition.
    It has no effect if we're already stopped.
    """
    logger.debug('test_stop')
    stop_state_names = ['paused', 'running']
    valid_stop_states = daq_lcls2.state_enum.include(stop_state_names)
    invalid_stop_states = daq_lcls2.state_enum.exclude(stop_state_names)
    # Check for the correct transition where applicable
    for state in valid_stop_states:
        daq_lcls2._state_transition(state)
        daq_lcls2.stop()
        sig_wait_value(
            daq_lcls2.transition_sig,
            daq_lcls2.transition_enum.endstep,
        )
    # Make sure we won't transition to the "starting" state,
    # which is the normal destination for stop, but shouldn't happen
    # if the run isn't going!
    for state in invalid_stop_states:
        daq_lcls2._state_transition(state)
        daq_lcls2.stop()
        sig_wait_value(
            daq_lcls2.state_sig,
            daq_lcls2.state_enum.starting,
            assert_success=False,
        )
        assert daq_lcls2.state_sig.get() == state


def test_begin_infinite(daq_lcls2: DaqLCLS2):
    """
    Regardless of our config, begin_infinite should keep going.
    Arguments for duration/events are ignored.
    """
    logger.debug('test_begin_infinite')
    daq_lcls2._state_transition('connected')
    daq_lcls2.configure(duration=1)
    daq_lcls2.begin_infinite(events=1, duration=0.1)
    time.sleep(5)
    assert daq_lcls2.state_sig.get() == daq_lcls2.state_enum.running


def test_end_run(daq_lcls2: DaqLCLS2):
    """
    End run brings down to the endrun transition.
    It has no effect if a run isn't open.
    """
    logger.debug('test_end_run')
    end_run_state_names = ['starting', 'paused', 'running']
    valid_end_states = daq_lcls2.state_enum.include(end_run_state_names)
    invalid_end_states = daq_lcls2.state_enum.exclude(end_run_state_names)
    # Check for the correct transition where applicable
    for state in valid_end_states:
        daq_lcls2._state_transition(state)
        daq_lcls2.end_run()
        sig_wait_value(
            daq_lcls2.transition_sig,
            daq_lcls2.transition_enum.endrun,
        )
    # Make sure we won't transition to the "configured" state,
    # which is the normal destination for end_run, but shouldn't happen
    # if the run isn't going!
    for state in invalid_end_states:
        daq_lcls2._state_transition(state)
        daq_lcls2.stop()
        sig_wait_value(
            daq_lcls2.state_sig,
            daq_lcls2.state_enum.configured,
            assert_success=False,
        )
        assert daq_lcls2.state_sig.get() == state


def test_read(daq_lcls2: DaqLCLS2):
    """
    Read has the following additional behavior here:
    - stop an "infinite" run (to allow certain sequencer-timed plans)
    """
    logger.debug('test_read')
    daq_lcls2._state_transition('connected')
    daq_lcls2.begin_infinite()
    daq_lcls2.read()
    sig_wait_value(
        daq_lcls2.transition_sig,
        daq_lcls2.transition_enum.endstep,
    )


def test_pause_resume(daq_lcls2: DaqLCLS2):
    """
    Pause brings us to the "paused" state if we were "running".
    Resume brings us to the "running" state from the "paused" state.
    The intention is for "resume" to only be called after "pause"
    but this will be confusing for the user, so we'll make it
    behave like a bare "kickoff" in all other cases.
    """
    logger.debug('test_pause_resume')
    daq_lcls2._state_transition('connected')
    daq_lcls2.begin_infinite()
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.running)
    daq_lcls2.pause()
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.paused)
    daq_lcls2.resume()
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.running)
    daq_lcls2.end_run()
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.configured)
    daq_lcls2.resume()
    sig_wait_value(daq_lcls2.state_sig, daq_lcls2.state_enum.running)


def test_collect(daq_lcls2: DaqLCLS2):
    """
    From the bluesky flyer interface docs:
    - Yield dictionaries that are partial Event documents.
    - They should contain the keys "time", "data", and "timestamps".
      A uid is added by the RunEngine.

    Note that the current implementation yields nothing, which also
    is enough to pass this test.
    """
    logger.debug('test_collect')
    for event in daq_lcls2.collect():
        assert 'time' in event
        assert 'data' in event
        assert 'timestamp' in event


def test_complete(daq_lcls2: DaqLCLS2):
    logger.debug('test_complete')
    """
    From the bluesky flyer interface docs:
    - Return a Status and mark it done when acquisition has completed.
    """
    logger.debug('test_complete')
    daq_lcls2._state_transition('connected')
    with assert_timespan(min=1, max=2):
        daq_lcls2.kickoff(duration=1).wait(timeout=1)
        status = daq_lcls2.complete()
        status.wait(timeout=3)


def test_describe_collect(daq_lcls2: DaqLCLS2):
    """
    This is like describe_configuration, but for collect.

    For now, this just needs to not throw any errors.
    """
    logger.debug('test_describe_collect')
    daq_lcls2.describe_collect()


def test_step_scan(daq_lcls2: DaqLCLS2, RE: RunEngine):
    """
    This puts it all together:
    - The daq should be usable in a normal step scan as a det
    - During the scan, the daq should run at each point
    - The run should end at the end of the scan
    """
    logger.debug('test_step_scan')
    counter = 0

    def enable_counter(value, **kwargs):
        nonlocal counter
        if value == daq_lcls2.transition_enum.enable:
            counter += 1

    daq_lcls2._state_transition('connected')
    daq_lcls2.configure(events=1)
    cbid = daq_lcls2.transition_sig.subscribe(enable_counter)
    RE(bp.scan([daq_lcls2], motor, 0, 10, 11))
    daq_lcls2.transition_sig.unsubscribe(cbid)
    assert counter == 11
    sig_wait_value(
        daq_lcls2.transition_sig,
        daq_lcls2.transition_enum.endrun,
    )


def test_fly_scan(daq_lcls2: DaqLCLS2, RE: RunEngine):
    """
    Make sure we can also use the DAQ as a standard flyer:
    - Configure DAQ for infinite run
    - fly_during a normal scan
    - only 1 enable transition should be seen
    - run should be ended
    """
    logger.debug('test_fly_scan')
    counter = 0

    def enable_counter(value, **kwargs):
        nonlocal counter
        if value == daq_lcls2.transition_enum.enable:
            counter += 1

    daq_lcls2._state_transition('connected')
    daq_lcls2.configure(events=0)
    cbid = daq_lcls2.transition_sig.subscribe(enable_counter)
    RE(
        bpp.stage_wrapper(
            bpp.fly_during_wrapper(
                bp.scan([], motor, 0, 10, 11),
                [daq_lcls2],
            ),
            [daq_lcls2],
        )
    )
    daq_lcls2.transition_sig.unsubscribe(cbid)
    assert counter == 1
    sig_wait_value(
        daq_lcls2.transition_sig,
        daq_lcls2.transition_enum.endrun,
    )


@pytest.mark.timeout(20)
def test_transition_errors(daq_lcls2: DaqLCLS2):
    """
    Many methods should propagate a DaqStateTransitionError:
    - _state_transition
    - kickoff
    - trigger
    - begin
    - stop
    - end_run
    - pause
    - resume
    """
    logger.debug('test_transition_errors')
    # Test the error sim as a sanity check before going in
    daq_lcls2._control.sim_queue_error('sanity')
    assert daq_lcls2._control.setState('connected', {}) == 'sanity'

    daq_lcls2._control.sim_queue_error('generic')
    with pytest.raises(DaqStateTransitionError):
        daq_lcls2._state_transition('connected')

    daq_lcls2._state_transition('connected')

    daq_lcls2._control.sim_queue_error('cannot configure')
    with pytest.raises(DaqStateTransitionError):
        daq_lcls2.kickoff().wait(timeout=1)

    daq_lcls2._control.sim_queue_error('cannot configure')
    with pytest.raises(DaqStateTransitionError):
        daq_lcls2.trigger().wait(timeout=1)

    daq_lcls2._control.sim_queue_error('cannot configure')
    with pytest.raises(DaqStateTransitionError):
        daq_lcls2.begin(wait=True)

    daq_lcls2._state_transition('running')

    daq_lcls2._control.sim_queue_error('cannot disable')
    with pytest.raises(DaqStateTransitionError):
        daq_lcls2.stop()

    daq_lcls2._control.sim_queue_error('cannot disable')
    with pytest.raises(DaqStateTransitionError):
        daq_lcls2.end_run()

    daq_lcls2._control.sim_queue_error('cannot disable')
    with pytest.raises(DaqStateTransitionError):
        daq_lcls2.pause()

    daq_lcls2._state_transition('paused')
    daq_lcls2._control.sim_queue_error('cannot enable')
    with pytest.raises(DaqStateTransitionError):
        daq_lcls2.resume()
