"""
Module that defines the controls python interface for the LCLS2 DAQ.

This is the second such interface. The original interface by Chris Ford
is still distributed by the DAQ code at psdaq.control.BlueskyScan.

This updated interface was created to meet expectations about specifics
of what the interface should be as established in lcls1, largely for
convenience of the end user. This should also make things more uniform
between the lcls1 and the lcls2 usages of the DAQ.

Some notes about the implementation:
- DaqLCLS2._monitor_thread is the only communication point for reading
  state from the DAQ. Here, we call a DaqControl API method called
  "monitorStatus" which blocks and reads queued messages sent to our
  ZMQ SUB socket from the DAQ. We unpack each of these messages and
  fan out the information to each of the corresponding signals in a
  background thread so we can write asynchronous code based on the
  updating values of our various signals using the normal ophyd
  subscription methods.
- DaqLCLS2._get_status_for is a convenience method for setting up
  Status objects tied to the DAQ reaching specific states
  and through specific transitions. This means we can schedule
  code to be run when the daq reaches specific states e.g. when
  it hits running or when it undergoes specific transitions e.g.
  endrun, or more simply we can wait for specific states and
  transitions.
- DaqLCLS2._state_transition is the main communication point for
  sending state to the DAQ. This is where we pass all information
  to the DAQ with the exception of the record/no record
  configuration. We assemble information like the extra controls
  and the events for the step length here and make sure the DAQ
  recieves them, as well as most of the cosmetic DAQ
  configuration values. Ultimately this method causes the DAQ
  to transition, e.g. to configure itself, to start a run,
  to pause a run, etc.
- DaqLCLS2.configure is the only place where we might find
  ourselves setting the DAQ's recording state (if requested).

See the following URL for a description of the DAQ state machine:
https://confluence.slac.stanford.edu/display/PSDMInternal/Finite+State+Machine
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from functools import cache
from numbers import Real
from typing import Any, Union, get_type_hints

from bluesky import RunEngine
from ophyd.device import Component as Cpt
from ophyd.signal import Signal
from ophyd.status import Status
from ophyd.utils import StatusTimeoutError, WaitTimeoutError
from ophyd.utils.errors import InvalidState
from pcdsutils.enum import HelpfulIntEnum

from ..exceptions import DaqStateTransitionError, DaqTimeoutError
from .interface import (CONFIG_VAL, ControlsArg, DaqBase, EnumId, Sentinel,
                        TernaryBool, get_controls_name, get_controls_value,
                        typing_check)

try:
    from psdaq.control.ControlDef import ControlDef
    from psdaq.control.DaqControl import DaqControl
except ImportError:
    ControlDef = None
    DaqControl = None

logger = logging.getLogger(__name__)


class DaqLCLS2(DaqBase):
    """
    The LCLS2 DAQ as a bluesky-compatible object.

    This uses the ``psdaq.control.DaqControl`` module to send ZMQ commands
    to the DAQ instance.

    It can be used as a ``Reader`` or as a ``Flyer`` in a ``bluesky`` plan.

    Parameters
    ----------
    platform : int
        Required argument to specify which daq platform we're connecting to.
    host : str
        The hostname of the DAQ host we are connecting to.
    timeout : int
        How many milliseconds to wait for various DAQ communications before
        reporting an error.
    RE: ``RunEngine``, optional
        Set ``RE`` to the session's main ``RunEngine``
    hutch_name: str, optional
        Define a hutch name to use instead of shelling out to get_hutch_name.
    sim: bool, optional
        If True, simulate the daq. Defaults to False.
    """
    step_value_sig = Cpt(
        Signal, value=1, kind='normal',
        doc='The index of the step we are on, according to the DAQ.',
    )
    state_sig = Cpt(
        Signal, value=None, kind='normal',
        doc='The current state according to the DAQ state machine.',
    )
    transition_sig = Cpt(
        Signal, value=None, kind='normal',
        doc='The current or last transition according to the DAQ.',
    )
    transition_elapsed_sig = Cpt(
        Signal, value=0.0, kind='normal',
        doc='The number of seconds elapsed in the current transition.',
    )
    transition_total_sig = Cpt(
        Signal, value=0.0, kind='normal',
        doc='The expected total transition time according to the DAQ.',
    )
    config_alias_sig = Cpt(
        Signal, value='None', kind='normal',
        doc='The config alias as reported by the DAQ, e.g. "BEAM".',
    )
    recording_sig = Cpt(
        Signal, value=False, kind='normal',
        doc='If True, the DAQ is configured to save the data.',
    )
    bypass_activedet_sig = Cpt(
        Signal, value=False, kind='normal',
        doc='If True, the DAQ is... bypassing the active detector?',
    )
    experiment_name_sig = Cpt(
        Signal, value='None', kind='normal',
        doc='The name of the active experiment as reported by the DAQ.',
    )
    run_number_sig = Cpt(
        Signal, value=0, kind='normal',
        doc='The current run number as reported by the DAQ.',
    )
    last_run_number_sig = Cpt(
        Signal, value=0, kind='normal',
        doc='The run number of the most recent completed run.',
    )

    group_mask_cfg = Cpt(
        Signal, value=None, kind='config',
        doc='Bitmask used by the DAQ to select active groups.',
    )
    detname_cfg = Cpt(
        Signal, value='scan', kind='config',
        doc=(
            'Name of the scan variables in the DAQ. '
            'There is currently no reason to change this.'
        ),
    )
    scantype_cfg = Cpt(
        Signal, value='scan', kind='config',
        doc=(
            'Name of the scan type in the DAQ. '
            'There is currently no reason to change this.'
        ),
    )
    serial_number_cfg = Cpt(
        Signal, value='1234', kind='config',
        doc=(
            'Serial number of the... scan? In the DAQ. '
            'There is currently no reason to change this.'
        ),
    )
    alg_name_cfg = Cpt(
        Signal, value='raw', kind='config',
        doc=(
            'Another standard DAQ data field. '
            'There is currently no reason to change this.'
        ),
    )
    alg_version_cfg = Cpt(
        Signal, value=[1, 0, 0], kind='config',
        doc=(
            'The version number associated with the alg name. '
            'There is currently no reason to change this.'
        ),
    )

    last_err_sig = Cpt(
        Signal, value=None, kind='omitted',
        doc=(
            'Signal that holds the text from the last error reported by '
            'the DAQ.'
        ),
    )
    last_warning_sig = Cpt(
        Signal, value=None, kind='omitted',
        doc=(
            'Signal that holds the text from the last warning reported by '
            'the DAQ.'
        ),
    )
    last_file_report_sig = Cpt(
        Signal, value=None, kind='omitted',
        doc=(
            'Signal that holds the text from the last file report sent '
            'by the DAQ.'
        ),
    )
    step_done_sig = Cpt(
        Signal, value=False, kind='omitted',
        doc='Is set to True when a step is done.',
    )
    last_transition_err_sig = Cpt(
        Signal, value=None, kind='omitted',
        doc='Signal that holds the text from the last transition error.',
    )
    configures_seen_sig = Cpt(
        Signal, value=0, kind='omitted',
        doc=(
            'A counter of how many time we see configure transitions. '
            'May be reset to 0 periodically.'
        )
    )
    configures_requested_sig = Cpt(
        Signal, value=0, kind='omitted',
        doc=(
            'A counter of how many times we request a configure transitions. '
            'May be reset to 0 periodically.'
        )
    )

    # Require transition if we change any value used in the transition
    requires_configure_transition = {
        'controls',
        'detname',
        'scantype',
        'serial_number',
        'alg_name',
        'alg_version',
    }

    def __init__(
        self,
        platform: int,
        host: str,
        timeout: int,
        RE: RunEngine | None = None,
        hutch_name: str | None = None,
        sim: bool = False,
    ):
        logger.debug(
            "DaqLCLS2.__init__"
            "(platform=%s, host=%s, timeout=%s, "
            "RE=%s, hutch_name=%s, sim=%s)",
            platform,
            host,
            timeout,
            RE,
            hutch_name,
            sim,
        )
        super().__init__(RE=RE, hutch_name=hutch_name, platform=platform)
        self.state_sig.put(self.state_enum.reset)
        self.transition_sig.put(self.transition_enum.reset)
        self.group_mask_cfg.put(1 << platform)
        self._update_default_config(self.group_mask_cfg)
        if sim:
            CtrlCls = SimDaqControl
        else:
            CtrlCls = DaqControl
        self._control = CtrlCls(
            host=host,
            platform=platform,
            timeout=timeout,
        )
        self._start_monitor_thread()

    @property
    @cache
    def state_enum(self) -> type[HelpfulIntEnum]:
        """
        An enum of LCLS2 DAQ states.

        This includes every node in the DAQ state machine ordered from
        completely off to actively collecting data. That is to say,
        higher numbered states are consistently more active than
        lower-numbered states, and transitions tend to take us to the
        next state up or down.

        Returns
        -------
        Enum : type[HelpfulIntEnum]
            The enum class that can be queried for individual DAQ states.
        """
        return HelpfulIntEnum('PsdaqState', ControlDef.states)

    @property
    @cache
    def transition_enum(self) -> type[HelpfulIntEnum]:
        """
        An enum of LCLS DAQ transitions.

        This includes every edge in the DAQ state machine.
        This is roughly ordered in a similar increasing way as state_enum,
        but this is by convention and not by design and should not be
        relied upon.

        This does not include information about how the nodes are connected.

        Returns
        -------
        Enum : type[HelpfulIntEnum]
            The enum class that can be queried for individual DAQ transitions.
        """
        return HelpfulIntEnum('PsdaqTransition', ControlDef.transitions)

    def _start_monitor_thread(self) -> None:
        """
        Monitor the DAQ state in a background thread.
        """
        thread = threading.Thread(target=self._monitor_thread, args=())
        thread.daemon = True
        thread.start()

    def _monitor_thread(self) -> None:
        """
        Monitors the DAQ's ZMQ subscription messages, puts into our signals.

        The LCLS2 DAQ has ZMQ PUB nodes that we can SUB to to get updates
        about the status of the DAQ.

        This thread takes the contents of those messages and uses them to
        update our signal components, so that the rest of this class can
        be written like a normal ophyd device: e.g. we'll be able to
        call subscribe and write event-driven callbacks for various
        useful quantities.
        """
        logger.debug("DaqLCLS2._monitor_thread()")
        first_loop = True
        while not self._destroyed:
            try:
                if first_loop:
                    command, *args = self._control.getStatus()
                    # Remove the platform dictionary in middle of tuple
                    args = args[:3] + args[4:]
                    first_loop = False
                else:
                    command, *args = self._control.monitorStatus()
                logger.debug(
                    "Received command %s with args %s from monitor.",
                    command,
                    args,
                )
                if command == 'error':
                    self.last_err_sig.put(args[0])
                elif command == 'warning':
                    self.last_warning_sig.put(args[0])
                elif command == 'fileReport':
                    self.last_file_report_sig.put(args[0])
                elif command == 'progress':
                    transition, elapsed, total, *_ = args
                    self.transition_sig.put(
                        self.transition_enum[transition]
                    )
                    self.transition_elapsed_sig.put(elapsed)
                    self.transition_total_sig.put(total)
                elif command == 'step':
                    self.step_value_sig.put(self.step_value_sig.get() + 1)
                    self.step_done_sig.put(bool(args[0]))
                else:
                    # Last case is normal status
                    transition = command
                    (state, config_alias, recording, bypass_activedet,
                     experiment_name, run_number, last_run_number) = args
                    if transition == 'endrun':
                        self.step_value_sig.put(1)
                    if transition == 'endstep':
                        self.step_done_sig.put(False)
                    trans_enum = self.transition_enum[transition]
                    self.transition_sig.put(trans_enum)
                    if trans_enum == self.transition_enum.configure:
                        self.configures_seen_sig.put(
                            self.configures_seen_sig.get() + 1
                        )
                    self.state_sig.put(
                        self.state_enum[state]
                    )
                    self.config_alias_sig.put(config_alias)
                    self.recording_sig.put(recording)
                    self.bypass_activedet_sig.put(bypass_activedet)
                    self.experiment_name_sig.put(experiment_name)
                    self.run_number_sig.put(run_number)
                    self.last_run_number_sig.put(last_run_number)
                    self.transition_elapsed_sig.put(0)
                    self.transition_total_sig.put(0)
            except Exception as exc:
                logger.debug("Exception in monitor thread: %s", exc)

    @state_sig.sub_value
    def _configured_cb(
        self,
        value: HelpfulIntEnum | None,
        **kwargs,
    ) -> None:
        """
        Callback on the state signal to update the configured signal.

        The LCLS2 DAQ is considered configured based on the state machine.

        Parameters
        ----------
        value : Optional[HelpfulIntEnum]
            The last updated value from state_sig
        """
        if value is None:
            self.configured_sig.put(False)
        else:
            self.configured_sig.put(
                value >= self.state_enum.configured
            )

    @property
    def state(self) -> str:
        """
        API to show the state as reported by the DAQ.

        Returns
        -------
        state : str
            The string state name of the DAQ's current state.
        """
        return self.state_sig.get().name

    def wait(
        self,
        timeout: float | None = None,
        end_run: bool = False,
    ) -> None:
        """
        Pause the thread until the DAQ is done aquiring.

        Parameters
        ----------
        timeout: ``float``, optional
            Maximum time to wait in seconds.
        end_run: ``bool``, optional
            If ``True``, end the run after we're done waiting.
        """
        logger.debug(
            "DaqLCLS2.wait(timeout=%s, end_run=%s)",
            timeout,
            end_run,
        )
        done_status = self._get_done_status(timeout=timeout, check_now=True)
        try:
            done_status.wait()
        except (StatusTimeoutError, WaitTimeoutError):
            msg = (
                f'Timeout after {timeout} seconds waiting for daq to '
                'finish acquiring.'
            )
            raise DaqTimeoutError(msg) from None
        if end_run:
            self.end_run()

    def _get_status_for(
        self,
        state: Iterator[EnumId] | None = None,
        transition: Iterator[EnumId] | None = None,
        timeout: float | None = None,
        check_now: bool = True,
    ) -> Status:
        """
        Return a status object for DAQ state transitions.

        This status object will be marked done when we're at the given state
        or when we're doing the given transition, if either state or
        transition was given.

        If both state and transition are given, then we need to be at both
        the given state and the given transition to mark the status as done.

        State and transition are both iterators so we can check for multiple
        states. This works in an "any" sort of fashion: we need to be at
        any one of the requested states, not at all of them.

        If neither state nor transition are provided, the status will be
        marked done at the next state or transition change, or immediately
        if check_now is True.

        Parameters
        ----------
        state : Optional[Iterator[EnumId]], optional
            The states that we'd like a status for.
            This can be e.g. a list of integers, strings, or enums.
        transition : Optional[Iterator[EnumId]], optional
            The transitions that we'd like a status for.
            This can be e.g. a list of integers, strings, or enums.
        timeout : Optional[float], optional
            The duration to wait before marking the status as failed.
            If omitted, the status will not time out.
        check_now : bool, optional
            If True, we'll check the states and transitions immediately.
            If False, we'll require the system to change into the states
            and transitions we're looking for.

        Returns
        -------
        status : Status
            A status that will be marked successful after the corresponding
            states or transitions are reached.
        """
        logger.debug(
            "DaqLCLS2._get_status_for"
            "(state=%s, transition=%s, timeout=%s, check_now=%s)",
            state,
            transition,
            timeout,
            check_now,
        )
        any_change = state is None and transition is None
        if state is None:
            state = {None}
            state_arg = False
        else:
            state = self.state_enum.include(state)
            state_arg = True
        if transition is None:
            transition = {None}
            trans_arg = False
        else:
            transition = self.transition_enum.include(transition)
            trans_arg = True

        def check_state(value: Any, old_value: Any, **kwargs) -> None:
            """Call success if this value and last transition are correct."""
            nonlocal last_state
            if value == old_value and not check_now:
                return
            with lock:
                if (
                    any_change
                    or (value in state and last_transition in transition)
                ):
                    success()
                else:
                    last_state = value

        def check_transition(value: Any, old_value: Any, **kwargs) -> None:
            """Call success if this value and last state are correct."""
            nonlocal last_transition
            if value == old_value and not check_now:
                return
            with lock:
                if (
                    any_change
                    or (value in transition and last_state in state)
                ):
                    success()
                else:
                    last_transition = value

        def success() -> None:
            """Set the status as successfully finished if needed."""
            try:
                status.set_finished()
            except InvalidState:
                ...

        def clean_up(status: Status) -> None:
            """
            Undo the subscriptions once the status is done.

            Runs on successes, failures, and timeouts.
            """
            if any_change or state_arg:
                self.state_sig.unsubscribe(state_cbid)
            if any_change or trans_arg:
                self.transition_sig.unsubscribe(transition_cbid)

        last_state = None
        last_transition = None
        lock = threading.Lock()
        status = Status(self, timeout=timeout)
        if any_change or state_arg:
            state_cbid = self.state_sig.subscribe(
                check_state,
                run=check_now,
            )
        if any_change or trans_arg:
            transition_cbid = self.transition_sig.subscribe(
                check_transition,
                run=check_now,
            )
        status.add_callback(clean_up)
        return status

    def _get_done_status(
        self,
        timeout: float | None = None,
        check_now: bool = True,
    ) -> Status:
        """
        Return a status that is marked successful when the DAQ is done.

        The DAQ is done acquiring if the most recent transition was not
        "beginrun", "beginstep", or "enable", which indicate that we're
        transitioning toward a running state.

        Parameters
        ----------
        timeout : Optional[float], optional
            The duration to wait before marking the status as failed.
            If omitted, the status will not time out.
        check_now : bool, optional
            If True, we'll check for the daq to be done right now.
            If False, we'll wait for a transition to a done state.

        Returns
        -------
        done_status : Status
            A status that is marked successful once the DAQ is done
            acquiring.
        """
        logger.debug(
            "DaqLCLS2._get_done_status(timeout=%s, check_now=%s)",
            timeout,
            check_now,
        )
        return self._get_status_for(
            transition=self.transition_enum.exclude(
                ['beginrun', 'beginstep', 'enable']
            ),
            timeout=timeout,
            check_now=check_now,
        )

    def _state_transition(
        self,
        state: EnumId,
        timeout: float | None = None,
        wait: bool = True,
    ) -> Status:
        """
        Cause a daq state transition appropriately.

        This passes extra data if we need to do the 'configure' or 'beginstep'
        transitions.

        The status will have a DaqStateTransitionError applied to it if the
        DAQ reports a failed transition, or a StatusTimeoutError applied to it
        if the given timeout expires. The exception will be raised by this
        method if wait=True, otherwise it will simply be applied to the
        status object and it will be the responsibility of the caller to
        inspect and handle the error. To inspect the error, the caller can
        either:
        - call status.exception() to return an exception or None
        - call status.wait() to wait for the status to finish and raise
          the exception if one has been applied.

        Parameters
        ----------
        state : EnumId
            A valid enum identifier for the target state. This should be a
            str, int, or Enum that corresponds with an element of
            self.state_enum.
        timeout : Optional[float], optional
            The duration to wait before marking the transition as failed.
            If omitted, the transition will not time out.
        wait : bool, optional
            If True, the default, block the thread until the transition
            completes or times out.

        Returns
        -------
        transition_status : Status
            A status object that is marked done when the transition has
            completed.
        """
        logger.debug(
            "DaqLCLS2._state_transition(state=%s, timeout=%s, wait=%s)",
            state,
            timeout,
            wait,
        )
        # Normalize state
        state = self.state_enum.from_any(state)
        # Determine what extra info to send to the DAQ
        phase1_info = {}
        if self.state_sig.get() < self.state_enum.configured <= state:
            # configure transition
            phase1_info['configure'] = self._get_phase1('Configure')
        if self.state_sig.get() < self.state_enum.paused <= state:
            # beginstep transition
            phase1_info['beginstep'] = self._get_phase1('BeginStep')
        if self.state_sig.get() < self.state_enum.running <= state:
            # enable transition:
            phase1_info['enable'] = {
                # this is the event count, 0 means run forever
                'readout_count': self.events_cfg.get(),
                'group_mask': self.group_mask_cfg.get(),
            }
        # Get a status to track the transition's success or failure
        status = self._get_status_for(
            state=[state],
            timeout=timeout,
        )
        # Set the transition in background thread, can be blocking
        trans_thread = threading.Thread(
            target=self._transition_thread,
            args=(state.name, phase1_info, status),
        )
        trans_thread.daemon = True
        trans_thread.start()
        # Handle duration ourselves in another thread for LCLS1 compat
        if (
            state == self.state_enum.running
            and self.events_cfg.get() == 0
            and self.duration_cfg.get() > 0
        ):
            logger.debug("Starting duration handler")
            duration_thread = threading.Thread(
                target=self._handle_duration_thread,
                args=(self.duration_cfg.get(), status)
            )
            duration_thread.daemon = True
            duration_thread.start()

        # Keep count to check for out of process configures
        if 'configure' in phase1_info:
            self.configures_requested_sig.put(
                self.configures_requested_sig.get() + 1
            )

        if wait:
            status.wait()
        return status

    def _transition_thread(
        self,
        state: str,
        phase1_info: dict[str, Any],
        status: Status,
    ) -> None:
        """
        Do the raw setState command.

        This is intended for use in a background thread because setState
        can block. A method is added here because we'd like to keep
        track of the return value of setState, which is an error message.

        Parameters
        ----------
        state : str
            A state name that psdaq is expecting.
        phase1_info : dict[str, Any]
            A dictionary that maps transition names to extra data that the
            DAQ can use.
        status : Status
            The status returned by _state_transition, so that we can
            mark it as failed if there is a problem here.
        """
        logger.debug(
            "DaqLCLS2._transition_thread(state=%s, phase1_info=%s)",
            state,
            phase1_info,
        )
        error_msg = self._control.setState(state, phase1_info)
        self.last_transition_err_sig.put(error_msg)
        if error_msg is not None:
            logger.debug("Setting exception in transition thread")
            status.set_exception(
                DaqStateTransitionError(
                    f'Error transitioning to {state}: {error_msg}'
                )
            )

    def _handle_duration_thread(
        self,
        duration: float,
        running_status: Status,
    ) -> None:
        """
        Wait a fixed amount of time, then stop the daq.

        The LCLS1 DAQ supported a duration argument that allowed us to
        request fixed-length runs instead of fixed-events runs.
        This is used to emulate that behavior.

        This avoids desynchronous behavior like starting the DAQ again
        at an inappropriate time after a cancelled run by ending early
        if the DAQ stops by any other means.

        Parameters
        ----------
        duration : float
            The amount of time to wait in seconds.
        running_status : Status
            The status returned by _state_transition, so that we can
            start the timer appropriately.
            Note: this is the state transition to "running"
        """
        logger.debug(
            "DaqLCLS2._handle_duration_thread(duration=%s)",
            duration,
        )
        try:
            running_status.wait()
        except Exception:
            logger.debug("Never made it to running, abort duration thread")
            return

        end_status = self._get_status_for(
            state=['starting'],
            transition=['endstep'],
            check_now=False,
        )

        # Handle timeouts and waits ourselves to avoid annoying ophyd message
        end_event = threading.Event()

        def done(*args, **kwargs):
            end_event.set()

        end_status.add_callback(done)
        end_event.wait(timeout=duration)

        if end_status.done:
            logger.debug("Duration thread ending, DAQ already stopped.")
        else:
            logger.debug("Duration thread expired, stopping the DAQ.")
            try:
                end_status.set_finished()
            except Exception:
                pass
            # Time to stop the DAQ
            self._state_transition(
                'starting',
                wait=True,
                timeout=self.begin_timeout_cfg.get(),
            )

    def _get_phase1(self, transition: str) -> dict[str, Any]:
        """
        For a given transition, get the extra data we need to send to the DAQ.

        This currently adds a hex data block for Configure and BeginStep
        transitions, and is built using a number of our configuration
        values.

        Parameters
        ----------
        transition : str
            The name of the transition from
            psdaq.controls.ControlDef.transitionId

        Returns
        -------
        phase1_info : dict[str, Any]
            The data to send to the DAQ.
        """
        logger.debug("DaqLCLS2._get_phase1(transition=%s)", transition)
        if transition == 'Configure':
            phase1_key = 'NamesBlockHex'
        elif transition == 'BeginStep':
            phase1_key = 'ShapesDataBlockHex'
        else:
            raise RuntimeError('Only Configure and BeginStep are supported.')

        data = {
            'motors': self._get_motors_for_transition(),
            'timestamp': 0,
            'detname': self.detname_cfg.get(),
            'dettype': 'scan',
            'scantype': self.scantype_cfg.get(),
            'serial_number': self.serial_number_cfg.get(),
            'alg_name': self.alg_name_cfg.get(),
            'alg_version': self.alg_version_cfg.get(),
        }
        logger.debug(
            'Assembling block for transition=%s, data=%s',
            transition,
            data,
        )
        try:
            data['transitionid'] = ControlDef.transitionId[transition]
        except KeyError as exc:
            raise RuntimeError(f'Invalid transition {transition}') from exc

        if transition == "Configure":
            data["add_names"] = True
            data["add_shapes_data"] = False
        else:
            data["add_names"] = False
            data["add_shapes_data"] = True

        data["namesid"] = ControlDef.STEPINFO
        block = self._control.getBlock(data=data)
        return {phase1_key: block}

    def _get_motors_for_transition(self) -> dict[str, Any]:
        """
        Return the appropriate values for the DAQ's "motors" data.

        This is similar to the controls from the LCLS1 DAQ.
        It includes supplementary positional data from configured beamline
        devices, as well as the DAQ step.

        Returns
        -------
        motors : dict[str, Any]
            Raw key-value pairs that the DAQ is expecting.
            These represent the name of a value as will be recorded along with
            the data stream as well as the corresponding value itself.
        """
        controls = self.controls_cfg.get()

        if not isinstance(controls, (list, tuple)):
            raise RuntimeError(
                f'Expected controls={controls} to be list or tuple'
            )

        # Always includes a step value, and let the user override it
        step_value = self.step_value_sig.get()

        for ctrl in controls:
            name = get_controls_name(ctrl)
            if name == ControlDef.STEP_VALUE:
                step_value = get_controls_value(ctrl)

        data = {
            'step_value': step_value,
            'step_docstring': (
                f'{{"detname": "{self.detname_cfg.get()}", }}'
                f'{{"scantype": "{self.scantype_cfg.get()}", }}'
                f'{{"step": {step_value}}}'
            )
        }

        # Add all the other controls/motors
        for ctrl in controls:
            name = get_controls_name(ctrl)
            if name != ControlDef.STEP_VALUE:
                data[name] = get_controls_value(ctrl)

        return data

    def begin(
        self,
        wait: bool = False,
        end_run: bool = False,
        events: int | None | Sentinel = CONFIG_VAL,
        duration: Real | None | Sentinel = CONFIG_VAL,
        record: bool | TernaryBool | None | Sentinel = CONFIG_VAL,
        controls: ControlsArg | None | Sentinel = CONFIG_VAL,
        motors: ControlsArg | None | Sentinel = CONFIG_VAL,
        begin_timeout: Real | None | Sentinel = CONFIG_VAL,
        begin_sleep: Real | None | Sentinel = CONFIG_VAL,
        group_mask: int | None | Sentinel = CONFIG_VAL,
        detname: str | None | Sentinel = CONFIG_VAL,
        scantype: str | None | Sentinel = CONFIG_VAL,
        serial_number: str | None | Sentinel = CONFIG_VAL,
        alg_name: str | None | Sentinel = CONFIG_VAL,
        alg_version: list[int] | None | Sentinel = CONFIG_VAL,
    ) -> None:
        """
        Interactive starting of the DAQ.

        All kwargs are passed through to configure as appropriate and are
        reverted afterwards.

        Parameters
        ----------
        wait : bool, optional
            If True, wait for the daq to stop.
        end_run : bool, optional
            If True, end the run when the daq stops.
        events : int or None, optional
            The number of events to take per step. Incompatible with the
            "duration" argument. Defaults to 0, and having both events
            and duration configured to 0 gives us an endless run (that
            can be terminated manually).
            Supplying an argument to "events" will reset "duration" to 0.
            If events is 0 and duration is nonzero, events will be ignored.
        duration : int, float, or None, optional
            How long to acquire data at each step in seconds.
            Incompatible with the "events" argument. Defaults to 0,
            and having both events and duration configured to 0 gives us
            an endless run (that can be terminated manually). Supplying
            an argument to "duration" will reset "events" to 0.
            If duration is 0 and events is nonzero, duration will be
            ignored.
        record : bool or None, optional
            Whether or not to save data during the DAQ run. Defaults to
            "None", which means that we'll keep the DAQ's recording
            state at whatever it is at the start of the run.
            Changing the DAQ recording state cannot be done during a run,
            as it will require a configure transition.
        controls : list or tuple of valid objects, or None, optional
            The objects to include per-step in the DAQ data stream.
            These must implement the "name" attribute and either the
            "position" attribute or the "get" method to retrieve their
            current value. To enforce an alternate name, you can pass a tuple
            instead of an object where the first element of the tuple is
            the replacement name. The tuple syntax can also be used to send
            primitive constants to the DAQ if the constant is an int, float,
            or str.
            If None or empty, we'll only include the default DAQ step counter,
            which will always be included.
        motors : list or dict of signals or positioners, or None, optional
            Alias of "controls" for backwards compatibility.
        begin_timeout : float or None, optional
            How long to wait before marking a begin run as a failure and
            raising an exception.
        begin_sleep : float or None, optional
            How long to wait before starting a run.
        group_mask : int or None, optional
            Bitmask that is used by the DAQ. This docstring writer is not
            sure exactly what it does. The default is all zeroes with a
            "1" bitshifted left by the platform number.
        detname : str or None, optional
            The name associated with the controls data in the DAQ.
            Defaults to "scan".
        scantype : str or None, optional
            Another string associated with the runs produced by this
            object in the DAQ. Defaults to "scan".
        serial_number : str or None, optional
            Another string associated with the runs produced by this
            object in the DAQ. Defaults to "1234".
        alg_name : str or None, optional
            Another string associated with the runs produced by this
            object in the DAQ. Defaults to "raw".
        alg_version : list of int, or None, optional
            The version numbers [major, minor, bugfix] associated with
            alg_name. Defaults to [1, 0, 0].
        """
        logger.debug(
            "DaqLCLS2.begin(wait=%s, end_run=%s, "
            "events=%s, duration=%s, record=%s, controls=%s, motors=%s, "
            "begin_timeout=%s, begin_sleep=%s, group_mask=%s, detname=%s, "
            "scantype=%s, serial_number=%s, alg_name=%s, alg_version=%s, "
            ")",
            wait,
            end_run,
            events,
            duration,
            record,
            controls,
            motors,
            begin_timeout,
            begin_sleep,
            group_mask,
            detname,
            scantype,
            serial_number,
            alg_name,
            alg_version,
        )
        super().begin(
            wait=wait,
            end_run=end_run,
            events=events,
            duration=duration,
            record=record,
            controls=controls,
            motors=motors,
            begin_timeout=begin_timeout,
            begin_sleep=begin_sleep,
            group_mask=group_mask,
            detname=detname,
            scantype=scantype,
            serial_number=serial_number,
            alg_name=alg_name,
            alg_version=alg_version,
        )

    def _end_run_callback(self, status: Status) -> None:
        """
        Callback for a status to end the run once the status completes.

        The status parameter is unused, but is passed in as self by
        the Status when this method is called.

        Regardless of the input, this will end the run.
        """
        logger.debug("DaqLCLS2._end_run_callback(status=%s)", status)
        self.end_run()

    def begin_infinite(self, **kwargs) -> None:
        """
        Start the DAQ in such a way that it runs until asked to stop.

        This is a shortcut included so that the user does not have to remember
        the specifics of how to get the daq to run indefinitely.

        kwargs are passed directly to begin, except for events and duration
        which cannot be specified here. These arguments will be ignored, as
        they need to be specified in a specific way to make the DAQ run
        infinitely.
        """
        logger.debug("DaqLCLS2.begin_infinite(kwargs=%s)", kwargs)
        kwargs['events'] = 0
        kwargs.pop('duration', None)
        self.begin(**kwargs)

    @property
    def _infinite_run(self) -> bool:
        """
        True if the DAQ is configured to run forever.
        """
        return self.events_cfg.get() == 0 and self.duration_cfg.get() == 0

    def stop(
        self,
        success: bool = False,
        timeout: float = 10.0,
        wait: bool = True,
    ) -> None:
        """
        Stop the current acquisition, ending it early.

        Parameters
        ----------
        success : bool, optional
            Flag set by bluesky to signify whether this was a good stop or a
            bad stop. Currently unused.
        timeout : float, optional
            How long before we consider the state transition to be failed
            in seconds. Defaults to a 10-second timeout so that it won't
            pause the thread forever.
        wait : bool, optional
            Wait for the transition to complete. Defaults to True.
            This also allows us to raise an exception if there is a
            transition error.
        """
        logger.debug("DaqLCLS2.stop(success=%s)", success)
        if self.state_sig.get() > self.state_enum.starting:
            self._state_transition('starting', timeout=timeout, wait=wait)

    def end_run(
        self,
        timeout: float = 10.0,
        wait: bool = True,
    ) -> None:
        """
        End the current run. This includes a stop if needed.

        Parameters
        ----------
        timeout : float, optional
            How long before we consider the state transition to be failed
            in seconds. Defaults to a 10-second timeout so that it won't
            pause the thread forever.
        wait : bool, optional
            Wait for the transition to complete. Defaults to True.
            This also allows us to raise an exception if there is a
            transition error.
        """
        logger.debug("DaqLCLS2.end_run()")
        if self.state_sig.get() > self.state_enum.configured:
            self._state_transition('configured', timeout=timeout, wait=wait)

    def trigger(self) -> Status:
        """
        Begin acquisition.

        Returns a status object that will be marked done when the daq has
        stopped acquiring.

        The status object will alternatively be marked done immediately if
        the DAQ is configured to run forever. This is so that the infinite-run
        behavior can be used in scans in conjunction with other time-bound
        triggers without freezing the scan indefinitely.

        This will raise a RuntimeError if the daq was never configured for
        events or duration.

        Returns
        -------
        done_status: ``Status``
            ``Status`` that will be marked as done when the daq has begun.
            This may fail with a DaqStateTransitionError or with a
            StatusTimeoutError as appropriate.
        """
        logger.debug("DaqLCLS2.trigger()")
        trigger_status = self._get_status_for(
            state=['starting'],
            transition=['endstep'],
            check_now=False,
            timeout=self.begin_timeout_cfg.get(),
        )

        def check_kickoff_fail(st: Status):
            if not st.success:
                try:
                    trigger_status.exception(st.exception())
                except InvalidState:
                    ...

        self.kickoff().add_callback(check_kickoff_fail)

        if self._infinite_run:
            logger.debug("Infinite run, setting status to finsihed.")
            try:
                trigger_status.set_finished()
            except InvalidState:
                ...
        return trigger_status

    def kickoff(self, **kwargs) -> Status:
        """
        Begin acquisition. This method is non-blocking.

        This will transition us into the "running" state, as long as we
        are connected or configured and not already running. In these
        cases we will raise a RuntimeError.

        This will cause the "configure", "beginrun", "beginstep", and "enable"
        transitions as needed, depending on which state we are starting from.

        This will also apply any defered configures if needed via calling
        "configure", even if no kwargs were passed.

        This is part of the ``bluesky`` ``Flyer`` interface.

        Parameters
        ----------
        kwargs : configure-compatible arguments
            These arguments are the last chance to configure the DAQ prior
            to starting the run. These will be reverted after the step,
            so that new kwargs can be provided in the next step.
            Note that changing the "record" state in the middle of a run
            is not allowed.

        Returns
        -------
        ready_status : ``Status``
            ``Status`` that will be marked as done when the daq has begun,
            or will have an exception applied if the state transition
            fails or times out. This will either be a
            DaqStateTransitionError or a StatusTimeoutError
        """
        logger.debug("DaqLCLS2.kickoff()")
        if self.state_sig.get() < self.state_enum.connected:
            raise RuntimeError('DAQ is not ready to run!')
        if self.state_sig.get() == self.state_enum.running:
            raise RuntimeError('DAQ is already running!')

        original_config = self.config

        def revert_cfg_after_step(status):
            self.preconfig(show_queued_cfg=False, **original_config)

        self.configure(**kwargs)
        end_run_status = self._get_status_for(
            transition=['endstep'],
            check_now=False,
        )
        kickoff_status = self._state_transition(
            'running',
            timeout=self.begin_timeout_cfg.get(),
            wait=False,
        )
        end_run_status.add_callback(revert_cfg_after_step)
        return kickoff_status

    @step_done_sig.sub_value
    def _step_ender(self, value: bool, **kwargs):
        """
        Formally end the step via asking for a transition back to "starting".

        Called whenenever a step is completed (e.g. we took enough events).
        """
        if value and self.state_sig.get() == self.state_enum.running:
            # This handles its own errors well enough
            # See _transition_thread
            self._state_transition(
                'starting',
                timeout=self.begin_timeout_cfg.get(),
                wait=False,
            )

    def complete(self) -> Status:
        """
        If the daq is freely running, this will `stop` the daq.
        Otherwise, we'll simply return the end_status object.

        Returns
        -------
        end_status: ``Status``
            ``Status`` that will be marked as done when the DAQ has finished
            acquiring
        """
        logger.debug("DaqLCLS2.complete()")
        done_status = self._get_done_status(check_now=True)
        if self._infinite_run:
            # Configured to run forever
            self.stop()
        return done_status

    def _enforce_config(self, name, value):
        """
        Raises a TypeError if the config argument has the wrong type.

        This is implemented by inspecting the type hint associated with
        the name parameter and comparing it with the type of the input
        value.

        Parameters
        ----------
        name : str
            The keyword-argument that must be passed in to "configure"
            or "preconfig" associated with value.
        value : Any
            The actual value that was passed into "configure" or "preconfig".
        """
        hint = get_type_hints(self.preconfig)[name]
        if not typing_check(value, hint):
            raise TypeError(
                f'Incorrect type for {name}={value}, expected {hint} '
                f'but got {type(value)}'
            )

    def preconfig(
        self,
        events: Union[int, None, Sentinel] = CONFIG_VAL,
        duration: Union[Real, None, Sentinel] = CONFIG_VAL,
        record: Union[bool, TernaryBool, None, Sentinel] = CONFIG_VAL,
        controls: Union[ControlsArg, None, Sentinel] = CONFIG_VAL,
        motors: Union[ControlsArg, None, Sentinel] = CONFIG_VAL,
        begin_timeout: Union[Real, None, Sentinel] = CONFIG_VAL,
        begin_sleep: Union[Real, None, Sentinel] = CONFIG_VAL,
        begin_throttle: Union[Real, None, Sentinel] = CONFIG_VAL,
        group_mask: Union[int, None, Sentinel] = CONFIG_VAL,
        detname: Union[str, None, Sentinel] = CONFIG_VAL,
        scantype: Union[str, None, Sentinel] = CONFIG_VAL,
        serial_number: Union[str, None, Sentinel] = CONFIG_VAL,
        alg_name: Union[str, None, Sentinel] = CONFIG_VAL,
        alg_version: Union[list[int], None, Sentinel] = CONFIG_VAL,
        show_queued_cfg: bool = True,
    ) -> None:
        """
        Adjust the configuration without causing a configure transition.

        This may be preferable over "configure" for interactive use for
        two reasons:
        1. A nice message is displayed instead of a return tuple of
           two dictionaries
        2. No real change happens to the DAQ when this method is called,
           at most this may schedule a configure transition for later.

        The behavior here is similar to putting to the cfg PVs, except
        here we add type checking and config printouts.

        This is called internally during "configure".

        Arguments that are not provided are not changed.
        Arguments that are passed as "None" will return to their
        default values.

        Parameters
        ----------
        events : int or None, optional
            The number of events to take per step. Incompatible with the
            "duration" argument. Defaults to "None", and running without
            configuring events or duration gives us an endless run (that
            can be terminated manually). Supplying an argument to "events"
            will reset "duration" to "None".
        duration : float or None, optional
            How long to acquire data at each step in seconds.
            Incompatible with the "events" argument. Defaults to "None",
            and running without configuring events or duration dives us
            an endless run (that can be terminated manually). Supplying
            an argument to "duration" will reset "events" to "None".
        record : bool or None, optional
            Whether or not to save data during the DAQ run. Defaults to
            "None", which means that we'll keep the DAQ's recording
            state at whatever it is at the start of the run.
            Changing the DAQ recording state cannot be done during a run,
            as it will require a configure transition.
        controls : list or tuple of valid objects, or None, optional
            The objects to include per-step in the DAQ data stream.
            These must implement the "name" attribute and either the
            "position" attribute or the "get" method to retrieve their
            current value. To enforce an alternate name, you can pass a tuple
            instead of an object where the first element of the tuple is
            the replacement name. The tuple syntax can also be used to send
            primitive constants to the DAQ if the constant is an int, float,
            or str.
            If None or empty, we'll only include the default DAQ step counter,
            which will always be included.
        motors : list or dict of signals or positioners, or None, optional
            Alias of "controls" for backwards compatibility.
        begin_timeout : float or None, optional
            How long to wait before marking a begin run as a failure and
            raising an exception.
        begin_sleep : float or None, optional
            How long to wait before starting a run.
        begin_throttle : float or None, optional
            Not implemented for LCLS2, would be a mandatory sleep between
            runs to avoid breaking the DAQ. Since this isn't implemented,
            it's only included here for internal API matching and isn't
            duplicated in the signatures for begin or configure.
        group_mask : int or None, optional
            Bitmask that is used by the DAQ. The default is all zeroes with
            a "1" bitshifted left by the platform number.
        detname : str or None, optional
            The name associated with the controls data in the DAQ.
            Defaults to "scan".
        scantype : str or None, optional
            Another string associated with the runs produced by this
            object in the DAQ. Defaults to "scan".
        serial_number : str or None, optional
            Another string associated with the runs produced by this
            object in the DAQ. Defaults to "1234".
        alg_name : str or None, optional
            Another string associated with the runs produced by this
            object in the DAQ. Defaults to "raw".
        alg_version : list of int, or None, optional
            The version numbers [major, minor, bugfix] associated with
            alg_name. Defaults to [1, 0, 0].
        show_queued_cfg: bool, optional
            If True, we'll show what the next configuration will be
            as a nice log message.
        """
        logger.debug(
            "DaqLCLS2.preconfig("
            "events=%s, duration=%s, record=%s, controls=%s, motors=%s, "
            "begin_timeout=%s, begin_sleep=%s, group_mask=%s, detname=%s, "
            "scantype=%s, serial_number=%s, alg_name=%s, alg_version=%s, "
            "show_queued_cfg=%s"
            ")",
            events,
            duration,
            record,
            controls,
            motors,
            begin_timeout,
            begin_sleep,
            group_mask,
            detname,
            scantype,
            serial_number,
            alg_name,
            alg_version,
            show_queued_cfg,
        )
        self._enforce_config('events', events)
        self._enforce_config('duration', duration)
        self._enforce_config('record', record)
        self._enforce_config('controls', controls)
        self._enforce_config('motors', motors)
        self._enforce_config('begin_timeout', begin_timeout)
        self._enforce_config('begin_sleep', begin_sleep)
        self._enforce_config('group_mask', group_mask)
        self._enforce_config('detname', detname)
        self._enforce_config('scantype', scantype)
        self._enforce_config('serial_number', serial_number)
        self._enforce_config('alg_name', alg_name)
        self._enforce_config('alg_version', alg_version)

        # Enforce only events or duration, not both
        if isinstance(events, int):
            duration = 0
        elif isinstance(duration, Real):
            duration = float(duration)
            events = 0
        # Handle motors as an alias for controls
        if not isinstance(motors, Sentinel):
            controls = motors
        # Call super
        super().preconfig(
            events=events,
            duration=duration,
            record=record,
            controls=controls,
            begin_timeout=begin_timeout,
            begin_sleep=begin_sleep,
            group_mask=group_mask,
            detname=detname,
            scantype=scantype,
            serial_number=serial_number,
            alg_name=alg_name,
            alg_version=alg_version,
            show_queued_cfg=show_queued_cfg,
        )

    def configure(
        self,
        events: int | None | Sentinel = CONFIG_VAL,
        duration: Real | None | Sentinel = CONFIG_VAL,
        record: bool | TernaryBool | None | Sentinel = CONFIG_VAL,
        controls: ControlsArg | None | Sentinel = CONFIG_VAL,
        motors: ControlsArg | None | Sentinel = CONFIG_VAL,
        begin_timeout: Real | None | Sentinel = CONFIG_VAL,
        begin_sleep: Real | None | Sentinel = CONFIG_VAL,
        group_mask: int | None | Sentinel = CONFIG_VAL,
        detname: str | None | Sentinel = CONFIG_VAL,
        scantype: str | None | Sentinel = CONFIG_VAL,
        serial_number: str | None | Sentinel = CONFIG_VAL,
        alg_name: str | None | Sentinel = CONFIG_VAL,
        alg_version: list[int] | None | Sentinel = CONFIG_VAL,
    ):
        """
        Adjusts the configuration, causing a "configure" transition if needed.

        A "configure" transition will be caused in the following cases:
        1. We are in the "connected" state
        2. We are in the "configured" state but an important configuration
           parameter has been changed. In this case, we will revert to the
           "connected" state and then return to the "configured" state.
        3. We are in the "configured" state but the configure was caused by
           some other process, for example, the DAQ GUI or a different
           Python session.

        In all other states, this will raise a "RuntimeError" if it decides
        that a "configure" transition is needed.

        Arguments that are not provided are not changed.
        Arguments that are passed as "None" will return to their
        default values.

        Parameters
        ----------
        events : int or None, optional
            The number of events to take per step. Incompatible with the
            "duration" argument. Defaults to "None", and running without
            configuring events or duration gives us an endless run (that
            can be terminated manually). Supplying an argument to "events"
            will reset "duration" to "None".
        duration : float or None, optional
            How long to acquire data at each step in seconds.
            Incompatible with the "events" argument. Defaults to "None",
            and running without configuring events or duration dives us
            an endless run (that can be terminated manually). Supplying
            an argument to "duration" will reset "events" to "None".
        record : bool or None, optional
            Whether or not to save data during the DAQ run. Defaults to
            "None", which means that we'll keep the DAQ's recording
            state at whatever it is at the start of the run.
            Changing the DAQ recording state cannot be done during a run,
            but it will not require a configure transition.
        controls : list or tuple of valid objects, or None, optional
            The objects to include per-step in the DAQ data stream.
            These must implement the "name" attribute and either the
            "position" attribute or the "get" method to retrieve their
            current value. To enforce an alternate name, you can pass a tuple
            instead of an object where the first element of the tuple is
            the replacement name. The tuple syntax can also be used to send
            primitive constants to the DAQ if the constant is an int, float,
            or str.
            If None or empty, we'll only include the default DAQ step counter,
            which will always be included.
        motors : list or dict of signals or positioners, or None, optional
            Alias of "controls" for backwards compatibility.
        begin_timeout : float or None, optional
            How long to wait before marking a begin run as a failure and
            raising an exception.
        begin_sleep : float or None, optional
            How long to wait before starting a run.
        group_mask : int or None, optional
            Bitmask that is used by the DAQ. This docstring writer is not
            sure exactly what it does. The default is all zeroes with a
            "1" bitshifted left by the platform number.
        detname : str or None, optional
            The name associated with the controls data in the DAQ.
            Defaults to "scan".
        scantype : str or None, optional
            Another string associated with the runs produced by this
            object in the DAQ. Defaults to "scan".
        serial_number : str or None, optional
            Another string associated with the runs produced by this
            object in the DAQ. Defaults to "1234".
        alg_name : str or None, optional
            Another string associated with the runs produced by this
            object in the DAQ. Defaults to "raw".
        alg_version : list of int, or None, optional
            The version numbers [major, minor, bugfix] associated with
            alg_name. Defaults to [1, 0, 0].

        Returns
        -------
        (old, new): tuple[dict, dict]
            The configurations before and after the function was called.
            This is used internally by bluesky when we include
            "configure" in a plan.
        """
        logger.debug("DaqLCLS2.configure, passing to super")
        old, new = super().configure(
            events=events,
            duration=duration,
            record=record,
            controls=controls,
            motors=motors,
            begin_timeout=begin_timeout,
            begin_sleep=begin_sleep,
            group_mask=group_mask,
            detname=detname,
            scantype=scantype,
            serial_number=serial_number,
            alg_name=alg_name,
            alg_version=alg_version,
        )
        other_proc_configured = (
            self.configures_seen_sig.get()
            != self.configures_requested_sig.get()
        )
        first_configure = self.configures_requested_sig.get() == 0
        # Cause a transition if we need to
        if any((
            self._queue_configure_transition,
            other_proc_configured,
            first_configure,
        )):
            if self.state_sig.get() < self.state_enum.connected:
                raise RuntimeError('Not ready to configure.')
            if self.state_sig.get() > self.state_enum.configured:
                raise RuntimeError(
                    'Cannot configure transition during an open run!'
                )
            if other_proc_configured:
                # Reset the counters now, before we count this configure
                self.configures_seen_sig.put(0)
                self.configures_requested_sig.put(0)
            if self.state_sig.get() == self.state_enum.configured:
                # Already configured, so we should unconfigure first
                self._state_transition(
                    'connected',
                    timeout=self.begin_timeout_cfg.get(),
                    wait=True,
                )
            self._state_transition(
                'configured',
                timeout=self.begin_timeout_cfg.get(),
                wait=True,
            )
            self._last_config = self.config
            self._queue_configure_transition = False

        # We need to adjust the recording state if configured
        # to do so and if this represents a change
        rec_cfg = self.record_cfg.get()
        if rec_cfg is not TernaryBool.NONE:
            new_rec = bool(rec_cfg)
            if self.recording_sig.get() != new_rec:
                if self.state_sig.get() > self.state_enum.configured:
                    raise RuntimeError(
                        'Cannot change recording state during an open run!'
                    )
                self._set_record_state(bool(rec_cfg))
        return old, new

    @property
    def record(self) -> bool:
        """
        ``True`` if the run will be recorded, ``False`` otherwise.

        If record is configured to be ``None``, we'll use the value selected
        in the GUI in lieu of any values from the Python, and that boolean
        will be returned here.

        You can check what has been configured in the python by checking
        daq.config['record'] or daq.record_cfg.get().

        You can set record via daq.record = True, for example, or by
        using daq.preconfig or daq.configure.
        """
        cfg_record = self.record_cfg.get()
        if cfg_record is TernaryBool.NONE:
            return self.recording_sig.get()
        return cfg_record.to_primitive()

    @record.setter
    def record(self, record: bool | None) -> None:
        self.preconfig(record=record, show_queued_cfg=False)

    def _set_record_state(self, record: bool) -> None:
        """
        Explicitly modify the DAQ's recording state and increment the counter.
        """
        self._control.setRecord(record)
        self.configures_requested_sig.put(
            self.configures_seen_sig.get() + 1
        )

    def stage(self) -> list[DaqLCLS2]:
        """
        Extend stage to save the "recording" state for post-run restoration.
        """
        objs = super().stage()
        self._pre_run_record = self.recording_sig.get()
        return objs

    def unstage(self) -> list[DaqLCLS2]:
        """
        Restore the pre-run "recording" state, if we do change it
        """
        objs = super().unstage()
        if self.recording_sig.get() != self._pre_run_record:
            self._set_record_state(self._pre_run_record)
        return objs

    def pause(self, timeout: float = 10.0, wait: bool = True) -> None:
        """
        Interrupt an ongoing step, to be resumed later.

        This may be called during a scan if the user uses ctrl+c.
        This puts the DAQ into the "paused" state.

        This is a no-op if we're not in the "running" state.

        Parameters
        ----------
        timeout : float, optional
            How long before we consider the state transition to be failed
            in seconds. Defaults to a 10-second timeout so that it won't
            pause the thread forever.
        wait : bool, optional
            Wait for the transition to complete. Defaults to True.
            This also allows us to raise an exception if there is a
            transition error.
        """
        if self.state_sig.get() == self.state_enum.running:
            self._state_transition('paused', timeout=timeout, wait=wait)

    def resume(self, timeout: float = 10.0, wait: bool = True) -> None:
        """
        The inverse of pause: return to a previously ongoing step.

        This may be called during RE.resume() after the user pauses a
        scan with ctrl+c.

        If called at any other time, when a run is not paused,
        this will act as a call to kickoff().
        Semantically this is the difference between restarting the
        ongoing step and resuming it.

        This always puts the DAQ into the "running" state.

        Parameters
        ----------
        timeout : float, optional
            How long before we consider the state transition to be failed
            in seconds. Defaults to a 10-second timeout so that it won't
            pause the thread forever.
        wait : bool, optional
            Wait for the transition to complete. Defaults to True.
            This also allows us to raise an exception if there is a
            transition error.
        """
        if self.state_sig.get() == self.state_enum.paused:
            self._state_transition('running', timeout=timeout, wait=wait)
        elif self.state_sig.get() < self.state_enum.paused:
            self.kickoff().wait(timeout=timeout)

    def run_number(self) -> int:
        """
        Determine the run number of the last run, or current run if running.

        This is a method and not a property for consistency with the other
        DAQ interfaces, which need to do some extra processing to come up
        with this number.

        Returns
        -------
        run_number : int
        """
        return max(self.run_number_sig.get(), self.last_run_number_sig.get())

    def status_info(self) -> dict[str, Any]:
        """
        Overide the default status info to display enums as strs.
        """
        status = super().status_info()
        for val in status.values():
            try:
                val['value'] = val['value'].name
            except (AttributeError, KeyError, TypeError):
                pass
        return status


class SimDaqControl:
    """
    Emulation of DaqControl for basic offline tests.
    """
    _tmap = {
        'reset': {
            'unallocated': 'rollcall',
        },
        'unallocated': {
            'allocated': 'alloc',
        },
        'allocated': {
            'unallocated': 'dealloc',
            'connected': 'connect',
        },
        'connected': {
            'allocated': 'disconnect',
            'configured': 'configure',
        },
        'configured': {
            'connected': 'unconfigure',
            'starting': 'beginrun',
        },
        'starting': {
            'configured': 'endrun',
            'paused': 'beginstep',
        },
        'paused': {
            'starting': 'endstep',
            'running': 'enable',
        },
        'running': {
            'paused': 'disable',
        },
    }

    def __init__(self, *args, **kwargs):
        logger.debug("SimDaqControl.__init__(%s, %s)", args, kwargs)
        if None in (ControlDef, DaqControl):
            raise RuntimeError(
                'Optional dependency psdaq is not installed, '
                'cannot run lcls2 daq'
            )
        self._lock = threading.RLock()
        self._new_status = threading.Event()
        self._headers = HelpfulIntEnum(
            'ValidHeaders',
            ['status', 'error', 'warning', 'filereport', 'progress', 'step']
        )
        self._states = HelpfulIntEnum('States', ControlDef.states)
        self._transitions = HelpfulIntEnum('Trans', ControlDef.transitions)
        self._recording = False
        self._experiment_name = 'tst0000'
        self._run_number = 0
        self._last_run_number = 0
        self._config_alias = 'TST'
        self._bypass_activedet = False
        self._elapsed = 0
        self._total = 0
        self._step_done = False
        self._error = ''
        self._warning = ''
        self._path = 'tst'
        self._cause_error = False
        self.sim_set_states('reset', 'reset')

    def getStatus(self) -> tuple[str, str, str, str, str, str, str, str]:
        """Return the current status."""
        return (
            self._transition,
            self._state,
            self._config_alias,
            self._recording,
            {'platform': 'dictionary'},
            self._bypass_activedet,
            self._experiment_name,
            self._run_number,
            self._last_run_number,
        )

    def monitorStatus(self) -> tuple[str, str, str, str, str, str, str, str]:
        """Wait, then return the next updated status when it changes."""
        logger.debug('SimDaqControl.monitorStatus() requested')
        self._new_status.wait()
        with self._lock:
            logger.debug('SimDaqControl sending new status')
            self._new_status.clear()
            if self._header == self._headers.status:
                status = [
                    self._transition,
                    self._state,
                    self._config_alias,
                    self._recording,
                    self._bypass_activedet,
                    self._experiment_name,
                    self._run_number,
                    self._last_run_number,
                ]
            elif self._header == self._headers.error:
                status = ['error', self._error]
            elif self._header == self._headers.warning:
                status = ['warning', self._warning]
            elif self._header == self._headers.filereport:
                status = ['fileReport', self._path]
            elif self._header == self._headers.progress:
                status = [
                    'progress',
                    self._transition,
                    self._elapsed,
                    self._total,
                ]
            elif self._header == self._headers.step:
                status = ['step', self._step_done]
            else:
                raise RuntimeError('Error in sim, bad header')
        while len(status) < 8:
            status.append('error')
        return tuple(status)

    def setState(
        self,
        state: EnumId,
        phase1_info: dict[str, Any],
    ) -> str | None:
        """
        Request the needed transitions to get to state.

        This may also cause additional state transitions e.g. if we're
        doing a fixed-length run.

        Returns a str if there is an error.
        """
        logger.debug('SimDaqControl.setState(%s, %s)', state, phase1_info)
        with self._lock:
            state = self._states.from_any(state)
            if state == self._states.reset:
                return self.sim_transition('reset')

            now = self._states.from_any(self._state)
            if state == now:
                return
            if state.value > now.value:
                goal_indices = range(now.value + 1, state.value + 1)
            else:
                goal_indices = range(now.value - 1, state.value - 1, -1)
            for goal in goal_indices:
                error = self.sim_transition(goal)
                if error is not None:
                    return error

            if state == self._states.running:
                # We need to schedule end_step
                self._step_done = False
                try:
                    events = phase1_info['enable']['readout_count']
                except KeyError:
                    events = 0
                if events > 0:
                    threading.Thread(
                        target=self._end_step_thread,
                        args=(events,)
                    ).start()

    def _end_step_thread(self, events: int) -> None:
        """The DAQ should stop after the step's events elapse"""
        time.sleep(events/120)
        if self._state == 'running':
            logger.debug('SimDaqControl ending step')
            self._step_done = True
            self.sim_new_status(self._headers.step)

    def getBlock(
        self,
        data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get relevant phase1_info for setState.

        This won't emulate the real daq's behavior, it's just for feeding back
        into the sim setState.
        """
        return data

    def setRecord(self, record: bool) -> None:
        """Match API for changing the recording state and emit update."""
        logger.debug("SimDaqControl.setRecord(%s)", record)
        with self._lock:
            self._recording = record
            self.sim_new_status(self._headers.status)

    def sim_set_states(
        self,
        transition: EnumId,
        state: EnumId,
    ) -> str | None:
        """Change the currently set state and emit update."""
        logger.debug("SimDaqControl.sim_set_state(%s, %s)", transition, state)
        with self._lock:
            if self._cause_error:
                self._cause_error = False
                self.sim_new_status(self._headers.error)
                logger.debug("Sim returning error: %s", self._error)
                return self._error
            self._transition = self._transitions.from_any(transition).name
            self._state = self._states.from_any(state).name
            if self._transition == self._transitions.beginrun:
                self._run_number += 1
            elif self._transition == self._transitions.endrun:
                self._last_run_number += 1
            self.sim_new_status(self._headers.status)

    def sim_transition(self, state: EnumId) -> str | None:
        """Internal transition, checks if valid."""
        logger.debug("SimDaqControl.sim_transition(%s)", state)
        with self._lock:
            goal = self._states.from_any(state)
            if goal == self._states.reset:
                return self.sim_set_states('reset', 'reset')
            now = self._states.from_any(self._state)
            try:
                transition = self._tmap[now.name][goal.name]
            except KeyError:
                raise RuntimeError(f'Invalid transition from {now} to {goal}')
            error = self.sim_set_states(transition, goal.name)
            if error is not None:
                return error

    def sim_new_status(self, header: HelpfulIntEnum) -> None:
        """Emit a status update."""
        logger.debug("SimDaqControl.sim_new_status(%s)", header)
        with self._lock:
            self._header = header
            self._new_status.set()

    def sim_queue_error(self, message: str) -> None:
        """The next requested transition will error."""
        logger.debug("SimDaqControl.sim_queue_error(%s)", message)
        with self._lock:
            self._cause_error = True
            self._error = message
