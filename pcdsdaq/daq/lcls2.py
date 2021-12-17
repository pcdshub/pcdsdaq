"""
Module that defines the controls python interface for the LCLS2 DAQ.

This is the second such interface. The original interface by Chris Ford
is still distributed by the DAQ code at psdaq.control.BlueskyScan.

This updated interface was created to meet expectations about specifics
of what the interface should be as established in lcls1, largely for
convenience of the end user. This should also make things more uniform
between the lcls1 and the lcls2 usages of the DAQ.
"""
from __future__ import annotations

import logging
import threading
from functools import cache
from numbers import Real
from typing import Any, Iterator, Optional, Union, get_type_hints

from bluesky import RunEngine
from ophyd.device import Component as Cpt
from ophyd.signal import Signal
from ophyd.status import DeviceStatus
from ophyd.utils import StatusTimeoutError, WaitTimeoutError
from ophyd.utils.errors import InvalidState

from .interface import (CONFIG_VAL, SENTINEL, ControlsArg, DaqBase, EnumId,
                        HelpfulIntEnum, get_controls_value, typing_check)

try:
    from psdaq.control.ControlDef import ControlDef
    from psdaq.control.DaqControl import DaqControl
except ImportError:
    # TODO reimplement this more like lcls1 to simplify sim injection
    DaqControl = None
    ControlDef = None

logger = logging.getLogger(__name__)
pydaq = None


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
    """
    # TODO double-check if timeout is int or float, what the units are,
    # and what the precise behavior is.
    step_value_sig = Cpt(Signal, value=1, kind='normal')
    state_sig = Cpt(Signal, value=None, kind='normal')
    transition_sig = Cpt(Signal, value=None, kind='normal')
    transition_elapsed_sig = Cpt(Signal, value=None, kind='normal')
    transition_total_sig = Cpt(Signal, value=None, kind='normal')
    config_alias_sig = Cpt(Signal, value=None, kind='normal')
    recording_sig = Cpt(Signal, value=None, kind='normal')
    bypass_activedet_sig = Cpt(Signal, value=None, kind='normal')
    experiment_name_sig = Cpt(Signal, value=None, kind='normal')
    run_number_sig = Cpt(Signal, value=None, kind='normal')
    last_run_number_sig = Cpt(Signal, value=None, kind='normal')

    group_mask_cfg = Cpt(Signal, value=None, kind='config')
    detname_cfg = Cpt(Signal, value='scan', kind='config')
    scantype_cfg = Cpt(Signal, value='scan', kind='config')
    serial_number_cfg = Cpt(Signal, value='1234', kind='config')
    alg_name_cfg = Cpt(Signal, value='raw', kind='config')
    alg_version_cfg = Cpt(Signal, value=[1, 0, 0], kind='config')

    last_err_sig = Cpt(Signal, value=None, kind='omitted')
    last_warning_sig = Cpt(Signal, value=None, kind='omitted')
    last_file_report_sig = Cpt(Signal, value=None, kind='omitted')
    step_done_sig = Cpt(Signal, value=None, kind='omitted')
    last_transition_err_sig = Cpt(Signal, value=None, kind='omitted')

    requires_configure_transition = {'record'}

    def __init__(
        self,
        platform: int,
        host: str,
        timeout: int,
        RE: Optional[RunEngine] = None,
        hutch_name: Optional[str] = None,
    ):
        if DaqControl is None:
            raise RuntimeError('psdaq is not installed, cannot use LCLS2 daq')
        super().__init__(RE=RE, hutch_name=hutch_name, platform=platform)
        self.state_sig.put(self.state_enum['reset'])
        self.transition_sig.put(self.transition_enum['reset'])
        self.group_mask_cfg.put(1 << platform)
        self._control = DaqControl(
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

    # TODO double-check if these need the type qualifier in the hint
    # It depends on if they have access to the helpful methods or not
    # If they have the methods, the hint should just be HelpfulIntEnum
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
        threading.Thread(target=self._monitor_thread, args=()).start()

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
        while not self._destroyed:
            try:
                info = self._control.monitorStatus()
                if info[0] == 'error':
                    self.last_err_sig.put(info[1])
                elif info[0] == 'warning':
                    self.last_warning_sig.put(info[1])
                elif info[0] == 'fileReport':
                    self.last_file_report_sig.put(info[1])
                elif info[0] == 'progress':
                    self.transition_sig.put(
                        self.transition_enum[self.info[1]]
                    )
                    self.transition_elapsed_sig.put(info[2])
                    self.transition_total_sig.put(info[3])
                elif info[0] == 'step':
                    self.step_value_sig.put(self.step_value_sig.get() + 1)
                    self.step_done_sig.put(info[1])
                else:
                    # Last case is normal status
                    if info[0] == 'endrun':
                        self.step_value_sig.put(1)
                    self.transition_sig.put(
                        self.transition_enum[info[0]]
                    )
                    self.state_sig.put(
                        self.state_enum[info[1]]
                    )
                    self.config_alias_sig.put(info[2])
                    self.recording_sig.put(info[3])
                    self.bypass_activedet_sig.put(info[4])
                    self.experiment_name_sig.put(info[5])
                    self.run_number_sig.put(info[6])
                    self.last_run_number_sig.put(info[7])
            except Exception:
                ...

    @state_sig.sub_value
    def _configured_cb(
        self,
        value: Optional[HelpfulIntEnum],
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
                value >= self.state_enum['configured']
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
        timeout: Optional[float] = None,
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
        done_status = self.get_done_status(timeout=timeout, check_now=True)
        done_status.wait()
        if end_run:
            self.end_run()

    def get_status_for(
        self,
        state: Optional[Iterator[EnumId]] = None,
        transition: Optional[Iterator[EnumId]] = None,
        timeout: Optional[float] = None,
        check_now: bool = True,
    ) -> DeviceStatus:
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
        status : DeviceStatus
            A status that will be marked successful after the corresponding
            states or transitions are reached.
        """
        if state is None:
            state = {None}
        else:
            state = {self.state_enum.from_any(s) for s in state}
        if transition is None:
            transition = {None}
        else:
            transition = {
                self.transition_enum.from_any(t) for t in transition
            }

        def check_state(value: Any, old_value: Any, **kwargs) -> None:
            """Call success if this value and last transition are correct."""
            nonlocal last_state
            if value == old_value and not check_now:
                return
            with lock:
                if value in state and last_transition in transition:
                    success()
                else:
                    last_state = value

        def check_transition(value: Any, old_value: Any, **kwargs) -> None:
            """Call success if this value and last state are correct."""
            nonlocal last_transition
            if value == old_value and not check_now:
                return
            with lock:
                if value in transition and last_state in state:
                    success()
                else:
                    last_transition = value

        def success() -> None:
            """Set the status as successfully finished if needed."""
            try:
                status.set_finished()
            except InvalidState:
                ...

        def clean_up(status: DeviceStatus) -> None:
            """
            Undo the subscriptions once the status is done.

            Runs on successes, failures, and timeouts.
            """
            self.state_sig.unsubscribe(state_cbid)
            self.transition_sig.unsubscribe(transition_cbid)

        last_state = None
        last_transition = None
        lock = threading.Lock()
        status = DeviceStatus(self, timeout=timeout)
        state_cbid = self.state_sig.subscribe(
            check_state,
            run=check_now,
        )
        transition_cbid = self.transition_sig.subscribe(
            check_transition,
            run=check_now,
        )
        status.add_callback(clean_up)
        return status

    def get_done_status(
        self,
        timeout: Optional[float] = None,
        check_now: bool = True,
    ) -> DeviceStatus:
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
        done_status : DeviceStatus
            A status that is marked successful once the DAQ is done
            acquiring.
        """
        return self.get_status_for(
            transition=self.transition_enum.exclude(
                ['beginrun', 'beginstep', 'enable']
            ),
            timeout=timeout,
            check_now=check_now,
        )

    def state_transition(
        self,
        state: EnumId,
        timeout: Optional[float] = None,
        wait: bool = True,
    ) -> DeviceStatus:
        """
        Cause a daq state transition appropriately.

        This passes extra data if we need to do the 'configure' or 'beginstep'
        transitions.

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
        transition_status : DeviceStatus
            A status object that is marked done when the transition has
            completed.
        """
        # Normalize state
        state = self.state_enum.from_any(state)
        # Determine what extra info to send to the DAQ
        phase1_info = {}
        if (
            self.state_sig.get()
            < self.state_enum['configure']
            <= state
        ):
            # configure transition
            phase1_info['configure'] = self._get_phase1('Configure')
        if (
            self.state_sig.get()
            < self.state_enum['starting']
            <= state
        ):
            # beginstep transition
            phase1_info['beginstep'] = self._get_phase1('BeginStep')
        if (
            self.state_sig.get()
            < self.state_enum['running']
            <= state
        ):
            # enable transition:
            phase1_info['enable'] = {
                # this is the event count, 0 means run forever
                'readout_count': self.events_cfg.get() or 0,
                'group_mask': self.group_mask_cfg.get(),
            }
        # Get a status to track the transition's success or failure
        status = self.get_status_for(
            state=[state],
            timeout=timeout,
        )
        # Set the transition in background thread, can be blocking
        threading.Thread(
            self._transition_thread,
            args=(state.name, phase1_info),
        ).start()
        # Handle duration ourselves in another thread for LCLS1 compat
        if (
            state == self.state_enum['running']
            and self.events_cfg.get() is None
            and self.duration_cfg.get() is not None
        ):
            threading.Thread(
                self._handle_duration_thread,
                args=(self.duration_cfg.get(), status)
            ).start()

        if wait:
            status.wait()
        return status

    def _transition_thread(
        self,
        state: str,
        phase1_info: dict[str, Any],
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
        """
        error_msg = self._control.setState(state, phase1_info)
        self.last_transition_err_sig.put(error_msg)

    # TODO could this be implemented better?
    def _handle_duration_thread(self, duration: float) -> None:
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
        """
        end_status = self.get_status_for(
            state=['starting'],
            transition=['endstep'],
            timeout=duration,
            check_now=False,
        )
        try:
            # If this succeeds, someone else stopped the DAQ
            # So in success, nothing to do
            end_status.wait()
        except (StatusTimeoutError, WaitTimeoutError):
            # The error means our wait expired
            # Time to stop the DAQ
            self.state_transition(
                'starting',
                wait=True,
                timeout=self.begin_timeout_cfg.get(),
            )

    def _get_phase1(self, transition: str) -> dict[str, any]:
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
        block = self._control.getBlock(transition=transition, data=data)
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

        # Always includes a step value, and let the user override it
        step_value = self.step_value_sig.get()

        if isinstance(controls, dict):
            try:
                step_value = get_controls_value(
                    controls[ControlDef.STEP_VALUE]
                )
            except KeyError:
                ...
        elif isinstance(controls, (list, tuple)):
            for ctrl in controls:
                if ctrl.name == ControlDef.STEP_VALUE:
                    step_value = get_controls_value(ctrl)
        elif controls is not None:
            raise RuntimeError(
                f'Expected controls={controls} to be dict, list, or None'
            )

        data = {
            'step_value': step_value,
            'step_docstring': (
                f'{{"detname": "{self.detname_cfg.get()}", }}'
                f'{{"scantype": "{self.scantype_cfg.get()}", }}'
                f'{{"step": {step_value}}}'
            )
        }

        # Add all the other controls/motors
        if isinstance(controls, dict):
            for key, ctrl in controls.items():
                if key != ControlDef.STEP_VALUE:
                    data[key] = get_controls_value(ctrl)
        if isinstance(controls, list):
            for ctrl in controls:
                if ctrl.name != ControlDef.STEP_VALUE:
                    data[ctrl.name] = get_controls_value(ctrl)

        return data

    # TODO refactor to show all the available arguments from configure
    def begin(
        self,
        wait: bool = False,
        end_run: bool = False,
        **kwargs,
    ) -> None:
        """
        Interactive starting of the DAQ.

        All kwargs are passed through to configure as appropriate and are
        reverted afterwards.

        Parameters
        ----------
        wait: bool, optional
            If True, wait for the daq to stop.
        end_run: bool, optional
            If True, end the run when the daq stops.
        """
        original_config = self.config
        self.preconfig(show_queued_cfg=False, **kwargs)
        super().begin(wait=wait, end_run=end_run)
        self.preconfig(show_queued_cfg=False, **original_config)

    def _end_run_callback(self, status: DeviceStatus) -> None:
        """
        Callback for a status to end the run once the status completes.

        The status parameter is unused, but is passed in as self by
        the DeviceStatus when this method is called.

        Regardless of the input, this will end the run.
        """
        self.end_run()

    # TODO decide if this needs more kwargs
    def begin_infinite(self) -> None:
        """
        Start the DAQ in such a way that it runs until asked to stop.

        This is a shortcut included so that the user does not have to remember
        the specifics of how to get the daq to run indefinitely.
        """
        self.begin(events=0)

    @property
    def _infinite_run(self) -> bool:
        """
        True if the DAQ is configured to run forever.
        """
        return self.events_cfg.get() == 0

    def stop(self, success: bool = False) -> None:
        """
        Stop the current acquisition, ending it early.

        Parameters
        ----------
        success : bool, optional
            Flag set by bluesky to signify whether this was a good stop or a
            bad stop. Currently unused.
        """
        if self.state_sig.get() > self.state_enum['starting']:
            self.state_transition('starting', wait=False)

    def end_run(self) -> None:
        """
        End the current run. This includes a stop if needed.
        """
        if self.state_sig.get() > self.state_enum['configured']:
            self.state_transition('configured', wait=False)

    def trigger(self) -> DeviceStatus:
        """
        Begin acquisition.

        Returns a status object that will be marked done when the daq has
        stopped acquiring.

        This will raise a RuntimeError if the daq was never configured for
        events or duration.

        Returns
        -------
        done_status: ``DeviceStatus``
            ``DeviceStatus`` that will be marked as done when the daq is done.
        """
        status = self.get_status_for(
            state=['starting'],
            transition=['endstep'],
            check_now=False,
            timeout=self.begin_timeout_cfg.get(),
        )
        self.kickoff()
        return status

    # TODO make sure this configures appropriately, it might not.
    # For example, if a reconfig is needed but we're in the "configured"
    # state or higher and not the "connected" state.
    def kickoff(self) -> DeviceStatus:
        """
        Begin acquisition. This method is non-blocking.

        This will transition us into the "running" state, as long as we
        are connected or configured and not already running. In these
        cases we will raise a RuntimeError.

        This will cause the "configure", "beginrun", "beginstep", and "enable"
        transitions as needed, depending on which state we are starting from.

        This is part of the ``bluesky`` ``Flyer`` interface.

        Returns
        -------
        ready_status: ``Status``
            ``Status`` that will be marked as done when the daq has begun.
        """
        if self.state_sig.get() < self.state_enum['connected']:
            raise RuntimeError('DAQ is not ready to run!')
        if self.state_sig.get() == self.state_enum['running']:
            raise RuntimeError('DAQ is already running!')
        return self.state_transition(
            'running',
            timeout=self.begin_timeout_cfg.get(),
            wait=False,
        )

    def complete(self) -> DeviceStatus:
        """
        If the daq is freely running, this will `stop` the daq.
        Otherwise, we'll simply return the end_status object.

        Returns
        -------
        end_status: ``Status``
            ``Status`` that will be marked as done when the DAQ has finished
            acquiring
        """
        done_status = self.get_done_status(check_now=True)
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
                f'Incorrect type for {name}={value}, expected {hint}'
            )

    def preconfig(
        self,
        events: Union[int, None, SENTINEL] = CONFIG_VAL,
        duration: Union[Real, None, SENTINEL] = CONFIG_VAL,
        record: Union[bool, None, SENTINEL] = CONFIG_VAL,
        controls: Union[ControlsArg, None, SENTINEL] = CONFIG_VAL,
        motors: Union[ControlsArg, None, SENTINEL] = CONFIG_VAL,
        begin_timeout: Union[Real, None, SENTINEL] = CONFIG_VAL,
        begin_sleep: Union[Real, None, SENTINEL] = CONFIG_VAL,
        group_mask: Union[int, None, SENTINEL] = CONFIG_VAL,
        detname: Union[str, None, SENTINEL] = CONFIG_VAL,
        scantype: Union[str, None, SENTINEL] = CONFIG_VAL,
        serial_number: Union[str, None, SENTINEL] = CONFIG_VAL,
        alg_name: Union[str, None, SENTINEL] = CONFIG_VAL,
        alg_version: Union[list[int], None, SENTINEL] = CONFIG_VAL,
        show_queued_config: bool = True,
    ) -> None:
        # TODO investigate dynamically populating docstrings with
        # values again for the BEGIN_TIMEOUT, etc.
        # TODO better docstring for group_mask
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
        controls : list or dict of signals or positioners, or None, optional
            The objects to include per-step in the DAQ data stream.
            These must implement either the "position" attribute or the
            "get" method to retrieve their current value.
            If a list, we'll use the object's "name" attribute to seed the
            data key. If a dict, we'll use the dictionary's keys instead.
            If None, we'll only include the default DAQ step counter,
            which will always be included.
        motors : list or dict of signals or positioners, or None, optional
            Alias of "controls" for backwards compatibility.
        begin_timeout : float or None, optional
            How long to wait before marking a begin run as a failure and
            raising an exception. The default is {BEGIN_TIMEOUT} seconds.
        begin_sleep : float or None, optional
            How long to wait before starting a run. The default is
            {BEGIN_SLEEP} seconds.
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
        show_queued_config: bool, optional
            If True, we'll show what the next configuration will be
            as a nice log message.
        """
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
            duration = None
        elif isinstance(duration, Real):
            duration = float(duration)
            events = None
        # Handle motors as an alias for controls
        if not isinstance(motors, SENTINEL):
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
            show_queued_cfg=show_queued_config,
        )

    # TODO make sure the configure timing is correct as described in docstring
    def configure(
        self,
        events: Union[int, None, SENTINEL] = CONFIG_VAL,
        duration: Union[Real, None, SENTINEL] = CONFIG_VAL,
        record: Union[bool, None, SENTINEL] = CONFIG_VAL,
        controls: Union[ControlsArg, None, SENTINEL] = CONFIG_VAL,
        motors: Union[ControlsArg, None, SENTINEL] = CONFIG_VAL,
        begin_timeout: Union[Real, None, SENTINEL] = CONFIG_VAL,
        begin_sleep: Union[Real, None, SENTINEL] = CONFIG_VAL,
        group_mask: Union[int, None, SENTINEL] = CONFIG_VAL,
        detname: Union[str, None, SENTINEL] = CONFIG_VAL,
        scantype: Union[str, None, SENTINEL] = CONFIG_VAL,
        serial_number: Union[str, None, SENTINEL] = CONFIG_VAL,
        alg_name: Union[str, None, SENTINEL] = CONFIG_VAL,
        alg_version: Union[list[int], None, SENTINEL] = CONFIG_VAL,
    ):
        """
        Adjusts the configuration, causing a "configure" transition if needed.

        A "configure" transition wil be caused in the following cases:
        1. We are in the "connected" state
        2. We are in the "configured" state but an important configuration
           parameter has been changed. In this case, we will revert to the
           "connected" state and then return to the "configured" state.

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
            as it will require a configure transition.
        controls : list or dict of signals or positioners, or None, optional
            The objects to include per-step in the DAQ data stream.
            These must implement either the "position" attribute or the
            "get" method to retrieve their current value.
            If a list, we'll use the object's "name" attribute to seed the
            data key. If a dict, we'll use the dictionary's keys instead.
            If None, we'll only include the default DAQ step counter,
            which will always be included.
        motors : list or dict of signals or positioners, or None, optional
            Alias of "controls" for backwards compatibility.
        begin_timeout : float or None, optional
            How long to wait before marking a begin run as a failure and
            raising an exception. The default is {BEGIN_TIMEOUT} seconds.
        begin_sleep : float or None, optional
            How long to wait before starting a run. The default is
            {BEGIN_SLEEP} seconds.
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
        if self._queue_configure_transition:
            if self.state_sig.get() < self.state_enum['connected']:
                raise RuntimeError('Not ready to configure.')
            if self.state_sig.get() > self.state_enum['configured']:
                raise RuntimeError(
                    'Cannot configure transition during an open run!'
                )
            if self.state_sig.get() == self.state_enum['configured']:
                # Already configured, so we should unconfigure first
                self.state_transition('connected', wait=True)
            if self.record_cfg.get() is not None:
                self._control.setRecord(self.record_cfg.get())
            self.state_transition('configured', wait=True)
            self._last_config = self.config
            self._queue_configure_transition = False
        return old, new

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
        return self.run_number_sig.get()
