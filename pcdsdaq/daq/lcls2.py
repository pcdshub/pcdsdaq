"""
Module that defines the controls python interface for the LCLS2 DAQ
"""
from __future__ import annotations

import logging
import threading
from functools import cache
from numbers import Real
from typing import Iterator, Optional, Type, Union, get_type_hints

from ophyd.device import Component as Cpt
from ophyd.signal import Signal
from ophyd.status import DeviceStatus, Status
from ophyd.utils import StatusTimeoutError, WaitTimeoutError
from ophyd.utils.errors import InvalidState

from .interface import (CONFIG_VAL, SENTINEL, ControlsArg, DaqBase, EnumId,
                        HelpfulIntEnum, get_controls_value, typing_check)

try:
    from psdaq.control.ControlDef import ControlDef
    from psdaq.control.DaqControl import DaqControl
except ImportError:
    DaqControl = None
    ControlDef = None

logger = logging.getLogger(__name__)
pydaq = None


class DaqLCLS2(DaqBase):
    step_value_sig = Cpt(
        Signal,
        value=1,
        kind='normal',
        name='step_value',
    )
    state_sig = Cpt(
        Signal,
        value=None,
        kind='normal',
        name='state',
    )
    transition_sig = Cpt(
        Signal,
        value=None,
        kind='normal',
        name='transition',
    )
    transition_elapsed_sig = Cpt(
        Signal,
        value=None,
        kind='normal',
        name='transition_elapsed',
    )
    transition_total_sig = Cpt(
        Signal,
        value=None,
        kind='normal',
        name='transition_total',
    )
    config_alias_sig = Cpt(
        Signal,
        value=None,
        kind='normal',
        name='config_alias',
    )
    recording_sig = Cpt(
        Signal,
        value=None,
        kind='normal',
        name='recording',
    )
    bypass_activedet_sig = Cpt(
        Signal,
        value=None,
        kind='normal',
        name='bypass_activedet',
    )
    experiment_name_sig = Cpt(
        Signal,
        value=None,
        kind='normal',
        name='experiment_name',
    )
    run_number_sig = Cpt(
        Signal,
        value=None,
        kind='normal',
        name='run_number',
    )
    last_run_number_sig = Cpt(
        Signal,
        value=None,
        kind='normal',
        name='last_run_number',
    )

    group_mask_cfg = Cpt(
        Signal,
        value=None,
        kind='config',
        name='group_mask',
    )
    detname_cfg = Cpt(
        Signal,
        value='scan',
        kind='config',
        name='detname',
    )
    scantype_cfg = Cpt(
        Signal,
        value='scan',
        kind='config',
        name='scantype',
    )
    serial_number_cfg = Cpt(
        Signal,
        value='1234',
        kind='config',
        name='serial_number',
    )
    alg_name_cfg = Cpt(
        Signal,
        value='raw',
        kind='config',
        name='alg_name',
    )
    alg_version_cfg = Cpt(
        Signal,
        value=[1, 0, 0],
        kind='config',
        name='alg_version_cfg',
    )

    last_err_sig = Cpt(
        Signal,
        value=None,
        kind='omitted',
        name='last_err',
    )
    last_warning_sig = Cpt(
        Signal,
        value=None,
        kind='omitted',
        name='last_warning',
    )
    last_file_report_sig = Cpt(
        Signal,
        value=None,
        kind='omitted',
        name='last_file_report',
    )
    step_done_sig = Cpt(
        Signal,
        value=None,
        kind='omitted',
        name='step_done',
    )
    last_transition_err_sig = Cpt(
        Signal,
        value=None,
        kind='omitted',
        name='last_transition_err',
    )

    requires_configure_transition = {'record'}

    def __init__(self, platform, host, timeout, RE=None, hutch_name=None):
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
    def state_enum(self) -> Type[HelpfulIntEnum]:
        return HelpfulIntEnum('PsdaqState', ControlDef.states)

    @property
    @cache
    def transition_enum(self) -> Type[HelpfulIntEnum]:
        return HelpfulIntEnum('PsdaqTransition', ControlDef.transitions)

    def _start_monitor_thread(self):
        """
        Monitor the DAQ state in a background thread.
        """
        threading.Thread(target=self._monitor_thread, args=()).start()

    def _monitor_thread(self):
        """
        Pick up our ZMQ subscription messages, put into our signals.
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
    def _configured_cb(self, value, **kwargs):
        """
        Callback on the state signal to update the configured signal.

        The LCLS2 DAQ is considered configured based on the state machine.
        """
        self.configured_sig.put(
            value >= self.state_enum['configured']
        )

    @property
    def state(self) -> str:
        """
        API to show the state as reported by the DAQ.
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
    ):
        """
        Return a status object for DAQ state transitions.

        This status object will be marked done when we're at the given state
        or when we're doing the given transition, if either state or
        transition was given.

        If both state and transition are given, then we need to arrive at
        the given state using the given transition to mark the status as
        done.

        State and transition are both lists so we can check for multiple
        states.
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

        def check_state(value, old_value, **kwargs):
            nonlocal last_state
            if value == old_value and not check_now:
                return
            with lock:
                if value in state and last_transition in transition:
                    success()
                else:
                    last_state = value

        def check_transition(value, old_value, **kwargs):
            nonlocal last_transition
            if value == old_value and not check_now:
                return
            with lock:
                if value in transition and last_state in state:
                    success()
                else:
                    last_transition = value

        def success():
            try:
                status.set_finished()
            except InvalidState:
                ...

        def clean_up(status):
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
    ):
        """
        The DAQ is done acquiring if the most recent transition was not
        "beginrun", "beginstep", or "enable".
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
        Undergo a daq state transition appropriately.

        This passes extra data if we need to do the 'configure' or 'beginstep'
        transitions.
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

    def _transition_thread(self, state, phase1_info):
        error_msg = self._control.setState(state, phase1_info)
        self.last_transition_err_sig.put(error_msg)

    def _handle_duration_thread(self, duration):
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

    def _get_phase1(self, transition):
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

    def _get_motors_for_transition(self):
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

    def _get_block(self, transition, data):
        raise NotImplementedError(
            'Have not done this one yet'
        )

    def begin(self, wait: bool = False, end_run: bool = False, **kwargs):
        original_config = self.config
        self.preconfig(show_queued_cfg=False, **kwargs)
        super().begin(wait=wait, end_run=end_run)
        self.preconfig(show_queued_cfg=False, **original_config)

    def _end_run_callback(self, status):
        self.end_run()

    def begin_infinite(self):
        self.begin(events=0)

    @property
    def _infinite_run(self):
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
        Call `stop`, then mark the run as finished.
        """
        if self.state_sig.get() > self.state_enum['configured']:
            self.state_transition('configured', wait=False)

    def trigger(self) -> Status:
        """
        Begin acquisition.

        Returns a status object that will be marked done when the daq has
        stopped acquiring.

        This will raise a RuntimeError if the daq was never configured for
        events or duration.

        Returns
        -------
        done_status: ``Status``
            ``Status`` that will be marked as done when the daq is done.
        """
        status = self.get_status_for(
            state=['starting'],
            transition=['endstep'],
            check_now=False,
            timeout=self.begin_timeout_cfg.get(),
        )
        self.kickoff()
        return status

    def kickoff(self) -> DeviceStatus:
        """
        Begin acquisition. This method is non-blocking.
        See `begin` for a description of the parameters.

        This method does not supply arguments for configuration parameters, it
        supplies arguments directly to ``pydaq.Control.begin``. It will
        configure before running if there are queued configuration changes.

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
        alg_version: Union[list[str], None, SENTINEL] = CONFIG_VAL,
        show_queued_config: bool = True,
    ):
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

    # TODO fix type hinting for default of _CONFIG_VAL instead of None
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
        alg_version: Union[list[str], None, SENTINEL] = CONFIG_VAL,
    ):
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

    def run_number(self):
        """
        Determine the run number of the last run, or current run if running.
        """
        return self.run_number_sig.get()
