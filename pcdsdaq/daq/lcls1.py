"""
Module that defines the controls python interface for the LCLS1 DAQ.

Note: this not yet ready for use. It was added alongside lcls2.py, but
it has not been finished or tested.

This is an experimental refactoring of the code found in original.py.
"""
from __future__ import annotations

import enum
import functools
import logging
import os
import threading
import time
from importlib import import_module
from typing import Any

from bluesky import RunEngine
from ophyd.device import Component as Cpt
from ophyd.signal import Signal
from ophyd.status import Status
from ophyd.utils import StatusTimeoutError, WaitTimeoutError

from .. import ext_scripts
from ..ami import AmiDet, set_monitor_det, set_pyami_filter
from ..exceptions import DaqStateTransitionError, DaqTimeoutError
from .interface import (CONFIG_VAL, ControlsArg, DaqBase, Sentinel,
                        get_controls_value)

logger = logging.getLogger(__name__)
pydaq = None


def check_connect(f):
    """
    Decorator to ensure that the `DaqLCLS1` is connected.
    """
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        logger.debug('Checking for daq connection')
        if not self.connected:
            msg = 'DAQ is not connected. Attempting to connect...'
            logger.info(msg)
            self.connect()
        if self.connected:
            logger.debug('Daq is connected')
            return f(self, *args, **kwargs)
        else:
            err = 'Could not connect to DAQ'
            logger.error(err)
            raise RuntimeError(err)
    return wrapper


class DaqLCLS1(DaqBase):
    """
    The LCLS1 daq as a ``bluesky``-compatible object.

    This uses the ``pydaq`` module to connect with a running daq instance,
    controlling it via socket commands.

    It can be used as a ``Reader`` in a ``bluesky`` plan to take data at
    discrete scan points.

    It can be used as a ``Flyer`` in a ``bluesky`` plan to have the daq start
    at the beginning of the run and end at the end of the run.

    Unlike normal ``bluesky`` readable devices or flyers, this has no data to
    report to the ``RunEngine`` on the ``read`` or ``collect`` calls. No data
    will pass into the python layer from the daq.

    Parameters
    ----------
    RE: ``RunEngine``, optional
        Set ``RE`` to the session's main ``RunEngine``

    hutch_name: str, optional
        Define a hutch name to use instead of shelling out to get_hutch_name.
    """
    use_l3t_cfg = Cpt(Signal, value=False, kind='config')
    begin_sleep_cfg = Cpt(Signal, value=0, kind='config')

    state_enum = enum.Enum(
        'PydaqState',
        'Disconnected Connected Configured Open Running',
        start=0,
    )
    requires_configure_transition = {'record', 'use_l3t'}

    def __init__(
        self,
        RE: RunEngine | None = None,
        hutch_name: str | None = None,
    ):
        if pydaq is None:
            globals()['pydaq'] = import_module('pydaq')
        super().__init__(RE=RE, hutch_name=hutch_name)
        self._control = None
        self._reset_begin()
        self._host = os.uname()[1]
        self._re_cbid = None
        self._pre_run_state = None
        self._last_stop = 0
        self._check_run_number_has_failed = False

    # Convenience properties
    @property
    def connected(self) -> bool:
        """
        ``True`` if the daq is connected, ``False`` otherwise.
        """
        return self._control is not None

    @property
    def state(self) -> str:
        """
        State as reported by the daq. Can be any of the following:
        - ``Disconnected``: No active session in python
        - ``Connected``:    Active session in python
        - ``Configured``:   Connected, and the daq has been configured
        - ``Open``:         We are in the middle of a run
        - ``Running``:      We are collecting data in a run
        """
        if self.connected:
            logger.debug('calling Daq.control.state()')
            num = self._control.state()
            return self._state_enum(num).name
        else:
            return 'Disconnected'

    # Interactive methods
    def connect(self) -> None:
        """
        Connect to the live DAQ, giving full control to the Python process.

        To undo this, you may call `disconnect`.
        """
        logger.debug('Daq.connect()')
        err = False
        conn = False
        if self._control is None:
            for plat in range(6):
                try:
                    logger.debug(('instantiate Daq.control '
                                  '= pydaq.Control(%s, %s)'),
                                 self._host, plat)
                    self._control = pydaq.Control(self._host, platform=plat)
                    logger.debug('Daq.control.connect()')
                    self._control.connect()
                    logger.info('Connected to DAQ')
                    conn = True
                    break
                except Exception as exc:
                    if 'query' in str(exc):
                        err = True
                        logger.error("Failed to connect: DAQ is not allocated!")
            if not (err or conn):
                err = True
                logger.error(
                    "Failed to connect: DAQ is not running on this "
                    "machine, and is not allocated!"
                )
            if err:
                logger.debug('del Daq.control')
                del self._control
                self._control = None
        else:
            logger.info('Connect requested, but already connected to DAQ')

    def disconnect(self) -> None:
        """
        Disconnect from the live DAQ, giving control back to the GUI.

        This is the opposite of `connect`.
        """
        logger.debug('Daq.disconnect()')
        if self._control is not None:
            self.end_run()
            self._control.disconnect()
        del self._control
        self._control = None
        self.preconfig(**self.default_config)
        self.configured_sig.put(False)
        logger.info('DAQ is disconnected.')

    @check_connect
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
        logger.debug('Daq.wait()')
        if self.state == 'Running':
            if not self._infinite_run:
                status = self._get_end_status()
                try:
                    status.wait(timeout=timeout)
                except (StatusTimeoutError, WaitTimeoutError):
                    msg = (f'Timeout after {timeout} seconds waiting for daq '
                           'to finish acquiring.')
                    raise DaqTimeoutError(msg) from None
            else:
                raise RuntimeError('Cannot wait, daq configured to run '
                                   'forever.')
        if end_run:
            self.end_run()

    def begin(
        self,
        events: int | None | Sentinel = CONFIG_VAL,
        duration: int | None | Sentinel = CONFIG_VAL,
        record: bool | None | Sentinel = CONFIG_VAL,
        use_l3t: bool | None | Sentinel = CONFIG_VAL,
        controls: ControlsArg | None | Sentinel = CONFIG_VAL,
        wait: bool = False,
        end_run: bool = False,
    ):
        """
        Start the daq and block until the daq has begun acquiring data.

        Optionally block with ``wait=True`` until the daq has finished aquiring
        data. If blocking, a ``ctrl+c`` will end the run and clean up.

        If omitted, any argument that is shared with `configure`
        will fall back to the configured value.

        Internally, this calls `kickoff` and manages its ``Status`` object.

        Parameters
        ----------
        events: ``int``, optional
            Number events to take in the daq.

        duration: ``int``, optional
            Time to run the daq in seconds, if ``events`` was not provided.

        record: ``bool``, optional
            If ``True``, we'll configure the daq to record data before this
            run.

        use_l3t: ``bool``, optional
            If ``True``, we'll run with the level 3 trigger. This means that
            if we specified a number of events, we will wait for that many
            "good" events as determined by the daq.

        controls: ``dict{name: device}`` or ``list[device...]``, optional
            If provided, values from these will make it into the DAQ data
            stream as variables. We will check ``device.position`` and
            ``device.value`` for quantities to use and we will update these
            values each time begin is called. To provide a list, all devices
            must have a ``name`` attribute.

        wait: ``bool``, optional
            If ``True``, wait for the daq to finish aquiring data. A
            ``KeyboardInterrupt`` (``ctrl+c``) during this wait will end the
            run and clean up.

        end_run: ``bool``, optional
            If ``True``, we'll end the run after the daq has stopped.
        """
        logger.debug(('DaqLCLS1.begin(events=%s, duration=%s, record=%s, '
                      'use_l3t=%s, controls=%s, wait=%s)'),
                     events, duration, record, use_l3t, controls, wait)
        try:
            if record is not CONFIG_VAL and record != self.record:
                old_record = self.record
                self.preconfig(record=record, show_queued_cfg=False)
            return super().begin(
                events=events,
                duration=duration,
                record=record,
                use_l3t=use_l3t,
                controls=controls,
                wait=wait,
                end_run=end_run,
            )
        finally:
            try:
                self.preconfig(record=old_record, show_queued_cfg=False)
            except NameError:
                pass

    @property
    def _begin_timeout(self) -> int:
        return self.begin_timeout_cfg.get() + self.begin_throttle_cfg.get()

    def begin_infinite(
        self,
        record: bool | None | Sentinel = CONFIG_VAL,
        use_l3t: bool | None | Sentinel = CONFIG_VAL,
        controls: ControlsArg | None | Sentinel = CONFIG_VAL,
    ) -> None:
        """
        Start the daq to run forever in the background.
        """
        self.begin(events=0, record=record, use_l3t=use_l3t,
                   controls=controls, wait=False, end_run=False)

    def _ender_thread(self) -> None:
        """
        End the run when the daq stops aquiring
        """
        self.wait()
        self.end_run()

    @check_connect
    def stop(self, success: bool = False) -> None:
        """
        Stop the current acquisition, ending it early.

        Parameters
        ----------
        success : bool, optional
            Flag set by bluesky to signify whether this was a good stop or a
            bad stop. Currently unused.
        """
        logger.debug('Daq.stop()')
        self._control.stop()
        self._reset_begin()
        self._last_stop = time.time()

    @check_connect
    def end_run(self) -> None:
        """
        Call `stop`, then mark the run as finished.
        """
        logger.debug('Daq.end_run()')
        self.stop()
        self._control.endrun()

    # Reader interface
    @check_connect
    def trigger(self) -> Status:
        """
        Begin acquisition. This method blocks until the run begins.

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
        """
        cfg = self.config
        if all(cfg[key] is None for key in ('events', 'duration')):
            raise RuntimeError('Cannot start daq in scan step, did not '
                               'configure events or duration.')
        self.begin()
        return self._get_end_status()

    # Flyer interface
    @check_connect
    def kickoff(
        self,
        events: int | None | Sentinel = CONFIG_VAL,
        duration: int | None | Sentinel = CONFIG_VAL,
        use_l3t: bool | None | Sentinel = CONFIG_VAL,
        controls: ControlsArg | None | Sentinel = CONFIG_VAL,
    ) -> Status:
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
        logger.debug('Daq.kickoff()')

        self._check_duration(duration)
        if self._queue_configure_transition or not self.configured:
            try:
                self.configure()
            except DaqStateTransitionError:
                err = ('Illegal reconfigure with {} during an open run. End '
                       'the current run with daq.end_run() before running '
                       'with a new configuration'.format(self.config))
                logger.debug(err, exc_info=True)
                raise DaqStateTransitionError(err)

        check_run_number = all((self.state == 'Configured',
                                self.config['record'],
                                not self._check_run_number_has_failed))
        if check_run_number:
            try:
                prev_run = self.run_number()
                next_run = prev_run + 1
            except Exception:
                logger.debug('Error getting run number in kickoff',
                             exc_info=True)
                next_run = None
                # Only try this once if it fails to prevent repeated timeouts
                self._check_run_number_has_failed = True
        else:
            next_run = None

        def start_thread(control, status, events, duration, use_l3t, controls,
                         run_number):
            tmo = self._begin_timeout
            dt = 0.1
            logger.debug('Make sure daq is ready to begin')
            # Stop and start if we already started
            if self.state in ('Open', 'Running'):
                self.stop()
            # It can take up to 0.4s after a previous begin to be ready
            while tmo > 0:
                if self.state in ('Configured', 'Open'):
                    break
                else:
                    tmo -= dt
            if self.state in ('Configured', 'Open'):
                begin_args = self._begin_args(events, duration, use_l3t,
                                              controls)
                if run_number is not None:
                    logger.info('Beginning daq run %s', run_number)

                logger.debug('daq.control.begin(%s)', begin_args)
                dt = time.time() - self._last_stop
                tmo = self.begin_throttle_cfg.get() - dt
                if tmo > 0:
                    time.sleep(tmo)
                control.begin(**begin_args)
                # Cache these so we know what the most recent begin was told
                self._begin = dict(events=events, duration=duration,
                                   use_l3t=use_l3t, controls=controls)
                logger.debug('Marking kickoff as complete')
                status.set_finished()
            else:
                logger.debug('Marking kickoff as failed')
                status.set_exception(RuntimeError('Daq begin failed!'))

        begin_status = Status(self)
        watcher = threading.Thread(target=start_thread,
                                   args=(self._control, begin_status, events,
                                         duration, use_l3t, controls,
                                         next_run))
        watcher.start()
        return begin_status

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
        logger.debug('Daq.complete()')
        end_status = self._get_end_status()
        if self._infinite_run:
            # Configured to run forever
            self.stop()
        return end_status

    def _get_end_status(self) -> Status:
        """
        Return a `Status` object that will be marked done when the DAQ has
        finished acquiring.

        This will be marked as done immediately if the daq is configured to run
        forever, because waiting for the end doesn't make sense in this case.

        Returns
        -------
        end_status: `Status`
        """
        logger.debug('Daq._get_end_status()')

        events = self._events
        duration = self._duration

        if not self._infinite_run:
            logger.debug('Getting end status for events=%s, duration=%s',
                         events, duration)

            def finish_thread(control, status):
                try:
                    logger.debug('Daq.control.end()')
                    control.end()
                except RuntimeError:
                    pass  # This means we aren't running, so no need to wait
                self._last_stop = time.time()
                self._reset_begin()
                status.set_finished()
                logger.debug('Marked acquisition as complete')
            end_status = Status(self)
            watcher = threading.Thread(target=finish_thread,
                                       args=(self._control, end_status))
            watcher.start()
            return end_status
        else:
            # Configured to run forever, say we're done so we can wait for just
            # the other things in the scan
            logger.debug('Returning finished status for infinite run with '
                         'events=%s, duration=%s', events, duration)
            status = Status(self)
            status.set_finished()
            return status

    def preconfig(
        self,
        events: int | None | Sentinel = CONFIG_VAL,
        duration: int | None | Sentinel = CONFIG_VAL,
        record: bool | None | Sentinel = CONFIG_VAL,
        use_l3t: bool | None | Sentinel = CONFIG_VAL,
        controls: ControlsArg | None | Sentinel = CONFIG_VAL,
        begin_sleep: int | None | Sentinel = CONFIG_VAL,
        show_queued_cfg: bool = True,
    ) -> None:
        """
        Queue configuration parameters for next call to `configure`.

        These will be overridden by arguments passed directly to `configure`.
        These will be cleared after each call to `configure`.

        This can be used to `configure` the `DaqLCLS1` object without
        connecting.

        This will display the next queued configuration using logger.info,
        assuming the logger has been configured.
        """
        # Only one of (events, duration) should be preconfigured.
        if events is not CONFIG_VAL:
            duration = CONFIG_VAL

        return super().preconfig(
            events=events,
            duration=duration,
            record=record,
            use_l3t=use_l3t,
            controls=controls,
            begin_sleep=begin_sleep,
            show_queued_cfg=show_queued_cfg,
        )

    @check_connect
    def configure(
        self,
        events: int | None | Sentinel = CONFIG_VAL,
        duration: int | None | Sentinel = CONFIG_VAL,
        record: bool | None | Sentinel = CONFIG_VAL,
        use_l3t: bool | None | Sentinel = CONFIG_VAL,
        controls: ControlsArg | None | Sentinel = CONFIG_VAL,
        begin_sleep: int | None | Sentinel = CONFIG_VAL,
    ) -> tuple[dict, dict]:
        """
        Changes the daq's configuration for the next run.

        All arguments omitted from the method call will default to the last
        configured value in the python session.

        This is the method that directly interfaces with the daq. If you simply
        want to get a configuration ready for later, use `preconfig`.

        Parameters
        ----------
        events: ``int``, optional
            If provided, the daq will run for this many events before
            stopping, unless we override in `begin`.
            If not provided, we'll use the ``duration`` argument instead.
            Defaults to its last configured value, or ``None`` on the first
            configure.

        duration: ``int``, optional
            If provided, the daq will run for this many seconds before
            stopping, unless we override in `begin`.
            If not provided, and ``events`` was also not provided, an empty
            call like ``begin()`` will run indefinitely. You can also achieve
            this behavior by passing events=None and/or duration=None, Defaults
            to its last configured value, or ``None`` on the first configure.

        record: ``bool``, optional
            If ``True``, we'll record the data. If ``False``, we'll run without
            recording. If ``None``, we'll use the option selected in the DAQ
            GUI. Defaults to the its last configured value, or ``None`` on the
            first configure.

        use_l3t: ``bool``, optional
            If ``True``, an ``events`` argument to begin will be reinterpreted
            to only count events that pass the level 3 trigger. Defaults to
            its last configured value, or ``False`` on the first configure.

        controls: ``dict{name: device}`` or ``list[device...]``, optional
            If provided, values from these will make it into the DAQ data
            stream as variables. We will check ``device.position`` and
            ``device.value`` for quantities to use and we will update these
            values each time begin is called. To provide a list, all devices
            must have a ``name`` attribute. Defaults to its last configured
            value, or no controls values on the first configure.

        begin_sleep: ``int``, optional
            The amount of time to wait after the DAQ returns begin is done.
            This is a hack because the DAQ often says that a begin transition
            is done without actually being done, so it needs a short delay.
            Defaults to its last configured value, or 0 on the first
            configure.

        Returns
        -------
        old, new: ``tuple`` of ``dict``
            The old configuration and the new configuration. These dictionaries
            are verbose, containing all configuration values and the timestamps
            at which they were configured, as specified by ``bluesky``.
        """
        logger.debug('Daq.configure(events=%s, duration=%s, record=%s, '
                     'use_l3t=%s, controls=%s, begin_sleep=%s)',
                     events, duration, record, use_l3t, controls, begin_sleep)
        state = self.state
        if state not in ('Connected', 'Configured'):
            err = f'Cannot configure from state {state}!'
            raise DaqStateTransitionError(err)

        self._check_duration(duration)

        old, new = super().configure(
            events=events,
            duration=duration,
            record=record,
            use_l3t=use_l3t,
            controls=controls,
            begin_sleep=begin_sleep,
        )

        config = self.config

        events = config['events']
        duration = config['duration']
        record = config['record']
        use_l3t = config['use_l3t']
        controls = config['controls']
        begin_sleep = config['begin_sleep']

        logger.debug('Updated with queued config, now we have: '
                     'events=%s, duration=%s, record=%s, '
                     'use_l3t=%s, controls=%s, begin_sleep=%s',
                     events, duration, record, use_l3t, controls, begin_sleep)

        config_args = self._config_args(record, use_l3t, controls)
        try:
            logger.debug('Daq.control.configure(%s)',
                         config_args)
            self._control.configure(**config_args)
            self.config_info(header='Daq configured:')
            self._last_config = self.config
            self._queue_configure_transition = False
            self.configred_sig.put(True)
        except Exception as exc:
            msg = 'Failed to configure!'
            logger.debug(msg, exc_info=True)
            raise RuntimeError(msg) from exc
        return old, new

    def _config_args(
        self,
        record: bool,
        use_l3t: bool,
        controls: ControlsArg,
    ):
        """
        For a given set of arguments to `configure`, return the arguments that
        should be sent to ``pydaq.Control.configure``.

        Returns
        -------
        config_args: dict
        """
        logger.debug('Daq._config_args(%s, %s, %s)',
                     record, use_l3t, controls)
        config_args = {}
        if record is not None:
            config_args['record'] = bool(record)
        if use_l3t:
            config_args['l3t_events'] = 0
        else:
            config_args['events'] = 0
        if controls is not None:
            config_args['controls'] = self._ctrl_arg(controls)
        return config_args

    def _ctrl_arg(self, controls: ControlsArg) -> list[tuple[str, Any]]:
        """
        Assemble the list of ``(str, val)`` pairs from a ``{str: device}``
        dictionary or a device ``list``

        Returns
        -------
        ctrl_arg: ``list[(str, val), ...]``
        """
        ctrl_arg = []
        if isinstance(controls, list):
            names = [dev.name for dev in controls]
            devices = controls
        elif isinstance(controls, dict):
            names = controls.keys()
            devices = controls.values()
        for name, device in zip(names, devices):
            val = get_controls_value(device)
            try:
                val = val[0]
            except Exception:
                pass
            ctrl_arg.append((name, val))
        return ctrl_arg

    def _begin_args(
        self,
        events: int | None | Sentinel = CONFIG_VAL,
        duration: int | None | Sentinel = CONFIG_VAL,
        use_l3t: bool | None | Sentinel = CONFIG_VAL,
        controls: ControlsArg | None | Sentinel = CONFIG_VAL,
    ) -> dict[str, Any]:
        """
        For a given set of arguments to `begin`, return the arguments that
        should be sent to ``pydaq.Control.begin``

        Returns
        -------
        begin_args: ``dict``
        """
        logger.debug('Daq._begin_args(%s, %s, %s, %s)',
                     events, duration, use_l3t, controls)
        begin_args = {}
        # Handle default args for events and duration
        if events is CONFIG_VAL and duration is CONFIG_VAL:
            # If both are omitted, use last configured values
            events = self.config['events']
            duration = self.config['duration']
        if events not in (None, CONFIG_VAL):
            # We either passed the events arg, or loaded from config
            if use_l3t in (None, CONFIG_VAL) and self.configured:
                use_l3t = self.config['use_l3t']
            if use_l3t:
                begin_args['l3t_events'] = events
            else:
                begin_args['events'] = events
        elif duration not in (None, CONFIG_VAL):
            # We either passed the duration arg, or loaded from config
            secs = int(duration)
            nsec = int((duration - secs) * 1e9)
            begin_args['duration'] = [secs, nsec]
        else:
            # We passed None somewhere/everywhere
            begin_args['events'] = 0  # Run until manual stop
        if controls is CONFIG_VAL:
            controls = self.config['controls']
        if controls is not None:
            begin_args['controls'] = self._ctrl_arg(controls)
        return begin_args

    def _check_duration(self, duration: int | None | Sentinel):
        if duration not in (None, CONFIG_VAL) and duration < 1:
            msg = ('Duration argument less than 1 is unreliable. Please '
                   'use the events argument to specify the length of '
                   'very short runs.')
            raise RuntimeError(msg)

    @property
    def _events(self) -> int | None:
        """
        For the current `begin` cycle, how many ``events`` we told the daq to
        run for.
        """
        events = self._begin['events']
        if events is CONFIG_VAL:
            events = self.config['events']
        return events

    @property
    def _duration(self) -> int | None:
        """
        For the current `begin` cycle, how long we told the daq to run for in
        seconds.
        """
        duration = self._begin['duration']
        if duration is CONFIG_VAL:
            duration = self.config['duration']
        return duration

    @property
    def _infinite_run(self) -> bool:
        """
        True if configured for an infinite run.
        """
        if self._events is None and self._duration is None:
            return True
        return self._events in (-1, 0)

    def _reset_begin(self) -> None:
        """
        Reset ``_begin`` to starting values for when we aren't running.
        """
        self._begin = dict(events=None, duration=None, use_l3t=None,
                           controls=None)

    def run_number(self, hutch_name: str | None = None):
        """
        Determine the run number of the last run, or current run if running.

        This requires you to be on an NFS-mounted host. If hutch can be
        determined from the get_hutch_name script from engineering_tools, then
        you don't need to pass in a hutch name.

        This is a method and not a property because all properties are
        run when you try to tab complete, and this isn't necessarily an
        instant check. It can also display log messages, which would be
        annoying on tab complete.

        Parameters
        ----------
        hutch_name: ``str``, optional
            The hutch to check the run number for. If omitted, we'll guess
            the hutch based on your session details.

        Returns
        -------
        run_number: ``int``
            The current run number, or previous run if not recording.

        Raises
        ------
        RuntimeError:
            if we have no access to NFS
        ValueError:
            if an invalid hutch was passed
        subprocess.TimeoutExpired:
            if the get run number script fails
        """
        try:
            hutch_name = hutch_name or self.hutch_name
            if hutch_name is None:
                hutch_name = ext_scripts.get_hutch_name()
            hutch_name = hutch_name.lower()
            if hutch_name not in ('amo', 'sxr', 'xpp', 'xcs', 'mfx', 'cxi',
                                  'mec', 'tst'):
                raise ValueError(
                    f"{hutch_name} is not a valid hutch, cannot determine run number"
                )
            if self.state in ('Open', 'Running') and self.config['record']:
                return ext_scripts.get_run_number(hutch=hutch_name, live=True)
            else:
                return ext_scripts.get_run_number(hutch=hutch_name, live=False)
        except FileNotFoundError:
            raise RuntimeError('No nfs access, cannot determine run number.')

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    def set_filter(
        self,
        *args,
        event_codes: list[int] | None = None,
        operator: str = '&',
        or_bykik: bool = False,
    ) -> None:
        """
        Set up the l3t filters.

        These connect through pyami to call set_l3t or clear_l3t. The function
        takes in arbitrary dets whose prefixes are the ami names, along with
        low and highs.

        Event codes are handled as a special case, since you always want high
        vs low.

        .. note::
            If or_bykik is True, this will treat bykik at an l3t pass! This is
            so you don't lose your off shots when the l3t trigger is in veto
            mode.

        Parameters
        ----------
        *args: (`AmiDet`, ``float``, ``float``) n times
            A sequence of (detector, low, high), which create filters that make
            sure the detector is between low and high. You can omit the first
            `AmiDet` as a shorthand for the current monitor, assuming a monitor
            has been set with `Daq.set_monitor` or `set_monitor_det`.

        event_codes: ``list``, optional
            A list of event codes to include in the filter. l3pass will be when
            the event code is present.

        operator: ``str``, optional
            The operator for combining the detector ranges and event codes.
            This can either be ``|`` to ``or`` the conditions together, so
            l3pass will happen if any filter passes, or it can be left at
            the default ``&`` to ``and`` the conditions together, so l3pass
            will only happen if all filters pass.

        or_bykik: ``bool``, optional
            False by default, appends an ``or`` condition that marks l3t pass
            when we see the bykik event code. This makes sure the off shots
            make it into the data if we're in l3t veto mode.
        """

        return set_pyami_filter(*args, event_codes=event_codes,
                                operator=operator, or_bykik=or_bykik)

    def set_monitor(self, det: AmiDet) -> None:
        """
        Pick the ami monitor det.
        """
        return set_monitor_det(det)

    set_monitor.__doc__ = set_monitor_det.__doc__
