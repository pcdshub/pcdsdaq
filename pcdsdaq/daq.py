"""
This module defines a control interface for the LCLS1 DAQ.
"""
from __future__ import annotations

import enum
import functools
import logging
import os
import time
import threading
from enum import Enum, IntEnum
from functools import cache
from importlib import import_module
from typing import Any, ClassVar, Iterator, Optional, Type

from bluesky import RunEngine

from ophyd.device import Component as Cpt, Device
from ophyd.ophydobj import Kind
from ophyd.signal import AttributeSignal, Signal
from ophyd.status import DeviceStatus, Status
from ophyd.utils import StatusTimeoutError, WaitTimeoutError
from ophyd.utils.errors import InvalidState

from . import ext_scripts
from .ami import set_ami_hutch, set_pyami_filter, set_monitor_det

try:
    from psdaq.control.DaqControl import DaqControl
    from psdaq.control.ControlDef import ControlDef
except ImportError:
    DaqControl = None
    ControlDef = None

logger = logging.getLogger(__name__)
pydaq = None

# Wait up to this many seconds for daq to be ready for a begin call
BEGIN_TIMEOUT = 15
# Do not allow begins within this many seconds of a stop
BEGIN_THROTTLE = 1

# Not-None sentinal for default value when None has a special meaning
# Indicates that the last configured value should be used
_CONFIG_VAL = object()


def check_connect(f):
    """
    Decorator to ensure that the `Daq` is connected before running a method.
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


class DaqTimeoutError(Exception):
    pass


class Daq(Device):
    """
    Base class to define shared DAQ API

    All subclasses should implement the "not implemented" methods here.

    Also defines some shared features so that different DAQ versions
    do not have to reinvent the wheel for basic API choices.
    """
    state_sig = Cpt(AttributeSignal, 'state', kind='normal', name='state')
    configured_sig = Cpt(
        Signal,
        value=False,
        kind='normal',
        name='configured',
    )

    events_cfg = Cpt(Signal, value=None, kind='config', name='events')
    duration_cfg = Cpt(Signal, value=None, kind='config', name='duration')
    record_cfg = Cpt(Signal, value=None, kind='config', name='record')
    controls_cfg = Cpt(Signal, value=None, kind='config', name='controls')
    begin_timeout_cfg = Cpt(
        Signal,
        value=BEGIN_TIMEOUT,
        kind='config',
        name='begin_timeout',
    )
    begin_sleep_cfg = Cpt(Signal, value=0, kind='config', name='begin_sleep')

    # Define these in subclass
    state_enum: ClassVar[Enum]
    requires_configure_transition: ClassVar[set[str]]

    # Variables from init
    _RE: Optional[RunEngine]
    hutch_name: Optional[str]
    platform: Optional[int]
    _queue_configure_transition: bool


    def __init__(
        self,
        RE: Optional[RunEngine] = None,
        hutch_name: Optional[str] = None,
        platform: Optional[int] = None,
        *,
        name: str ='daq',
    ):
        self._RE = RE
        self.hutch_name = hutch_name
        self.platform = platform
        self._queue_configure_transition = True
        super().__init__(name=name)
        register_daq(self)

    # Convenience properties
    @property
    def configured(self) -> bool:
        """
        ``True`` if the daq is configured, ``False`` otherwise.
        """
        return self.configured_sig.get()

    @property
    @cache
    def default_config(self) -> dict[str, Any]:
        """
        The default configuration defined in the class definition.
        """
        default = {}
        for walk in self.walk_components():
            if walk.item.kind == Kind.config:
                default[walk.item.name] = walk.item.kwargs['value']
        return default

    @property
    def config(self) -> dict[str, Any]:
        """
        The current configuration, e.g. the last call to `configure`
        """
        if self.configured:
            cfg = self.read_configuration()
            return {key: value['value'] for key, value in cfg.items()}
        else:
            return self.default_config.copy()

    @property
    def state(self) -> str:
        """
        API to show the state as reported by the DAQ.
        """
        raise NotImplementedError('Please implement state in subclass.')

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
        raise NotImplementedError('Please implement wait in subclass.')

    def begin(self, wait: bool = False, end_run: bool = False, **kwargs):
        """
        Start the daq.
        
        This is the equivalent of "kickoff" but for interactive sessions.
        All kwargs except for "wait" and "end_run" are passed through to
        kickoff.

        Parameters
        ----------
        wait : ``bool``, optional
            If True, wait for the daq to be done running.
        end_run : ``bool``, optional
            If True, end the daq after we're done running.
        """
        logger.debug(f'Daq.begin(kwargs={kwargs})')
        try:
            kickoff_status = self.kickoff(**kwargs)
            try:
                kickoff_status.wait(timeout=self.begin_timeout_cfg.get())
            except (StatusTimeoutError, WaitTimeoutError) as exc:
                raise DaqTimeoutError(
                    f'Timeout after {self.begin_timeout_cfg.get()} seconds '
                    'waiting for daq to begin.'
                ) from exc

            # In some daq configurations the begin status returns very early,
            # so we allow the user to configure an emperically derived extra
            # sleep.
            time.sleep(self.begin_sleep_cfg.get())
            if wait:
                self.wait(end_run=end_run)
            elif end_run:
                threading.Thread(
                    target=self.wait,
                    args=(),
                    kwargs={'end_run': end_run},
                ).start()
        except KeyboardInterrupt:
            if end_run:
                logger.info('%s.begin interrupted, ending run', self.name)
                self.end_run()
            else:
                logger.info('%s.begin interrupted, stopping', self.name)
                self.stop()

    def begin_infinite(self):
        raise NotImplementedError(
            'Please implement begin_infinite in subclass.'
        )

    def stop(self, success: bool = False) -> None:
        """
        Stop the current acquisition, ending it early.

        Parameters
        ----------
        success : bool, optional
            Flag set by bluesky to signify whether this was a good stop or a
            bad stop. Currently unused.
        """
        raise NotImplementedError('Please implement stop in subclass.')

    def end_run(self) -> None:
        """
        Call `stop`, then mark the run as finished.
        """
        raise NotImplementedError('Please implement end_run in subclass.')

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
            ``Status`` that will be marked as done when the daq has begun.
        """
        raise NotImplementedError('Please implement trigger in subclass.')

    def read(self):
        """
        Return data about the status of the daq.

        This also stops if running so you can use this device in a bluesky scan
        and wait for "everything else" to be done, then stop the daq
        afterwards.
        """
        if self.state == 'Running':
            self.stop()
        return super().read()

    def kickoff(self) -> Status:
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
        raise NotImplementedError('Please implement kickoff in subclass.')

    def complete(self) -> Status:
        """
        If the daq is freely running, this will `stop` the daq.
        Otherwise, we'll simply collect the end_status object.

        Returns
        -------
        end_status: ``Status``
            ``Status`` that will be marked as done when the DAQ has finished
            acquiring
        """
        raise NotImplementedError('Please implement complete in subclass.')

    def collect(self):
        """
        Collect data as part of the ``bluesky`` ``Flyer`` interface.

        As per the ``bluesky`` interface, this is a generator that is expected
        to output partial event documents. However, since we don't have any
        events to report to python, this will be a generator that immediately
        ends.
        """
        logger.debug('Daq.collect()')
        yield from ()

    def describe_collect(self):
        """
        As per the ``bluesky`` interface, this is how you interpret the null
        data from `collect`. There isn't anything here, as nothing will be
        collected.
        """
        logger.debug('Daq.describe_collect()')
        return {}

    def preconfig(self, show_queued_cfg=True, **kwargs):
        """
        Write to the configuration signals without executing any transitions.

        Will store the boolean _queue_configure_transition if any
        configurations were changed that require a configure transition. 
        """
        for key, value in kwargs.items():
            if value is _CONFIG_VAL:
                continue
            try:
                sig = getattr(self, key + '_cfg')
            except AttributeError as exc:
                raise ValueError(
                    f'Did not find config parameter {key}'
                ) from exc
            if isinstance(sig, Signal) and sig.kind == 'config':
                sig.put(value)
            else:
                raise ValueError(
                    f'{key} is not a config parameter!'
                )
            if key in self.requires_configure_transition:
                self._queue_configure_transition = True

        if show_queued_cfg:
            self.config_info(self.config, 'Queued config:')

    def configure(self, **kwargs) -> tuple[dict, dict]:
        """
        Write to the configuration signals and execute a configure transition.

        Must be extended in subclass to cause the configure transition when
        needed and to reset the _queue_configure_transition attribute.
        """
        old = self.read_configuration()
        self.preconfig(show_queued_cfg=False, **kwargs)
        return old, self.read_configuration()

    def config_info(self, config=None, header='Config:'):
        """
        Show the config information as a logger.info message.

        This will print to the screen if the logger is configured correctly.

        Parameters
        ----------
        config: ``dict``, optional
            The configuration to show. If omitted, we'll use the current
            config.

        header: ``str``, optional
            A prefix for the config line.
        """
        if config is None:
            config = self.config

        txt = []
        for key, value in config.items():
            if value is not None:
                txt.append('{}={}'.format(key, value))
        if header:
            header += ' '
        logger.info(header + ', '.join(txt))

    @property
    def record(self) -> bool:
        """
        If ``True``, we'll configure the daq to record data. If ``False``, we
        will configure the daq to not record data.

        Setting this is the equivalent of scheduling a `configure` call to be
        executed later, e.g. ``configure(record=True)``, or putting to the
        record_cfg signal.
        """
        return self.record_cfg.get()

    @record.setter
    def record(self, record):
        self.preconfig(record=record, show_queued_cfg=False)

    def stage(self):
        """
        ``bluesky`` interface for preparing a device for action.

        This sets up the daq to end runs on run stop documents.
        It also caches the current state, so we know what state to return to
        after the ``bluesky`` scan.
        If a run is already started, we'll end it here so that we can start a
        new run during the scan.

        Returns
        -------
        staged: ``list``
            list of devices staged
        """
        logger.debug('Daq.stage()')
        self._pre_run_state = self.state
        if self._re_cbid is None:
            self._re_cbid = self._RE.subscribe(self._re_manage_runs)
        self.end_run()
        return [self]

    def _re_manage_runs(self, name, doc):
        """
        Callback for the RunEngine to manage run stop.
        """
        if name == 'stop':
            self.end_run()

    def unstage(self):
        """
        ``bluesky`` interface for undoing the `stage` routine.

        Returns
        -------
        unstaged: ``list``
            list of devices unstaged
        """
        logger.debug('Daq.unstage()')
        if self._re_cbid is not None:
            self._RE.unsubscribe(self._re_cbid)
            self._re_cbid = None
        # If we're still running, end now
        if self.state in ('Open', 'Running'):
            self.end_run()
        # Return to running if we already were (to keep AMI running)
        if self._pre_run_state == 'Running':
            self.begin_infinite()
        # For other states, end_run was sufficient.
        # E.g. do not disconnect, or this would close the open plots!
        return [self]

    # TODO see if pause/resume need to be bifurcated between lcls1 and lcls2
    def pause(self):
        """
        ``bluesky`` interface for determining what to do when a plan is
        interrupted. This will call `stop`, but it will not call `end_run`.
        """
        logger.debug('Daq.pause()')
        if self.state == 'Running':
            self.stop()

    def resume(self):
        """
        ``bluesky`` interface for determining what to do when an interrupted
        plan is resumed. This will call `begin`.
        """
        logger.debug('Daq.resume()')
        if self.state == 'Open':
            self.begin()

    def run_number(self, hutch_name=None):
        ... # TODO determine how to handle this one
        # LCLS1 uses an external script to get the run number because
        # the pydaq implementation does not work if done during a run
        # LCLS2 might have a better way


class DaqLCLS1(Daq):
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
    use_l3t_cfg = Cpt(Signal, value=False, kind='config', name='use_l3t')
    begin_sleep_cfg = Cpt(Signal, value=0, kind='config', name='begin_sleep')

    state_enum = enum.Enum(
        'PydaqState',
        'Disconnected Connected Configured Open Running',
        start=0,
    )
    requires_configure_transition = {'record', 'use_l3t'}

    def __init__(self, RE=None, hutch_name=None):
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
    def connected(self):
        """
        ``True`` if the daq is connected, ``False`` otherwise.
        """
        return self._control is not None

    @property
    def state(self):
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
    def connect(self):
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
                        logger.error(('Failed to connect: DAQ is not '
                                      'allocated!'))
            if not (err or conn):
                err = True
                logger.error(('Failed to connect: DAQ is not running on this '
                              'machine, and is not allocated!'))
            if err:
                logger.debug('del Daq.control')
                del self._control
                self._control = None
        else:
            logger.info('Connect requested, but already connected to DAQ')

    def disconnect(self):
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
    def wait(self, timeout=None, end_run=False):
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

    def begin(self, events=_CONFIG_VAL, duration=_CONFIG_VAL,
              record=_CONFIG_VAL, use_l3t=_CONFIG_VAL, controls=_CONFIG_VAL,
              wait=False, end_run=False):
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
            if record is not _CONFIG_VAL and record != self.record:
                old_record = self.record
                self.preconfig(record=record, show_queued_cfg=False)
            return super().begin(
                events=events,
                duration=duration,
                record=record,
                use_l3t=use_l3t,
                controls=controls,
                wait=wait,
            )
        finally:
            try:
                self.preconfig(record=old_record, show_queued_cfg=False)
            except NameError:
                pass

    @property
    def _begin_timeout(self):
        return BEGIN_TIMEOUT + BEGIN_THROTTLE

    def begin_infinite(self, record=_CONFIG_VAL, use_l3t=_CONFIG_VAL,
                       controls=_CONFIG_VAL):
        """
        Start the daq to run forever in the background.
        """
        self.begin(events=0, record=record, use_l3t=use_l3t,
                   controls=controls, wait=False, end_run=False)

    def _ender_thread(self):
        """
        End the run when the daq stops aquiring
        """
        self.wait()
        self.end_run()

    @check_connect
    def stop(self, success: bool = False):
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
    def end_run(self):
        """
        Call `stop`, then mark the run as finished.
        """
        logger.debug('Daq.end_run()')
        self.stop()
        self._control.endrun()

    # Reader interface
    @check_connect
    def trigger(self):
        """
        Begin acquisition. This method blocks until the run begins.

        Returns a status object that will be marked done when the daq has
        stopped acquiring.

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
    def kickoff(self, events=_CONFIG_VAL, duration=_CONFIG_VAL,
                use_l3t=_CONFIG_VAL, controls=_CONFIG_VAL):
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
            except StateTransitionError:
                err = ('Illegal reconfigure with {} during an open run. End '
                       'the current run with daq.end_run() before running '
                       'with a new configuration'.format(self.config))
                logger.debug(err, exc_info=True)
                raise StateTransitionError(err)

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
                tmo = BEGIN_THROTTLE - dt
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

        begin_status = Status(obj=self)
        watcher = threading.Thread(target=start_thread,
                                   args=(self._control, begin_status, events,
                                         duration, use_l3t, controls,
                                         next_run))
        watcher.start()
        return begin_status

    def complete(self):
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

    def _get_end_status(self):
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
            end_status = Status(obj=self)
            watcher = threading.Thread(target=finish_thread,
                                       args=(self._control, end_status))
            watcher.start()
            return end_status
        else:
            # Configured to run forever, say we're done so we can wait for just
            # the other things in the scan
            logger.debug('Returning finished status for infinite run with '
                         'events=%s, duration=%s', events, duration)
            status = Status(obj=self)
            status.set_finished()
            return status

    def preconfig(self, events=_CONFIG_VAL, duration=_CONFIG_VAL,
                  record=_CONFIG_VAL, use_l3t=_CONFIG_VAL,
                  controls=_CONFIG_VAL, begin_sleep=_CONFIG_VAL,
                  show_queued_cfg=True):
        """
        Queue configuration parameters for next call to `configure`.

        These will be overridden by arguments passed directly to `configure`.
        These will be cleared after each call to `configure`.

        This can be used to `configure` the `Daq` object without connecting.

        This will display the next queued configuration using logger.info,
        assuming the logger has been configured.
        """
        # Only one of (events, duration) should be preconfigured.
        if events is not _CONFIG_VAL:
            duration = _CONFIG_VAL

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
    def configure(self, events=_CONFIG_VAL, duration=_CONFIG_VAL,
                  record=_CONFIG_VAL, use_l3t=_CONFIG_VAL,
                  controls=_CONFIG_VAL, begin_sleep=_CONFIG_VAL):
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
            err = 'Cannot configure from state {}!'.format(state)
            raise StateTransitionError(err)

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
            self._queue_configure_transition = False
            self.configred_sig.put(True)
        except Exception as exc:
            msg = 'Failed to configure!'
            logger.debug(msg, exc_info=True)
            raise RuntimeError(msg) from exc
        return old, new

    def _config_args(self, record, use_l3t, controls):
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

    def _ctrl_arg(self, controls):
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
            try:
                val = device.position
            except AttributeError:
                val = device.get()
            try:
                val = val[0]
            except Exception:
                pass
            ctrl_arg.append((name, val))
        return ctrl_arg

    def _begin_args(self, events, duration, use_l3t, controls):
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
        if events is _CONFIG_VAL and duration is _CONFIG_VAL:
            # If both are omitted, use last configured values
            events = self.config['events']
            duration = self.config['duration']
        if events not in (None, _CONFIG_VAL):
            # We either passed the events arg, or loaded from config
            if use_l3t in (None, _CONFIG_VAL) and self.configured:
                use_l3t = self.config['use_l3t']
            if use_l3t:
                begin_args['l3t_events'] = events
            else:
                begin_args['events'] = events
        elif duration not in (None, _CONFIG_VAL):
            # We either passed the duration arg, or loaded from config
            secs = int(duration)
            nsec = int((duration - secs) * 1e9)
            begin_args['duration'] = [secs, nsec]
        else:
            # We passed None somewhere/everywhere
            begin_args['events'] = 0  # Run until manual stop
        if controls is _CONFIG_VAL:
            controls = self.config['controls']
        if controls is not None:
            begin_args['controls'] = self._ctrl_arg(controls)
        return begin_args

    def _check_duration(self, duration):
        if duration not in (None, _CONFIG_VAL) and duration < 1:
            msg = ('Duration argument less than 1 is unreliable. Please '
                   'use the events argument to specify the length of '
                   'very short runs.')
            raise RuntimeError(msg)

    @property
    def _events(self):
        """
        For the current `begin` cycle, how many ``events`` we told the daq to
        run for.
        """
        events = self._begin['events']
        if events is _CONFIG_VAL:
            events = self.config['events']
        return events

    @property
    def _duration(self):
        """
        For the current `begin` cycle, how long we told the daq to run for in
        seconds.
        """
        duration = self._begin['duration']
        if duration is _CONFIG_VAL:
            duration = self.config['duration']
        return duration

    @property
    def _infinite_run(self):
        if self._events is None and self._duration is None:
            return True
        return self._events in (-1, 0)

    def _reset_begin(self):
        """
        Reset ``_begin`` to starting values for when we aren't running.
        """
        self._begin = dict(events=None, duration=None, use_l3t=None,
                           controls=None)

    def run_number(self, hutch_name=None):
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
                raise ValueError(('{} is not a valid hutch, cannot determine '
                                  'run number'.format(hutch_name)))
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

    def set_filter(self, *args, event_codes=None, operator='&',
                   or_bykik=False):
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

    def set_monitor(self, det):
        return set_monitor_det(det)
    set_monitor.__doc__ = set_monitor_det.__doc__


class DaqLCLS2(Daq):
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
    # TODO add all the config options as components

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

    requires_configure_transition = {}
    all_transitions = set() # TODO fill this in
    _infinite_run: bool

    def __init__(self, platform, host, timeout, RE=None, hutch_name=None):
        if DaqControl is None:
            raise RuntimeError('psdaq is not installed, cannot use LCLS2 daq')
        super().__init__(RE=RE, hutch_name=hutch_name, platform=platform)
        self.state_sig.put(self.state_enum.from_any('reset'))
        self.transition_sig.put(self.transition_enum.from_any('reset'))
        self._control = DaqControl(host=host, platform=platform, timeout=timeout)
        self._infinite_run = False
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
                        self.transition_enum.from_any(self.info[1])
                    )
                    self.transition_elapsed_sig.put(info[2])
                    self.transition_total_sig.put(info[3])
                elif info[0] == 'step':
                    self.step_done_sig.put(info[1])
                else:
                    # Last case is normal status
                    self.transition_sig.put(
                        self.transition_enum.from_any(info[0])
                    )
                    self.state_sig.put(
                        self.state_enum.from_any(info[1])
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
            value >= self.state_enum.from_any('configured')
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
        done_status = self.get_done_status(timeout=timeout)
        done_status.wait()
        if end_run:
            self.end_run()

    def get_status_for(
        self,
        state: Optional[Iterator[Any]] = None,
        transition: Optional[Iterator[Any]] = None,
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


    def get_done_status(self, timeout: Optional[float] = None):
        """
        The DAQ is done acquiring if the most recent transition was not
        "beginrun", "beginstep", or "enable".
        """
        return self.get_status_for(
            transition=self.transition_enum.exclude(
                ['beginrun', 'beginstep', 'enable']
            ),
            timeout=timeout,
            check_now=True,
        )

    def begin_infinite(self):
        raise NotImplementedError(
            'Please implement begin_infinite in subclass.'
        )
        self._infinite_run = True

    def stop(self, success: bool = False) -> None:
        """
        Stop the current acquisition, ending it early.

        Parameters
        ----------
        success : bool, optional
            Flag set by bluesky to signify whether this was a good stop or a
            bad stop. Currently unused.
        """
        if self.state_sig.get() in self.state_enum.include(
            ['paused', 'running']
        ):
            self._control.setTransition('endstep')

    def end_run(self) -> None:
        """
        Call `stop`, then mark the run as finished.
        """
        if self.state_sig.get() in self.state_enum.include(
            ['starting', 'paused', 'running']
        ):
            self._control.setTransition('endrun')

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

    def kickoff(self) -> Status:
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
        if self.state_sig.get() < self.state_enum.from_any('connected'):
            raise RuntimeError('DAQ is not ready to run!')
        if self.state_sig.get() == self.state_enum.from_any('running'):
            raise RuntimeError('DAQ is already running!')
        phase1_info = {}
        # TODO where do I put the events per step
        data = {
            'motors': self._get_motors_for_configure(),
            'timestamp': 0,
            'detname': self.detname_sig.get(),
            'dettype': 'scan',
            'scantype': self.scan_type_sig.get(),
            'serial_number': 1234,
            'alg_name': 'raw',
            'alg_version': [1, 0, 0],
        },
        if self.state_cfg.get() == self.state_enum.from_any('connected'):
            # Add info for the Configure transition
            phase1_info['configure'] = {
                'NamesBlockHex': self._getBlock(
                    transition='Configure',
                    data=data,
                ),
            }
        if self.state_cfg.get() in self.state_enum.include(
            ['connected', 'configured', 'starting']
        ):
            # Add info for the BeginStep transition
            phase1_info['beginstep'] = {
                'ShapesDataBlockHex': self._getBlock(
                    transition='BeginStep',
                    data=data,
                ),
            }
        status = self.get_status_for(
            state=['running'],
            timeout=self.begin_timeout_cfg.get(),
        )
        # TODO handle state transitions in background thread to not block
        self._control.setState('running', phase1_info)
        return status


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
        done_status = self.get_done_status()
        if self._infinite_run:
            # Configured to run forever
            self.stop()
        return done_status

    def configure(self):
        ...

    def stage(self):
        ...

    def unstage(self):
        ...

    def run_number(self):
        ...


# TODO replace Any with the correct type hint, here and elsewhere
class HelpfulIntEnum(IntEnum):
    def from_any(self, identifier: Any) -> Type[HelpfulIntEnum]:
        """
        Try all the ways to interpret identifier as the enum
        """
        try:
            return self[identifier]
        except KeyError:
            return self(identifier)

    def include(
        self,
        identifiers: Iterator[Any],
    ) -> set[Type[HelpfulIntEnum]]:
        """
        Return all enum values matching the ones given.
        """
        return {self.from_any(ident) for ident in identifiers}

    def exclude(
        self,
        identifiers: Iterator[Any],
    ) -> set[Type[HelpfulIntEnum]]:
        """
        Return all enum values other than the ones given.
        """
        return set(self.__members__.values()) - self.include(identifiers)


class StateTransitionError(Exception):
    pass


_daq_instance = None


def register_daq(daq):
    """
    Called by `Daq` at the end of ``__init__`` to save our one daq instance as
    the real `Daq`. There will always only be one `Daq`.

    Parameters
    ----------
    daq: `Daq`
    """
    global _daq_instance
    _daq_instance = daq
    if daq.hutch_name is not None:
        set_ami_hutch(daq.hutch_name.lower())


def get_daq():
    """
    Called by other modules to get the registered `Daq` instance.

    Returns
    -------
    daq: `Daq`
    """
    return _daq_instance
