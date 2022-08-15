"""
This module defines a control interface for the LCLS1 DAQ.
"""
import enum
import functools
import logging
import os
import time
import threading
from importlib import import_module

from ophyd.status import Status, wait as status_wait

from . import ext_scripts
from .ami import set_pyami_filter, set_monitor_det

logger = logging.getLogger(__name__)
pydaq = None

# Wait up to this many seconds for daq to be ready for a begin call
BEGIN_TIMEOUT = 2
# Do not allow begins within this many seconds of a stop
BEGIN_THROTTLE = 1


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


class Daq:
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
    """
    _state_enum = enum.Enum('PydaqState',
                            'Disconnected Connected Configured Open Running',
                            start=0)
    default_config = dict(events=None,
                          duration=None,
                          use_l3t=False,
                          record=False,
                          controls=None,
                          begin_sleep=0)
    name = 'daq'
    parent = None

    def __init__(self, RE=None):
        if pydaq is None:
            globals()['pydaq'] = import_module('pydaq')
        super().__init__()
        self._control = None
        self._config = None
        self._desired_config = {}
        self._reset_begin()
        self._host = os.uname()[1]
        self._RE = RE
        self._re_cbid = None
        self._config_ts = {}
        self._update_config_ts()
        self._pre_run_state = None
        self._last_stop = 0
        register_daq(self)

    # Convenience properties
    @property
    def connected(self):
        """
        ``True`` if the daq is connected, ``False`` otherwise.
        """
        return self._control is not None

    @property
    def configured(self):
        """
        ``True`` if the daq is configured, ``False`` otherwise.
        """
        return self._config is not None

    @property
    def config(self):
        """
        The current configuration, e.g. the last call to `configure`
        """
        if self.configured:
            return self._config.copy()
        else:
            return self.default_config.copy()

    @property
    def next_config(self):
        """
        The next queued configuration.

        This can be different than `config` if we have queued up a
        configuration to be run on the next begin.
        """
        cfg = self.config
        cfg.update(self._desired_config)
        return cfg

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
        self._desired_config = self._config or {}
        self._config = None
        logger.info('DAQ is disconnected.')

    @check_connect
    def wait(self, timeout=None):
        """
        Pause the thread until the DAQ is done aquiring.

        Parameters
        ----------
        timeout: ``float``
            Maximum time to wait in seconds.
        """
        logger.debug('Daq.wait()')
        if self.state == 'Running':
            if self._events or self._duration:
                status = self._get_end_status()
                status_wait(status, timeout=timeout)
            else:
                raise RuntimeError('Cannot wait, daq configured to run '
                                   'forever.')

    def begin(self, events=None, duration=None, record=None, use_l3t=None,
              controls=None, wait=False, end_run=False):
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
        logger.debug(('Daq.begin(events=%s, duration=%s, record=%s, '
                      'use_l3t=%s, controls=%s, wait=%s)'),
                     events, duration, record, use_l3t, controls, wait)
        try:
            if record is not None and record != self.record:
                old_record = self.record
                self.preconfig(record=record, show_queued_cfg=False)
            begin_status = self.kickoff(events=events, duration=duration,
                                        use_l3t=use_l3t, controls=controls)
            status_wait(begin_status, timeout=self._begin_timeout)
            # In some daq configurations the begin status returns very early,
            # so we allow the user to configure an emperically derived extra
            # sleep.
            time.sleep(self.config['begin_sleep'])
            if wait:
                self.wait()
                if end_run:
                    self.end_run()
            if end_run and not wait:
                threading.Thread(target=self._ender_thread, args=()).start()
        except KeyboardInterrupt:
            self.end_run()
            logger.info('%s.begin interrupted, ending run', self.name)
        finally:
            try:
                self.preconfig(record=old_record, show_queued_cfg=False)
            except NameError:
                pass

    @property
    def _begin_timeout(self):
        return BEGIN_TIMEOUT + BEGIN_THROTTLE

    def begin_infinite(self, record=None, use_l3t=None, controls=None):
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
    def stop(self):
        """
        Stop the current acquisition, ending it early.
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
        cfg = self.next_config
        if all(cfg[key] is None for key in ('events', 'duration')):
            raise RuntimeError('Cannot start daq in scan step, did not '
                               'configure events or duration.')
        self.begin()
        return self._get_end_status()

    def read(self):
        """
        Return data. There is no data implemented yet.

        This also stops if running so you can use this device in a bluesky scan
        and wait for "everything else" to be done, then stop the daq
        afterwards.
        """
        if self.state == 'Running':
            self.stop()
        return {}

    def describe(self):
        """
        Explain what read returns. There is nothing  yet.
        """
        return {}

    # Flyer interface
    @check_connect
    def kickoff(self, events=None, duration=None, use_l3t=None, controls=None):
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
        if self._desired_config or not self.configured:
            try:
                self.configure()
            except StateTransitionError:
                err = ('Illegal reconfigure with {} during an open run. End '
                       'the current run with daq.end_run() before running '
                       'with a new configuration'.format(self._desired_config))
                logger.debug(err, exc_info=True)
                raise StateTransitionError(err)

        def start_thread(control, status, events, duration, use_l3t, controls):
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
                if self.config['record']:
                    try:
                        prev_run = self.run_number()
                        next_run = prev_run + 1
                        logger.info('Beginning daq run %s', next_run)
                    except (RuntimeError, ValueError):
                        logger.debug('Error getting run number in kickoff',
                                     exc_info=True)
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
                status._finished(success=True)
            else:
                logger.debug('Marking kickoff as failed')
                status._finished(success=False)

        begin_status = Status(obj=self)
        watcher = threading.Thread(target=start_thread,
                                   args=(self._control, begin_status, events,
                                         duration, use_l3t, controls))
        watcher.start()
        return begin_status

    def complete(self):
        """
        If the daq is freely running, this will `stop` the daq.
        Otherwise, we'll simply collect the end_status object.

        Returns
        -------
        end_status: ``Status``
            ``Status`` that will be marked as done when the DAQ has finished
            acquiring
        """
        logger.debug('Daq.complete()')
        end_status = self._get_end_status()
        if not (self._events or self._duration):
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

        if self._events or self._duration:
            def finish_thread(control, status):
                try:
                    logger.debug('Daq.control.end()')
                    control.end()
                except RuntimeError:
                    pass  # This means we aren't running, so no need to wait
                self._last_stop = time.time()
                self._reset_begin()
                status._finished(success=True)
                logger.debug('Marked acquisition as complete')
            end_status = Status(obj=self)
            watcher = threading.Thread(target=finish_thread,
                                       args=(self._control, end_status))
            watcher.start()
            return end_status
        else:
            # Configured to run forever, say we're done so we can wait for just
            # the other things in the scan
            return Status(obj=self, done=True, success=True)

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

    def preconfig(self, events=None, duration=None, record=None, use_l3t=None,
                  controls=None, begin_sleep=None, show_queued_cfg=True):
        """
        Queue configuration parameters for next call to `configure`.

        These will be overridden by arguments passed directly to `configure`.
        These will be cleared after each call to `configure`.

        This can be used to `configure` the `Daq` object without connecting.

        This will display the next queued configuration using logger.info,
        assuming the logger has been configured.
        """
        # Only one of (events, duration) should be preconfigured.
        if events is not None:
            self._desired_config['events'] = events
            self._desired_config['duration'] = None
        elif duration is not None:
            self._desired_config['events'] = None
            self._desired_config['duration'] = duration

        for arg, name in zip((record, use_l3t, controls, begin_sleep),
                             ('record', 'use_l3t', 'controls', 'begin_sleep')):
            if arg is not None:
                self._desired_config[name] = arg

        if show_queued_cfg:
            self.config_info(self.next_config, 'Queued config:')

    @check_connect
    def configure(self, events=None, duration=None, record=None,
                  use_l3t=None, controls=None, begin_sleep=None):
        """
        Changes the daq's configuration for the next run.

        This is the method that directly interfaces with the daq. If you simply
        want to get a configuration ready for later, use `preconfig`.

        Parameters
        ----------
        events: ``int``, optional
            If provided, the daq will run for this many events before
            stopping, unless we override in `begin`.
            If not provided, we'll use the ``duration`` argument instead.

        duration: ``int``, optional
            If provided, the daq will run for this many seconds before
            stopping, unless we override in `begin`.
            If not provided, and ``events`` was also not provided, an empty
            call like ``begin()`` will run indefinitely.

        record: ``bool``, optional
            If ``True``, we'll record the data. Otherwise, we'll run without
            recording. Defaults to ``False``, or the last set value for
            ``record``.

        use_l3t: ``bool``, optional
            If ``True``, an ``events`` argument to begin will be reinterpreted
            to only count events that pass the level 3 trigger. Defaults to
            its last configured value, or ``False``.

        controls: ``dict{name: device}`` or ``list[device...]``, optional
            If provided, values from these will make it into the DAQ data
            stream as variables. We will check ``device.position`` and
            ``device.value`` for quantities to use and we will update these
            values each time begin is called. To provide a list, all devices
            must have a ``name`` attribute.

        begin_sleep: ``int``, optional
            The amount of time to wait after the DAQ returns begin is done.
            This is a hack because the DAQ often says that a begin transition
            is done without actually being done, so it needs a short delay.

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
        old = self.read_configuration()

        self.preconfig(events=events, duration=duration, record=record,
                       use_l3t=use_l3t, controls=controls,
                       begin_sleep=begin_sleep, show_queued_cfg=False)
        config = self.next_config

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
            # self._config should reflect exactly the arguments to configure,
            # this is different than the arguments that pydaq.Control expects
            self._config = dict(events=events, duration=duration,
                                record=record, use_l3t=use_l3t,
                                controls=controls, begin_sleep=begin_sleep)
            self._update_config_ts()
            self.config_info(header='Daq configured:')
        except Exception as exc:
            self._config = None
            msg = 'Failed to configure!'
            logger.debug(msg, exc_info=True)
            raise RuntimeError(msg) from exc
        new = self.read_configuration()
        self._desired_config = {}
        return old, new

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
    def record(self):
        """
        If ``True``, we'll configure the daq to record data. If ``False``, we
        will configure the daq to not record data.

        Setting this is the equivalent of scheduling a `configure` call to be
        executed later, e.g. ``configure(record=True)``
        """
        return self.next_config['record']

    @record.setter
    def record(self, record):
        self.preconfig(record=record)

    def _update_config_ts(self):
        """
        Create timestamps and update the ``bluesky`` readback for
        `read_configuration`
        """
        for k, v in self.config.items():
            old_value = self._config_ts.get(k, {}).get('value')
            if old_value is None or v != old_value:
                self._config_ts[k] = dict(value=v,
                                          timestamp=time.time())

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
        config_args['record'] = record
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
                val = device.value
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
        if events is None and duration is None:
            events = self.config['events']
            duration = self.config['duration']
        if events is not None:
            if use_l3t is None and self.configured:
                use_l3t = self.config['use_l3t']
            if use_l3t:
                begin_args['l3t_events'] = events
            else:
                begin_args['events'] = events
        elif duration is not None:
            secs = int(duration)
            nsec = int((duration - secs) * 1e9)
            begin_args['duration'] = [secs, nsec]
        else:
            begin_args['events'] = 0  # Run until manual stop
        if controls is None:
            ctrl_dict = self.config['controls']
            if ctrl_dict is not None:
                begin_args['controls'] = self._ctrl_arg(ctrl_dict)
        else:
            begin_args['controls'] = self._ctrl_arg(controls)
        return begin_args

    def _check_duration(self, duration):
        if duration is not None and duration < 1:
            msg = ('Duration argument less than 1 is unreliable. Please '
                   'use the events argument to specify the length of '
                   'very short runs.')
            raise RuntimeError(msg)

    def read_configuration(self):
        """
        ``bluesky`` interface for checking the current configuration

        Returns
        -------
        config: ``dict``
            Mapping of config key to current configured value and timestamp
            when it was last set.
        """
        logger.debug('Daq.read_configuration()')
        return self._config_ts.copy()

    def describe_configuration(self):
        """
        ``bluesky`` interface for describing how to interpret the configured
        values

        Returns
        -------
        config_desc: ``dict``
            Mapping of config key to field metadata.
        """
        logger.debug('Daq.describe_configuration()')
        try:
            controls_shape = [len(self.config['controls']), 2]
        except (TypeError, RuntimeError, AttributeError):
            controls_shape = []
        return dict(events=dict(source='daq_events_in_run',
                                dtype='number',
                                shape=[]),
                    duration=dict(source='daq_run_duration',
                                  dtype='number',
                                  shape=[]),
                    use_l3t=dict(source='daq_use_l3trigger',
                                 dtype='number',
                                 shape=[]),
                    record=dict(source='daq_record_run',
                                dtype='number',
                                shape=[]),
                    controls=dict(source='daq_control_vars',
                                  dtype='array',
                                  shape=controls_shape),
                    begin_sleep=dict(source='daq_begin_sleep',
                                     dtype='number',
                                     shape=[]),
                    )

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
        Callback for the RunEngine to manage run start and stop.
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
        # Return to the state we had at stage
        if self._pre_run_state == 'Disconnected':
            self.disconnect()
        elif self._pre_run_state == 'Running':
            self.begin_infinite()
        # For other states, end_run was sufficient.
        return [self]

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

    @property
    def _events(self):
        """
        For the current `begin` cycle, how many ``events`` we told the daq to
        run for.
        """
        return self._begin['events'] or self.config['events']

    @property
    def _duration(self):
        """
        For the current `begin` cycle, how long we told the daq to run for in
        seconds.
        """
        return self._begin['duration'] or self.config['duration']

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
        """
        try:
            if hutch_name is None:
                hutch_name = ext_scripts.hutch_name()
            if hutch_name not in ('tmo', 'rix', 'txi', 'xpp', 'xcs', 'mfx',
                                  'cxi', 'mec', 'det', 'tst'):
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

    def set_filter(self, *args, event_codes=None, operator='&', or_bykik=True):
        return set_pyami_filter(*args, event_codes=event_codes,
                                operator=operator, or_bykik=or_bykik)
    set_filter.__doc__ = set_pyami_filter.__doc__

    def set_monitor(self, det):
        return set_monitor_det(det)
    set_monitor.__doc__ = set_monitor_det.__doc__


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


def get_daq():
    """
    Called by other modules to get the registered `Daq` instance.

    Returns
    -------
    daq: `Daq`
    """
    return _daq_instance
