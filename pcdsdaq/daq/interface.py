"""
Module that defines the Daq control interface and supplies some basic tools.

This interface is intended to be shared between all implementations of
Daq control.

The interfaces defined here have three primary sources:
1. Items that are required to use the Daq device in a Bluesky scan
2. Items that are maintained from an older version of python daq control
3. Items that shape the Daq in the image of an Ophyd device for ease of
   management and consistently with other beamline devices.
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum, IntEnum
from functools import cache
from typing import Any, ClassVar, Iterator, NewType, Optional, Type, Union

from bluesky import RunEngine
from ophyd.device import Component as Cpt
from ophyd.device import Device
from ophyd.ophydobj import Kind
from ophyd.positioner import PositionerBase
from ophyd.signal import AttributeSignal, Signal
from ophyd.status import DeviceStatus
from ophyd.utils import StatusTimeoutError, WaitTimeoutError

from ..ami import set_ami_hutch

logger = logging.getLogger(__name__)

# Wait up to this many seconds for daq to be ready for a begin call
BEGIN_TIMEOUT = 15
# Do not allow begins within this many seconds of a stop
BEGIN_THROTTLE = 1

# Not-None sentinal for default value when None has a special meaning
# Indicates that the last configured value should be used
SENTINEL = NewType('SENTINEL')
CONFIG_VAL = SENTINEL()


class DaqTimeoutError(Exception):
    pass


class DaqBase(Device):
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
    _last_config: dict[str, Any]
    _queue_configure_transition: bool

    def __init__(
        self,
        RE: Optional[RunEngine] = None,
        hutch_name: Optional[str] = None,
        platform: Optional[int] = None,
        *,
        name: str = 'daq',
    ):
        self._RE = RE
        self.hutch_name = hutch_name
        self.platform = platform
        self._last_config = {}
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

    def trigger(self) -> DeviceStatus:
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

    def read(self) -> dict[str, dict[str, Any]]:
        """
        Return data about the status of the daq.

        This also stops if running so you can use this device in a bluesky scan
        and wait for "everything else" to be done, then stop the daq
        afterwards.
        """
        if self.state == 'Running':
            self.stop()
        return super().read()

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
        raise NotImplementedError('Please implement kickoff in subclass.')

    def complete(self) -> DeviceStatus:
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

    def describe_collect(self) -> dict:
        """
        As per the ``bluesky`` interface, this is how you interpret the null
        data from `collect`. There isn't anything here, as nothing will be
        collected.
        """
        logger.debug('Daq.describe_collect()')
        return {}

    def preconfig(self, show_queued_cfg: bool = True, **kwargs):
        """
        Write to the configuration signals without executing any transitions.

        Will store the boolean _queue_configure_transition if any
        configurations were changed that require a configure transition.
        """
        for key, value in kwargs.items():
            if isinstance(value, SENTINEL):
                continue
            try:
                sig = getattr(self, key + '_cfg')
            except AttributeError as exc:
                raise ValueError(
                    f'Did not find config parameter {key}'
                ) from exc
            if isinstance(value, None):
                value = self.default_config[key]
            if isinstance(sig, Signal) and sig.kind == 'config':
                sig.put(value)
            else:
                raise ValueError(
                    f'{key} is not a config parameter!'
                )

        if self._last_config:
            self._queue_configure_transition = False
            for key in self.requires_configure_transition:
                if self.config[key] != self._last_config[key]:
                    self._queue_configure_transition = True
                    break
        else:
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

    def config_info(
        self,
        config: Optional[dict[str, Any]] = None,
        header: str = 'Config:',
    ) -> None:
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
    def record(self, record: bool):
        self.preconfig(record=record, show_queued_cfg=False)

    def stage(self) -> list[DaqBase]:
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

    def _re_manage_runs(self, name: str, doc: dict[str, Any]):
        """
        Callback for the RunEngine to manage run stop.
        """
        if name == 'stop':
            self.end_run()

    def unstage(self) -> list[DaqBase]:
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
        if self.state in ('Open', 'Running', 'running'):
            self.end_run()
        # Return to running if we already were (to keep AMI running)
        if self._pre_run_state in ('Running', 'running'):
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

    def run_number(self) -> int:
        """
        Return the number of the current run, or the previous run otherwise.
        """
        raise NotImplementedError('Please implement run_number in subclass.')


def typing_check(value, hint):
    # This works for basic types
    try:
        return isinstance(value, hint)
    except TypeError:
        ...
    # This works for unions of basic types
    try:
        return isinstance(value, hint.__args__)
    except TypeError:
        ...
    # This works for unions that include subscripted generics
    cls_to_check = []
    for arg in hint.__args__:
        try:
            cls_to_check.append(arg.__origin__)
        except AttributeError:
            cls_to_check.append(arg)
    return isinstance(value, cls_to_check)


ControlsObject = Union[PositionerBase, Signal]
ControlsArg = Union[
    list[ControlsObject],
    dict[str, ControlsObject],
]

EnumId = Union[Type[Enum], int, str]


class HelpfulIntEnum(IntEnum):
    def from_any(self, identifier: EnumId) -> Type[HelpfulIntEnum]:
        """
        Try all the ways to interpret identifier as the enum
        """
        try:
            return self[identifier]
        except KeyError:
            return self(identifier)

    def include(
        self,
        identifiers: Iterator[EnumId],
    ) -> set[Type[HelpfulIntEnum]]:
        """
        Return all enum values matching the ones given.
        """
        return {self.from_any(ident) for ident in identifiers}

    def exclude(
        self,
        identifiers: Iterator[EnumId],
    ) -> set[Type[HelpfulIntEnum]]:
        """
        Return all enum values other than the ones given.
        """
        return set(self.__members__.values()) - self.include(identifiers)


class StateTransitionError(Exception):
    pass


def get_controls_value(obj: ControlsObject) -> Any:
    try:
        return obj.position
    except Exception:
        return obj.get()


_daq_instance = None


# TODO this breaks lcls2 daq, need to fix
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
