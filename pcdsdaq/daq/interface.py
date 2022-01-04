"""
Module that defines the Daq control interface and supplies some basic tools.

This interface is intended to be shared between all implementations of
Daq control.

The interfaces defined here have three primary sources:
1. Items that are required to use the Daq device in a Bluesky scan
2. Items that are maintained from an older version of python daq control
3. Items that shape the Daq in the image of an Ophyd device for ease of
   management and consistency with other beamline devices.
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum, IntEnum
from functools import cache
from typing import Any, ClassVar, Generator, Iterator, NewType, Optional, Union

from bluesky import RunEngine
from ophyd.device import Component as Cpt
from ophyd.device import Device
from ophyd.ophydobj import Kind, OphydObject
from ophyd.positioner import PositionerBase
from ophyd.signal import AttributeSignal, Signal
from ophyd.status import DeviceStatus
from ophyd.utils import StatusTimeoutError, WaitTimeoutError

from ..ami import set_ami_hutch

logger = logging.getLogger(__name__)

# Not-None sentinal for default value when None has a special meaning
# Indicates that the last configured value should be used
SENTINEL = NewType('SENTINEL', object)
CONFIG_VAL = SENTINEL(object())

# The DAQ singleton
_daq_instance = None

# Types
# Type hint for objects that have .position or .get
ControlsObject = Union[PositionerBase, Signal]
# Type hint for valid arguments to configure(controls=)
ControlsArg = Union[
    list[ControlsObject],
    dict[str, ControlsObject],
]
# Type hint for valid identifiers for an enum
EnumId = Union[Enum, int, str]


class HelpfulIntEnum(IntEnum):
    """
    IntEnum subclass with some utility extensions.
    """
    def from_any(self, identifier: EnumId) -> HelpfulIntEnum:
        """
        Try all the ways to interpret identifier as the enum.

        This is intended to consolidate the try/except tree typically used
        to interpret external input as an enum.

        Parameters
        ----------
        identifier : EnumId
            Any str, int, or Enum value that corresponds with a valid value
            on this HelpfulIntEnum instance.

        Returns
        -------
        enum : HelpfulIntEnum
            The corresponding enum object associated with the identifier.
        """
        try:
            return self[identifier]
        except KeyError:
            return self(identifier)

    def include(
        self,
        identifiers: Iterator[EnumId],
    ) -> set[HelpfulIntEnum]:
        """
        Returns all enum values matching the identifiers given.

        This is a shortcut for calling self.from_any many times and
        assembling a set of the results.

        Parameters
        ----------
        identifiers : Iterator[EnumId]
            Any iterable that contains strings, ints, and Enum values that
            correspond with valid values on this HelpfulIntEnum instance.

        Returns
        -------
        enums : set[HelpfulIntEnum]
            A set whose elements are the enum objects associated with the
            input identifiers.
        """
        return {self.from_any(ident) for ident in identifiers}

    def exclude(
        self,
        identifiers: Iterator[EnumId],
    ) -> set[HelpfulIntEnum]:
        """
        Return all enum values other than the ones given.

        Parameters
        ----------
        identifiers : Iterator[EnumId]
            Any iterable that contains strings, ints, and Enum values that
            correspond with valid values on this HelpfulIntEnum instance.

        Returns
        -------
        enums : set[HelpfulIntEnum]
            A set whose elements are the valid enum objects not associated
            with the input identifiers.
        """
        return set(self.__members__.values()) - self.include(identifiers)


class DaqError(Exception):
    """
    Base class for DAQ exceptions.

    External users can try/except on this exception class as a catch-all for
    DAQ-specific exceptions.
    """
    ...


class DaqTimeoutError(DaqError):
    """
    Exception raised when the DAQ times out.

    This encompasses cases where we ask for a specific action, but we observe
    that nothing has happened for too long a duration, so we don't know
    if the operation will ever complete.
    """
    ...


class DaqStateTransitionError(DaqError):
    """
    Exception raised when a DAQ state transition fails.

    This is distinct from a timeout where we aren't sure what's happened.
    This is the case where we know that something definitely has gone wrong.
    """
    ...


# Helper functions
def get_daq() -> Optional[DaqBase]:
    """
    Called by other modules to get the registered `DaqBase` instance.

    This will return None if there is no such registered instance.

    Returns
    -------
    daq: `DaqBase`
    """
    return _daq_instance


def register_daq(daq: DaqBase) -> None:
    """
    Called by `DaqBase` at the end of ``__init__`` to save our one daq
    instance as the real `DaqBase`. There will always only be one `DaqBase`.

    Parameters
    ----------
    daq: `DaqBase`
    """
    global _daq_instance
    _daq_instance = daq
    if daq.hutch_name is not None:
        set_ami_hutch(daq.hutch_name.lower())


def get_controls_value(obj: ControlsObject) -> Any:
    """
    Return the primary value associated with a controls object.

    In the case of positioners, this will be the .position attribute.
    In the case of signals, this will be the return value from .get.

    Parameters
    ----------
    obj : ControlsObject
        The positioner or signal to extract a value from.

    Returns
    -------
    val: Any
        The value associated with that signal. Most commonly this will
        be a float, but it could be any Python type.
    """
    try:
        return obj.position
    except Exception:
        return obj.get()


def typing_check(value: Any, hint: Any) -> bool:
    """
    A best-effort check if value matches the given type hint.

    This is not expected to work outside of the context of this module
    and its behavior is subject to change without notice.

    The intended use case is for parsing through the type annotations on
    the configure methods.

    Parameters
    ----------
    value : Any
        Any value to check.
    hint : Any
        Any type hint to check against.

    Returns
    -------
    ok : bool
        True if the value matches the hint, False otherwise.
    """
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


def clip_name(obj: OphydObject):
    """
    Remove everything after and including the last underscore in a name.

    This lets me have nice looking read keys without needing to override
    legacy api from before these classes were ophyd-ized.
    """
    obj.name = '_'.join(obj.name.split('_')[:-1])


# Base classes
class DaqBase(Device):
    """
    Base class to define shared DAQ API

    All subclasses should implement the "not implemented" methods here.

    Also defines some shared features so that different DAQ versions
    do not have to reinvent the wheel for basic API choices.
    """
    state_sig = Cpt(AttributeSignal, 'state', kind='normal')
    configured_sig = Cpt(Signal, value=False, kind='normal')

    events_cfg = Cpt(Signal, value=None, kind='config')
    duration_cfg = Cpt(Signal, value=None, kind='config')
    record_cfg = Cpt(Signal, value=None, kind='config')
    controls_cfg = Cpt(Signal, value=None, kind='config')
    begin_timeout_cfg = Cpt(Signal, value=15, kind='config')
    begin_throttle_cfg = Cpt(Signal, value=1, kind='config')
    begin_sleep_cfg = Cpt(Signal, value=0, kind='config')

    # Define these in subclass
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
        logger.debug(
            'DaqBase.__init__(%s, %s, %s, %s)',
            RE,
            hutch_name,
            platform,
            name,
        )
        self._RE = RE
        self.hutch_name = hutch_name
        self.platform = platform
        self._last_config = {}
        self._queue_configure_transition = True
        super().__init__(name=name)
        for cpt_name in self.component_names:
            clip_name(getattr(self, cpt_name))
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
            return {key: info['value'] for key, info in cfg.items()}
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
        kickoff, in case the DAQ API requires parameters to be started.

        In this base class, we handle:
        - calling kickoff
        - waiting for the kickoff to complete
        - waiting for begin_sleep
        - waiting for the acquisition to run if requested
        - ending the run if requested
        - stopping or ending the run on ctrl+c as appropriate

        Parameters
        ----------
        wait : ``bool``, optional
            If True, wait for the daq to be done running.
        end_run : ``bool``, optional
            If True, end the daq after we're done running.
        """
        logger.debug(
            'DaqBase.begin(wait=%s, end_run=%s, kwargs=%s)',
            wait,
            end_run,
            kwargs,
        )
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
        """
        Start the DAQ in such a way that it runs until asked to stop.

        This is a shortcut included so that the user does not have to remember
        the specifics of how to get the daq to run indefinitely.
        """
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
        End the current run. This includes a stop if needed.
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
        afterwards. This is occasionally used in sequencer-guided scans.
        """
        logger.debug("DaqBase.read()")
        if self.state.lower() == 'running':
            logger.debug("Stopping DAQ in DaqBase.read()")
            self.stop()
        return super().read()

    def kickoff(self, **kwargs) -> DeviceStatus:
        """
        Begin acquisition. This method is non-blocking.

        Bluesky will not pass in any parameters, but kwargs can be forwarded
        to the DAQ API if needed. All kwargs should have a cooresponding
        configuration signal so we know what values to use during a bluesky
        scan.

        This is part of the ``bluesky`` ``Flyer`` interface.

        Returns
        -------
        ready_status: ``Status``
            ``Status`` that will be marked as done when the daq has begun.
        """
        raise NotImplementedError('Please implement kickoff in subclass.')

    def complete(self) -> DeviceStatus:
        """
        Return a status that will be marked as done after acquisition stops.

        If the daq is freely running, this will `stop` the daq.
        Otherwise, we'll simply collect the end_status object and wait for
        the acquisition to end as scheduled.

        This is part of the ``bluesky`` ``Flyer`` interface.

        Returns
        -------
        end_status: ``Status``
            ``Status`` that will be marked as done when the DAQ has finished
            acquiring
        """
        raise NotImplementedError('Please implement complete in subclass.')

    def collect(self) -> Generator[None, None, None]:
        """
        Collect data as part of the ``bluesky`` ``Flyer`` interface.

        As per the ``bluesky`` interface, this is a generator that is expected
        to output partial event documents. However, since we don't have any
        events to report to python, this will be a generator that immediately
        ends.
        """
        logger.debug('DaqBase.collect()')
        yield from ()

    def describe_collect(self) -> dict:
        """
        Interpret the data from `collect`.

        There isn't anything here, as nothing will be collected.

        Returns
        -------
        desc : dict
            An empty dictionary.
        """
        logger.debug('DaqBase.describe_collect()')
        return {}

    def preconfig(self, show_queued_cfg: bool = True, **kwargs) -> None:
        """
        Write to the configuration signals without executing any transitions.

        None values are interpreted as "return to the default config"
        CONFIG_VAL sentinels are interpreted as "do not change anything"

        Will store the boolean _queue_configure_transition if any
        configurations were changed that require a configure transition.

        Parameters
        ----------
        show_queued_cfg : bool, optional
            If True, gives a nice printout of the new configuration.
        """
        logger.debug(
            "DaqBase.preconfig(show_queued_cfg=%s, kwargs=%s)",
            show_queued_cfg,
            kwargs,
        )
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

        kwargs are passed straight through to preconfig.

        Returns
        -------
        (old, new) : tuple
            The previous configuration and the new configuration after calling
            this method. This is a requirement of the bluesky interface.
        """
        logger.debug("DaqBase.configure(kwargs=%s)", kwargs)
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
        logger.debug(
            "DaqBase.config_info(config=%s, header=%s)",
            config,
            header,
        )
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
    def record(self) -> Optional[bool]:
        """
        Whether or not to record data.

        If ``True``, we'll configure the daq to record data.
        If ``False``, we'll configure the daq to not record data.
        If ``None``, we'll keep the current record/norecord state.

        Setting this is the equivalent of scheduling a `configure` call to be
        executed later, e.g. ``configure(record=True)``, or putting to the
        record_cfg signal.
        """
        return self.record_cfg.get()

    @record.setter
    def record(self, record: Optional[bool]):
        self.preconfig(record=record, show_queued_cfg=False)

    # TODO evaluate if lcls2 daq needs to set self to connected state
    # on stage/unstage as in Chris Ford's implementation
    # should there be a way from the Python to get to connected state
    # in lcls2?
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
        logger.debug('DaqBase.stage()')
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
        logger.debug('DaqBase.unstage()')
        if self._re_cbid is not None:
            self._RE.unsubscribe(self._re_cbid)
            self._re_cbid = None
        # If we're still running, end now
        if self.state.lower() in ('open', 'running', 'starting', 'paused'):
            logger.debug("Ending run in DaqBase.unstage()")
            self.end_run()
        # Return to running if we already were (to keep AMI running)
        if self._pre_run_state.lower() == 'running':
            logger.debug("Starting infinite run in DaqBase.unstage()")
            self.begin_infinite()
        # For other states, end_run was sufficient.
        # E.g. do not disconnect, or this would close the open plots!
        return [self]

    def pause(self) -> None:
        """
        Called when a bluesky plan is interrupted.

        This will call `stop`, but it will not call `end_run`.

        This is not to be confused with the "paused" state in the lcls2 DAQ,
        which is semantically different. For LCLS2, this ends the step
        drops us to the 'starting' state.
        """
        logger.debug('DaqBase.pause()')
        if self.state.lower() in ('running', 'paused'):
            self.stop()

    # TODO this might need to call kickoff for lcls2 instead of begin
    def resume(self) -> None:
        """
        Called when an interrupted bluesky plan is resumed.

        This will call `begin`.
        """
        logger.debug('DaqBase.resume()')
        if self.state == 'Open':
            self.begin()

    def run_number(self) -> int:
        """
        Return the number of the current run, or the previous run otherwise.
        """
        raise NotImplementedError('Please implement run_number in subclass.')
