import logging
import time
from importlib import import_module
from threading import Thread

import numpy as np
from ophyd.device import Component as Cpt
from ophyd.device import Device, Staged
from ophyd.signal import Signal
from ophyd.status import Status
from ophyd.utils.errors import ReadOnlyError
from toolz.itertoolz import partition

from .ext_scripts import get_ami_proxy, get_hutch_name

logger = logging.getLogger(__name__)
L3T_DEFAULT = '/reg/neh/operator/{}opr/l3t/amifil.l3t'

# Set uninitialized globals for style-checker
pyami = None
pyami_connected = None
ami_proxy = None
l3t_file = None
monitor_det = None
last_filter_string = None
hutch_name = None


# Define default starting values. Can also use to reset module.
def _reset_globals():
    defaults = dict(pyami=None,
                    pyami_connected=False,
                    ami_proxy=None,
                    l3t_file=None,
                    monitor_det=None,
                    last_filter_string=None,
                    hutch_name=None)
    globals().update(defaults)


_reset_globals()


def auto_setup_pyami():
    """
    Does a best-guess at the ami configuration, if it has not yet been setup.

    The steps are:

    1. check hutch name
    2. determine ami proxy and register it
    3. setup detault l3t file
    4. makes sure pyami is imported and connected to the ami proxy

    This will be called the first time pyami is needed. We don't import at the
    top of this file because we need to be able to import this file even if
    pyami isn't in the environment, which is semi-frequent.
    """
    global pyami
    global pyami_connected

    if None in (ami_proxy, l3t_file):
        # get_hutch_name fails if not on nfs
        # or on a bad nfs day, so only do if 100% needed
        hutch = hutch_name or get_hutch_name()

    if ami_proxy is None:
        proxy = get_ami_proxy(hutch)
        set_pyami_proxy(proxy)

    if l3t_file is None:
        set_l3t_file(L3T_DEFAULT.format(hutch))

    if pyami is None:
        logger.debug('importing pyami')
        pyami = import_module('pyami')

    if not pyami_connected:
        logger.debug('initializing pyami')
        try:
            pyami.connect(ami_proxy)
            pyami_connected = True
        except Exception:
            pyami_connected = False
            raise


def set_ami_hutch(name):
    """
    Pick the hutch name to skip a shell out to get_hutch_name.

    Parameters
    ----------
    name: ``str``
        Name of the hutch
    """
    global hutch_name
    hutch_name = name.lower()


def set_pyami_proxy(proxy):
    """
    Pick the hostname or group to use for the pyami connection.

    Parameters
    ----------
    proxy: ``str`` or ``int``
        Either the server name or group number
    """
    global ami_proxy
    ami_proxy = proxy


def set_l3t_file(filename):
    """
    Pick the file to write out for the l3t trigger

    Parameters
    ----------
    filename: ``str``
        Full file path
    """
    global l3t_file
    l3t_file = filename


def set_monitor_det(det):
    """
    Designate one `AmiDet` as the monitor.

    The monitor det is the default normalization detector and the default
    filtering detector when no detector is provided.

    Parameters
    ----------
    det: `AmiDet` or ``bool``
        The detector to set as the monitor. Alternatively, pass in ``False`` to
        disable the monitor det.
    """
    global monitor_det
    if det:
        monitor_det = det
    else:
        monitor_det = None


def set_pyami_filter(*args, event_codes=None, operator='&', or_bykik=False):
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
    global last_filter_string

    auto_setup_pyami()
    filter_string = dets_filter(*args, event_codes=event_codes,
                                operator=operator, or_bykik=or_bykik)
    if filter_string is None:
        pyami.clear_l3t()
    else:
        pyami.set_l3t(filter_string, l3t_file)
        last_filter_string = filter_string


def dets_filter(*args, event_codes=None, operator='&', or_bykik=True):
    """
    Return valid l3t/pyami filter strings in a useful format.

    The function takes in arbitrary dets whose prefixes are the ami names,
    along with low and highs. Event codes are handled as a special case, since
    you always want high vs low.

    .. note::
        By default this will treat bykik at an l3t pass! This is so you don't
        lose your off shots when the l3t trigger is in veto mode. You can
        disable this with ``or_bykik=False``, but this will remain the default
        behavior for backwards compatibility and to prevent someone from losing
        shots that they wanted in the data.

    Parameters
    ----------
    *args: (`AmiDet`, ``float``, ``float``) n times
        A sequence of (detector, low, high), which create filters that make
        sure the detector is between low and high. You can omit the first
        `AmiDet` as a shorthand for the current monitor, assuming a monitor has
        been set with `Daq.set_monitor` or `set_monitor_det`.

    event_codes: ``list``, optional
        A list of event codes to include in the filter. l3pass will be when the
        event code is present.

    operator: ``str``, optional
        The operator for combining the detector ranges and event codes. This
        can either be ``|`` to ``or`` the conditions together, so l3pass will
        happen if any filter passes, or it can be left at the default ``&`` to
        ``and`` the conditions together, so l3pass will only happen if all
        filters pass.

    or_bykik: ``bool``, optional
        True by default, appends an ``or`` condition that marks l3t pass when
        we see the bykik event code. This makes sure the off shots make it into
        the data if we're in l3t veto mode.

    Returns
    -------
    filter_string: ``str``
        A valid filter string for `AmiDet` or for ``pyami.set_l3t``
    """
    filter_strings = []
    if len(args) % 3 == 2:
        # One arg missing, add the monitor det as first arg
        if monitor_det is None:
            raise RuntimeError('Did not recieve args multiple of 3, but ',
                               'monitor_det is not set. Aborting.')
        else:
            args = [monitor_det] + list(args)
    for det, lower, upper in partition(3, args):
        if isinstance(det, str):
            ami_name = det
        elif isinstance(det, AmiDet):
            ami_name = det.prefix
        else:
            raise TypeError('Must use AmiDet or string for filtering!')
        filter_strings.append(basic_filter(ami_name, lower, upper))
    if event_codes is not None:
        for code in event_codes:
            filter_strings.append(evr_filter(code))
    if len(filter_strings) == 0:
        return None
    else:
        base = concat_filter_strings(filter_strings, operator=operator)
        if or_bykik:
            bykik = evr_filter(162)
            return concat_filter_strings([base, bykik], operator='|')
        else:
            return base


def basic_filter(ami_name, lower, upper):
    """
    Helper function for creating an ami filter string.

    Parameters
    ----------
    ami_name: ``str``
        The name of the value in ami

    lower: ``float``
        The lower bound for the value to pass

    upper: ``float``
        The upper bound for the value to pass

    Returns
    -------
    filter_string: ``str``
    """
    return f'{lower}<{ami_name}<{upper}'


def evr_filter(event_code):
    """
    Helper function that creates a filter for a certain event code.

    Parameters
    ----------
    event_code: ``int``
        The event code to create a filter for

    Returns
    -------
    filter_string: ``str``
    """
    evr_base = 'DAQ:EVR:Evt{}'
    return basic_filter(evr_base.format(event_code), 0.1, 2)


def concat_filter_strings(filter_strings, operator='&'):
    """
    Helper function to combine ami filter strings

    Parameters
    ----------
    filter_strings: ``list``
        The valid filter strings to combine

    operator: ``str``
        The operator to place between the filter strings. This can either be
        ``&`` or ``|``, for ``and`` or ``or`` respectively.
    """
    if len(filter_strings) == 0:
        raise ValueError('filter_strings must have at least one element')
    elif len(filter_strings) == 1:
        return filter_strings[0]
    else:
        sep = ')' + operator + '('
        return '(' + sep.join(filter_strings) + ')'


class AmiDet(Device):
    """
    Detector that gets data from pyami scalars.

    The data will be in the form of an accumulated mean, rms, and number
    of entries used in the calculations. The raw data is not avaiable via
    pyami.

    This only supports scalars. The array features are known to crash both the
    python session and active ami clients, so don't use them.

    Parameters
    ----------
    prefix: ``str``
        The ami name to use to retrieve the data.

    name: ``str``, required keyword
        The shorter name to use to label the data.

    filter_str: ``str``, optional
        If provided, we'll filter the incoming data using this filter string.
        If omitted or None, we'll use the last set_l3t string.
        If False, but not None, we'll do no filtering at all. This includes the
        empty string.

    min_duration: ``float``, optional
        If provided, we'll wait this many seconds before declaring the
        acquisition as complete. Otherwise, we'll stop acquring on read.

    normalize: ``bool`` or ``AmiDet``, optional
        Determines the normalization behavior of this detector. The default is
        ``True``, which means normalize to the current ``monitor_det``. See
        `set_monitor_det`. ``False`` means do not normalize. You can also pass
        in any other detector to normalize against something that is not the
        ``monitor_det``.
    """
    mean = Cpt(Signal, value=0., kind='hinted')
    err = Cpt(Signal, value=0., kind='hinted')
    entries = Cpt(Signal, value=0, kind='normal')
    mean_raw = Cpt(Signal, value=0., kind='normal')
    err_raw = Cpt(Signal, value=0., kind='normal')
    mean_mon = Cpt(Signal, value=0., kind='normal')
    err_mon = Cpt(Signal, value=0., kind='normal')
    entries_mon = Cpt(Signal, value=0., kind='normal')
    mon_prefix = Cpt(Signal, value='', kind='normal')
    rms = Cpt(Signal, value=0., kind='omitted')

    def __init__(self, prefix, *, name, filter_string=None, min_duration=0,
                 normalize=True):
        auto_setup_pyami()
        self._entry = None
        self._monitor = None
        self.filter_string = filter_string
        self.min_duration = min_duration
        self.normalize = normalize
        super().__init__(prefix, name=name)

    def stage(self):
        """
        Called early in a bluesky scan to initialize the pyami.Entry object.

        Note that pyami.Entry objects begin accumulating data immediately.

        This will be when the filter_string is used to determine how to filter
        the pyami data. Setting the filter_string after stage is called will
        have no effect.

        Internally this creates a new pyami.Entry object. These objects start
        accumulating data immediately.
        """
        if self.filter_string is None and last_filter_string is not None:
            self._entry = pyami.Entry(self.prefix, 'Scalar',
                                      last_filter_string)
        elif self.filter_string:
            self._entry = pyami.Entry(self.prefix, 'Scalar',
                                      self.filter_string)
        else:
            self._entry = pyami.Entry(self.prefix, 'Scalar')
        if self.normalize:
            if isinstance(self.normalize, AmiDet):
                self._monitor = self.normalize
            else:
                self._monitor = monitor_det
            if self._monitor is not None:
                self.mon_prefix.put(self._monitor.prefix)
        return super().stage()

    def unstage(self):
        """
        Called late in a bluesky scan to remove the pyami.Entry object and the
        monitor.
        """
        self._entry = None
        if self._monitor is not None and self._monitor is not self:
            self._monitor.unstage()
            unstaged = super().unstage() + [self._monitor]
        else:
            unstaged = super().unstage()
        self._monitor = None
        self.mon_prefix.put('')
        return unstaged

    def trigger(self):
        """
        Called during a bluesky scan to clear the accumulated pyami data.

        This must be done because the pyami.Entry objects continually
        accumulate data forever. You can stop it by deleting the objects
        as in `unstage`, and you can clear it here to at least start from a
        clean slate.

        If min_duration is zero, this will return a status already marked done
        and successful. Otherwise, this will return a status that will be
        marked done after min_duration seconds.

        If there is a normalization detector in use and it has not been staged,
        it will be staged during the first trigger in a scan.
        """
        if self._entry is None:
            raise RuntimeError('AmiDet %s(%s) was never staged!', self.name,
                               self.prefix)
        if self._monitor is not None and self._monitor is not self:
            if self._monitor._staged != Staged.yes:
                self._monitor.unstage()
                self._monitor.stage()
            monitor_status = self._monitor.trigger()
        else:
            monitor_status = None
        self._entry.clear()
        if self.min_duration:
            def inner(duration, status):
                time.sleep(duration)
                status.set_finished()
            status = Status(obj=self)
            Thread(target=inner, args=(self.min_duration, status)).start()
        else:
            status = Status(obj=self)
            status.set_finished()
        if monitor_status is None:
            return status
        else:
            return status & monitor_status

    def get(self, *args, **kwargs):
        self._get_data()
        return super().get(*args, **kwargs)

    def read(self, *args, **kwargs):
        self._get_data()
        return super().read(*args, **kwargs)

    def _get_data(self):
        """
        Helper function that stuffs ami data into this device's signals.

        Parameters
        ----------
        del_entry: ``bool``
            If ``True``, we'll clear the accumulated data after getting it.
        """
        if self._entry is None:
            raise RuntimeError('Must stage AmiDet to begin accumulating data')

        data = self._entry.get()
        self.mean_raw.put(data['mean'])
        self.rms.put(data['rms'])
        self.entries.put(data['entries'])
        # Calculate the standard error because old python did
        if data['entries']:
            data['err'] = data['rms']/np.sqrt(data['entries'])
        else:
            data['err'] = 0
        self.err_raw.put(data['err'])

        def adj_error(det_mean, det_err, mon_mean, mon_err):
            return det_err/mon_mean + mon_err * (det_mean/mon_mean)**2

        if self._monitor is None:
            self.mean.put(data['mean'])
            self.err.put(data['err'])
            self.mean_mon.put(0)
            self.err_mon.put(0)
            self.entries_mon.put(0)
        elif self._monitor is self:
            self.mean.put(1)
            if data['mean'] == 0:
                self.err.put(np.nan)
            else:
                self.err.put(adj_error(data['mean'], data['err'],
                                       data['mean'], data['err']))
            self.mean_mon.put(data['mean'])
            self.err_mon.put(data['err'])
            self.entries_mon.put(data['entries'])
        else:
            mon_data = self._monitor.get()
            if mon_data.mean_raw == 0:
                self.mean.put(np.nan)
                self.err.put(np.nan)
            else:
                self.mean.put(data['mean']/mon_data.mean_raw)
                self.err.put(adj_error(data['mean'], data['err'],
                                       mon_data.mean_raw,
                                       mon_data.err_raw))
            self.mean_mon.put(mon_data.mean_raw)
            self.err_mon.put(mon_data.err_raw)
            self.entries_mon.put(mon_data.entries)

    def put(self, *args, **kwargs):
        raise ReadOnlyError('AmiDet is read-only')

    def set_det_filter(self, *args, event_codes=None, operator='&'):
        """
        Set the filter on this detector only.

        This lets you override the l3t filter for a single AmiDet. Call with
        no arguments to revert to the last l3t filter. Call with a simple
        ``False`` to disable filtering on this detector. Call as you would to
        set the l3t filter to setup a normal filtering override.

        Parameters
        ----------
        *args: (``AmiDet``, ``float``, ``float``) n times
            A sequence of (detector, low, high), which create filters that make
            sure the detector is between low and high. If instead, the first
            argument is ``False``, we'll disable filtering on this detector.

        event_codes: ``list``, optional
            A list of event codes to include in the filter. l3pass will be when
            the event code is present.

        operator: ``str``, optional
            The operator for combining the detector ranges and event codes.
            This can either be ``|`` to ``or`` the conditions together, so
            l3pass will happen if any filter passes, or it can be left at the
            default ``&`` to ``and`` the conditions together, so l3pass will
            only happen if all filters pass.
        """
        if len(args) == 1 and not args[0]:
            self.filter_string = False
        else:
            self.filter_string = dets_filter(*args, event_codes=event_codes,
                                             operator=operator)
