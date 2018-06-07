import logging

from bluesky.callbacks.core import CallbackBase
from ophyd.device import Device, Component as Cpt
from ophyd.signal import EpicsSignal

from .daq import get_daq

logger = logging.getLogger(__name__)


class ScanVars(Device, CallbackBase):
    i_step = Cpt(EpicsSignal, ':ISTEP')
    is_scan = Cpt(EpicsSignal, ':ISSCAN')
    var0 = Cpt(EpicsSignal, ':SCANVAR00')
    var1 = Cpt(EpicsSignal, ':SCANVAR01')
    var2 = Cpt(EpicsSignal, ':SCANVAR02')
    var0_max = Cpt(EpicsSignal, ':MAX00')
    var1_max = Cpt(EpicsSignal, ':MAX01')
    var2_max = Cpt(EpicsSignal, ':MAX02')
    var0_min = Cpt(EpicsSignal, ':MIN00')
    var1_min = Cpt(EpicsSignal, ':MIN01')
    var2_min = Cpt(EpicsSignal, ':MIN02')
    n_steps = Cpt(EpicsSignal, ':NSTEPS')
    n_shots = Cpt(EpicsSignal, ':NSHOTS')

    def __init__(self, prefix, *, name, RE, **kwargs):
        super().__init__(prefix, name=name, **kwargs)
        self._cbid = None
        self._RE = RE

    def enable(self):
        if self._cbid is None:
            self._cbid = self._RE.subscribe(self)

    def disable(self):
        if self._cbid is not None:
            self._RE.unsubscribe(self._cbid)
            self._cbid = None

    def start(self, doc):
        """
        Initialize the scan variables at the start of a run.

        This inspects the metadata dictionary and will set reasonable values if
        this metadata dictionary is well-formed as in ``bluesky`` built-ins
        like ``scan``. It also inspects the daq object.
        """
        logger.debug('Seting up scan var pvs')
        try:
            self.i_step.put(1)
            self.is_scan.put(1)
            # inspect the doc
            # first, check for motor names
            try:
                motors = doc['motors']
                for i, name in enumerate(motors[:3]):
                    sig = getattr(self, 'var{}'.format(i))
                    sig.put(name)
            except KeyError:
                logger.debug('Skip var names, no "motors" in start doc')

            # second, check for start/stop points
            try:
                motor_info = doc['plan_pattern_args']['args']
                for i, (_, start, stop) in enumerate(motor_info[:3]):
                    sig_max = getattr(self, 'var{}max'.format(i))
                    sig_min = getattr(self, 'var{}min'.format(i))
                    sig_max.put(max(start, stop))
                    sig_min.put(min(start, stop))
            except KeyError:
                logger.debug(('Skip max/min, no "plan_pattern_args" "args" in '
                              'start doc'))

            # last, check for number of steps
            try:
                num = doc['plan_args']['num']
                self.n_steps.put(num)
            except KeyError:
                logger.debug('Skip n_steps, no "plan_args" "num" in start doc')

            # inspect the daq
            daq = get_daq()
            if daq.config['events'] is not None:
                self.n_shots.put(daq.config['events'])
        except Exception as exc:
            err = 'Error setting up scan var pvs: %s'
            logger.error(err, exc)
            logger.debug(err, exc, exc_info=True)

    def event(self, doc):
        """
        Update the step counter at each scan step.

        This actually sets the step counter for the next scan step, because
        this runs immediately after a scan step and recieves an event doc from
        the step that just ran.
        """
        self.i_step.put(doc['seq_num']+1)

    def stop(self, doc):
        """
        Set all fields to their default values at the end of a run.

        These are all 0 for the numeric fields and empty strings for the string
        fields.
        """
        self.i_step.put(0)
        self.is_scan.put(0)
        self.var0.put('')
        self.var1.put('')
        self.var2.put('')
        self.var0_max.put(0)
        self.var1_max.put(0)
        self.var2_max.put(0)
        self.var0_min.put(0)
        self.var1_min.put(0)
        self.var2_min.put(0)
        self.n_steps.put(0)
        self.n_shots.put(0)