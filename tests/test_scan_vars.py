import logging

from bluesky.callbacks.core import CallbackBase
from bluesky.plan_stubs import create, read, save
from bluesky.plans import count, scan
from bluesky.preprocessors import run_wrapper, stage_wrapper
from ophyd.signal import Signal
from ophyd.sim import det1, det2, motor, motor1, motor2, motor3

import pcdsdaq.daq
from pcdsdaq.scan_vars import ScanVars

logger = logging.getLogger(__name__)


class FakeSignal(Signal):
    def __init__(self, prefix, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Placeholder for the make_fake_device in next ophyd
for cpt_name in ScanVars.component_names:
    cpt = getattr(ScanVars, cpt_name)
    cpt.cls = FakeSignal


# Lets check the setup a bit, but doing reflexive checks on istep, etc. is
# counterproductive because it's literally an inversion of the normal code.
class CheckVals(CallbackBase):
    def __init__(self, scan_vars):
        self.scan_vars = scan_vars
        self.plan = None

    def start(self, doc):
        logger.debug(doc)
        if self.plan == 'scan':
            assert self.scan_vars.var0.get() == 'motor1'
            assert self.scan_vars.var1.get() == 'motor2'
            assert self.scan_vars.var0_max.get() == 10
            assert self.scan_vars.var0_min.get() == 0
            assert self.scan_vars.var1_max.get() == 20
            assert self.scan_vars.var1_min.get() == 0

        if self.plan in ('scan', 'count'):
            assert self.scan_vars.n_steps.get() == 11

        if self.plan == 'custom':
            assert self.scan_vars.n_shots.get() == 0
        else:
            assert self.scan_vars.n_shots.get() == 120


def test_scan_vars(RE, daq):
    logger.debug('test_scan_vars')

    daq.configure(events=120)

    scan_vars = ScanVars('TST', name='tst', RE=RE)
    scan_vars.enable()

    check = CheckVals(scan_vars)
    RE.subscribe(check)

    check.plan = 'scan'
    RE(scan([det1, det2], motor1, 0, 10, motor2, 20, 0,
            motor3, 0, 1, motor, 0, 1, 11))

    check.plan = 'count'
    RE(count([det1, det2], 11))

    def custom(detector):
        for i in range(3):
            yield from create()
            yield from read(detector)
            yield from save()

    check.plan = 'custom'
    daq.configure(duration=4)
    RE(stage_wrapper(run_wrapper(custom(det1)), [det1]))

    scan_vars.disable()

    # Last, let's force an otherwise uncaught error to cover the catch-all
    # try-except block to make sure the log message doesn't error
    scan_vars.start({'motors': 4})


def test_scan_vars_no_daq(RE):
    logger.debug('test_scan_vars_no_daq')

    # If no daq has ever been loaded, we should cover an extra line
    pcdsdaq.daq._daq_instance = None
    scan_vars = ScanVars('TST', name='tst', RE=RE)
    scan_vars.start({})
