"""
This module defines bluesky preprocessors for using the daq as a flyer.
"""
from bluesky.plan_stubs import configure
from bluesky.preprocessors import fly_during_wrapper, stage_wrapper
from bluesky.utils import make_decorator

from .daq import get_daq
from .exceptions import DaqNotRegisteredError


def daq_during_wrapper(plan, record=None, use_l3t=False, controls=None):
    """
    Run a plan with the `Daq`.

    This can be used with an ordinary ``bluesky`` plan that you'd like the daq
    to run along with. This also stages the daq so that the run start/stop
    will be synchronized with the bluesky runs.

    This must be applied outside the ``run_wrapper``. All configuration must
    be done by supplying config kwargs to this wrapper.

    The `daq_during_decorator` is the same as the `daq_during_wrapper`,
    but it is meant to be used as a function decorator.

    Parameters
    ----------
    plan: ``plan``
        The ``plan`` to use the daq in

    record: ``bool``, optional
        If ``True``, we'll record the data. Otherwise, we'll run without
        recording. Defaults to ``False``, or the last set value for
        ``record``.

    use_l3t: ``bool``, optional
        If ``True``, an ``events`` argument to begin will be reinterpreted
        to only count events that pass the level 3 trigger. Defaults to
        ``False``.

    controls: ``dict{name: device}`` or ``list[device...]``, optional
        If provided, values from these will make it into the DAQ data
        stream as variables. We will check ``device.position`` and
        ``device.value`` for quantities to use and we will update these
        values each time begin is called. To provide a list, all devices
        must have a ``name`` attribute.
    """
    daq = get_daq()
    if daq is None:
        raise DaqNotRegisteredError(
            "A daq object has not yet been registered.  Scans are unavailable "
            "until one is initialized and registered."
        )
    yield from configure(daq, events=None, duration=None, record=record,
                         use_l3t=use_l3t, controls=controls)
    yield from stage_wrapper(fly_during_wrapper(plan, flyers=[daq]), [daq])


daq_during_decorator = make_decorator(daq_during_wrapper)
