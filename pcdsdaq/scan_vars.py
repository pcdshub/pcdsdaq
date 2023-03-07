import logging
from typing import Any

from bluesky.callbacks.core import CallbackBase
from ophyd.device import Component as Cpt
from ophyd.device import Device
from ophyd.signal import EpicsSignal
from toolz import partition

from .daq import get_daq

logger = logging.getLogger(__name__)


class ScanVars(Device, CallbackBase):
    """
    Collection of PVs to keep track of the scan state.

    Use `enable` to set up automatic updating of these PVs
    during a ``RunEngine`` scan. This relies on proper
    metadata like the metadata in the built in ``scan``
    and ``count`` plans to populate the PVS.

    Use `disable` to remove this from the ``RunEngine``.

    Parameters
    ----------
    prefix: ``str``
        The PV prefix, e.g. ``XPP:SCAN``

    name: ``str``, required keyword
        A name to refer to this object by

    RE: ``RunEngine``, required keyword
        The ``RunEngine`` instance associated with the session.

    i_start: ``int``, optional
        The starting count for the i_step tracker. This defaults to zero,
        which is offset by one from the one-indexed bluesky counter.
    """
    MAX_VARS = 3

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

    def __init__(self, prefix, *, name, RE, i_start=0, **kwargs):
        super().__init__(prefix, name=name, **kwargs)
        self._cbid = None
        self._RE = RE
        self._i_start = i_start
        self.seen_max_vars_warning = False

    def enable(self):
        """
        Enable automatic updating of PVs during a scan.
        """
        if self._cbid is None:
            self._cbid = self._RE.subscribe(self)

    def disable(self):
        """
        Disable automatic updating of PVs during a scan.
        """
        if self._cbid is not None:
            self._RE.unsubscribe(self._cbid)
            self._cbid = None

    def start(self, doc: dict[str, Any]):
        """
        Initialize the scan variables at the start of a run.

        This inspects the metadata dictionary and will set reasonable values if
        this metadata dictionary is well-formed as in ``bluesky`` built-ins
        like ``scan``. It also inspects the daq object.
        """
        logger.debug('Seting up scan var pvs')
        self.seen_max_vars_warning = False
        try:
            self.i_step.put(self._i_start)
            self.is_scan.put(1)
            # inspect the doc
            # check for motor names
            try:
                motors = doc['motors']
                for i, name in enumerate(motors[:3]):
                    sig = getattr(self, f'var{i}')
                    sig.put(name)
            except KeyError:
                logger.debug('Skip var names, no "motors" in start doc')

            # check for a top level number of points, this is often present
            try:
                num_points = doc['num_points']
            except KeyError:
                logger.debug('No num_points, skip num from top-level key.')
                has_top_level_num = False
            else:
                self.n_steps.put(num_points)
                has_top_level_num = True

            # in this block we find the min/max and number of points
            # check the plan pattern, determines how we read the args
            # inner_product is mot, start, stop, (repeat), num
            # outer_product is mot, start, stop, num, (repeat) + snakes
            # inner_list_product and outer_list_product are mot, list (repeat)
            # there are other patterns, but that's all we'll handle for now
            try:
                plan_pattern = doc['plan_pattern']
                plan_pattern_args = doc['plan_pattern_args']
            except KeyError:
                logger.debug('No plan pattern, skip max/min/num from shape')
                if not has_top_level_num:
                    logger.error(
                        'Scan PVs will be missing any max/min/num info for '
                        'this scan due to invalid or missing metadata.'
                    )
            else:
                try:
                    if plan_pattern == 'inner_product':
                        self.setup_inner_product(plan_pattern_args)
                    elif plan_pattern == 'outer_product':
                        self.setup_outer_product(plan_pattern_args)
                    elif plan_pattern == 'inner_list_product':
                        self.setup_inner_list_product(plan_pattern_args)
                    elif plan_pattern == 'outer_list_product':
                        self.setup_outer_list_product(plan_pattern_args)
                    else:
                        logger.error(
                            'Encountered unknown plan type, '
                            'did not set up min/max/num scan PVs.'
                        )
                        logger.debug(
                            'No scan var setup for plan_pattern %s',
                            plan_pattern,
                        )
                except Exception:
                    logger.error(
                        'Error setting up min/max/num scan PVs'
                    )
                    logger.debug(
                        'Error setting up plan_pattern %s',
                        plan_pattern,
                        exc_info=True,
                    )

            # inspect the daq
            daq = get_daq()
            if daq is None:
                logger.debug('Skip n_shots, no daq')
            else:
                if daq.config['events'] is None:
                    logger.debug('Skip n_shots, daq configured for duration')
                else:
                    self.n_shots.put(daq.config['events'])
        except Exception as exc:
            err = 'Error setting up scan var pvs: %s'
            logger.error(err, exc)
            logger.debug(err, exc, exc_info=True)

    def update_min_max(self, start: float, stop: float, index: int) -> None:
        """
        Helper function for updating the min and max PVs for the scan table.
        """
        if index >= self.MAX_VARS:
            # Once per start doc, warn about too many vars
            if not self.seen_max_vars_warning:
                self.seen_max_vars_warning = True
                logger.warning(
                    f'There are only PVs allocated for {self.MAX_VARS} '
                    'motors in the scan PVs, additional motors are omitted.'
                )
            return
        sig_max = getattr(self, f'var{index}_max')
        sig_min = getattr(self, f'var{index}_min')
        sig_max.put(max(start, stop))
        sig_min.put(min(start, stop))

    def setup_inner_product(self, plan_pattern_args: dict[str, Any]) -> None:
        """
        Handle max, min, number of steps for inner_product scans.

        These are the plans whose arguments are (mot, start, stop) repeat,
        then a num later, such as the normal scan.
        """
        # check for start/stop points
        per_motor = partition(3, plan_pattern_args['args'])
        for index, (_, start, stop) in enumerate(per_motor):
            self.update_min_max(start, stop, index)

        # check for number of steps
        num = plan_pattern_args['num']
        self.n_steps.put(num)

    def setup_outer_product(self, plan_pattern_args: dict[str, Any]) -> None:
        """
        Handle max, min, number of steps for outer_product scans.

        These are the plans whose arguments are (mot, start, stop, num)
        repeat, with snake directions intersperced starting after the second
        num (or not), such as grid_scan.
        """
        # check for start/stop points
        args = plan_pattern_args['args']
        # Either we have 4n args
        # Or we have 5n-1 args if we have snakes on all but the first motor
        # Removes the snakes if they are here for some uniformity
        if len(args) % 4 == 0:
            # Just split into sets of 4
            per_motor = partition(4, args)
        elif (len(args) + 1) % 5 == 0:
            # Remove the 9th, 14th, 19th...
            keep_elems = (
                elem for num, elem in enumerate(args)
                if num < 9 or (num + 1) % 5 != 0
            )
            per_motor = partition(4, keep_elems)
        else:
            raise RuntimeError('Unexpected number of arguments')
        product_num = 1
        for index, (_, start, stop, num) in enumerate(per_motor):
            self.update_min_max(start, stop, index)
            # check for number of steps: a product of all the steps!
            product_num *= num
        self.n_steps.put(product_num)

    def setup_inner_list_product(
        self,
        plan_pattern_args: dict[str, Any],
    ) -> None:
        """
        Handle max, min, number of steps for inner_list_product scans.

        These are the plans whose arguments are (mot, list) repeat,
        where every list needs to have the same length because it's a 1D
        scan with multiple motors, such as list_scan.
        """
        # check for start/stop points
        per_motor = partition(2, plan_pattern_args['args'])
        for index, (_, points) in enumerate(per_motor):
            self.update_min_max(min(points), max(points), index)
            # On the first loop, cache the number of points
            if index == 0:
                self.n_steps.put(len(points))

    def setup_outer_list_product(
        self,
        plan_pattern_args: dict[str, Any],
    ) -> None:
        """
        Handle max, min, number of steps for outer_list_product scans.

        These are the plans whose arguments are (mot, list) repeat,
        where the lists can be any length because it's a multi-dimensional
        mesh scan, like list_grid_scan.
        """
        # check for start/stop points
        per_motor = partition(2, plan_pattern_args['args'])
        product_num = 1
        for index, (_, points) in enumerate(per_motor):
            self.update_min_max(min(points), max(points), index)
            # check for number of steps: a product of all the steps!
            product_num *= len(points)
        self.n_steps.put(product_num)

    def event(self, doc):
        """
        Update the step counter at each scan step.

        This actually sets the step counter for the next scan step, because
        this runs immediately after a scan step and recieves an event doc from
        the step that just ran.
        """
        self.i_step.put(doc['seq_num']-1 + self._i_start)

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
