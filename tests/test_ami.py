import importlib
import logging

import pytest
from bluesky.callbacks import collector
from bluesky.plans import count

import pcdsdaq.ami
import pcdsdaq.sim.pyami as sim_pyami
from pcdsdaq.ami import (AmiDet, auto_setup_pyami, concat_filter_strings,
                         dets_filter, set_monitor_det, set_pyami_filter)

logger = logging.getLogger(__name__)


def test_ami_basic(ami_det):
    logger.debug('test_ami_basic')
    ami_det.stage()
    ami_det.trigger()
    stats = ami_det.get()
    assert stats.entries > 0
    ami_det._entry._values = []
    # Should not error with no values collected
    stats = ami_det.get()


@pytest.mark.timeout(60)
def test_ami_scan(ami_det, RE):
    logger.debug('test_ami_scan')
    ami_det.min_duration = 1
    ami_det.filter_string = '4<x<5'
    mean_list = []
    coll = collector(ami_det.mean.name, mean_list)
    num = 5
    RE(count([ami_det], num=num), {'event': coll})
    assert len(mean_list) == num


def test_normalize_scan(ami_det, ami_det_2, RE):
    logger.debug('test_normalize_scan')
    ami_det.min_duration = 1
    ami_det.normalize = ami_det_2
    set_monitor_det(ami_det_2)
    RE(count([ami_det, ami_det_2], num=5))
    RE(count([ami_det], num=5))
    RE(count([ami_det_2], num=5))


def test_normalize_error(ami_det, ami_det_2):
    logger.debug('test_normalize_error')
    set_monitor_det(ami_det_2)
    ami_det_2.stage()
    ami_det_2._entry._values = []
    with pytest.raises(RuntimeError):
        ami_det.get()
    ami_det.stage()
    ami_det.get()
    assert ami_det.mean_mon.get() == 0


def test_ami_stage(ami_det):
    logger.debug('test_ami_stage')
    assert ami_det._entry is None
    ami_det.stage()
    assert ami_det._entry is not None
    ami_det.unstage()
    assert ami_det._entry is None
    set_pyami_filter(ami_det, 0, 1)
    ami_det.stage()
    assert ami_det._entry is not None
    ami_det.unstage()
    assert ami_det._entry is None
    ami_det.filter_string = '0<x<1'
    ami_det.stage()
    assert ami_det._entry is not None
    ami_det.unstage()
    assert ami_det._entry is None


def test_ami_trigger_errors(ami_det):
    logger.debug('test_ami_trigger_errors')
    with pytest.raises(RuntimeError):
        ami_det.trigger()


def test_ami_errors(ami_det):
    logger.debug('test_ami_errors')
    with pytest.raises(Exception):
        ami_det.put(4)


def test_no_pyami():
    logger.debug('test_no_pyami')
    pcdsdaq.ami.pyami = None
    with pytest.raises(ImportError):
        AmiDet('NOPYAMI', name='nopyami')


def test_set_monitor_det(ami_det):
    logger.debug('test_set_monitor_det')
    set_monitor_det(ami_det)
    assert pcdsdaq.ami.monitor_det is ami_det
    set_monitor_det(False)
    assert pcdsdaq.ami.monitor_det is None


def test_set_pyami_filter_clear(ami_det):
    logger.debug('test_set_pyami_filter_clear')
    set_pyami_filter()
    assert sim_pyami.clear_l3t_count == 1


def test_set_pyami_filter_one(ami_det):
    logger.debug('test_set_pyami_filter_one')
    set_pyami_filter(ami_det, 0, 1)
    assert sim_pyami.set_l3t_count == 1


def test_pyami_filter_string(ami_det):
    logger.debug('test_set_pyami_filter_string')
    set_pyami_filter('DET:NAME', 0, 1)
    assert sim_pyami.set_l3t_count == 1


def test_pyami_filter_error(ami_det):
    logger.debug('test_pyami_filter_error')
    with pytest.raises(TypeError):
        set_pyami_filter(None, 0, 1)


def test_set_pyami_filter_two(ami_det):
    logger.debug('test_set_pyami_filter_two')
    set_pyami_filter(ami_det, 0, 1, ami_det, 2, 3)
    assert sim_pyami.set_l3t_count == 1


def test_set_pyami_filter_evr(ami_det):
    logger.debug('test_set_pyami_filter_evr')
    set_pyami_filter(event_codes=[162, 163])
    assert sim_pyami.set_l3t_count == 1


def test_set_pyami_filter_all(ami_det):
    logger.debug('test_set_pyami_filter_all')
    set_pyami_filter(ami_det, 0, 1, ami_det, 2, 3, event_codes=[162, 163])
    assert sim_pyami.set_l3t_count == 1


def test_set_pyami_filter_daq(daq, ami_det):
    logger.debug('test_set_pyami_filter_daq')
    daq.set_filter()
    assert sim_pyami.clear_l3t_count == 1


def test_set_monitor_daq(daq, ami_det):
    logger.debug('test_set_monitor_daq')
    daq.set_monitor(ami_det)
    assert pcdsdaq.ami.monitor_det == ami_det


def test_dets_filter_default_arg(ami_det):
    logger.debug('test_dets_filter_default_arg')
    with pytest.raises(RuntimeError):
        dets_filter(0, 1)
    set_monitor_det(ami_det)
    assert ami_det.prefix in dets_filter(0, 1)


def test_dets_filter_no_kick(ami_det):
    logger.debug('test_dets_filter_no_kick')
    dets_filter('name', 0, 1, or_bykik=False)


def test_set_det_filter(ami_det):
    logger.debug('test_set_det_filter')
    ami_det.set_det_filter('test', 0, 1)
    assert ami_det.filter_string
    ami_det.set_det_filter(False)
    assert not ami_det.filter_string


def test_concat_error():
    logger.debug('test_concat_error')
    with pytest.raises(Exception):
        concat_filter_strings([])


def test_auto_setup_pyami(sim, monkeypatch):
    logger.debug('test_auto_setup_pyami')
    pcdsdaq.ami._reset_globals()

    def fake_hutch_name(*args, **kwargs):
        return 'tst'

    def fake_get_proxy(*args, **kwargs):
        return 'tst-proxy'

    def fake_import(module):
        if module == 'pyami':
            return sim_pyami
        else:
            return importlib.import_module(module)

    monkeypatch.setattr(pcdsdaq.ami, 'get_hutch_name', fake_hutch_name)
    monkeypatch.setattr(pcdsdaq.ami, 'get_ami_proxy', fake_get_proxy)
    monkeypatch.setattr(pcdsdaq.ami, 'import_module', fake_import)

    auto_setup_pyami()

    # Now make sure we error if things are bad
    pcdsdaq.ami._reset_globals()
    sim_pyami.connect_success = False

    with pytest.raises(RuntimeError):
        auto_setup_pyami()
