import pytest

from pcdsdaq.daq import DaqLCLS2


@pytest.fixture(scope='function')
def daq_lcls2(RE):
    return DaqLCLS2(
        platform=0,
        host='tst',
        timeout=1000,
        RE=RE,
        hutch_name='tst',
        sim=True,
    )


def test_state():
    # Should work at any point
    1/0


def test_preconfig():
    # Should work at any point
    1/0


def test_record():
    # Should work at any point
    1/0


def test_run_number():
    # Should work at any point
    1/0


def test_stage_unstage():
    # Should work at any point
    1/0


def test_configure():
    # Test this after preconfig
    1/0


def test_config_info():
    # Test this after preconfig
    1/0


def test_default_config():
    # Test this after configure
    1/0


def test_configured():
    # Test this after configure
    1/0


def test_config():
    # Test this after configure
    1/0


def test_kickoff():
    # Test this after configure
    1/0


def test_wait():
    # Test this after kickoff
    1/0


def test_trigger():
    # Test this after kickoff
    1/0


def test_begin():
    # Test this after wait
    1/0


def test_stop():
    # Test this after begin
    1/0


def test_begin_infinite():
    # Test this after stop
    1/0


def test_end_run():
    # Test this after stop
    1/0


def test_read():
    # Test this after begin_infinite
    1/0


def test_pause_resume():
    # Test after begin_infinite
    1/0


def test_collect():
    # Test this after kickoff
    1/0


def test_complete():
    # Test this after collect
    1/0


def test_describe_collect():
    # Test this after collect
    1/0


def test_step_scan():
    # Test at the end
    1/0


def test_fly_scan():
    # Test at the end
    1/0
