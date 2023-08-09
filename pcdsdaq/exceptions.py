class PcdsdaqError(Exception):
    """Base class for pcdsdaq-related errors."""


class DaqNotRegisteredError(PcdsdaqError):
    """The DAQ has not yet been registered."""
