class DaqError(Exception):
    """
    Base class for DAQ or pcdsdaq-specific exceptions.

    External users can try/except on this exception class as a catch-all for
    DAQ-specific exceptions.
    """


class DaqNotRegisteredError(DaqError):
    """The DAQ has not yet been registered and scans are unavailable."""


class DaqTimeoutError(DaqError):
    """
    Exception raised when the DAQ times out.

    This encompasses cases where we ask for a specific action, but we observe
    that nothing has happened for too long a duration, so we don't know
    if the operation will ever complete.
    """


class DaqStateTransitionError(DaqError):
    """
    Exception raised when a DAQ state transition fails.

    This is distinct from a timeout where we aren't sure what's happened.
    This is the case where we know that something definitely has gone wrong.
    """
