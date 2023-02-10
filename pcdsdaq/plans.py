"""
Plans relevant to running the DAQ in coordination with other specialized
beamline equipment relevant to run control
"""
import bluesky.plan_stubs as bps


def sequencer_mode(daq, sequencer, iterations, sequence_wait=250):
    """
    Configure the DAQ to be controlled by the EventSequencer
    """
    # Configure the DAQ to be run-forever. It will be run by the EventSequencer
    yield from bps.configure(daq, events=0)
    # Configure the EventSequencer to run for a specified number of sequences
    yield from bps.configure(sequencer, play_mode=1, rep_count=iterations)
    # Configure the EventSequencer to wait for the DAQ to arm itself
    sequencer.DEFAULT_SLEEP = sequence_wait
    # TODO: It would be nice to add additional checks here to ensure that our
    # configuration is reasonable
    # * Is the DAQ configured to readout on an event code?
    # * Does the Sequencer have the DAQ's readout event code in its' sequence?
