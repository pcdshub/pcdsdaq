# WIP need to implement this simulation
class DaqControl:
    def monitorStatus(self):
        raise NotImplementedError()

    def setState(self, state, phase1_info):
        raise NotImplementedError()

    def getBlock(self, transition, data):
        raise NotImplementedError()

    def setRecord(self, record):
        raise NotImplementedError()

    def sim_transition(self, transition, state):
        raise NotImplementedError()
