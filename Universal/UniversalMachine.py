class ExtendedCell:
    def __init__(self):
        self.LocalAnalogMemory = []
        self.LocalLogicMemory = []
        self.LocalAnalogOutputUnit = []
        self.LocalLogicUnit = []
        self.LocalCommunicationControlUnit = []


class CNN_UM:
    def __init__(self, size):
        self.GlobalAnalogProgrammingUnit = []
        self.Cells = []
        for i in range(size * size):
            self.Cells.append(ExtendedCell())
