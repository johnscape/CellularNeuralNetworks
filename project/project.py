from dataset import CreateDataset
from cnn import CellularNetwork

from project.training import StartTraining


def CreateModel(file=None):
    cnn = CellularNetwork()
    cnn.SetSimTime(100)
    cnn.SetTimeStep(0.1)

    if file is not None:
        cnn.LoadNetwork(file)

    return cnn


CreateDataset()
red_model = CreateModel("red_network")
green_model = CreateModel("green_network")
blue_model = CreateModel("blue_network")
base_model = CreateModel()
StartTraining([red_model, green_model, blue_model], [False, False, False], 1000, 100)
