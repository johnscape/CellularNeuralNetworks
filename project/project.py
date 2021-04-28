from dataset import CreateDataset
from cnn import CellularNetwork
from testing import CreateNormalFromRGB

from training import StartTraining


def CreateModel(file=None):
    cnn = CellularNetwork()
    cnn.SetSimTime(100)
    cnn.SetTimeStep(0.1)

    if file is not None:
        cnn.LoadNetwork(file)

    return cnn


CreateDataset()
red_model = CreateModel("red_network.npz")
green_model = CreateModel("green_network.npz")
blue_model = CreateModel("blue_network.npz")
base_model = CreateModel()
StartTraining([red_model, green_model, blue_model], [True, False, True], 10000, 100)
CreateNormalFromRGB([red_model, green_model, blue_model])
