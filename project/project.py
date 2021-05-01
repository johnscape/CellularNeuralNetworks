from dataset import CreateDataset
from cnn import CellularNetwork, BoundaryTypes
from testing import CreateNormalFromRGB

from training import StartTraining


def CreateModel(file=None) -> CellularNetwork:
    """
    Creates a Cellular Neural Network, with the parameters defined in the documentation
    @param file: The file to load the existing template values from. If set to None,
    the template will be randomly generated
    @return: The CNN created to generate normal maps.
    """
    cnn = CellularNetwork()
    cnn.SetSimTime(100)
    cnn.SetTimeStep(0.1)
    cnn.Boundary = BoundaryTypes.ZERO_FLUX

    if file is not None:
        cnn.LoadNetwork(file)

    return cnn


targeted_size = 32

CreateDataset(targetSize=targeted_size, clearIfExists=False)
red_model = CreateModel("red_network.npz")
green_model = CreateModel("green_network.npz")
blue_model = CreateModel("blue_network.npz")
base_model = CreateModel()
StartTraining([red_model, green_model, blue_model], [False, False, False], 5000, 100, targeted_size)
CreateNormalFromRGB([red_model, green_model, blue_model], targeted_size)
