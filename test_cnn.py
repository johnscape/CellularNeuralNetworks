import pytest
import numpy as np
from cnn import *

imgToCellData = [
    (np.zeros((15, 15)), np.ones((15, 15))),
    (255 * np.ones((15, 15)), -1 * np.ones((15, 15)))
]

cellToImgData = [
    (np.ones((15, 15)), np.zeros((15, 15))),
    (-1 * np.ones((15, 15)), 255 * np.ones((15, 15)))
]


@pytest.mark.parametrize("inp, expected", imgToCellData)
def test_image_to_cell(inp, expected):
    r = ImageToCell(inp)
    assert np.array_equal(r, expected) is True, "Error was " + str(ImageToCell(inp)[0])


@pytest.mark.parametrize("inp, expected", cellToImgData)
def test_cell_to_image(inp, expected):
    r = CellToImage(inp)
    assert np.array_equal(r, expected) is True


def test_standard_cnnnonlinearity():
    inp = np.zeros((3, 3))
    exp = np.zeros((3, 3))

    inp[0, 0] = -1
    inp[0, 1] = 1
    inp[0, 2] = 5

    exp[0, 0] = -1
    exp[0, 1] = 1
    exp[0, 2] = 1

    inp[1, 0] = 0
    inp[1, 1] = 0.2
    inp[1, 2] = -0.2

    exp[1, 0] = 0
    exp[1, 1] = 0.2
    exp[1, 2] = -0.2

    inp[2, 0] = -5
    inp[2, 1] = 2.2
    inp[2, 2] = -1.1

    exp[2, 0] = -1
    exp[2, 1] = 1
    exp[2, 2] = -1

    ans = StandardCNNNonlinearity(inp)

    assert np.array_equal(ans, exp) == True

