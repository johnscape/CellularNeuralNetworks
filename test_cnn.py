import pytest
import numpy as np
from cnn import ImageToCell, CellToImage, StandardCNNNonliearity

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


def test_standard_cnnnonliearity():
    assert StandardCNNNonliearity(15) == 1
