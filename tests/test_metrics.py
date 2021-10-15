import pytest
import torch

# Dimensions.
img_size = 12

# Define test scenarios.
S_BATCH = torch.zeros((1, 1, img_size, img_size))
S_BATCH = torch.zeros((1, 1, img_size, img_size))
A_BATCH_1 = torch.zeros((1, 1, img_size, img_size))

@pytest.mark.localisation
def test_pointing_game():
    pass
