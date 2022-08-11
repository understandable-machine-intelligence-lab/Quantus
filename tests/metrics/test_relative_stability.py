import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
import quantus


'''
Following scenarios are to be tested for each Relative Stability metric

    - Pre-computed perturbations
    - Pre-computed perturbations shape = x_batch.shape
    - Different perturb functions

    - Pre-computed explanations
    - only a_batch or as_batch given
    - Pre-computed explanations perturbed.shape = Pre-computed explanations shape
    - Different XAI methods
    
    - return_aggregate True/False
    - abs True/False
    - channel first/channel last inputs
    - normalize True/False
    

There are no desired values, the tests are rather just to make sure no exceptions occur during intended usage 🤷🤷  
'''


@pytest.mark.robustness
@pytest.mark.parametrize(

)
def test_relative_input_stability(
    model, data: np.ndarray, params, expected
):
    pass


@pytest.mark.robustness
@pytest.mark.parametrize(

)
def test_relative_output_stability():
    pass


@pytest.mark.robustness
@pytest.mark.parametrize(

)
def test_relative_representation_stability():
    pass