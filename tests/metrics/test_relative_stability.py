import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
import quantus
from quantus import relative_output_stability_objective, relative_stability_objective
import jax.numpy as jnp
import pytest

'''
Following scenarios are to be tested for each Relative Stability metric

    - Test output shapes of relative stability objectives, which are to maximize (and if they do not fail on expected input shapes)
    - Pre-computed perturbations
    - Pre-computed perturbations shape = x_batch.shape
    - Different perturb functions

    - Pre-computed explanations
    - only a_batch or as_batch given
    - Pre-computed explanations perturbed.shape = Pre-computed explanations shape
    - Different XAI methods
    
    - return_aggregate True/False
    - abs True/False
    - normalize True/False
    
    
We do not test, do not assert:
    - channel order of inputs/model -> it's users responsibility to provide inputs and model, which are compatible
    - perturb_func, explain_func -> if they're not callable, anyway Error will be raised when calling to
    - Meaningful values for num_perturbations, their bounds, if perturbations caused enough changes 
        -> the library is ment for scientific usage, we assume domain experts know what they're doing
    

There are no desired values, the tests are rather just to make sure no exceptions occur during intended usage 🤷🤷  
'''


@pytest.mark.robustness
def test_relative_stability_objective():
    x = np.random.random((5, 24, 24, 1))

    res = relative_stability_objective(x, x, x, x, 0.00001)

    assert res.shape == (5,), 'Must output same dimension as inputs batch axis'


@pytest.mark.robustness
def test_relative_output_stability_objective():
    h = np.random.random((5, 10))
    a = np.random.random((5, 24, 24, 1))

    res = relative_output_stability_objective(h, h, a, a, 0.00001)

    assert res.shape == (5,), 'Must output same dimension as inputs batch axis'


@pytest.mark.robustness
@pytest.mark.parametrize(
    'model,data,params',
    [
        (       # no explain func, no pre computed explanations
                lazy_fixture('load_mnist_model_tf'),
                lazy_fixture('load_mnist_images'),
                {}
        )
    ]
)
def test_invalid_kwargs(
        model, data, params
):
    with pytest.raises(ValueError):
        ris = quantus.RelativeInputStability(**params)
        ris(model, data['x_batch'], data['y_batch'], **params)


@pytest.mark.robustness
@pytest.mark.parametrize(
    'model,data,params',
    [
        (
                lazy_fixture('load_mnist_model_tf'),
                lazy_fixture('load_mnist_images_tf'),
                {
                    'explain_func': quantus.explain,
                }
        )
    ]
)
def test_relative_input_stability_pre_computed_perturbations(model, data, params):
    ris = quantus.RelativeInputStability(**params)
    x = data['x_batch']
    xs = np.asarray([quantus.random_noise(x) for _ in range(5)])

    result = ris(model, x, data['y_batch'], xs_batch=xs, **params)
    print(f'{result = }')

    assert (result != jnp.nan).all(), 'Probably divided by 0'


@pytest.mark.robustness
@pytest.mark.parametrize(
    'model,data,params',
    [
        (
                lazy_fixture('load_mnist_model_tf'),
                lazy_fixture('load_mnist_images_tf'),
                {
                    'explain_func': quantus.explain,
                }
        )
    ]
)
def test_relative_input_stability_compute_perturbations(model, data, params):
    ris = quantus.RelativeInputStability(**params)
    x = data['x_batch']

    result = ris(model, x, data['y_batch'], **params)
    print(f'{result = }')

    assert (result != jnp.nan).all(), 'Probably divided by 0'


@pytest.mark.robustness
# @pytest.mark.parametrize()
def test_relative_output_stability():
    pass


@pytest.mark.robustness
# @pytest.mark.parametrize()
def test_relative_representation_stability():
    pass
