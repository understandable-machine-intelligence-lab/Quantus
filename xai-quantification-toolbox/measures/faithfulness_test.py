from sklearn.metrics import auc
import numpy as np
from .base import *
from ..helpers.utils import check_if_fitted


class FaithfulnessTest(Measure):

    def __init__(self,
                 **params):
        self.params = params
        self.mask_value = params.get("mask_value", 0)
        self.mask_strategy = params.get("mask_strategy", "4x4")
        self.mask_order = params.get("mask_order", "deletion")
        super(FaithfulnessTest, self).__init__()

        #segmentation_fn

    def perturb(self):
        pass

    def fit(self,
            model: Union[tf.keras.models.Model, torch.nn.Module, None],
            inputs: Union[tf.Tensor, torch.Tensor, np.array],
            targets: Union[tf.Tensor, torch.Tensor, np.array, None],
            attributions: Union[tf.Tensor, torch.Tensor, np.array, None]):
        """
        For one model, one set of inputs (and targets) and one explanation method with one attribution per input,
        compute the area under the curve (AUC). """

        assert inputs is not None, "Specify the input."
        assert targets is not None, "For each input, specify the target class you wish to explain against."
        assert check_if_fitted(model), 'The model must be fitted.'
        self.model = model

        # Load explanations if not already provided.
        if not attributions:
            attributions = AttributionLoader(inputs, targets)

        # FIXME. Per batch or input?
        for x, y, a in (inputs, targets, attributions):

            # Get indices of sorted attributions (descending).
            a_ix = np.argsort(a)[::-1]

            # Create n masked versions of input x.
            input_segmentations = segmentation_fn(x, a_ix, self.mask_value, self.mask_strategy, self.mask_order)

            # Predict on x.
            f_x = model.predict(x)

            f_x_is = []
            for x_i in input_segmentations:

                # Predict with model, store function value.
                f_x_i = model.predict(x_i)

                # Store the difference in function value.
                f_x_is.append(abs(f_x, f_x_i))

            # Calculate AUC per input.

            #if not np.shape(inputs)[0] == 1:
            #    for x_batch, y_batch in inputs, targets:
            #        for x, y in zip(x_batch, y_batch)

        self.scores_ = {k: fun() for k in methods.keys()}
        self.faithfulness_auc = auc(x, y)

    @property
    def score_(self):
        pass