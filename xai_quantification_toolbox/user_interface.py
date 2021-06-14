
import xaiquantificationtoolbox

# Use case: one model, one input, one explanation.

#model, input, attributionSSSS

# create_quantifier - loaders the model, interpreting the model
# decide what evaluation to do
measures = Pipeline([FaithfulnessTest(**params),
                     StabilityTest(**params),
                     SparsenessTest(**params),
                     PointingGame(**params),
                     ModelRandomisationTest(**params)])

# Example params - FaithfulnessTest (aka pixel-flipping)
params = {"mask_value": 0,
          "mask_strategy": 4,  # 4x4 patches
          "mask_order": "deletion"}

# Quantifier:
# - what type of evaluations do you want to run?
# - how do you want to store it?
# - other settings related to running it
quantifier = xaiquantificationtoolbox.Quantifier(measure=measures, # Optional, we provide default setting
                                                 io_object=h5py.File(...),
                                                 checkpoints=..)
# quantifier = xaiquantificationtoolbox.create_quantifier()

# Common data objects (image classification example)
# stored it locally in folders
# tfds tf.data.Dataset
# pytorch torch.utils.data.Dataloader
# np.array
# [nr_samples, dim1, dim2, channels]

# Options: (input)
# either give us a preprocessed tensor e.g., np. tf.Tensor or torch.Tensor ...
# or Dataset(path="...", subfolder/partition=["train", "validation", "test"],
#               index="...", ...
# in the end we need a tuple like this:
# (inputs, targets)

# Options: (attributions), name="Explanation"
# either we are given a content_like(inputs) and infer mapping to input
# or AttributionLoader(path="store_explanations_here",
#                      methods=["Gradient", "LIME", "Occlusion"]
# ..
# (attributions [methods, inputs], inputs, targets)
# (["Gradient", "LIME"]

evaluation = quantifier.run_evaluation(data=test_set,  #(inputs, targets),  #or inputs=inputs, targets=targets or DataLoader
                                       model=model,
                                       attributions=attributions)  # or AttributionLoader(

# (10, 224, 224, 3)
# (10, 224, 224, 3), (5, 10, 224, 224, 3)

# ... parameter checking ... for measures - do we provide everything we need to run evalaution

#methods = [Gradient(params), # attribute name ->
#           LIME(params),
#           Smoothgrad(std=0.1), name="Smoothgrad_std=0.1"
#           DeepLIFT(params)]
#           IntegratedGradients(params),

#"FaithfulnessTest" [len(methods), nr_images, array_per_image]

measures = ["FaithfulnessTest", #[[0, 0., ]] [0.9]
            "StabilityTest",
            SparsenessTest(params),
            PointingGame(params)]


evaluation.faithfulness_auc_ # [0.5, 0.2]
evaluation.summary_table_ # pd.DataFrame(...)
evaluation.scores_  #-> Dict[TestName, Outcome]
evaluation.get_params_  #-> Dict
evaluation.get_methods_  #-> XAI methods
evaluation.set_methods_  #-> change names of XAI methods

#### Plot ...
evaluation.scores_.plot()

# class Measure():
# reqs: return a list/ np.array with at least one value

# class SparsenessTest(Measure):
#
# ....