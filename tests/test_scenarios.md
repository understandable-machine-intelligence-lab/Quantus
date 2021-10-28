# Quantus - test scenarios

## Metrics

### 1. Localisation

PointingGame
key params: None

Cases:
- Attributions all in gt
- Attributions none in gt
- Attributions are half in gt

AttributionLocalization
key params: weighted

Cases:
- Attributions all in gt - same size
- Attributions all in gt - mask is smaller
- Attributions none in gt - same size
- Attributions none in gt - mask is smaller
- Attributions are partly in gt - same size
- Attributions are partly in gt - mask is smaller

TopKIntersection
key params: k, concept_influence

- Attributions all in gt - two different k -> same outcome
- Attributions are not in top k - two different k ->
- Attributions are all in gt PLUS extra - concept influence True/ False -> different outcomes

RelevanceRankAccuracy
key params: None

Cases:
- Attributions all in gt -> 1
- Attributions none in gt -> 0
- Attributions are half in gt -> 0.5

AUC
key params: None

Cases: 
- Attributions are ranked same but one normalized and one nonnormalized -> same outome

RelevanceMassAccuracy
key params: None

Cases:
- Attributions are all inside segmentation mask - different sizes s_batch -> different outcomes

### 2. Complexity

Sparseness
key params: None

Cases:
- Attributions are all zeros but 1 element vs half half --> different gini scores

Complexity
key params: None

Cases:
- Attributions are close to uniform -> entropy should be high

EffectiveComplexity
key params: eps

Cases:
- Attributions are all smaller than eps -> 0
- Attributions are all larger than eps -> len(attributions)

### 3. Randomisation

- Expectation is that similarity decreases as the network is increasingly randomised with gradient explanation

ModelParameterRandomisation
key params: layer_order

Cases:
- Check that 1 is not in similarity list
- Check that different layer_order param setting has different outcomes

RandomLogit
key params: num_classes

Cases:
- Check that 1 is not in similarity list


### 4. Robustness

LocalLipschitzEstimate
key params: perturb_std

Cases: 
- set perturb_std = 0.0, warning produced
- we set perturb_std = 0.001 and then set perturb_std = 0.1, that average lle of the second case is smaller than the first

MaxSensitivity
key params: nr_samples

Cases:
- set perturb_std = 0.0, warning produced
- nr_samples=5, nr_samples=100, stds getting smaller then the larger then nr_samples
- we set perturb_std = 0.001 and then set perturb_std = 0.1, that average lle of the second case is smaller than the first

AvgSensitivity
key params: nr_samples

Cases:
- set perturb_std = 0.0, warning produced
- nr_samples=5, nr_samples=100, stds getting smaller then the larger then nr_samples
- we set perturb_std = 0.001 and then set perturb_std = 0.1, that average lle of the second case is smaller than the first

Continuity
key params: nr_steps, perturb_baseline, perturb_func, nr_patches

Cases:
- nr_steps=1, nr_steps=100, then the std of patch sums (across test samples) should be smaller when nr_steps is large
- normalize and not normalize that average patch sums (across test samples) are differenet

### 5. Axiomatic

Completeness
key params: perturb_baseline

Cases:
- test it with IntGrad - perturb_baseline = "black" 


NonSensitivity
key params: 

Cases:
- nonsensitivity (# indices) should get larger as we increase the number of non-zero attributions within the ground truth
mask of a (well-trained) model that is assumed to place evidence onto the object it is classifying
- all attributions are larger than eps, -> zero attributions is a non-zero set -> perturb  
  -> 224*224

  
### 6. Faithfulness








                        # DEBUG.
                        # a_perturbed[:,
                        # top_left_x: top_left_x + self.patch_size,
                        # top_left_y: top_left_y + self.patch_size,] = 0
                        # plt.imshow(a_perturbed.reshape(224, 224))
                        # plt.show()

