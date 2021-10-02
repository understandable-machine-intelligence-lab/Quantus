### Tests



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

Sparseness
key params: None

Cases:
- Attributions are all zeros but 1 element vs half half --> different gini scores


Complexity
key params: None

Cases:
- Attributions are close to uniform -> entropy should be high

Complexity
key params: eps

Cases:
- Attributions are all smaller than eps -> 0
- Attributions are all larger than eps -> len(attributions)