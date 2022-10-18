## User guidelines

Just 'throwing' some metrics at your explanations and considering the job done is not a very productive approach.
Before evaluating your explanations, make sure to:

* Always read the original publication to understand the context that the metric was introduced in - it may differ from your specific task and/ or data domain
* Spend time on understanding and investigating how the hyperparameters of metrics can influence the evaluation outcome. Some parameters that usually influence results significantly include:
  * the choice of perturbation function
  * whether normalisation is applied and the choice of the normalisation function
  * whether unsigned or signed attributions are considered
* Establish evidence that your chosen metric is well-behaved in your specific setting, e.g., include a random explanation (as a control variant) to verify the metric
* Reflect on the metric's underlying assumptions, e.g., most perturbation-based metrics don't account for nonlinear interactions between features
* Ensure that your model is well-trained, as a poor behaving model, e.g., a non-robust model will have useless explanations
* Each metric measures different properties of explanations, and especially the various categories (faithfulness, localisation, ...) can be viewed as different facettes of evaluation,
but a single metric never suffices as a sole criterion for the quality of an explanation method


## Disclaimers

**1. Implementation may differ from the original author(s)**

Note that the implementations of metrics in this library have not been verified by the original authors. 
Thus any metric implementation in this library may differ from the original authors. 
It is moreover likely that differences exist since 
* the source code of original publication is most often not made publicly available 
* sometimes the mathematical definition of the metric is missing  
* the description of hyperparameter choice was left out. 

This leaves room for (subjective) interpretations.

**2. Discrepancy in operationalisation is likely**

Metrics for XAI methods are often empirical interpretations (or translations) of qualities that researcher(s) stated 
were important for explanations to fulfil. Hence there may be a discrepancy between what the author claims to measure by 
the proposed metric and what is actually measured, e.g., using entropy as an operationalisation of explanation complexity.

**3. Hyperparameters may (and should) change depending on the application/ task and dataset/ domain**

Metrics are often designed with a specific use case in mind, most commonly for an image classification setting. 
Thus it is not always clear how to change the hyperparameters to make them suitable for another setting. 
Pay careful attention to how your hyperparameters should be tuned and what a proper baseline value could be in your context

**4. Evaluation of explanations must be understood in its context; its application and of its kind**

 What evaluation metric to use can depend on the following factors:  
 * **The type of explanation:** e.g., an explanation by example cannot be evaluated 
 the same way as attribution-based or feature-importance methods
 * **The application/ task:** we may not require the explanations to fulfil 
 certain criteria in some context compared to others, e.g., multi-label 
 vs. single label classification 
 * **The dataset/ domain:** e.g, text vs. images, or if different dependency structures between features exist, 
 as well as the preprocessing of the data, leading to differences on what the model
 may perceive, and how attribution methods can react to that
 * **The user:** most evaluation metrics are founded from principles of what 
 a user may expect from explanations, even in the seemingly objective
 measures. E.g., localisation asks for the explanation to be focused on objects expected to be important, 
 and may fail independent of the explanation if the model simply does not consider those objects, 
 while robustness asks to explain similarly over things we 
 think looks similar, not considering how the model represents the data manifold etc. 
 Thus it is important to define what attribution quality means for each experimental setting.

**5. Evaluation (and explanations) can be unreliable if the model is not robust**

Evaluation can fail (depending on the evaluation method) if you explain a poorly trained model. 
If the model is not robust, then explanations cannot be expected to be meaningful or interpretable [1, 2]. 
If the model achieves high predictive performance, but for the wrong reasons (e.g., Clever Hans effects, Backdoor issues) 
[3, 4], unexpected effects on localisation metrics are likely.

**6. Evaluation outcomes can be true to the data or true to the model**

Generally, explanations should depend on both the data and the model.
However, both are difficult to measure at the same time, and
the interpretation of evaluation outcomes will differ depending on whether we prioritise 
that attributions are faithful to data or to the model [5, 6]. As explained in [5], 
imagine if a model is trained to use only one of two highly correlated features. 
The explanation might then rightly point out that this one feature is important 
(and that the other correlated feature is not). But if we were to re-train the model, 
the model might now pick the other feature as basis for prediction, for which the explanation 
will consequently tell another story --- that the other feature is important. Since the 
explanation function have returned conflicting information about what features are important 
--- we might now believe that the explanation function in itself is unstable. But this may 
not necessarily be true --- in this case, the explanation has remained faithful to the model 
but not the data. As such, in the context of evaluation, to avoid misinterpretation of results, 
it may therefore be important to articulate what you care most about explaining.

**References**

[1] P. Chalasani, J. Chen, A. R. Chowdhury, X. Wu, and S. Jha, “Concise explanations of neural  networks using adversarial training,” in Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, pp. 1383–1391, 2020.

[2] N. Bansal, C. Agarwal, and A. Nguyen, “SAM: the sensitivity of attribution methods to  hyperparameters,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition,  CVPR Workshops 2020, Seattle, WA, USA, June 14-19, 2020. Computer Vision Foundation IEEE, pp. 11–21, 2020.

[3] S. Lapuschkin, S. Wäldchen, A. Binder, G. Montavon, W. Samek, and K.-R. Müller, “Unmasking clever hans predictors and assessing what machines really learn,” Nature Communications, vol. 10, p. 1096, 2019.

[4] C. J. Anders, L. Weber, D. Neumann, W. Samek, K.-R. Müller, and S. Lapuschkin, “Finding  and removing clever hans: Using explanation methods to debug and improve deep models,”  Information Fusion, vol. 77, pp. 261–295, 2022.

[5] P. Sturmfels, S. Lundberg, and S. Lee. "Visualizing the impact of feature attribution baselines." Distill 5, no. 1: e22, 2020.

[6] D. Janzing, L. Minorics, and P. Blöbaum. "Feature relevance quantification in explainable AI: A causal problem." In International Conference on Artificial Intelligence and Statistics, pp. 2907-2916. PMLR, 2020.


