## User guidelines  
  
Just 'throwing' some metrics at your XAI explanations and consider the job done, is an approach doomed to fail.   
Before evaluating your explanations, make sure to:  
  
* Always read the original publication to understand the context that the metric was introduced in - it may differ from your specific task and/ or data domain   
* Spend time on understanding and investigate how the hyperparameters of the metrics influence the evaluation outcome; does changing the perturbation function fundamentally change scores?   
* Establish evidence that your chosen metric is well-behaved in your specific setting e.g., include a random explanation (as a control variant) to verify the metric  
* Reflect on the metric's underlying assumptions e.g., most perturbation-based metrics don't account for nonlinear interactions between features  
* Ensure that your model is well-trained, a poor behaving model e.g., a non-robust model will have useless explanations  
  
## Disclaimers  
  
1. Implementation may differ from the original author(s)  
  
Any metric implementation in this library may differ from the original authors.   
It is moreover likely that differences exist since 1) the source code of original publication is most often not made publicly available, 2) sometimes the mathematical definition of the metric is missing and/ or 3) the description of hyperparameter choice was left out.  
This leave room for (subjective) interpretations.   
Note that the implementations of metrics in this library have not been verified by the original authors.  
  
2. Discrepancy in operationalisation is likely
  
Metrics for XAI methods are often empirical interpretations (or translations) of qualities that some researcher(s) claimed were important for explanations to fulfill.   
Hence it may be a discrepancy between what the author claims to measure by the proposed metric and what is actually measured e.g., using entropy as an operationalisation of explanation complexity.     
  
3. Hyperparameters may (and should) change depending on application/ task and dataset/ domain  
  
Metrics are often designed with a specific use case in mind e.g., in an image classification setting. Thus it is not always clear how to change the hyperparameters to make them suitable for another setting.   
Pay careful attention to how your hyperparameters should be tuned; what is a proper baseline value in your context i.e., that represents the notion of “missingness”?  
  
4.  Evaluation of explanations must be understood in its context; its application of and of its kind. The importance of defining what attribution quality means for each separate application and data domain. 
  
What evaluation metric to use is completely dependent on: 1) the type of explanation (explanation by example cannot be evaluated the same way as attribution-based/ feature-importance methods), 2) the application/ task: we may not require the explanations to fulfil certain criteria in some context compared to others e.g., multi-label vs single label classification 3) the dataset/ domain: text vs images e.g, different dependency structures between features exist, and preprocessing of the data, leading to differences on what the model may perceive, and how attribution methods can react to that (prime example: MNIST in range  [0,1] vs [-1,1] and any NN and 4) the user (most evaluation metrics are founded from principles of what a user want from its explanation e.g., even in the seemingly objective measures we are enforcing our preferences e.g., in TCAV "explain in a language we can understand", object localisation "explain over objects we think are important", robustness "explain similarly over things we think looks similar" etc etc..)

5. Evaluation (and explanations) will be unreliable if the model is not robust
   
Evaluation will fail if you explain with a poor model. If the model is not robust, then explanations cannot be expected to be meaningful.
Other metrics like Localisation metrics would also fail if the model achieves a high performance, but for the wrong reason (e.g., Clever Hans, Backdoor issues).

6. Evaluation outcomes can be true to data or true to model
   
Interpretation of evaluation will differ depending on if we want to be faithful to data or to the model. Faithful to data or model? A model is trained to use only one of two highly correlated features. The explanation might rightly point out that the model was trained so that one feature is important but the other correlated feature is not. But if we re-train the model, it might now pick the other feature and with the previous attributions, we might now think that the attributions are not good.