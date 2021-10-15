## User guidelines

Just 'throwing' some metrics at your XAI explanations and consider the job done, is an approach not very productive.
Before evaluating your explanations, make sure to:

* Always read the original publication to understand the context that the metric was introduced in - it may differ from your specific task and/ or data domain
* Spend time on understanding and investigate how the hyperparameters of the metrics influence the evaluation outcome; does changing the perturbation function fundamentally change scores?
* Establish evidence that your chosen metric is well-behaved in your specific setting e.g., include a random explanation (as a control variant) to verify the metric
* Reflect on the metric's underlying assumptions e.g., most perturbation-based metrics don't account for nonlinear interactions between features
* Ensure that your model is well-trained, a poor behaving model e.g., a non-robust model will have useless explanations
