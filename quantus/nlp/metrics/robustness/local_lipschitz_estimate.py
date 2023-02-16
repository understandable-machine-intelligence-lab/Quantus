from __future__ import annotations

from quantus.nlp.metrics.batched_text_classification_metric import (
    BatchedTextClassificationMetric,
)


class LocalLipschitzEstimate(BatchedTextClassificationMetric):
    """
    Implementation of the Local Lipschitz Estimate (or Stability) test by Alvarez-Melis et al., 2018a, 2018b.

    This tests asks how consistent are the explanations for similar/neighboring examples.
    The test denotes a (weaker) empirical notion of stability based on discrete,
    finite-sample neighborhoods i.e., argmax_(||f(x) - f(x')||_2 / ||x - x'||_2)
    where f(x) is the explanation for input x and x' is the perturbed input.

    References:
        1) David Alvarez-Melis and Tommi S. Jaakkola. "On the robustness of interpretability methods."
        arXiv preprint arXiv:1806.08049 (2018).

        2) David Alvarez-Melis and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." NeurIPS (2018): 7786-7795.
    """
