"""This module contains the implementation of the MST-C metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Quantus is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# Quantus project URL:
# https://github.com/understandable-machine-intelligence-lab/Quantus

import sys
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import ConvexHull, QhullError
from sklearn.neighbors import kneighbors_graph

from quantus.helpers import warn
from quantus.helpers.enums import (
    DataType,
    EvaluationCategory,
    ModelType,
    ScoreDirection,
)
from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.base import Metric

if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final


@final
class MSTC(Metric[List[float]]):
    """
    Implementation of the MST-C metric.

    MST-C measures the spatial spread and cohesion of an attribution map.
    The absolute value of the attribution map is taken before salient points
    are selected and the spread and cohesion components are calculated.

    The spread component is defined as::

        q_spread = 1 / sqrt(A_hull)

    where ``A_hull`` is the area of the convex hull containing the selected
    salient attribution points.

    The cohesion component is defined as::

        q_cohesion = |V| / L_T

    where ``|V|`` is the number of selected salient attribution points and
    ``L_T`` is the total length of their minimum spanning tree.

    The unscaled MST-C score is::

        MST-C = q_spread * q_cohesion

    When ``scale_score=True``, the final score is::

        MST-C_scaled = sqrt(height**2 + width**2) * MST-C * 100

    Higher scores indicate attribution points that are spatially more compact
    and cohesive.

    Notes
    -----
    MST-C operates on one two-dimensional attribution map per sample.
    Attributions should therefore be channel-aggregated before being supplied
    to the metric, for example with shape ``(batch, height, width)``.

    Quantus may internally expand these attribution maps to
    ``(batch, 1, height, width)``. The singleton attribution-channel dimension
    is removed in :meth:`custom_batch_preprocess`.

    Attribution signs are discarded by applying the absolute-value operation
    before the MST-C calculation.

    When ``auto_increase_k=False`` and the constructed k-NN graph is
    disconnected, the calculation continues using the minimum spanning forest.
    Distances between disconnected components are therefore not included in
    the cohesion term, and a warning is emitted unless warnings are disabled.

    Attributes
    ----------
    name:
        The name of the metric.
    data_applicability:
        Data types supported by the metric.
    model_applicability:
        Model types supported by the metric.
    score_direction:
        Direction in which the score should be interpreted.
    evaluation_category:
        Explanation-quality category measured by the metric.
    """

    name = "MST-C"
    data_applicability = {DataType.IMAGE}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.HIGHER
    evaluation_category = EvaluationCategory.COMPLEXITY

    def __init__(
        self,
        threshold: float = 80.0,
        k: int = 20,
        auto_increase_k: bool = False,
        scale_score: bool = True,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = None,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        threshold:
            Percentile used to select salient attribution points after taking the absolute value. It
            must satisfy ``0 <= threshold < 100``, default=80.0.
        k:
            Initial number of nearest neighbours used to construct the k-NN
            graph, default=20.
        auto_increase_k:
            Indicates whether ``k`` should be increased automatically until the
            k-NN graph becomes connected, default=False.
        scale_score:
            Indicates whether the final score is multiplied by the image
            diagonal and by 100, default=True.
        normalise:
            Indicates whether attribution normalisation is applied,
            default=True.
        normalise_func:
            Attribution normalisation function used when ``normalise=True``.
        normalise_func_kwargs:
            Keyword arguments passed to ``normalise_func``.
        return_aggregate:
            Indicates whether the scores are aggregated across instances.
        aggregate_func:
            Callable used to aggregate the evaluation scores.
        default_plot_func:
            Callable used to plot metric results.
        disable_warnings:
            Indicates whether metric warnings are disabled, default=False.
        display_progressbar:
            Indicates whether a progress bar is displayed, default=False.
        kwargs:
            Optional keyword arguments.
        """

        self._validate_parameters(
            threshold=threshold,
            k=k,
            auto_increase_k=auto_increase_k,
            scale_score=scale_score,
        )

        super().__init__(
            abs=True,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        self.threshold = float(threshold)
        self.k = int(k)
        self.auto_increase_k = auto_increase_k
        self.scale_score = scale_score

        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("threshold", "k"),
                data_domain_applicability="Image",
                citation=(
                    "Mohammad Mahdi Mesgari, Jackie Ma, Wojciech Samek, "
                    "Sebastian Lapuschkin, Leander Weber. "
                    "'Structural Compactness as a Complementary Criterion "
                    "for Explanation Quality.' arXiv preprint "
                    "arXiv:2603.29491 (2026)."
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> List[float]:
        """Evaluate MST-C for a batch of explanations."""

        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            batch_size=batch_size,
            **kwargs,
        )

    def custom_batch_preprocess(
        self,
        *,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Remove the singleton attribution-channel dimension added by Quantus.

        MST-C requires one channel-aggregated two-dimensional attribution map
        per sample.
        """

        a_batch = np.asarray(a_batch)

        if a_batch.ndim == 4:
            if a_batch.shape[1] != 1:
                raise ValueError(
                    "MST-C requires channel-aggregated attributions with one "
                    "attribution channel, but received shape "
                    f"{a_batch.shape}. Aggregate the attribution channels before "
                    "calling the metric."
                )
            a_batch = a_batch[:, 0, :, :]

        if a_batch.ndim != 3:
            raise ValueError(
                "MST-C expects attribution maps with shape "
                "(batch, height, width) after preprocessing, but received "
                f"shape {a_batch.shape}."
            )

        return {"a_batch": a_batch}

    def evaluate_batch(self, a_batch: np.ndarray, **kwargs) -> List[float]:
        """
        Compute MST-C for one batch of attribution maps.

        Parameters
        ----------
        a_batch:
            Batch of spatial attribution maps with shape
            ``(batch, height, width)``.
        kwargs:
            Unused keyword arguments.

        Returns
        -------
        scores_batch:
            One MST-C score for each attribution map. Undefined scores are
            represented by ``np.nan``.
        """

        if a_batch.ndim != 3:
            raise ValueError(
                "MST-C expects a_batch with shape (batch, height, width), "
                f"but received {a_batch.shape}."
            )

        return [self._evaluate_map(a) for a in a_batch]

    def _evaluate_map(self, attribution: np.ndarray) -> float:
        """Compute MST-C for one two-dimensional attribution map."""

        # MST-C is sign-invariant: discard attribution polarity first.
        attribution = np.abs(np.asarray(attribution, dtype=np.float64))

        height, width = attribution.shape
        image_diagonal = float(np.hypot(height, width))

        finite_mask = np.isfinite(attribution)
        finite_values = attribution[finite_mask]

        if finite_values.size == 0:
            if not self.disable_warnings:
                warn.warn_mst_c_invalid("attribution map contains no finite values")
            return float("nan")

        # An all-zero attribution map contains no salient structure.
        if np.max(finite_values) == 0.0:
            return 0.0

        cutoff = np.percentile(
            finite_values,
            self.threshold,
        )

        coordinates = np.argwhere(finite_mask & (attribution >= cutoff)).astype(
            np.float64
        )

        n_points = coordinates.shape[0]

        # At least two points are required to obtain a non-zero MST length.
        if n_points < 2:
            if not self.disable_warnings:
                warn.warn_mst_c_invalid(
                    f"only {n_points} salient point(s) found after thresholding"
                )
            return float("nan")

        # -----------------------------------------------------
        # Attribution spread.
        #
        # q_spread = 1 / sqrt(A_hull)
        #
        # The image diagonal is used when a non-degenerate
        # two-dimensional convex hull cannot be constructed.
        # -----------------------------------------------------
        sqrt_hull_area = image_diagonal

        if n_points >= 3 and np.linalg.matrix_rank(coordinates - coordinates[0]) >= 2:
            try:
                # In two dimensions, ConvexHull.volume is the
                # enclosed area; ConvexHull.area is the perimeter.
                hull_area = float(ConvexHull(coordinates).volume)

                if np.isfinite(hull_area) and hull_area > 1e-12:
                    sqrt_hull_area = float(np.sqrt(hull_area))

            except QhullError:
                # Retain the image-diagonal fallback.
                pass

        q_spread = 1.0 / sqrt_hull_area

        # -----------------------------------------------------
        # Attribution cohesion.
        #
        # q_cohesion = |V| / L_T
        # -----------------------------------------------------
        graph, n_components = self._construct_knn_graph(coordinates)

        if n_components != 1 and not self.disable_warnings:
            warn.warn_disconnected_graph(n_components=n_components)

        mst_length = float(minimum_spanning_tree(graph).sum())

        if not np.isfinite(mst_length) or mst_length <= 0.0:
            if not self.disable_warnings:
                warn.warn_mst_c_invalid(f"MST length is invalid: {mst_length}")
            return float("nan")

        q_cohesion = n_points / mst_length

        # Final MST-C score.

        score = q_spread * q_cohesion

        if self.scale_score:
            score *= image_diagonal * 100.0

        return float(score)

    def _construct_knn_graph(
        self,
        points: np.ndarray,
    ) -> Tuple[csr_matrix, int]:
        """
        Construct the symmetric k-NN graph used for the MST calculation.

        If requested, increase ``k`` until the graph is connected or until all
        other points are included as neighbours.
        """

        n_points = points.shape[0]
        k_current = min(self.k, n_points - 1)

        graph = self._make_symmetric_knn_graph(points, k_current)
        n_components = int(
            connected_components(
                graph,
                directed=False,
                return_labels=False,
            )
        )

        while self.auto_increase_k and n_components != 1 and k_current < n_points - 1:
            k_current += 1
            graph = self._make_symmetric_knn_graph(points, k_current)
            n_components = int(
                connected_components(
                    graph,
                    directed=False,
                    return_labels=False,
                )
            )

        return graph, n_components

    @staticmethod
    def _make_symmetric_knn_graph(
        points: np.ndarray,
        k: int,
    ) -> csr_matrix:
        """Return an undirected distance-weighted k-nearest-neighbour graph."""

        graph = kneighbors_graph(
            points,
            n_neighbors=k,
            mode="distance",
            include_self=False,
        )

        # The sklearn graph is directed. Keep an edge whenever either endpoint
        # selected the other as a neighbour, without doubling edge weights.
        return graph.maximum(graph.T).tocsr()

    @staticmethod
    def _validate_parameters(
        threshold: float,
        k: int,
        auto_increase_k: bool,
        scale_score: bool,
    ) -> None:
        """Validate MST-C-specific constructor parameters."""

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not np.isfinite(threshold)
            or not 0.0 <= threshold < 100.0
        ):
            raise ValueError("'threshold' must satisfy 0 <= threshold < 100.")

        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("'k' must be an integer greater than or equal to 1.")

        if not isinstance(auto_increase_k, bool):
            raise TypeError("'auto_increase_k' must be a boolean.")

        if not isinstance(scale_score, bool):
            raise TypeError("'scale_score' must be a boolean.")
