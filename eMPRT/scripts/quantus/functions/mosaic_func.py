"""This module contains a collection of mosaic creation functions, i.e., group images within a grid structure."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import math
import random
from typing import List, Tuple, Optional, Union, Any

import numpy as np


def build_single_mosaic(mosaic_images_list: List[np.ndarray]) -> np.ndarray:
    """
    Frame a list of 4 images into a 2x2 mosaic image.

    Parameters
    ----------
    mosaic_images_list: List[np.array]
        A list of four images.

    Returns
    -------
    mosaic: np.ndarray
         The single 2x2 mosaic built from a list of images.
    """
    first_row = np.concatenate((mosaic_images_list[0], mosaic_images_list[1]), axis=1)
    second_row = np.concatenate((mosaic_images_list[2], mosaic_images_list[3]), axis=1)
    mosaic = np.concatenate((first_row, second_row), axis=2)
    return mosaic


def mosaic_creation(
    images: np.ndarray,
    labels: np.ndarray,
    mosaics_per_class: int,
    seed: Optional[int] = None,
) -> Tuple[
    Any, List[Tuple[Any, ...]], List[Tuple[Any, ...]], List[Tuple[int, ...]], List[Any]
]:
    """
    Build a mosaic dataset from an image dataset (images). Each mosaic corresponds to a 2x2 grid. Each one
    is composed by four images: two belonging to the target class and the other two are chosen randomly from
    the rest of the classes.

    Parameters
    ----------
    images: np.ndarray
         A np.ndarray which contains the input data.
    labels: np.ndarray
         A np.ndarray which contains the labels from the input data.
    mosaics_per_class: integer
        An integer indicating the number of mosaics per class.
    seed: integer
        An integer used to generate a random number (optional)..

    Returns
    -------
    all_mosaics: np.ndarray
         a np.ndarray which contains the mosaic data
    mosaic_indices_list: a List[Tuple[int, int, int, int]] which contains the image indices corresponding to the images
                         composing each mosaic
    mosaic_labels_list (List[Tuple[Union[int, str], ...]]): a List[Tuple[Union[int, str], ...]] which contains the labels of the images composing each
                        mosaic. Each tuple contains four values referring to (top_left_label, top_right_label,
                        bottom_left_label, bottom_right_label)
    p_batch_list (List[Tuple[int, int, int, int]]): a List[Tuple[int, int, int, int]] which contains the positions of the target class within the mosaic.
                  Each tuple contains 0 and 1 values (non_target_class and target_class) referring to (top_left, top_right, bottom_left, bottom_right).
    target_list (List[Union[int, str]]): a List[Union[int, str]] which contains the target class of each mosaic.
    """

    args = []
    if seed:
        args = [seed]
    rng = random.Random(*args)

    mosaics_images_list = []
    mosaic_indices_list = []
    mosaic_labels_list = []
    p_batch_list = []
    target_list = []
    total_labels = list(np.unique(labels))

    for target_class in total_labels:
        outer_classes = total_labels.copy()
        outer_classes.remove(target_class)

        target_class_images = images[labels == target_class]
        target_class_image_indices = np.where(labels == target_class)[0]
        target_class_images_and_indices = list(
            zip(target_class_images, target_class_image_indices)
        )

        no_repetitions = int(
            math.ceil((2 * mosaics_per_class) / len(target_class_images))
        )
        total_target_class_images_and_indices = (
            target_class_images_and_indices * no_repetitions
        )
        rng.shuffle(total_target_class_images_and_indices)

        no_outer_images_per_class = int(
            math.ceil((2 * mosaics_per_class) / len(outer_classes))
        )
        total_outer_images_and_indices = []
        total_outer_labels = []
        for outer_class in outer_classes:
            outer_class_images = images[labels == outer_class]
            outer_class_images_indices = np.where(labels == outer_class)[0]
            outer_class_images_and_indices = list(
                zip(outer_class_images, outer_class_images_indices)
            )

            current_outer_images_and_indices = rng.choices(
                outer_class_images_and_indices, k=no_outer_images_per_class
            )
            total_outer_images_and_indices += current_outer_images_and_indices
            total_outer_labels += [outer_class] * no_outer_images_per_class

        total_outer = list(zip(total_outer_images_and_indices, total_outer_labels))
        rng.shuffle(total_outer)

        iter_images_and_indices = iter(total_target_class_images_and_indices)
        iter_outer = iter(total_outer)
        for _ in range(mosaics_per_class):
            mosaic_elems = [
                (next(iter_images_and_indices), target_class),
                (next(iter_images_and_indices), target_class),
                next(iter_outer),
                next(iter_outer),
            ]
            rng.shuffle(mosaic_elems)

            current_mosaic = build_single_mosaic([elem[0][0] for elem in mosaic_elems])
            mosaics_images_list.append(current_mosaic)

            mosaic_indices = tuple(elem[0][1] for elem in mosaic_elems)
            mosaic_indices_list.append(mosaic_indices)

            current_targets = tuple(elem[1] for elem in mosaic_elems)
            mosaic_labels_list.append(current_targets)

            current_p_batch = tuple(
                int(elem[1] == target_class) for elem in mosaic_elems
            )
            p_batch_list.append(current_p_batch)

            target_list.append(target_class)

    all_mosaics = np.array(mosaics_images_list)

    return (
        all_mosaics,
        mosaic_indices_list,
        mosaic_labels_list,
        p_batch_list,
        target_list,
    )
