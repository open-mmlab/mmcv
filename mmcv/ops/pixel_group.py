# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import numpy as np
import torch
from torch import Tensor

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['pixel_group'])

def estimate_confidence(label: torch.Tensor, score: torch.Tensor,
                        label_num: int) -> List[List[float]]:

    import torch_npu
    point_vector = torch.zeros((label_num, 2),
                               dtype=torch.float32).to(score.device)

    label_flat = label.flatten()
    score_flat = score.flatten()

    mask = label_flat > 0
    valid_labels = label_flat[mask]
    valid_scores = score_flat[mask]

    point_vector.index_add_(
        0, valid_labels,
        torch.stack((valid_scores, torch.ones_like(valid_scores)), dim=1))

    valid_mask = point_vector[:, 1] > 0
    point_vector[valid_mask, 0] /= point_vector[valid_mask, 1]

    point_vector_list = point_vector.tolist()
    for l in range(1, label_num):
        coords = (label == l).nonzero(as_tuple=False).float()
        coords = coords[:, [1, 0]]
        point_vector_list[l].extend(coords.flatten().tolist())

    return point_vector_list

def pixel_group(
    score: Union[np.ndarray, Tensor],
    mask: Union[np.ndarray, Tensor],
    embedding: Union[np.ndarray, Tensor],
    kernel_label: Union[np.ndarray, Tensor],
    kernel_contour: Union[np.ndarray, Tensor],
    kernel_region_num: int,
    distance_threshold: float,
) -> List[List[float]]:
    """Group pixels into text instances, which is widely used text detection
    methods.

    Arguments:
        score (np.array or torch.Tensor): The foreground score with size hxw.
        mask (np.array or Tensor): The foreground mask with size hxw.
        embedding (np.array or torch.Tensor): The embedding with size hxwxc to
            distinguish instances.
        kernel_label (np.array or torch.Tensor): The instance kernel index with
            size hxw.
        kernel_contour (np.array or torch.Tensor): The kernel contour with
            size hxw.
        kernel_region_num (int): The instance kernel region number.
        distance_threshold (float): The embedding distance threshold between
            kernel and pixel in one instance.

    Returns:
        list[list[float]]: The instance coordinates and attributes list. Each
        element consists of averaged confidence, pixel number, and coordinates
        (x_i, y_i for all pixels) in order.
    """
    assert isinstance(score, (torch.Tensor, np.ndarray))
    assert isinstance(mask, (torch.Tensor, np.ndarray))
    assert isinstance(embedding, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_label, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_contour, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_region_num, int)
    assert isinstance(distance_threshold, float)

    if isinstance(score, np.ndarray):
        score = torch.from_numpy(score)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if isinstance(embedding, np.ndarray):
        embedding = torch.from_numpy(embedding)
    if isinstance(kernel_label, np.ndarray):
        kernel_label = torch.from_numpy(kernel_label)
    if isinstance(kernel_contour, np.ndarray):
        kernel_contour = torch.from_numpy(kernel_contour)

    if score.device.type == 'npu':
        import torch_npu
        embedding_dim = embedding.shape[2]
        kernel_vector = torch.zeros((kernel_region_num, embedding_dim),
                                    dtype=torch.float32).to(score.device)

        for label in range(1, kernel_region_num):
            label_mask = (kernel_label == label)
            label_embeddings = embedding[label_mask]
            kernel_vector[label, :] = label_embeddings.sum(dim=0)
            vector_sum = label_mask.sum()
            kernel_vector[label, :] /= vector_sum

            kernel_cv = kernel_vector[label, :]
            valid_mask = (mask == 1) & (kernel_label == 0)
            valid_embeddings = embedding[valid_mask]
            distances = torch.sum((valid_embeddings - kernel_cv)**2, dim=1)
            within_threshold = distances < distance_threshold**2

            kernel_label[valid_mask] = torch.where(within_threshold, label,
                                                   kernel_label[valid_mask])

        return estimate_confidence(kernel_label, score, kernel_region_num)

    if torch.__version__ == 'parrots':
        label = ext_module.pixel_group(
            score,
            mask,
            embedding,
            kernel_label,
            kernel_contour,
            kernel_region_num=kernel_region_num,
            distance_threshold=distance_threshold)
        label = label.tolist()
        label = label[0]
        list_index = kernel_region_num
        pixel_assignment = []
        for x in range(kernel_region_num):
            pixel_assignment.append(
                np.array(
                    label[list_index:list_index + int(label[x])],
                    dtype=np.float))
            list_index = list_index + int(label[x])
    else:
        pixel_assignment = ext_module.pixel_group(score, mask, embedding,
                                                  kernel_label, kernel_contour,
                                                  kernel_region_num,
                                                  distance_threshold)
    return pixel_assignment