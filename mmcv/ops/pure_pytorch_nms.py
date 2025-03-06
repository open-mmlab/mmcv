from typing import Union

import numpy as np
import torch

array_like_type = Union[torch.Tensor, np.ndarray]


def nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Pure PyTorch implementation of NMS without CUDA dependencies.
    
    Args:
        boxes (torch.Tensor): Boxes in shape (N, 4). 
        scores (torch.Tensor): Scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        
    Returns:
        torch.Tensor: Indices of kept boxes.
    """
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64, device=boxes.device)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = torch.argsort(scores, descending=True)
    
    keep = []
    while order.size(0) > 0:
        i = order[0].item()
        keep.append(i)
        
        if order.size(0) == 1:
            break
        
        # Compute IoU of the picked box with the rest
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU < threshold
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def soft_nms_pytorch(boxes: torch.Tensor, 
                    scores: torch.Tensor, 
                    iou_threshold: float = 0.3, 
                    sigma: float = 0.5, 
                    min_score: float = 1e-3, 
                    method: str = 'linear') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of Soft-NMS without CUDA dependencies.
    
    Args:
        boxes (torch.Tensor): Boxes in shape (N, 4).
        scores (torch.Tensor): Scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        sigma (float): Parameter for Gaussian method.
        min_score (float): Score filter threshold.
        method (str): 'linear', 'gaussian' or 'naive'.
        
    Returns:
        tuple: (kept_boxes, kept_indices)
    """
    if boxes.shape[0] == 0:
        return torch.zeros((0, 5), device=boxes.device), torch.tensor([], dtype=torch.int64, device=boxes.device)
    
    device = boxes.device
    N = boxes.shape[0]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Clone scores and initialize tensor for keeping track of indices
    scores_modified = scores.clone()
    indices = torch.arange(N, device=device)
    
    # Loop until scores are below threshold or we've processed all boxes
    i = 0
    while i < scores_modified.shape[0]:
        # Pick the box with highest score
        max_score_idx = torch.argmax(scores_modified)
        boxes[max_score_idx:max_score_idx+1]
        
        # Swap the box with the first position
        if max_score_idx != i:
            boxes[i], boxes[max_score_idx] = boxes[max_score_idx].clone(), boxes[i].clone()
            scores_modified[i], scores_modified[max_score_idx] = scores_modified[max_score_idx].clone(), scores_modified[i].clone()
            indices[i], indices[max_score_idx] = indices[max_score_idx].clone(), indices[i].clone()
            areas[i], areas[max_score_idx] = areas[max_score_idx].clone(), areas[i].clone()
        
        # Move to next position
        i += 1
        
        # If no more boxes remain, exit loop
        if i >= scores_modified.shape[0]:
            break
        
        # Calculate IoU of current box with the rest
        xx1 = torch.max(boxes[i-1, 0], boxes[i:, 0])
        yy1 = torch.max(boxes[i-1, 1], boxes[i:, 1])
        xx2 = torch.min(boxes[i-1, 2], boxes[i:, 2])
        yy2 = torch.min(boxes[i-1, 3], boxes[i:, 3])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        iou = inter / (areas[i-1] + areas[i:] - inter)
        
        # Apply different soft-NMS methods to decay scores
        if method == 'naive':
            # Simply set scores to 0 if IoU > threshold
            decay = (iou <= iou_threshold).float()
        elif method == 'linear':
            # Linear decay: (1 - IoU) if IoU > threshold
            decay = torch.ones_like(iou)
            decay[iou > iou_threshold] = 1 - iou[iou > iou_threshold]
        elif method == 'gaussian':
            # Gaussian decay: exp(-IoU² / sigma)
            decay = torch.exp(-torch.pow(iou, 2) / sigma)
        else:
            raise ValueError(f"Method '{method}' not recognized. Use 'naive', 'linear', or 'gaussian'.")
        
        # Apply decay to scores
        scores_modified[i:] = scores_modified[i:] * decay
        
        # Remove boxes with scores below min_score
        keep_indices = torch.where(scores_modified[i:] >= min_score)[0] + i
        
        if keep_indices.shape[0] == 0:
            break
        
        # Keep only boxes above min_score
        boxes = torch.cat([boxes[:i], boxes[keep_indices]])
        scores_modified = torch.cat([scores_modified[:i], scores_modified[keep_indices]])
        indices = torch.cat([indices[:i], indices[keep_indices]])
        areas = torch.cat([areas[:i], areas[keep_indices]])
    
    # Prepare output: concatenate boxes and scores
    dets = torch.cat([boxes, scores_modified.unsqueeze(1)], dim=1)
    
    # Only keep boxes with scores above min_score
    keep = scores_modified >= min_score
    return dets[keep], indices[keep]


def nms_match_pytorch(dets: torch.Tensor, iou_threshold: float) -> list[torch.Tensor]:
    """
    Pure PyTorch implementation of NMS match without CUDA dependencies.
    
    Args:
        dets (torch.Tensor): Det boxes with scores, shape (N, 5).
        iou_threshold (float): IoU threshold for NMS.
        
    Returns:
        list[torch.Tensor]: The outer list corresponds different
        matched group, the inner Tensor corresponds the indices for a group
        in score order.
    """
    if dets.shape[0] == 0:
        return []
    
    scores = dets[:, 4]
    boxes = dets[:, :4]
    
    # Sort boxes by scores
    _, order = torch.sort(scores, descending=True)
    
    # Initialize list to store matched groups
    matched_groups = []
    unmatched = torch.ones(dets.shape[0], dtype=torch.bool, device=dets.device)
    
    # Compute areas once for all boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Process boxes in order of descending scores
    for i in range(order.size(0)):
        if not unmatched[order[i]]:
            continue
        
        # Create a new group with this box
        group = [order[i].item()]
        unmatched[order[i]] = False
        
        # Find boxes that match with this one
        box_i = boxes[order[i]]
        
        # Process remaining boxes
        for j in range(i + 1, order.size(0)):
            if not unmatched[order[j]]:
                continue
            
            box_j = boxes[order[j]]
            
            # Compute IoU
            xx1 = torch.max(box_i[0], box_j[0])
            yy1 = torch.max(box_i[1], box_j[1])
            xx2 = torch.min(box_i[2], box_j[2])
            yy2 = torch.min(box_i[3], box_j[3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            iou = inter / (areas[order[i]] + areas[order[j]] - inter)
            
            # If IoU exceeds threshold, add box to this group
            if iou > iou_threshold:
                group.append(order[j].item())
                unmatched[order[j]] = False
        
        # Sort group by score and add to matched groups
        group_scores = scores[group]
        _, group_order = torch.sort(group_scores, descending=True)
        sorted_group = [group[idx] for idx in group_order.tolist()]
        matched_groups.append(torch.tensor(sorted_group, dtype=torch.long, device=dets.device))
    
    return matched_groups


def nms_quadri_pytorch(dets: torch.Tensor, 
                       scores: torch.Tensor, 
                       iou_threshold: float, 
                       labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of Quadrilateral NMS without CUDA dependencies.
    
    Args:
        dets (torch.Tensor): Quadri boxes in shape (N, 8). 
            They are expected to be in (x1, y1, ..., x4, y4) format.
        scores (torch.Tensor): Scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        labels (torch.Tensor, optional): Boxes' label in shape (N,).
        
    Returns:
        tuple: (kept_boxes_with_scores, kept_indices)
    """
    if dets.shape[0] == 0:
        return dets, torch.tensor([], dtype=torch.int64, device=dets.device)
    
    device = dets.device
    
    # Set up for multi-label case
    multi_label = labels is not None
    if multi_label:
        dets_with_labels = torch.cat((dets, labels.unsqueeze(1)), 1)
    else:
        dets_with_labels = dets
    
    # Sort scores
    _, order = scores.sort(0, descending=True)
    dets_with_labels.index_select(0, order)
    
    # Convert quadrilaterals to polygons for IoU computation
    num_dets = dets.shape[0]
    keep = torch.zeros(num_dets, dtype=torch.bool, device=device)
    
    # Loop through boxes
    for i in range(num_dets):
        if i >= order.shape[0]:
            break
        
        # Current box
        current_idx = order[i]
        keep[current_idx] = True
        
        # Skip if we've processed all boxes
        if i == num_dets - 1:
            break
            
        # Get current quad points
        quad_i = dets[current_idx].reshape(4, 2)
        
        # Check against remaining boxes
        for j in range(i + 1, num_dets):
            # Get comparison quad
            compare_idx = order[j]
            
            # If boxes have different labels and we're using multi_label, they don't suppress each other
            if multi_label and labels[current_idx] != labels[compare_idx]:
                continue
                
            quad_j = dets[compare_idx].reshape(4, 2)
            
            # Compute IoU between two quadrilaterals
            # This is an approximation - proper quad IoU is complex
            # Convert to pixel masks and compute IoU
            # For simplicity, we use bounding box IoU as an approximation
            
            # Get bounding box for each quad
            x1_i, y1_i = torch.min(quad_i[:, 0]), torch.min(quad_i[:, 1])
            x2_i, y2_i = torch.max(quad_i[:, 0]), torch.max(quad_i[:, 1])
            
            x1_j, y1_j = torch.min(quad_j[:, 0]), torch.min(quad_j[:, 1])
            x2_j, y2_j = torch.max(quad_j[:, 0]), torch.max(quad_j[:, 1])
            
            # Compute IoU of bounding boxes
            xx1 = torch.max(x1_i, x1_j)
            yy1 = torch.max(y1_i, y1_j)
            xx2 = torch.min(x2_i, x2_j)
            yy2 = torch.min(y2_i, y2_j)
            
            w = torch.max(torch.tensor(0., device=device), xx2 - xx1)
            h = torch.max(torch.tensor(0., device=device), yy2 - yy1)
            
            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            area_j = (x2_j - x1_j) * (y2_j - y1_j)
            inter = w * h
            
            # IoU = intersection / union
            iou = inter / (area_i + area_j - inter)
            
            # If IoU exceeds threshold, suppress the box
            if iou > iou_threshold:
                keep[compare_idx] = False
    
    # Get indices of kept boxes
    keep_inds = torch.nonzero(keep).squeeze(1)
    
    # Sort by original order
    _, order_keep = torch.sort(order.new_tensor([torch.where(order == i)[0].item() for i in keep_inds]))
    keep_inds = keep_inds[order_keep]
    
    # Prepare output
    dets_out = torch.cat((dets[keep_inds], scores[keep_inds].reshape(-1, 1)), dim=1)
    
    return dets_out, keep_inds