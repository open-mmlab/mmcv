import math

import torch


def roi_pool_pytorch(features: torch.Tensor,
                    rois: torch.Tensor,
                    output_size: tuple[int, int],
                    spatial_scale: float = 1.0) -> torch.Tensor:
    """
    Pure PyTorch implementation of ROI Pooling without CUDA dependencies.
    
    Args:
        features (torch.Tensor): Input features of shape [N, C, H, W]
        rois (torch.Tensor): RoIs of shape [K, 5], format as [batch_idx, x1, y1, x2, y2]
        output_size (tuple[int, int]): Output size [h, w]
        spatial_scale (float): Scale factor to map input coordinates to feature map coordinates
        
    Returns:
        torch.Tensor: Pooled features of shape [K, C, output_size[0], output_size[1]]
    """
    if rois.size(0) == 0:
        return torch.zeros(
            (0, features.size(1), output_size[0], output_size[1]),
            dtype=features.dtype,
            device=features.device)
    
    num_rois = rois.size(0)
    channel_size = features.size(1)
    height, width = features.size(2), features.size(3)
    
    # Scale RoIs to feature map size
    scaled_rois = rois.clone()
    scaled_rois[:, 1:5] = scaled_rois[:, 1:5] * spatial_scale
    
    # Convert to integers (floor for x1, y1 and ceil for x2, y2 to ensure the RoI covers the original region)
    roi_batch_inds = scaled_rois[:, 0].long()
    x1 = scaled_rois[:, 1].floor().long()
    y1 = scaled_rois[:, 2].floor().long()
    x2 = scaled_rois[:, 3].ceil().long()
    y2 = scaled_rois[:, 4].ceil().long()
    
    # Clamp to ensure indices are within feature map bounds
    x1 = torch.clamp(x1, min=0, max=width - 1)
    y1 = torch.clamp(y1, min=0, max=height - 1)
    x2 = torch.clamp(x2, min=0, max=width - 1)
    y2 = torch.clamp(y2, min=0, max=height - 1)
    
    # Calculate output height and width
    output_h, output_w = output_size
    
    # Initialize output tensor
    outputs = torch.zeros(num_rois, channel_size, output_h, output_w, device=features.device)
    
    # For each RoI
    for roi_idx in range(num_rois):
        batch_idx = roi_batch_inds[roi_idx]
        roi_width = max(x2[roi_idx] - x1[roi_idx] + 1, 1)
        roi_height = max(y2[roi_idx] - y1[roi_idx] + 1, 1)
        
        # Calculate bin size
        bin_h = roi_height / output_h
        bin_w = roi_width / output_w
        
        # For each output bin
        for bin_i in range(output_h):
            for bin_j in range(output_w):
                # Calculate bin boundaries
                start_h = min(y1[roi_idx] + math.floor(bin_i * bin_h), height - 1)
                end_h = min(y1[roi_idx] + math.ceil((bin_i + 1) * bin_h), height)
                start_w = min(x1[roi_idx] + math.floor(bin_j * bin_w), width - 1)
                end_w = min(x1[roi_idx] + math.ceil((bin_j + 1) * bin_w), width)
                
                # Skip empty bins
                if start_h >= end_h or start_w >= end_w:
                    outputs[roi_idx, :, bin_i, bin_j] = 0
                    continue
                
                # Max pooling over the bin
                outputs[roi_idx, :, bin_i, bin_j] = torch.max(
                    torch.max(
                        features[batch_idx, :, start_h:end_h, start_w:end_w],
                        dim=2
                    )[0],
                    dim=2
                )[0]
    
    return outputs


def roi_align_pytorch(features: torch.Tensor,
                     rois: torch.Tensor,
                     output_size: tuple[int, int],
                     spatial_scale: float = 1.0,
                     sampling_ratio: int = 0,
                     aligned: bool = False) -> torch.Tensor:
    """
    Pure PyTorch implementation of ROI Align without CUDA dependencies.
    
    Args:
        features (torch.Tensor): Input features of shape [N, C, H, W]
        rois (torch.Tensor): RoIs of shape [K, 5], format as [batch_idx, x1, y1, x2, y2]
        output_size (tuple[int, int]): Output size [h, w]
        spatial_scale (float): Scale factor to map input coordinates to feature map coordinates
        sampling_ratio (int): Number of sampling points per bin (0 means adaptive)
        aligned (bool): If True, shift the bounding box coordinates by -0.5 for better alignment
        
    Returns:
        torch.Tensor: Aligned features of shape [K, C, output_size[0], output_size[1]]
    """
    if rois.size(0) == 0:
        return torch.zeros(
            (0, features.size(1), output_size[0], output_size[1]),
            dtype=features.dtype,
            device=features.device)
    
    num_rois = rois.size(0)
    channel_size = features.size(1)
    height, width = features.size(2), features.size(3)
    
    # Convert output_size to list if it's a scalar
    if isinstance(output_size, int):
        output_size = [output_size, output_size]
    output_h, output_w = output_size
    
    # Scale and shift ROIs
    roi_batch_inds = rois[:, 0].long()
    offset = 0.5 if aligned else 0.0
    
    # Apply spatial scale and offset
    roi_start_w = (rois[:, 1] - offset) * spatial_scale
    roi_start_h = (rois[:, 2] - offset) * spatial_scale
    roi_end_w = (rois[:, 3] - offset) * spatial_scale
    roi_end_h = (rois[:, 4] - offset) * spatial_scale
    
    # Calculate ROI dimensions
    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    
    # Ensure minimum size
    roi_width = torch.clamp(roi_width, min=1.0)
    roi_height = torch.clamp(roi_height, min=1.0)
    
    # Calculate bin sizes
    bin_h = roi_height / output_h
    bin_w = roi_width / output_w
    
    # Determine sampling ratio if adaptive
    if sampling_ratio <= 0:
        # Adaptive sampling - use at least 1 sample per bin
        roi_bin_grid_h = torch.ceil(roi_height / output_h)
        roi_bin_grid_w = torch.ceil(roi_width / output_w)
    else:
        # Fixed sampling
        roi_bin_grid_h = torch.full_like(roi_width, sampling_ratio)
        roi_bin_grid_w = torch.full_like(roi_width, sampling_ratio)
    
    # Initialize output tensor
    outputs = torch.zeros(num_rois, channel_size, output_h, output_w, device=features.device)
    
    # For each ROI
    for roi_idx in range(num_rois):
        # Get batch index for this ROI
        batch_idx = roi_batch_inds[roi_idx]
        
        # Skip invalid batch indices
        if batch_idx < 0 or batch_idx >= features.size(0):
            continue
        
        # Calculate grid size for this ROI
        grid_h = int(roi_bin_grid_h[roi_idx])
        grid_w = int(roi_bin_grid_w[roi_idx])
        grid_h = max(grid_h, 1)
        grid_w = max(grid_w, 1)
        
        # Calculate count for averaging
        count = grid_h * grid_w
        
        # For each output bin
        for bin_i in range(output_h):
            for bin_j in range(output_w):
                # Initialize accumulators
                output_val = torch.zeros(channel_size, device=features.device)
                
                # Start position for this bin
                start_h = roi_start_h[roi_idx] + bin_i * bin_h[roi_idx]
                start_w = roi_start_w[roi_idx] + bin_j * bin_w[roi_idx]
                
                # Grid step size
                step_h = bin_h[roi_idx] / grid_h
                step_w = bin_w[roi_idx] / grid_w
                
                # For each grid point
                for grid_i in range(grid_h):
                    for grid_j in range(grid_w):
                        # Calculate sample point
                        h = start_h + step_h * (grid_i + 0.5)
                        w = start_w + step_w * (grid_j + 0.5)
                        
                        # Skip points outside feature map
                        if h < 0 or h >= height or w < 0 or w >= width:
                            continue
                        
                        # Bilinear interpolation
                        h_low = int(h)
                        w_low = int(w)
                        h_high = min(h_low + 1, height - 1)
                        w_high = min(w_low + 1, width - 1)
                        
                        # Calculate interpolation weights
                        lh = h - h_low
                        lw = w - w_low
                        hh = 1 - lh
                        hw = 1 - lw
                        
                        # Get feature values at four corners
                        v1 = features[batch_idx, :, h_low, w_low]
                        v2 = features[batch_idx, :, h_low, w_high]
                        v3 = features[batch_idx, :, h_high, w_low]
                        v4 = features[batch_idx, :, h_high, w_high]
                        
                        # Bilinear interpolation
                        w1 = hh * hw
                        w2 = hh * lw
                        w3 = lh * hw
                        w4 = lh * lw
                        
                        # Accumulate weighted sum
                        output_val += w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                
                # Average over grid points
                outputs[roi_idx, :, bin_i, bin_j] = output_val / count
    
    return outputs