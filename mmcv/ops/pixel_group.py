import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['pixel_group'])


def pixel_group(score, mask, embedding, kernel_label, kernel_contour,
                kernel_region_num, distance_threshold):
    """Group pixels into text instances, which is widely used text detection
    methods.

    Arguments:
        score (np.array): The foreground score with size hxw.
        mask (np.array): The foreground mask with size hxw.
        embedding (np.array): The emdedding with size hxwxc to
            distinguish instances.
        kernel_label (np.array): The instance kernel index with size hxw.
        kernel_contour (np.array): The kernel contour with size hxw.
        kernel_region_num (int): The instance kernel region number.
        distance_threshold (float): The embedding distance threshold between
            kernel and pixel in one instance.

    Returns:
        List[List[float]]: The instance coordinate list. Each element consists
            of averaged confidence, pixel number, and coordinates
            (x_i, y_i for all pixels) in order.
    """
    score_t = torch.from_numpy(score)
    mask_t = torch.from_numpy(mask)
    embedding_t = torch.from_numpy(embedding)
    kernel_label_t = torch.from_numpy(kernel_label)
    kernel_contour_t = torch.from_numpy(kernel_contour)

    pixel_assignment = ext_module.pixel_group(score_t, mask_t, embedding_t,
                                              kernel_label_t, kernel_contour_t,
                                              int(kernel_region_num),
                                              float(distance_threshold))
    return pixel_assignment
