# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'rans_encode_with_indexes', 'rans_decode_with_indexes',
    'pmf_to_quantized_cdf'
])


def rans_encode_with_indexes(symbols: Tensor,
                             indexes: Tensor,
                             cdfs: Tensor,
                             cdfs_sizes: Tensor,
                             offsets: Tensor,
                             num_threads: int = 1) -> bytes:
    """This function is used to encode symbols with cdfs and offsets.

    The indexes is used to select the corresponding cdf and offset
    for each symbol.

    We support multi-threads to encode symbols. And the number of
    threads controlled by num_threads will be stored in the
    encoded bitstreams.

    All args should be on the same device, continuous memory and
    int32 type.

    Args:
        symbols (Tensor): The symbols to be encoded. The shape is (N,).
        indexes (Tensor): The indexes to select the corresponding cdf
            and offset for each symbol. The shape is (N,).
        cdfs (Tensor): The cdfs used to encode symbols.
            The shape is (M, max(cdfs_sizes)).
        cdfs_sizes (Tensor): The cdfs sizes. The shape is (M,).
        offsets (Tensor): The offsets used to encode symbols.
            The shape is (M,).
        num_threads (int): The number of threads to use. Default: 1.
    Returns:
        bytes: The encoded symbols.
    """
    return ext_module.rans_encode_with_indexes(symbols, indexes, cdfs,
                                               cdfs_sizes, offsets,
                                               num_threads)


def rans_decode_with_indexes(encoded: bytes, indexes: Tensor, cdfs: Tensor,
                             cdfs_sizes: Tensor, offsets: Tensor) -> Tensor:
    """This function is used to decode symbols with cdfs and offsets.

    The indexes is used to select the corresponding cdf and offset for
    each symbol.
    All tensor args should be on the same device, continuous memory
    and int32 type.

    Args:
        encoded (bytes): The encoded bitstreams.
        indexes (Tensor): The indexes to select the corresponding cdf
            and offset for each symbol. The shape is (N,).
        cdfs (Tensor): The cdfs used to encode symbols.
            The shape is (M, max(cdfs_sizes)).
        cdfs_sizes (Tensor): The cdfs sizes. The shape is (M,).
        offsets (Tensor): The offsets used to encode symbols.
            The shape is (M,).

    Returns:
        Tensor: The decoded symbols. The shape is (N,).
    """
    return ext_module.rans_decode_with_indexes(encoded, indexes, cdfs,
                                               cdfs_sizes, offsets)


def pmf_to_quantized_cdf(pmfs: Tensor, pmf_lengths: Tensor,
                         tail_masses: Tensor) -> Tensor:
    """This function is used to convert pmfs to quantized cdfs.

    The quantized cdfs will be used to encode symbols.
    Args:
        pmfs (Tensor): The pmfs to be converted.
            The shape is (N, max(pmf_lengths)). dtype should be float.
        pmf_lengths (Tensor): The pmf lengths.
            The shape is (N,). dtype should be int.
        tail_masses (Tensor): The tail masses.
            The shape is (N,). dtype should be float.
    Returns:
        Tensor: The quantized cdfs.
            The shape is (N, max(pmf_lengths) + 1). dtype is int.
    """
    return ext_module.pmf_to_quantized_cdf(pmfs, pmf_lengths, tail_masses)
