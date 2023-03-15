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
                             num_threads: int = 1):
    return ext_module.rans_encode_with_indexes(symbols, indexes, cdfs,
                                               cdfs_sizes, offsets,
                                               num_threads)


rans_decode_with_indexes = ext_module.rans_decode_with_indexes
pmf_to_quantized_cdf = ext_module.pmf_to_quantized_cdf
