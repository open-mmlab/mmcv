# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import time
from multiprocessing import cpu_count

import pytest
import torch

from mmcv.ops import (pmf_to_quantized_cdf, rans_decode_with_indexes,
                      rans_encode_with_indexes)

EXTEND_TIMES = 100
DEVICES = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
NUM_THREDS = [i for i in [1, 2, 4] if i <= cpu_count()]


@pytest.fixture(scope='module')
def rans_input_data():
    inputs = torch.load('tests/data/for_rans/rans_encoder_inputs.pt')
    symbols = inputs['symbols'].repeat(EXTEND_TIMES)
    indexes = inputs['indexes'].repeat(EXTEND_TIMES)
    cdfs = inputs['quantized_cdfs']
    cdfs_sizes = inputs['cdfs_sizes']
    offsets = inputs['offsets']
    return symbols, indexes, cdfs, cdfs_sizes, offsets


@pytest.fixture(scope='module')
def rans_output_data():
    with open('tests/data/for_rans/rans_encoder_output.pkl', 'rb') as f:
        encoded = pickle.load(f)
    return encoded


@pytest.fixture(scope='module')
def pmf_input_data():
    inputs = torch.load('tests/data/for_rans/pmf_to_quantized_cdf_inputs.pt')
    pmfs = inputs['pmfs']
    pmf_lengths = inputs['pmf_lengths']
    tail_masses = inputs['tail_masses']
    return pmfs, pmf_lengths, tail_masses


@pytest.fixture(scope='module')
def pmf_output_data():
    output = torch.load('tests/data/for_rans/pmf_to_quantized_cdf_output.pt')
    quantized_cdfs = output['quantized_cdfs']
    return quantized_cdfs


def _move_data_to_target_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return [d.to(device) for d in data]


def _calculate_mibs(fn, device, num_threads, num_symbols, args):
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()
    for _ in range(100):
        fn(*args)
    if torch.cuda.is_available():
        end_event.record()
        end_event.synchronize()
        latency = start_event.elapsed_time(end_event) / 1000
    else:
        end_time = time.time()
        latency = (end_time - start_time)
    mibs = num_symbols / latency * 100 / 1024 / 1024
    print(f'\ndevice {device}, num_threads {num_threads}: \
            {fn.__name__}: {latency:.4f}s, {mibs:.2f}MiB/s')


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('num_threads', NUM_THREDS)
def test_rans_encode_with_indexes(rans_input_data, rans_output_data, device,
                                  num_threads):
    data = _move_data_to_target_device(rans_input_data, device)
    symbols, indexes, cdfs, cdfs_sizes, offsets = data
    gt_encoded = rans_output_data
    encoded = rans_encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes,
                                       offsets, num_threads)
    assert len(gt_encoded) * 0.9 < len(encoded) / EXTEND_TIMES < len(
        gt_encoded) * 1.1
    _calculate_mibs(rans_encode_with_indexes, device, num_threads,
                    symbols.shape[0], data + [num_threads])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('num_threads', NUM_THREDS)
def test_rans_decode_with_indexes(rans_input_data, device, num_threads):
    data = _move_data_to_target_device(rans_input_data, device)
    symbols, indexes, cdfs, cdfs_sizes, offsets = data
    encoded = rans_encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes,
                                       offsets, num_threads)
    decoded = rans_decode_with_indexes(encoded, indexes, cdfs, cdfs_sizes,
                                       offsets)
    assert (decoded == symbols).all()
    _calculate_mibs(rans_decode_with_indexes, device, num_threads,
                    symbols.shape[0], [encoded] + data[1:])


@pytest.mark.parametrize('encode_device', DEVICES)
@pytest.mark.parametrize('decode_device', DEVICES)
def test_rans_cross_decode_with_indexes(rans_input_data, encode_device,
                                        decode_device):
    data = _move_data_to_target_device(rans_input_data, encode_device)
    symbols, indexes, cdfs, cdfs_sizes, offsets = data
    encoded = rans_encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes,
                                       offsets)
    data = _move_data_to_target_device(rans_input_data, decode_device)
    symbols, indexes, cdfs, cdfs_sizes, offsets = data
    decoded = rans_decode_with_indexes(encoded, indexes, cdfs, cdfs_sizes,
                                       offsets)
    assert (decoded == symbols).all()


@pytest.mark.parametrize('device', DEVICES)
def test_pmf_to_quantized_cdf(pmf_input_data, pmf_output_data, device):
    data = _move_data_to_target_device(pmf_input_data, device)
    pmfs, pmf_lengths, tail_masses = data
    gt_quantized_cdfs = _move_data_to_target_device(pmf_output_data, device)
    quantized_cdfs = pmf_to_quantized_cdf(pmfs, pmf_lengths, tail_masses)
    assert torch.allclose(quantized_cdfs, gt_quantized_cdfs)
    _calculate_mibs(pmf_to_quantized_cdf, device, -1, pmfs.numel(), data)
