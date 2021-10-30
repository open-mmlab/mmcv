import itertools

import torch

from mmcv.cnn.bricks import ConditionalRandomField


def compute_score(crf, emission, tag):
    # emission: (seq_length, num_tags)
    assert emission.dim() == 2
    assert emission.size(0) == len(tag)
    assert emission.size(1) == crf.num_tags
    assert all(0 <= t < crf.num_tags for t in tag)

    # Add transitions score
    score = crf.start_transitions[tag[0]] + crf.end_transitions[tag[-1]]
    for cur_tag, next_tag in zip(tag, tag[1:]):
        score += crf.transitions[cur_tag, next_tag]

    # Add emission score
    for emit, t in zip(emission, tag):
        score += emit[t]
    return score


def test_ConditionalRandomField():
    # test ConditionalRandomField
    seq_length, batch_size = 3, 2
    crf = ConditionalRandomField(num_tags=5, batch_first=False)
    emissions = torch.rand(seq_length, batch_size, crf.num_tags)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]],
                        dtype=torch.uint8).transpose(0, 1)
    best_tags = crf.decode(emissions, mask=mask)
    # shape: (batch_size, seq_length, num_tags)
    emissions = emissions.transpose(0, 1)
    # shape: (batch_size, seq_length)
    mask = mask.transpose(0, 1)

    for emission, best_tag, mask_ in zip(emissions, best_tags, mask):
        seq_len = mask_.sum()
        assert len(best_tag) == seq_len
        assert all(isinstance(t, int) for t in best_tag)
        emission = emission[:seq_len]
        manual_best_tag = max(
            itertools.product(range(crf.num_tags), repeat=seq_len),
            key=lambda t: compute_score(crf, emission, t))
        assert tuple(best_tag) == manual_best_tag
