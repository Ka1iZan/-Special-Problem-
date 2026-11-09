import torch
from spner.prototypes import centroid, shared_prototype, cosine_drift, alignment_loss, drift_sampling_probs

def test_centroid():
    x = torch.tensor([[1., 2.], [3., 4.]])
    assert torch.allclose(centroid(x), torch.tensor([2., 3.]))

def test_equal_language_weight_shared_prototype():
    p = torch.tensor([[0., 2.], [2., 0.]])
    assert torch.allclose(shared_prototype(p), torch.tensor([1., 1.]))

def test_zero_drift_for_identical_vectors():
    p = torch.tensor([1., 2., 3.])
    assert torch.isclose(cosine_drift(p, p), torch.tensor(0.), atol=1e-6)

def test_alignment_prefers_correct_near_prototype():
    prototypes = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    good = torch.tensor([[1., 0.]])
    bad = torch.tensor([[0., 1.]])
    y = torch.tensor([0])
    assert alignment_loss(good, prototypes, y) < alignment_loss(bad, prototypes, y)

def test_drift_softmax_monotonic():
    d = torch.tensor([0.0, 0.1, 0.5])
    p = drift_sampling_probs(d)
    assert p[2] > p[1] > p[0]
