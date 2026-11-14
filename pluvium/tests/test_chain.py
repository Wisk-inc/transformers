# Copyright 2026 Nathan. Apache-2.0.
import itertools
import math

import numpy as np
import torch

from pluvium.chain import first_entry, hmm_nll, marginals_np, state_marginals, truncated_log_prob
from pluvium.products import _window_any


def brute_nll(trans, init, mask, conf):
    a, pi = torch.softmax(trans, -1), torch.softmax(init, -1)
    b, t, p, k = mask.shape
    total = 0.0
    for bi in range(b):
        for pi_ in range(p):
            s = 0.0
            for path in itertools.product(range(k), repeat=t):
                pr = pi[bi, pi_, path[0]]
                for h in range(1, t):
                    pr = pr * a[bi, h, pi_, path[h - 1], path[h]]
                for h in range(t):
                    pr = pr * conf[path[h]][mask[bi, h, pi_]].sum()
                s = s + pr
            total = total - math.log(float(s))
    return total


def test_hmm_matches_enumeration():
    torch.manual_seed(1)
    for k in (2, 4):
        trans, init = torch.randn(2, 4, 3, k, k), torch.randn(2, 3, k)
        conf = torch.softmax(torch.randn(k, k) * 2, -1)
        mask = torch.nn.functional.one_hot(torch.randint(0, k, (2, 4, 3)), k).bool()
        mask[0, 1, 2] = True                                   # an unknown hour
        nll, count = hmm_nll(trans, init, mask, conf)
        assert abs(float(nll) - brute_nll(trans, init, mask, conf)) < 1e-3
        assert int(count) == 2 * 4 * 3 - 1


def test_marginals_agree_between_torch_and_numpy():
    torch.manual_seed(2)
    trans, init = torch.randn(1, 6, 5, 4, 4), torch.randn(1, 5, 4)
    a = torch.softmax(trans, -1)[0].permute(1, 0, 2, 3).numpy()
    pi = torch.softmax(init, -1)[0].numpy()
    m = state_marginals(trans, init)[0].permute(1, 0, 2).numpy()
    assert np.allclose(m, marginals_np(a, pi), atol=1e-6)


def test_window_probability_matches_simulation():
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(4), size=(1, 12, 4))                 # (P, T, K, K)
    pi = rng.dirichlet(np.ones(4), size=1)
    target = np.array([False, False, False, True])
    exact = _window_any(a, pi, target, 8)[0]
    hits, n = 0, 20000
    state = rng.choice(4, size=n, p=pi[0])
    seen = state == 3
    for t in range(1, 8):
        u = rng.random(n)
        cdf = np.cumsum(a[0, t][state], -1)
        state = (u[:, None] > cdf).sum(-1).clip(max=3)
        seen |= state == 3
    hits = seen.mean()
    assert abs(exact - hits) < 0.015
    onset = first_entry(a, np.eye(4)[:1], np.arange(4) > 0)
    assert np.all(onset >= 0) and onset.sum() <= 1 + 1e-9


def test_truncated_lognormal_normalised():
    y = torch.linspace(2.5, 7.6, 40001)
    p = torch.exp(truncated_log_prob(y, torch.tensor(1.0), torch.tensor(0.5), 2.5, 7.6))
    assert abs(float(torch.trapezoid(p, y)) - 1.0) < 1e-3
