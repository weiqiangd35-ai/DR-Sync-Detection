"""Vectorized CUSUM simulation.

Key fact: for a zero-start CUSUM the first-passage time over threshold h is
read off the same path for every h <= h_max, because nothing resets before
the first passage. One run to h_max therefore yields the whole ARL(h) (or
delay(h)) curve, and the ARL used throughout (E_0[T] from Z=0) is exactly
the mean of these first-passage times.
"""
import numpy as np


def _draw(cdf, n, rng):
    return np.minimum(np.searchsorted(cdf, rng.random(n)), len(cdf) - 1)


def first_passage(charts, prob, hgrid, N=6000, maxT=400000, seed=1,
                  want_argmax_at=None):
    """charts: list of (K,) LLR arrays; prob: (K,) law; hgrid: increasing.
    Returns T (len(hgrid), N) first-passage times (np.inf if not reached),
    and, if want_argmax_at is a threshold, the argmax chart index at that
    crossing for each path."""
    rng = np.random.default_rng(seed)
    A = np.stack([np.asarray(c, float) for c in charts])       # (M, K)
    cdf = np.cumsum(prob) / np.sum(prob)
    hgrid = np.asarray(hgrid, float)
    Z = np.zeros((A.shape[0], N))
    T = np.full((len(hgrid), N), np.inf)
    amax = np.full(N, -1)
    alive = np.ones(N, bool)
    hmax = hgrid[-1]
    t = 0
    while alive.any() and t < maxT:
        t += 1
        ai = np.flatnonzero(alive)
        idx = _draw(cdf, ai.size, rng)
        Za = np.maximum(Z[:, ai] + A[:, idx], 0.0)
        Z[:, ai] = Za
        top = Za.max(axis=0)
        # record first passages for every grid threshold crossed now
        newly = (top[None, :] >= hgrid[:, None]) & np.isinf(T[:, ai])
        if newly.any():
            r, c = np.nonzero(newly)
            T[r, ai[c]] = t
        if want_argmax_at is not None:
            hitm = (top >= want_argmax_at) & (amax[ai] < 0)
            if hitm.any():
                amax[ai[hitm]] = Za[:, hitm].argmax(axis=0)
        done = top >= hmax
        alive[ai[done]] = False
    return T, amax


def mean_se(x):
    x = x[np.isfinite(x)]
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)), int(x.size)


def calibrate(charts, prob, target, hlo, hhi, step=0.02, N=6000, seed=3):
    """ARL(h) curve from one run; log-linear interpolation to the target."""
    hgrid = np.arange(hlo, hhi + 1e-9, step)
    T, _ = first_passage(charts, prob, hgrid, N=N, seed=seed,
                         maxT=int(40 * target))
    arl = np.array([np.nanmean(np.where(np.isfinite(r), r, np.nan)) for r in T])
    trunc = np.isinf(T[-1]).sum()
    j = np.searchsorted(arl, target)
    j = min(max(j, 1), len(hgrid) - 1)
    h0, h1 = hgrid[j - 1], hgrid[j]
    a0, a1 = np.log(arl[j - 1]), np.log(arl[j])
    h = h0 + (np.log(target) - a0) * (h1 - h0) / (a1 - a0)
    se = float(np.std(T[j][np.isfinite(T[j])], ddof=1) / np.sqrt(N))
    return float(h), dict(hgrid=hgrid.tolist(), arl=arl.tolist(),
                          se_at_target=se, truncated=int(trunc))


def renewal(charts, p_sync, p_fail, fuse, lam, h, N=500, T=10000, seed=5):
    """N independent renewal processes of T slots each; failures arrive at
    rate lam during sync; CUSUM runs from 0 at failure onset and resets at
    alarm (instantaneous repair). Returns long-run Pf and its s.e. across
    the N independent processes."""
    rng = np.random.default_rng(seed)
    A = np.stack(charts)
    cs, cf = np.cumsum(p_sync), np.cumsum(p_fail)
    Z = np.zeros((A.shape[0], N))
    sync = np.ones(N, bool)
    fa = np.zeros(N)
    for _ in range(T):
        sync &= ~(rng.random(N) < lam)            # failure onset
        ds, df = _draw(cs, N, rng), _draw(cf, N, rng)
        d = np.where(sync, ds, df)
        fa += fuse[d]
        f = ~sync
        if f.any():
            Zf = np.maximum(Z[:, f] + A[:, d[f]], 0.0)
            hit = Zf.max(axis=0) >= h
            Zf[:, hit] = 0.0
            Z[:, f] = Zf
            idx = np.flatnonzero(f)[hit]
            sync[idx] = True
    per = fa / T
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(N))
