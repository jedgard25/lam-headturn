"""Splat cleanup: floaters -> hidden, splats poking out of the local surface -> pulled back onto it, blur-makers -> size-capped.
Everything is judged against each splat's own neighbourhood (k nearest in 3D), so it adapts to dense face vs sparse shoulders."""
import numpy as np
from scipy.spatial import cKDTree


def cleanup(g, k=12, iso=3.0, poke=1.5, pull=0.8, size_cap=2.0, stats=False):
    g = g.copy()
    P = g[:, :3].astype(np.float64)
    tree = cKDTree(P)
    d, idx = tree.query(P, k=k + 1)
    sp = d[:, 1:5].mean(1)                              # local spacing
    # 1) isolated floaters: far from everything relative to the typical spacing around them
    med = np.median(sp)
    nb_sp = sp[idx[:, 1:]].mean(1)
    floater = sp > iso * np.maximum(nb_sp, med * .5)
    # 2) local plane from neighbours; offset along the normal relative to spacing
    nbP = P[idx[:, 1:]]                                  # (N,k,3)
    m = nbP.mean(1)
    cov = np.einsum("nki,nkj->nij", nbP - m[:, None], nbP - m[:, None]) / k
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]                                    # smallest-eigenvalue direction
    off = np.einsum("ni,ni->n", P - m, nrm)
    rel = np.abs(off) / np.maximum(sp, 1e-6)
    poking = (rel > poke) & ~floater
    # pull them toward the plane, more strongly the further out they are (never past it)
    f = np.where(poking, pull * (1 - poke / np.maximum(rel, 1e-6)) + (pull * poke / np.maximum(rel, 1e-6)) * 0, 0)
    P2 = P - (f * off)[:, None] * nrm
    g[:, :3] = P2
    # 3) size cap: no axis wider than size_cap x local spacing (kills blur blobs and long streaks)
    cap = (size_cap * sp)[:, None]
    big = (g[:, 7:10] > cap).any(1)
    g[:, 7:10] = np.minimum(g[:, 7:10], cap)
    # floaters: fade out instead of deleting (keeps the 20018 layout the viewer expects)
    g[floater, 6] *= 0.0
    if stats:
        return g, dict(floaters=int(floater.sum()), poking=int(poking.sum()), capped=int(big.sum()), spacing=float(med))
    return g
