"""Triangulate LAM's canonical splat grid once -> faces.bin (uint32 triples). Splat i is the same face location in every output, so the
topology from one raw (un-warped) result works for every photo. Usage: python tools/make_faces.py raw.ply"""
import sys, numpy as np
from plyfile import PlyData
from scipy.spatial import Delaunay
v = PlyData.read(sys.argv[1])["vertex"]; P = np.stack([v["x"], v["y"], v["z"]], 1)
vis = np.nonzero(1 / (1 + np.exp(-v["opacity"])) > .15)[0]   # ~half the splats are near-invisible fillers: they would mix depth layers
F = vis[Delaunay(P[vis, :2]).simplices]                # frontal-view triangulation of the visible ones (indices map back to all splats)
L = np.max([np.linalg.norm(P[F[:, a]] - P[F[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
F = F[L < 10 * np.median(L)]                           # only drop the convex-hull spikes; z jumps are handled in the viewer (median depth)
F.astype("<u4").tofile("faces.bin"); print(len(P), "points,", len(F), "triangles")
