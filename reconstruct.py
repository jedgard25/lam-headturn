"""Headshot -> Gaussian .ply using LAM-20K CoreML (no FLAME template: offsets used as positions)."""
import io, sys, argparse
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

MODEL = "model/LAMReconstruct_fp16.mlpackage"  # fp16 rewrite of the int8 pkg (see dequant.py); int8 segfaults on GPU
SH0 = 0.28209479177387814
_model = None


def load_model():
    global _model
    if _model is None:
        import coremltools as ct
        import os  # ALL = GPU+CPU works with fp16; the int8 pkg segfaults on GPU, and CPU_AND_NE dies
        _model = ct.models.MLModel(MODEL, compute_units=ct.ComputeUnit[os.environ.get("LAM_UNITS", "ALL")])
    return _model


def prep(img: Image.Image, zoom=1.0) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h) / zoom
    l, t = (w - s) / 2, (h - s) / 3  # bias crop upward (faces sit high)
    return img.crop((l, t, l + s, t + s)).resize((518, 518), Image.LANCZOS)


def infer(img: Image.Image, zoom=1.0) -> np.ndarray:
    out = load_model().predict({"input_image": prep(img, zoom)})["gaussian_attributes"]
    return np.asarray(out, dtype=np.float32)[0]  # (20018, 14)


def to_ply(g: np.ndarray, pos_scale=1.0, flame_base=None) -> bytes:
    pos = g[:, 0:3] * pos_scale
    if flame_base is not None:
        pos = pos + flame_base
    rgb = np.clip(g[:, 3:6], 1e-4, 1 - 1e-4)
    op = np.clip(g[:, 6], 1e-4, 1 - 1e-4)
    sc = np.log(np.maximum(g[:, 7:10] * pos_scale, 1e-8))
    q = g[:, 10:14] / np.maximum(np.linalg.norm(g[:, 10:14], axis=1, keepdims=True), 1e-8)
    n = len(g)
    dt = [(k, "f4") for k in ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity",
                              "scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]]
    a = np.zeros(n, dtype=dt)
    a["x"], a["y"], a["z"] = pos.T
    a["f_dc_0"], a["f_dc_1"], a["f_dc_2"] = ((rgb - 0.5) / SH0).T
    a["opacity"] = np.log(op / (1 - op))
    a["scale_0"], a["scale_1"], a["scale_2"] = sc.T
    a["rot_0"], a["rot_1"], a["rot_2"], a["rot_3"] = q.T
    buf = io.BytesIO()
    PlyData([PlyElement.describe(a, "vertex")]).write(buf)
    return buf.getvalue()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("-o", default="out.ply")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--pos-scale", type=float, default=1.0)
    a = ap.parse_args()
    g = infer(Image.open(a.image), a.zoom)
    for i, n in enumerate(["pos","pos","pos","rgb","rgb","rgb","op","scl","scl","scl","q","q","q","q"]):
        pass
    print("pos  min/max", g[:, :3].min(0), g[:, :3].max(0))
    print("rgb  mean", g[:, 3:6].mean(0), " opacity mean", g[:, 6].mean())
    print("scale mean", g[:, 7:10].mean(0))
    open(a.o, "wb").write(to_ply(g, a.pos_scale))
    print("wrote", a.o)
