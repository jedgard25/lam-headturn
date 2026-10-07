"""Depth + landmark refinement of LAM splats (no FLAME needed).
1. Vision landmarks on the photo crop and on a frontal render of LAM's own splats -> smooth 2D warp (TPS) fixing head proportions.
2. Depth Anything V2 Small on the photo -> corrects splat depth over the face (calibrated to LAM's own z by robust regression).
"""
import json, os, subprocess, tempfile
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates
import splatrender as sr

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 518
KEYS = ["contour", "leftEye", "rightEye", "leftBrow", "rightBrow", "noseCrest", "outerLips", "innerLips", "leftPupil", "rightPupil", "nose"]
_depth = None


def landmarks(img: Image.Image):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "i.png"); img.convert("RGB").save(p)
        r = subprocess.run([os.path.join(HERE, "tools/landmarks"), p], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    if "error" in j:
        return None
    return j


def lm_array(j):
    return np.concatenate([np.array(j["pts"][k], dtype=np.float64).reshape(-1, 2) for k in KEYS if j["pts"].get(k)])


def depth_map(img: Image.Image) -> np.ndarray:
    global _depth
    if _depth is None:
        from transformers import pipeline
        _depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="mps" if sr.DEV == "mps" else "cpu")
    d = np.asarray(_depth(img.convert("RGB").resize((SIZE, SIZE)))["predicted_depth"], dtype=np.float32).squeeze()
    d = np.asarray(Image.fromarray(d).resize((SIZE, SIZE), Image.BILINEAR))
    return d  # relative disparity: bigger = nearer


def bg_color(img: Image.Image):
    """Flat background colour from the top rows and the upper side strips (the bottom border is usually clothing). Returns (rgb 0..1, std)."""
    a = np.asarray(img.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32) / 255.
    b = np.concatenate([a[:8].reshape(-1, 3), a[:SIZE // 2, :8].reshape(-1, 3), a[:SIZE // 2, -8:].reshape(-1, 3)])
    return np.median(b, 0), b.std(0).max() * 255


def photo_mask(img: Image.Image):
    """Foreground mask of a flat-background crop (the app composites the cutout onto a flat colour). None if the bg isn't flat."""
    a = np.asarray(img.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
    bg, sd = bg_color(img)
    if sd > 18:
        return None
    m = np.linalg.norm(a - bg * 255, axis=2) > 28
    return gaussian_filter(m.astype(np.float32), 2.0) > .5


def row_extents(mask, y0, y1, ts):
    """left/right silhouette x at rows interpolated between y0 (top of head) and y1 (chin)."""
    pts = []
    for t in ts:
        y = int(round(y0 + t * (y1 - y0)))
        xs = np.nonzero(mask[y])[0]
        if len(xs) < 4:
            return None
        pts += [[xs.min(), y], [xs.max(), y]]
    return np.array(pts, dtype=np.float64)


def level(photo: Image.Image):
    """LAM's head is canonical (upright); a tilted photo would make the warp bend an upright head into a tilted face.
    So level the photo by the eye line first (Vision's own roll is 0 for this request revision). Returns (photo, landmarks, valid, degrees)."""
    pl = landmarks(photo)
    if pl is None:
        return photo, None, None, 0.
    e = sorted([np.array(pl["pts"][k], dtype=np.float64).mean(0) for k in ("leftEye", "rightEye") if pl["pts"].get(k)], key=lambda q: q[0])
    ang = np.degrees(np.arctan2(e[1][1] - e[0][1], e[1][0] - e[0][0])) if len(e) == 2 else 0.
    if 1. < abs(ang) < 35.:
        c = tuple(lm_array(pl).mean(0))
        fill = tuple(int(round(v * 255)) for v in bg_color(photo)[0])
        lev = photo.convert("RGB").rotate(ang, resample=Image.BICUBIC, center=c, fillcolor=fill)
        pl2 = landmarks(lev)
        if pl2 is not None:
            valid = np.asarray(Image.new("L", photo.size, 255).rotate(ang, center=c), dtype=np.float32) / 255.  # 0 where the rotation had no pixels
            return lev, pl2, valid, ang
    return photo, pl, None, 0.


def refine(g: np.ndarray, photo: Image.Image, depth_w=0.6, smooth=2.0, zpow=0.3, jpow=1.0, fit_iters=250, merge=0.0, tidy=True, return_debug=False, fit_kw=None):
    g = g.copy()
    photo, pl, valid, _ = level(photo)
    import fit
    g, cam = sr.unflare(g)  # LAM's frustum-shaped shell -> parallel sides, same frontal picture
    rend, _ = fit.render_fast(g, cam, SIZE)
    rl = landmarks(Image.fromarray((rend * 255).astype(np.uint8)))
    if pl is None or rl is None:
        return (g, None) if return_debug else g
    T, S = lm_array(pl), lm_array(rl)
    n = min(len(T), len(S))
    if len(T) != len(S):
        return (g, None) if return_debug else g
    T0, S0 = T, S  # (kept for debug)
    # silhouette rows (hair + head outline) from the photo's flat-bg mask vs. the splats' own alpha
    pm = photo_mask(photo)
    if pm is not None:
        am = fit.render_fast(g, cam, SIZE)[1] > .5
        ts = np.linspace(.12, 1.0, 9)
        ys_p, ys_r = np.nonzero(pm.any(1))[0], np.nonzero(am[:int(S[:17, 1].max()) + 1].any(1))[0]
        if len(ys_p) and len(ys_r):
            Tp = row_extents(pm, ys_p.min(), T[:17, 1].max(), ts)
            Sp = row_extents(am, ys_r.min(), S[:17, 1].max(), ts)
            if Tp is not None and Sp is not None:
                T = np.vstack([T, Tp]); S = np.vstack([S, Sp])
    # identity anchors on a ring around the head so the warp fades out toward the body/edges
    ctr = S.mean(0); span = np.ptp(S, axis=0).max()
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = ctr + np.stack([np.cos(ang), np.sin(ang)], 1) * span * 1.6
    src = np.vstack([S, ring]); dst = np.vstack([T, ring])
    warp = RBFInterpolator(src, dst - src, kernel="thin_plate_spline", smoothing=smooth)
    # project splats with the render camera
    ctr3, dist = cam
    f = 1 / np.tan(sr.FOV / 2) * SIZE / 2
    P = g[:, :3] - ctr3
    zc = dist - P[:, 2]                      # distance in front of camera
    px = SIZE / 2 + f * P[:, 0] / zc; py = SIZE / 2 - f * P[:, 1] / zc
    uv = np.stack([px, py], 1)
    d = warp(uv); uv2 = uv + d
    # jacobian area ratio -> splat scale
    e = 1.0
    jx = (warp(uv + [e, 0]) + [e, 0] - d) / e; jy = (warp(uv + [0, e]) + [0, e] - d) / e
    jac = np.sqrt(np.clip(jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0], .25, 4.0))
    # face-width ratio: scale depth relief with the head so a rounder face is rounder in z too
    wr = np.ptp(T[:17, 0]) / max(np.ptp(S[:17, 0]), 1)
    # --- depth ---
    D = gaussian_filter(depth_map(photo), 3.0)
    dv = map_coordinates(D, [uv2[:, 1], uv2[:, 0]], order=1, mode="nearest")
    z0 = P[:, 2] * wr ** zpow
    # face mask = splats inside the warped jaw hull (+ forehead) and reasonably opaque
    from matplotlib.path import Path
    hull = Path(np.vstack([T[:17], T[:17, :1] * 0 + T[:17].mean(0)[0], ])[:17]) if False else None
    cont = T[:17]
    top = np.array([[cont[0, 0], cont[0, 1] - .45 * np.ptp(cont[:, 1])], [cont[-1, 0], cont[-1, 1] - .45 * np.ptp(cont[:, 1])]])
    poly = Path(np.vstack([cont, top[::-1]]))
    inface = poly.contains_points(uv2) & (g[:, 6] > .15) & (P[:, 2] > np.percentile(P[:, 2], 35))
    A = np.stack([dv[inface], np.ones(inface.sum())], 1)
    coef = np.linalg.lstsq(A, z0[inface], rcond=None)[0]
    for _ in range(3):  # robust refit
        r = A @ coef - z0[inface]; ok = np.abs(r) < 2.0 * r.std() + 1e-9
        coef = np.linalg.lstsq(A[ok], z0[inface][ok], rcond=None)[0]
    zd = coef[0] * dv + coef[1]
    # soft weight: full inside face polygon, fading out
    wmask = gaussian_filter(poly.contains_points(np.stack(np.meshgrid(np.arange(SIZE), np.arange(SIZE)), -1).reshape(-1, 2)).reshape(SIZE, SIZE).astype(np.float32), 12)
    w = depth_w * map_coordinates(wmask, [uv2[:, 1], uv2[:, 0]], order=1, mode="nearest")
    z1 = (1 - w) * z0 + w * zd
    if merge > 0:  # one surface: opaque face splats take the (robust) median depth of their screen-space neighbours, so layers can't slide apart at yaw
        from scipy.spatial import cKDTree
        sel = np.nonzero((w > .3 * depth_w) & (g[:, 6] > .1))[0]
        if len(sel) > 50:
            _, nb = cKDTree(uv2[sel]).query(uv2[sel], k=12)
            zm = np.median(z1[sel][nb], axis=1)
            z1[sel] = (1 - merge) * z1[sel] + merge * zm
    # new camera-space positions
    zc2 = dist - z1
    x2 = (uv2[:, 0] - SIZE / 2) / f * zc2; y2 = -(uv2[:, 1] - SIZE / 2) / f * zc2
    g[:, 0] = x2 + ctr3[0]; g[:, 1] = y2 + ctr3[1]; g[:, 2] = z1 + ctr3[2]
    g[:, 7:10] *= (jac ** jpow)[:, None]
    bg, bg_sd = bg_color(photo)
    if fit_iters and bg_sd <= 18:  # photometric polish needs the flat cutout background (the app's matte step provides it)
        import fit
        S2 = SIZE // 2
        ph = np.asarray(photo.convert("RGB").resize((S2, S2), Image.LANCZOS), dtype=np.float32) / 255.
        # shape + tone only: the old per-splat fit (fit.optimize) also moved depth/colour/opacity, which a single view can't constrain
        sil = None if pm is None else np.asarray(Image.fromarray(pm.astype(np.float32)).resize((S2, S2), Image.BILINEAR))
        rs = lambda m, n: np.asarray(Image.fromarray(m.astype(np.float32)).resize((n, n), Image.BILINEAR))
        kw = dict(fit_kw or {}); ckw = {k[2:]: kw.pop(k) for k in list(kw) if k.startswith("c_")}
        g = fit.shape_fit(g, ph, bg, cam, mask=sil, size=S2, iters=fit_iters, anchors=T[:len(T0)] * (S2 / SIZE),
                          valid=None if valid is None else rs(valid, S2), **kw)
        if ckw.get("iters", 300):  # then take the colours from the photo, through the viewer's own blend
            cs = int(ckw.pop("size", SIZE))
            g = fit.color_fit(g, np.asarray(photo.convert("RGB").resize((cs, cs), Image.LANCZOS), dtype=np.float32) / 255., bg, cam,
                              valid=None if valid is None else rs(valid, cs), size=cs, **ckw)
    if tidy:  # hide floaters, pull strays back to the local surface, cap oversized (blurry/streaky) splats
        import clean
        g = clean.cleanup(g)
    if return_debug:
        return g, dict(T=T, S=S, wr=wr, coef=coef, depth=D, inface=inface)
    return g
