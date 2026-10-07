"""Offline eval loop: photo -> matte -> LAM -> refine variants -> contact sheet rendered with the exact (viewer-matching) renderer.
  python evalloop.py samples/headshot.jpg -o /tmp/sheet.png --variants nofit,old,new
Rows = variants, columns = photo | yaw 0 | |render-photo| x3 | yaw -35 | yaw +35 | face crop at yaw 0. LAM output is cached next to the sheet."""
import argparse, os, subprocess, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import minimum_filter
import reconstruct, refine, fit, clean, splatrender as sr

HERE = os.path.dirname(os.path.abspath(__file__))


def flat_bg(path):
    """What index.html's prepare() does: Vision matte, slight erode, composite on the mean background colour."""
    img = Image.open(path).convert("RGB")
    with tempfile.TemporaryDirectory() as d:
        o = os.path.join(d, "o.png")
        r = subprocess.run([os.path.join(HERE, "tools/matte"), path, o], capture_output=True, text=True)
        if r.returncode or not os.path.exists(o):
            raise RuntimeError("matte failed: " + (r.stdout or r.stderr))
        a = np.asarray(Image.open(o).convert("RGBA").resize(img.size), dtype=np.float32)[..., 3] / 255.
    a = minimum_filter(a, size=2 * max(1, round(min(img.size) / 500)) * 2 + 1)
    rgb = np.asarray(img, dtype=np.float32)
    bg = rgb[a < .05].mean(0) if (a < .05).sum() > 100 else np.array([125, 120, 115], dtype=np.float32)
    return Image.fromarray((a[..., None] * rgb + (1 - a[..., None]) * bg).astype(np.uint8))


def variant(name, g, photo):
    if name == "raw":
        return sr.unflare(g)[0]
    if name == "nofit":
        return refine.refine(g, photo, fit_iters=0)
    if name.startswith("new"):  # "new" or "new@gain_bound=.2;grid=24" to try shape_fit settings
        kw = {k: float(v) if "." in v or "e" in v else int(v) for k, v in (kv.split("=") for kv in name.partition("@")[2].split(";") if kv)}
        return refine.refine(g, photo, fit_iters=200, fit_kw=kw)
    if name == "old":  # the previous per-splat fit, for A/B
        w = refine.refine(g, photo, fit_iters=0, tidy=False)
        S2 = refine.SIZE // 2
        ph = np.asarray(photo.convert("RGB").resize((S2, S2), Image.LANCZOS), dtype=np.float32) / 255.
        return clean.cleanup(fit.optimize(w, ph, refine.bg_color(photo)[0], sr.camera(g), size=S2, iters=200, opt_color=True, opt_scale=False))
    raise ValueError(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("-o", default="sheet.png")
    ap.add_argument("--variants", default="nofit,old,new"); ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--yaw", type=float, default=35.)
    a = ap.parse_args()
    comp = flat_bg(a.image)
    photo = reconstruct.prep(comp, 1.0)
    cache = os.path.splitext(a.o)[0] + "." + os.path.basename(a.image) + ".lam.npy"
    if os.path.exists(cache):
        g0 = np.load(cache)
    else:
        g0 = reconstruct.infer(comp, 1.0); np.save(cache, g0)
    S = a.size
    ph = np.asarray(photo.resize((S, S), Image.LANCZOS), dtype=np.float32) / 255.
    bg = tuple(refine.bg_color(photo)[0])
    rows = []
    for name in a.variants.split(","):
        t = time.time(); g = variant(name, g0, photo); dt = time.time() - t
        cam = sr.camera(g0)  # the camera refine fitted against (the viewer re-centres, which only rescales)
        front = sr.render(g, S, 0., bg=bg, cam=cam)[0]
        err = np.abs(front - ph)
        sides = [sr.render(g, S, np.radians(y), bg=bg, cam=cam)[0] for y in (-a.yaw, a.yaw)]
        big = sr.render(g, 2 * S, 0., bg=bg, cam=cam)[0]
        crop = big[int(.45 * S):int(1.45 * S), S // 2:S // 2 + S]
        row = np.concatenate([ph, front, np.clip(err * 3, 0, 1), *sides, crop], 1)
        im = Image.fromarray((row * 255).astype(np.uint8))
        z = g[:, 2]
        label = f"{name}  err={err.mean():.4f}  {dt:.1f}s  z p1/p99={np.percentile(z, 1):.3f}/{np.percentile(z, 99):.3f}"
        ImageDraw.Draw(im).text((6, 4), label, fill=(255, 255, 0))
        print(label); rows.append(np.asarray(im))
        reconstruct_out = os.path.splitext(a.o)[0] + f".{name}.ply"
        open(reconstruct_out, "wb").write(reconstruct.to_ply(g))
    Image.fromarray(np.concatenate(rows, 0)).save(a.o)
    print("wrote", a.o)
