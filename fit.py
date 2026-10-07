"""Photometric refinement of (already landmark-warped) splats: gradient-descend splat positions (and optionally colour/opacity)
so a frontal render matches the photo, with tethers to the LAM/warp/depth init so the result stays a plausible head."""
import numpy as np, torch
from scipy.spatial import cKDTree
import splatrender as sr

DEV = sr.DEV


def project(g, cam, size, yaw=0.0):
    c, dist = cam
    t = torch.tensor(g, dtype=torch.float32, device=DEV)
    P = t[:, :3] - torch.tensor(c, dtype=torch.float32, device=DEV)
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32, device=DEV)
    P = P @ Ry.T
    p = P.clone(); p[:, 2] -= dist
    f = 1 / np.tan(sr.FOV / 2) * size / 2
    zi = p[:, 2].clamp(max=-0.01)
    q = t[:, 10:14] / t[:, 10:14].norm(dim=1, keepdim=True).clamp_min(1e-8)
    R = Ry @ sr.quat_R(q)
    M = R * t[:, 7:10][:, None, :]
    S = M @ M.transpose(1, 2)
    J = torch.zeros(len(t), 2, 3, device=DEV)
    # image coords: u = c + f*x/(-z), v = c - f*y/(-z)  (v runs down, so the two rows have opposite signs)
    J[:, 0, 0] = -f / zi; J[:, 1, 1] = f / zi
    J[:, 0, 2] = f * p[:, 0] / zi ** 2; J[:, 1, 2] = -f * p[:, 1] / zi ** 2
    C = J @ S @ J.transpose(1, 2)
    C[:, 0, 0] += .3; C[:, 1, 1] += .3
    u = size / 2 + f * p[:, 0] / (-zi); v = size / 2 - f * p[:, 1] / (-zi)
    return u, v, -zi, C, f, J @ R, t[:, 7:10]


def soft_render(u, v, zc, Ci, rgb, op, bg, size, tau, K=13, return_cov=False):
    """Order-independent blended splatting (differentiable): colour = depth-weighted mean, coverage = 1-exp(-sum alpha)."""
    h = K // 2
    bx = torch.floor(u).long(); by = torch.floor(v).long()
    o = torch.arange(-h, h + 1, device=DEV)
    ox, oy = torch.meshgrid(o, o, indexing="xy")
    px = bx[:, None, None] + ox[None]; py = by[:, None, None] + oy[None]
    dx = (px.float() + .5) - u[:, None, None]; dy = (py.float() + .5) - v[:, None, None]
    r2 = Ci[:, 0, 0][:, None, None] * dx * dx + 2 * Ci[:, 0, 1][:, None, None] * dx * dy + Ci[:, 1, 1][:, None, None] * dy * dy
    a = torch.exp(-.5 * r2) * op[:, None, None] * (r2 < 9)
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    idx = torch.where(valid, py * size + px, torch.full_like(px, size * size)).reshape(-1)
    a = (a * valid).reshape(-1)
    w = a * torch.exp(-(zc - zc.min()) / tau)[:, None, None].expand_as(px).reshape(-1)
    n = size * size + 1
    num = torch.zeros(n, 3, device=DEV).index_add(0, idx, w[:, None] * rgb[:, None, None, :].expand(-1, K, K, -1).reshape(-1, 3))
    den = torch.zeros(n, device=DEV).index_add(0, idx, w)
    cov = 1 - torch.exp(-torch.zeros(n, device=DEV).index_add(0, idx, a))
    col = num / den.clamp_min(1e-8)[:, None]
    img = cov[:, None] * col + (1 - cov[:, None]) * bg[None]
    if return_cov:
        return img[:-1].reshape(size, size, 3), cov[:-1].reshape(size, size)
    return img[:-1].reshape(size, size, 3)


@torch.no_grad()
def render_fast(g, cam, size, bg=(0.07, 0.07, 0.07), K=25):
    """Approximate frontal render (rgb, coverage) ~10x faster than splatrender.render; good enough for landmark detection / silhouettes."""
    u, v, zc, C, f, A, s0 = project(g, cam, size)
    Ci = torch.linalg.inv(C)
    rgb = torch.tensor(np.clip(g[:, 3:6], 0, 1), dtype=torch.float32, device=DEV)
    op = torch.tensor(np.clip(g[:, 6], 0, 1), dtype=torch.float32, device=DEV)
    img, cov = soft_render(u, v, zc, Ci, rgb, op, torch.tensor(bg, dtype=torch.float32, device=DEV), size, float((zc.max() - zc.min()) * .03), K=K, return_cov=True)
    return img.cpu().numpy(), cov.cpu().numpy()


def blur(x, k=3):
    x = x.permute(2, 0, 1)[None]
    return torch.nn.functional.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)[0].permute(1, 2, 0)


def optimize(g, photo_rgb, bg, cam, size=259, iters=250, side_yaws=(), w_side=0.5, side_size=130, opt_color=True, opt_scale=True, col_bound=.15, scl_bound=.7, w_tether=1e-3, w_smooth=0.5, w_z=5.0, w_col=0.3, verbose=False):
    """g: (N,14) warped splats. photo_rgb: (size,size,3) float 0..1. Returns new g."""
    u0, v0, zc0, C, f, A, s0 = project(g, cam, size)
    eye = .3 * torch.eye(2, device=DEV)[None]
    ds = torch.zeros_like(s0, requires_grad=True)
    rgb0 = torch.tensor(np.clip(g[:, 3:6], 0, 1), dtype=torch.float32, device=DEV)
    op0 = torch.tensor(np.clip(g[:, 6], 1e-4, 1 - 1e-4), dtype=torch.float32, device=DEV)
    tgt = torch.tensor(photo_rgb, dtype=torch.float32, device=DEV); bgt = torch.tensor(bg, dtype=torch.float32, device=DEV)
    tau = float((zc0.max() - zc0.min()) * .03)
    du = torch.zeros_like(u0, requires_grad=True); dv = torch.zeros_like(u0, requires_grad=True); dz = torch.zeros_like(u0, requires_grad=True)
    drgb = torch.zeros_like(rgb0, requires_grad=True); dop = torch.zeros_like(op0, requires_grad=True)
    params = [{"params": [du, dv], "lr": .25}, {"params": [dz], "lr": zc0.std().item() * .02}]
    if opt_color:
        params += [{"params": [drgb], "lr": .02}, {"params": [dop], "lr": .08}]
    if opt_scale:
        params += [{"params": [ds], "lr": .02}]
    opt = torch.optim.Adam(params)
    xyz = torch.stack([u0, v0, zc0 * (size / 2 / f) * 0 + 0], 1).cpu().numpy()  # kNN graph in screen space of the init
    _, nb = cKDTree(np.stack([u0.cpu().numpy(), v0.cpu().numpy(), (zc0 * 200).cpu().numpy()], 1)).query(np.stack([u0.cpu().numpy(), v0.cpu().numpy(), (zc0 * 200).cpu().numpy()], 1), k=7)
    nb = torch.tensor(nb[:, 1:], device=DEV)
    logit0 = torch.log(op0 / (1 - op0))
    zs = zc0.std()
    sides = []
    if side_yaws:
        rs = side_size / size
        with torch.no_grad():
            for yw in side_yaws:
                us, vs, zcs, Cs, fs_, _, _ = project(g, cam, side_size, yaw=yw)
                prior = soft_render(us, vs, zcs, torch.linalg.inv(Cs), rgb0, op0, bgt, side_size, tau)
                sides.append((yw, torch.linalg.inv(Cs), prior))
    dist_ = cam[1]
    for it in range(iters):
        opt.zero_grad()
        u, v, zc = u0 + du, v0 + dv, zc0 + dz
        rgb = (rgb0 + col_bound * torch.tanh(drgb)).clamp(0, 1); op = torch.sigmoid(logit0 + dop)
        sc = s0 * torch.exp(scl_bound * torch.tanh(ds))
        Cc = (A * (sc ** 2)[:, None, :]) @ A.transpose(1, 2) + eye
        det = Cc[:, 0, 0] * Cc[:, 1, 1] - Cc[:, 0, 1] ** 2
        Ci = torch.stack([torch.stack([Cc[:, 1, 1], -Cc[:, 0, 1]], -1), torch.stack([-Cc[:, 0, 1], Cc[:, 0, 0]], -1)], 1) / det.clamp_min(1e-8)[:, None, None]
        img = soft_render(u, v, zc, Ci, rgb, op, bgt, size, tau)
        l_img = (img - tgt).abs().mean() + .5 * (blur(img) - blur(tgt)).abs().mean()
        lap = lambda d: ((d - d[nb].mean(1)) ** 2).mean()
        l_reg = w_tether * ((du ** 2 + dv ** 2).mean()) + w_smooth * (lap(du) + lap(dv)) + w_z * (dz / zs).pow(2).mean() * .1
        l_reg = l_reg + w_smooth * lap(dz / zs) * .5
        if opt_color:
            l_reg = l_reg + w_col * (torch.tanh(drgb) ** 2).mean() * .1 + 0.02 * (dop ** 2).mean() + 0.5 * ((drgb - drgb[nb].mean(1)) ** 2).mean()
        if opt_scale:
            l_reg = l_reg + 0.05 * (torch.tanh(ds) ** 2).mean()
        l_side = 0.
        for yw, Cis, prior in sides:  # novel-view consistency: stay close to the warp/depth prior seen from the side
            px = (u - size / 2) / f * zc; py = -(v - size / 2) / f * zc; Pz = dist_ - zc
            cy_, sy_ = float(np.cos(yw)), float(np.sin(yw))
            x2 = cy_ * px + sy_ * Pz; z2 = -sy_ * px + cy_ * Pz; zc2 = dist_ - z2
            fs2 = f * side_size / size
            us = side_size / 2 + fs2 * x2 / zc2; vs = side_size / 2 - fs2 * py / zc2
            im = soft_render(us, vs, zc2, Cis, rgb, op, bgt, side_size, tau)
            l_side = l_side + (im - prior).abs().mean() + .5 * (blur(im) - blur(prior)).abs().mean()
        (l_img + l_reg + (w_side * l_side if sides else 0.)).backward(); opt.step()
        if verbose and it % 50 == 0:
            print(it, float(l_img), float(l_reg))
    with torch.no_grad():
        u, v, zc = (u0 + du), (v0 + dv), (zc0 + dz)
        c, dist = cam
        x = (u - size / 2) / f * zc; y = -(v - size / 2) / f * zc
        out = g.copy()
        out[:, 0] = x.cpu().numpy() + c[0]; out[:, 1] = y.cpu().numpy() + c[1]; out[:, 2] = (dist - zc).cpu().numpy() + c[2]
        if opt_color:
            out[:, 3:6] = (rgb0 + col_bound * torch.tanh(drgb)).clamp(0, 1).cpu().numpy(); out[:, 6] = torch.sigmoid(logit0 + dop).cpu().numpy()
        if opt_scale:
            out[:, 7:10] = (s0 * torch.exp(scl_bound * torch.tanh(ds))).cpu().numpy()
    return out


def comp_render(u, v, rank, Ci, rgb, op, bg, size, K=13, M=48):
    """Sorted alpha-over compositing (the viewer's blend, unlike soft_render's depth-weighted mean), differentiable in u, v, rgb, op.
    rank: per-splat depth rank (0 = nearest). Each pixel keeps its M front-most contributions. Returns (img, coverage)."""
    N = len(u); h = K // 2
    bx = torch.floor(u.detach()).long(); by = torch.floor(v.detach()).long()
    o = torch.arange(-h, h + 1, device=DEV)
    ox, oy = torch.meshgrid(o, o, indexing="xy")
    px = bx[:, None, None] + ox[None]; py = by[:, None, None] + oy[None]
    dx = (px.float() + .5) - u[:, None, None]; dy = (py.float() + .5) - v[:, None, None]
    r2 = Ci[:, 0, 0][:, None, None] * dx * dx + 2 * Ci[:, 0, 1][:, None, None] * dx * dy + Ci[:, 1, 1][:, None, None] * dy * dy
    a = (torch.exp(-.5 * r2) * op[:, None, None]).clamp(max=.99)
    ok = (px >= 0) & (px < size) & (py >= 0) & (py < size) & (r2.detach() < 9) & (a.detach() > 1 / 255)
    sp = torch.arange(N, device=DEV)[:, None, None].expand_as(px)[ok]
    pix = (py * size + px)[ok]; a = a[ok]
    key = pix * N + rank[sp]                             # pixel-major, front-to-back within a pixel
    try:
        perm = torch.argsort(key)
        cnt = torch.bincount(pix, minlength=size * size)
    except (NotImplementedError, RuntimeError):          # integer sort/bincount missing on this backend: do the bookkeeping on cpu
        perm = torch.argsort(key.cpu()).to(DEV)
        cnt = torch.bincount(pix.cpu(), minlength=size * size).to(DEV)
    pix, sp, a = pix[perm], sp[perm], a[perm]
    slot = torch.arange(len(pix), device=DEV) - (torch.cumsum(cnt, 0) - cnt)[pix]
    keep = slot < M
    pix, sp, a, slot = pix[keep], sp[keep], a[keep], slot[keep]
    Aa = torch.zeros(size * size, M, device=DEV).index_put((pix, slot), a)
    T = torch.exp(torch.cumsum(torch.log1p(-Aa), 1))     # transmittance after each slot
    Tprev = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], 1)
    w = (Aa * Tprev)[pix, slot]
    cov = 1 - T[:, -1]
    if rgb is None:                                      # blend weights only: image = sum_pairs w * rgb[sp] + (1 - cov) * bg, linear in rgb
        return pix, sp, w, cov
    img = torch.zeros(size * size, 3, device=DEV).index_add(0, pix, w[:, None] * rgb[sp])
    img = img + (1 - cov)[:, None] * bg[None]
    return img.reshape(size, size, 3), cov.reshape(size, size)


def color_fit(g, photo_rgb, bg, cam, valid=None, size=518, iters=300, lam=.2, area_pow=1., op_pow=2., K=25, verbose=False):
    """Re-texture from the photo: with geometry and opacity frozen the frontal image is linear in the splat colours, so the blend
    weights are computed once (true sorted compositing) and the colours are fitted to the photo at full resolution.
    Splats the photo barely sees have almost no gradient and the L2 pull (lam) keeps them at their LAM colour; the pull grows with
    a splat's screen area (area_pow), so detail is painted by the small splats and big soft ones (lenses, haze) can't turn into blotches."""
    with torch.no_grad():
        u, v, zc, C, f, A, s0 = project(g, cam, size)
        pix, sp, w, cov = comp_render(u, v, torch.argsort(torch.argsort(zc)), torch.linalg.inv(C), None,
                                      torch.tensor(np.clip(g[:, 6], 0, 1), dtype=torch.float32, device=DEV), None, size, K=K)
        tgt = torch.tensor(photo_rgb, dtype=torch.float32, device=DEV).reshape(-1, 3)
        area = (C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] ** 2).clamp_min(1e-8).sqrt()
        opq = torch.tensor(np.clip(g[:, 6], .02, 1), dtype=torch.float32, device=DEV)
        wreg = (((area / area.median()) ** area_pow).clamp(.2, 100.) / opq ** op_pow)[:, None]  # faint splats (lens haze) keep their colour too
        base = (1 - cov)[:, None] * torch.tensor(bg, dtype=torch.float32, device=DEV)[None]
        vw = torch.ones(size * size, 1, device=DEV) if valid is None else torch.tensor(valid, dtype=torch.float32, device=DEV).reshape(-1, 1)
    rgb0 = torch.tensor(np.clip(g[:, 3:6], 0, 1), dtype=torch.float32, device=DEV)
    d = torch.zeros_like(rgb0, requires_grad=True)
    opt = torch.optim.Adam([d], lr=.02)
    for it in range(iters):
        opt.zero_grad()
        img = base + torch.zeros(size * size, 3, device=DEV).index_add(0, pix, w[:, None] * (rgb0 + d).clamp(0, 1)[sp])
        l_img = ((img - tgt) * vw).abs().mean()
        (l_img + lam * (wreg * d ** 2).mean()).backward(); opt.step()
        if verbose and it % 50 == 0:
            print(it, float(l_img))
    out = g.copy()
    out[:, 3:6] = (rgb0 + d).clamp(0, 1).detach().cpu().numpy()
    return out


def _bilin(u, v, size, n):
    """Bilinear taps of an n x n control grid spanning the image, at pixel coords (u, v): (flat idx (N,4), weights (N,4))."""
    gx = (u / size).clamp(0, 1) * (n - 1); gy = (v / size).clamp(0, 1) * (n - 1)
    x0 = gx.floor().clamp(max=n - 2).long(); y0 = gy.floor().clamp(max=n - 2).long()
    fx = gx - x0; fy = gy - y0
    idx = torch.stack([y0 * n + x0, y0 * n + x0 + 1, (y0 + 1) * n + x0, (y0 + 1) * n + x0 + 1], 1)
    wts = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
    return idx, wts


def _tv(G):
    return ((G[1:] - G[:-1]) ** 2).mean() + ((G[:, 1:] - G[:, :-1]) ** 2).mean()


def shape_fit(g, photo_rgb, bg, cam, mask=None, anchors=None, valid=None, w_anchor=.02, size=259, iters=200, grid=32, gain_grid=6, gain_bound=.25,
              w_sil=.5, w_smooth=.02, w_tether=1e-4, w_gain=.1, verbose=False, return_debug=False):
    """Shape-only photo fit. One frontal photo constrains where things sit on screen and the overall tone, nothing else, so that is
    all that is fitted: a smooth 2D warp (coarse control grid of screen-space displacements) and a smooth per-channel colour gain.
    Depth, opacity and each splat's own colour detail are left exactly as LAM / the depth refine made them.
    photo_rgb: (size,size,3) float 0..1. mask: optional (size,size) foreground mask of the photo.
    anchors: optional (L,2) pixel positions already aligned by the landmark warp; the fit is held near zero there so it fills in between
    landmarks instead of dragging eyes/mouth around. valid: optional (size,size) weight, 0 where the photo has no real pixels. Returns new g."""
    u0, v0, zc0, C, f, A, s0 = project(g, cam, size)
    Ci = torch.linalg.inv(C)
    rank = torch.argsort(torch.argsort(zc0))
    rgb0 = torch.tensor(np.clip(g[:, 3:6], 0, 1), dtype=torch.float32, device=DEV)
    op0 = torch.tensor(np.clip(g[:, 6], 0, 1), dtype=torch.float32, device=DEV)
    tgt = torch.tensor(photo_rgb, dtype=torch.float32, device=DEV); bgt = torch.tensor(bg, dtype=torch.float32, device=DEV)
    sil = None if mask is None else torch.tensor(mask, dtype=torch.float32, device=DEV)
    vw = torch.ones(size, size, device=DEV) if valid is None else torch.tensor(valid, dtype=torch.float32, device=DEV)
    anc = None
    if anchors is not None:
        at = torch.tensor(np.asarray(anchors), dtype=torch.float32, device=DEV)
        anc = _bilin(at[:, 0], at[:, 1], size, grid)
    W = torch.zeros(grid, grid, 2, device=DEV, requires_grad=True)             # warp, pixels
    Gn = torch.zeros(gain_grid, gain_grid, 3, device=DEV, requires_grad=True)  # log colour gain (pre-tanh)
    wi, ww = _bilin(u0, v0, size, grid); gi, gw = _bilin(u0, v0, size, gain_grid)
    samp = lambda G, i, w_: (G.reshape(-1, G.shape[-1])[i] * w_[..., None]).sum(1)
    gain = lambda: torch.exp(gain_bound * torch.tanh(samp(Gn, gi, gw)))
    down = lambda x, k: torch.nn.functional.avg_pool2d(x.permute(2, 0, 1)[None], k)[0]
    opt = torch.optim.Adam([{"params": [W], "lr": .3}, {"params": [Gn], "lr": .03}])
    for it in range(iters):
        opt.zero_grad()
        d = samp(W, wi, ww)
        img, cov = comp_render(u0 + d[:, 0], v0 + d[:, 1], rank, Ci, (rgb0 * gain()).clamp(0, 1), op0, bgt, size)
        diff = (img - tgt) * vw[..., None]
        l_img = diff.abs().mean() + sum(down(diff, k).abs().mean() for k in (4, 8))
        l_sil = ((cov - sil) * vw).abs().mean() if sil is not None else 0.
        l_reg = w_smooth * _tv(W) + w_tether * (W ** 2).mean() + w_gain * _tv(Gn) + 1e-3 * (Gn ** 2).mean()
        if anc is not None:
            l_reg = l_reg + w_anchor * (samp(W, *anc) ** 2).sum(1).mean()
        (l_img + w_sil * l_sil + l_reg).backward(); opt.step()
        if verbose and it % 25 == 0:
            print(it, float(l_img), float(l_sil), float(l_reg))
    with torch.no_grad():
        d = samp(W, wi, ww)
        # local area change of the warp -> splat scale
        ix, wx = _bilin(u0 + 1, v0, size, grid); iy, wy = _bilin(u0, v0 + 1, size, grid)
        jx = samp(W, ix, wx) - d; jy = samp(W, iy, wy) - d
        jac = ((1 + jx[:, 0]) * (1 + jy[:, 1]) - jx[:, 1] * jy[:, 0]).clamp(.5, 2.).sqrt()
        u, v = u0 + d[:, 0], v0 + d[:, 1]
        c, dist = cam
        out = g.copy()
        out[:, 0] = ((u - size / 2) / f * zc0).cpu().numpy() + c[0]; out[:, 1] = (-(v - size / 2) / f * zc0).cpu().numpy() + c[1]
        out[:, 3:6] = (rgb0 * gain()).clamp(0, 1).cpu().numpy()
        out[:, 7:10] = g[:, 7:10] * jac[:, None].cpu().numpy()
        if return_debug:
            img, cov = comp_render(u, v, rank, Ci, (rgb0 * gain()).clamp(0, 1), op0, bgt, size)
            return out, dict(img=img.cpu().numpy(), cov=cov.cpu().numpy(), warp=W.detach().cpu().numpy(), gain=gain().cpu().numpy())
    return out


@torch.no_grad()
def color_gain(g, photo_rgb, bg, cam, size=259, sigma=7.0, lo=.75, hi=1.35):
    """Cheap, no-iteration tone match: per-pixel ratio of blurred photo to blurred render, applied to splat colours.
    Fixes washed-out / mis-toned skin while leaving LAM's high-frequency texture (and its side-view consistency) untouched."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    img, cov = render_fast(g, cam, size, bg=tuple(bg), K=13)
    P = photo_rgb
    m = np.clip(cov, 0, 1)[..., None]
    num = np.stack([gaussian_filter(P[..., k] * m[..., 0], sigma) for k in range(3)], -1)
    den = np.stack([gaussian_filter(img[..., k] * m[..., 0], sigma) for k in range(3)], -1)
    gain = np.clip((num + 1e-3) / (den + 1e-3), lo, hi)
    u, v, zc, C, f, A, s0 = project(g, cam, size)
    u = u.cpu().numpy(); v = v.cpu().numpy()
    out = g.copy()
    for k in range(3):
        gk = map_coordinates(gain[..., k], [np.clip(v, 0, size - 1), np.clip(u, 0, size - 1)], order=1, mode="nearest")
        out[:, 3 + k] = np.clip(g[:, 3 + k] * gk, 0, 1)
    return out
