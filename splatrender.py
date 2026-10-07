"""Tiny torch Gaussian-splat renderer matching index.html's camera (FOV, dist=DIST_K*radius). Used for landmark mapping + validation."""
import numpy as np, torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
# One camera for the fits and the viewer (keep in sync with FOV / DISTK in index.html). A portrait-lens view: the photo is un-projected
# through this camera, and a close wide one (it was fov .8 at 3.2 radii) makes everything behind the face plane come out too big
# (back of the head / hair wider than the face, forehead looming). DIST_K keeps the old framing.
FOV = .3
DIST_K = 3.2 * np.tan(.4) / np.tan(FOV / 2)
SRC_K = 3.2  # LAM's shell is drawn as if seen from this many radii away (see unflare)


def unflare(g, cam=None, k=None, squash=.55):
    """LAM's raw output is a front shell that fans out toward the back like a camera frustum (top view: a V, not a U): seen through a
    close camera that looks right, in 3D the back of the head / top of the hair is far too big. Re-project it from that close camera
    (k radii away) to ours so the frontal picture is unchanged and the sides become parallel.
    The shell is also about twice too deep: its rim (the hair outline, which on a real head sits around ear depth) hangs ~2 face-widths
    behind the nose, so it swings out when the head turns or tilts. `squash` scales depth behind the face surface. Returns (g, cam of the input)."""
    cam = cam or camera(g)
    c, d1 = cam
    d0 = d1 / DIST_K * (k or SRC_K)
    g = g.copy()
    if squash != 1:
        zf = np.percentile(g[g[:, 6] > .3, 2], 75)                      # ~ the face surface
        g[:, 2] = np.where(g[:, 2] < zf, zf + (g[:, 2] - zf) * squash, g[:, 2])
    z = g[:, 2] - c[2]
    s = ((d1 - z) / np.maximum(d0 - z, 1e-3) * d0 / d1)[:, None]
    g[:, :2] = c[:2] + (g[:, :2] - c[:2]) * s
    g[:, 7:10] *= s
    return g, cam

def camera(g):
    c = g[:, :3].mean(0)
    radius = max(np.linalg.norm(g[:, :3] - c, axis=1).mean() * 2, .05)
    return c, radius * DIST_K

def quat_R(q):
    w, x, y, z = q.unbind(-1)
    R = torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)
    return R.reshape(-1, 3, 3)

@torch.no_grad()
def render(g, size=256, yaw=0.0, sz=1.0, bg=(0.07, 0.07, 0.07), cam=None, chunk=1024, alpha=False, full=None, origin=(0, 0)):
    """g: (N,14) pos3 rgb3 op1 scl3 q4 (LAM order). Returns (size,size,3) float numpy + camera used.
    full/origin: render a size x size window at pixel `origin` of a virtual full x full frame (a crop at the viewer's real resolution)."""
    c, dist = cam if cam is not None else camera(g)
    t = torch.tensor(g, dtype=torch.float32, device=DEV)
    pos = t[:, :3] - torch.tensor(c, dtype=torch.float32, device=DEV)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32, device=DEV)  # viewer's rotY
    pos = pos @ Ry.T
    p = pos.clone(); p[:, 2] -= dist
    full = full or size
    f = 1 / np.tan(FOV / 2) * full / 2
    z = p[:, 2]
    keep = z < -0.01
    q = t[:, 10:14] / t[:, 10:14].norm(dim=1, keepdim=True).clamp_min(1e-8)
    R = Ry @ quat_R(q)
    M = R * t[:, 7:10][:, None, :]
    S = M @ M.transpose(1, 2)
    zi = z.clamp(max=-0.01)
    J = torch.zeros(len(t), 2, 3, device=DEV)
    # image coords: u = c + f*x/(-z), v = c - f*y/(-z)  (v runs down, so the two rows have opposite signs)
    J[:, 0, 0] = -f / zi; J[:, 1, 1] = f / zi
    J[:, 0, 2] = f * p[:, 0] / zi**2; J[:, 1, 2] = -f * p[:, 1] / zi**2
    C = J @ S @ J.transpose(1, 2)
    C[:, 0, 0] += .3; C[:, 1, 1] += .3
    C = C * sz * sz
    px = full / 2 + f * p[:, 0] / (-zi) - origin[0]; py = full / 2 - f * p[:, 1] / (-zi) - origin[1]
    det = (C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] ** 2).clamp_min(1e-8)
    ic = torch.stack([C[:, 1, 1] / det, -C[:, 0, 1] / det, C[:, 1, 0] * 0 + C[:, 0, 0] / det], 1)  # xx, xy, yy of inverse
    order = torch.argsort(-z[keep])  # far (most negative z... ) -> viewer sorts by view z ascending = far first
    order = torch.argsort(z)  # most negative z = farthest first; we composite front-to-back so reverse
    order = order.flip(0)
    idx = torch.nonzero(keep)[:, 0]
    order = order[keep[order]]
    ys, xs = torch.meshgrid(torch.arange(size, device=DEV), torch.arange(size, device=DEV), indexing="ij")
    X = (xs.reshape(-1) + .5).float(); Y = (ys.reshape(-1) + .5).float()
    T = torch.ones(size * size, device=DEV); out = torch.zeros(size * size, 3, device=DEV)
    rgb = t[:, 3:6]; op = t[:, 6]
    for i in range(0, len(order), chunk):
        o = order[i:i + chunk]
        dx = X[:, None] - px[o][None]; dy = Y[:, None] - py[o][None]
        e = ic[o][None]
        r2 = e[..., 0] * dx * dx + 2 * e[..., 1] * dx * dy + e[..., 2] * dy * dy
        a = torch.exp(-.5 * r2) * op[o][None]
        a = torch.where(r2 > 9, torch.zeros_like(a), a).clamp(max=.999)
        Tc = torch.cumprod(1 - a, 1)
        Tprev = torch.cat([torch.ones_like(Tc[:, :1]), Tc[:, :-1]], 1) * T[:, None]
        w = a * Tprev
        out += w @ rgb[o]
        T = T * Tc[:, -1]
    if alpha:
        return (1 - T).reshape(size, size).cpu().numpy(), (c, dist)
    out += T[:, None] * torch.tensor(bg, device=DEV)[None]
    return out.reshape(size, size, 3).clamp(0, 1).cpu().numpy(), (c, dist)
