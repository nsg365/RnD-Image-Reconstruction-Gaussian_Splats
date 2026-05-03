import os
import cv2
import argparse
import numpy as np
import torch
import torch.optim as optim

parser = argparse.ArgumentParser(description="Train 2D Gaussian Splats on a single image")
parser.add_argument("--image",        type=str,   required=True)
parser.add_argument("--iterations",   type=int,   default=5000,   help="Training steps (default: 5000)")
parser.add_argument("--num_splats",   type=int,   default=10000,  help="Initial splat count (default: 10000)")
parser.add_argument("--max_dim",      type=int,   default=512,    help="Max render resolution (default: 512)")
parser.add_argument("--save_every",   type=int,   default=100,    help="Save preview every N steps")
parser.add_argument("--output_dir",   type=str,   default="../outputs")

# densification
parser.add_argument("--densify_from",     type=int,   default=300,    help="Start densification after N steps")
parser.add_argument("--densify_every",    type=int,   default=200,    help="Densify every N steps")
parser.add_argument("--densify_until",    type=int,   default=3500,   help="Stop densification after N steps")
parser.add_argument("--densify_grad_thr", type=float, default=0.0005, help="Position grad norm threshold for splitting")
parser.add_argument("--prune_opacity",    type=float, default=0.05,   help="Prune splats below this opacity")
parser.add_argument("--max_splats",       type=int,   default=30000,  help="Hard cap on splat count")

# init weighting
parser.add_argument("--detail_bias",  type=float, default=0.85,
                    help="Fraction of splats seeded on high-gradient pixels (0=uniform, 1=all-detail)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
os.makedirs(args.output_dir, exist_ok=True)

# load & resize
img_bgr = cv2.imread(args.image)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {args.image}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
h0, w0  = img_rgb.shape[:2]
scale   = args.max_dim / max(h0, w0)
if scale < 1.0:
    img_rgb = cv2.resize(img_rgb, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

target = torch.tensor(img_rgb, dtype=torch.float32, device=device)
H, W   = target.shape[:2]
print(f"Target size: {W}×{H}")

# intrinsics (all kept as float32 to avoid double-promotion in tensor ops)
fx = fy = np.float32(max(W, H) / (2.0 * np.tan(np.radians(30))))
cx = np.float32(W / 2.0)
cy = np.float32(H / 2.0)

R_cam = torch.eye(3,   dtype=torch.float32, device=device)
t_cam = torch.zeros(3, dtype=torch.float32, device=device)

# gradient / detail map for importance sampling
def build_detail_map(img: np.ndarray) -> np.ndarray:
    """Returns a (H,W) probability map: high where image has edges/detail."""
    gray   = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gx     = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy     = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag    = np.sqrt(gx**2 + gy**2)
    # blur to spread probability around edges a bit
    mag    = cv2.GaussianBlur(mag, (0, 0), sigmaX=3)
    mag   += 1e-3        # floor so flat regions still get a few splats
    return mag / mag.sum()

detail_prob = build_detail_map(img_rgb)          # (H, W) normalised

# Save the importance map for inspection
imp_vis = (detail_prob / detail_prob.max() * 255).astype(np.uint8)
cv2.imwrite(os.path.join(args.output_dir, "importance_map_broken.jpg"),
            cv2.applyColorMap(imp_vis, cv2.COLORMAP_INFERNO))
print("Saved importance_map.jpg")

#importance-weighted loss map (used during training)
# Softer version so we don't completely ignore flat areas
loss_weight = torch.tensor(
    detail_prob / detail_prob.max(), dtype=torch.float32, device=device
)                                                         # (H, W)  ∈ [ε, 1]
loss_weight = (loss_weight + 0.2).clamp(max=1.0)         # floor at 0.2 so sky still matters

# gradient-weighted splat initialisation
def init_splats(n: int, detail_bias: float):
    """
    Place `n` splats:
      detail_bias  fraction on high-gradient pixels (importance sampling)
      1 - detail_bias fraction uniformly at random
    """
    rng = np.random.default_rng(42)
    flat_prob = detail_prob.ravel()

    n_detail  = int(n * detail_bias)
    n_uniform = n - n_detail

    # detail samples
    idx_d  = rng.choice(H * W, size=n_detail, replace=True, p=flat_prob)
    vi_d   = (idx_d // W).astype(np.int32)
    ui_d   = (idx_d  % W).astype(np.int32)
    # add sub-pixel jitter so we don't stack exactly on same pixel
    pix_u_d = ui_d + rng.uniform(-0.5, 0.5, n_detail).astype(np.float32)
    pix_v_d = vi_d + rng.uniform(-0.5, 0.5, n_detail).astype(np.float32)

    # uniform samples
    pix_u_u = rng.uniform(0, W, n_uniform).astype(np.float32)
    pix_v_u = rng.uniform(0, H, n_uniform).astype(np.float32)

    pix_u = np.concatenate([pix_u_d, pix_u_u])
    pix_v = np.concatenate([pix_v_d, pix_v_u])

    Z0  = np.float32(1.0)
    X0  = (pix_u - cx) / fx * Z0
    Y0  = (pix_v - cy) / fy * Z0
    xyz_np = np.stack([X0, Y0, np.full(len(pix_u), Z0, dtype=np.float32)], axis=-1).astype(np.float32)

    ui = np.clip(pix_u.astype(np.int32), 0, W - 1)
    vi = np.clip(pix_v.astype(np.int32), 0, H - 1)
    color_np = img_rgb[vi, ui]

    return xyz_np, color_np

xyz_np, color_np = init_splats(args.num_splats, args.detail_bias)
N = xyz_np.shape[0]
print(f"Initialised {N} splats  ({int(N*args.detail_bias)} detail + {N - int(N*args.detail_bias)} uniform)")

# splat tensors (wrapped in a list so densification can replace them)
def make_params(xyz_np_, color_np_):
    """Create leaf tensors for a fresh set of splats."""
    n = xyz_np_.shape[0]
    xyz_       = torch.from_numpy(xyz_np_.astype(np.float32)).to(device).requires_grad_(True)
    color_raw_ = torch.logit(
        torch.from_numpy(color_np_.astype(np.float32)).to(device).clamp(1e-6, 1 - 1e-6)
    ).detach().requires_grad_(True)
    # start small — detail splats should be tight
    log_sx_  = torch.full((n,), -4.0, dtype=torch.float32, device=device, requires_grad=True)
    log_sy_  = torch.full((n,), -4.0, dtype=torch.float32, device=device, requires_grad=True)
    log_sz_  = torch.full((n,), -4.0, dtype=torch.float32, device=device, requires_grad=True)
    opacity_ = torch.full((n,),  0.0, dtype=torch.float32, device=device, requires_grad=True)
    return xyz_, color_raw_, log_sx_, log_sy_, log_sz_, opacity_

xyz, color_raw, log_sx, log_sy, log_sz, opacity = make_params(xyz_np, color_np)

# pixel grid
ys, xs = torch.meshgrid(
    torch.arange(H, device=device, dtype=torch.float32),
    torch.arange(W, device=device, dtype=torch.float32),
    indexing="ij",
)
pixels = torch.stack([xs, ys], dim=-1)   # (H, W, 2)

# render
def render():
    sx = torch.exp(log_sx); sy = torch.exp(log_sy); sz = torch.exp(log_sz)
    alpha = torch.sigmoid(opacity)
    c     = torch.sigmoid(color_raw)

    xyz_cam = xyz @ R_cam.T + t_cam
    xc, yc, zc = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    valid = zc > 0.01
    if not valid.any():
        return torch.zeros((H, W, 3), dtype=torch.float32, device=device)

    xv, yv, zv   = xc[valid], yc[valid], zc[valid]
    sxv, syv, szv = sx[valid], sy[valid], sz[valid]
    av, cv_       = alpha[valid], c[valid]

    u = xv * fx / zv + cx
    v = yv * fy / zv + cy
    Nv = xv.shape[0]

    J = torch.zeros((Nv, 2, 3), dtype=torch.float32, device=device)
    J[:, 0, 0] =  fx / zv;  J[:, 0, 2] = -fx * xv / (zv * zv)
    J[:, 1, 1] =  fy / zv;  J[:, 1, 2] = -fy * yv / (zv * zv)

    cov3D = torch.zeros((Nv, 3, 3), dtype=torch.float32, device=device)
    cov3D[:, 0, 0] = sxv * sxv
    cov3D[:, 1, 1] = syv * syv
    cov3D[:, 2, 2] = szv * szv

    W_mat  = R_cam.unsqueeze(0).expand(Nv, 3, 3)
    Sigma2 = J @ (W_mat @ cov3D @ W_mat.transpose(1, 2)) @ J.transpose(1, 2)
    Sigma2 = Sigma2 + torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0) * 0.3

    det = (Sigma2[:, 0, 0] * Sigma2[:, 1, 1] - Sigma2[:, 0, 1] ** 2).clamp(min=1e-8)
    inv = torch.zeros_like(Sigma2)
    inv[:, 0, 0] =  Sigma2[:, 1, 1] / det
    inv[:, 1, 1] =  Sigma2[:, 0, 0] / det
    inv[:, 0, 1] = -Sigma2[:, 0, 1] / det
    inv[:, 1, 0] = -Sigma2[:, 1, 0] / det

    rad = 3.0 * torch.sqrt(torch.maximum(Sigma2[:, 0, 0], Sigma2[:, 1, 1]))

    canvas = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    TILE = 64
    for y0 in range(0, H, TILE):
        y1 = min(H, y0 + TILE)
        for x0 in range(0, W, TILE):
            x1 = min(W, x0 + TILE)
            mask_t = (u + rad > x0) & (u - rad < x1) & \
                     (v + rad > y0) & (v - rad < y1)
            if not mask_t.any(): continue
            idx_ = torch.argsort(zv[mask_t])
            mu_t  = u[mask_t][idx_];  mv_t  = v[mask_t][idx_]
            inv_t = inv[mask_t][idx_]; a_t  = av[mask_t][idx_]; c_t = cv_[mask_t][idx_]
            px = pixels[y0:y1, x0:x1].reshape(-1, 2)
            dx = px[:, 0:1] - mu_t.unsqueeze(0)
            dy = px[:, 1:2] - mv_t.unsqueeze(0)
            maha  = (dx*dx*inv_t[:,0,0].unsqueeze(0) + dy*dy*inv_t[:,1,1].unsqueeze(0) +
                     2.0*dx*dy*inv_t[:,0,1].unsqueeze(0))
            gauss = torch.exp(-0.5 * maha) * a_t.unsqueeze(0)
            T_after  = torch.cumprod(1.0 - gauss, dim=1)
            T_before = torch.cat([torch.ones((px.shape[0], 1), dtype=torch.float32, device=device),
                                  T_after[:, :-1]], dim=1)
            canvas[y0:y1, x0:x1] = (T_before * gauss @ c_t).reshape(y1-y0, x1-x0, 3)

    return torch.clamp(canvas, 0.0, 1.0)

# optimiser
def make_optimizer():
    return optim.Adam([
        {"params": color_raw, "lr": 5e-3},
        {"params": xyz,       "lr": 3e-4},
        {"params": log_sx,    "lr": 4e-3},
        {"params": log_sy,    "lr": 4e-3},
        {"params": log_sz,    "lr": 4e-3},
        {"params": opacity,   "lr": 4e-3},
    ])

optimizer = make_optimizer()

# adaptive densification
# Accumulate position gradient norms across steps between densification events
xyz_grad_accum  = torch.zeros(xyz.shape[0], dtype=torch.float32, device=device)
grad_accum_count = 0

def densify_and_prune(step):
    """
    Split splats with high accumulated position-gradient (under-reconstructed),
    clone small-but-active ones in detail regions, prune transparent ones.
    Replaces global param tensors in-place.
    """
    global xyz, color_raw, log_sx, log_sy, log_sz, opacity
    global xyz_grad_accum, grad_accum_count, optimizer

    n_before = xyz.shape[0]

    # average accumulated gradient
    avg_grad = xyz_grad_accum / max(grad_accum_count, 1)   # (N,)

    # masks
    opac_val    = torch.sigmoid(opacity).detach()
    prune_mask  = opac_val < args.prune_opacity                    # too transparent
    split_mask  = (avg_grad > args.densify_grad_thr) & ~prune_mask # high-error → split
    # clone: active but splat is still small (detail regions not yet covered)
    sx_screen   = torch.exp(log_sx).detach() * fx / xyz[:, 2].detach().clamp(min=0.1)
    clone_mask  = (~split_mask) & (~prune_mask) & \
                  (avg_grad > args.densify_grad_thr * 0.3) & \
                  (sx_screen < 3.0)

    keep_mask   = ~prune_mask

    # collect kept splats
    def g(t, mask): return t[mask].detach()

    kxyz  = g(xyz,       keep_mask)
    kcr   = g(color_raw, keep_mask)
    klsx  = g(log_sx,    keep_mask)
    klsy  = g(log_sy,    keep_mask)
    klsz  = g(log_sz,    keep_mask)
    kop   = g(opacity,   keep_mask)

    # split: replace one large splat with two smaller offset ones
    if split_mask.any():
        sxyz = g(xyz,       split_mask)
        scr  = g(color_raw, split_mask)
        LOG_SCALE = torch.tensor(np.log(1.6), dtype=torch.float32, device=device)
        slsx = g(log_sx,    split_mask) - LOG_SCALE   # shrink
        slsy = g(log_sy,    split_mask) - LOG_SCALE
        slsz = g(log_sz,    split_mask) - LOG_SCALE
        sop  = g(opacity,   split_mask)
        # random small offset in XY
        noise = torch.randn_like(sxyz) * torch.exp(g(log_sx, split_mask)).unsqueeze(1) * 0.5
        noise[:, 2] = 0.0
        sxyz2 = sxyz + noise; sxyz3 = sxyz - noise

        kxyz = torch.cat([kxyz, sxyz2, sxyz3])
        kcr  = torch.cat([kcr,  scr,   scr  ])
        klsx = torch.cat([klsx, slsx,  slsx ])
        klsy = torch.cat([klsy, slsy,  slsy ])
        klsz = torch.cat([klsz, slsz,  slsz ])
        kop  = torch.cat([kop,  sop,   sop  ])

    # ── clone: duplicate small active splats with tiny offset ─────────────────
    if clone_mask.any():
        cxyz = g(xyz,       clone_mask)
        ccr  = g(color_raw, clone_mask)
        clsx = g(log_sx,    clone_mask)
        clsy = g(log_sy,    clone_mask)
        clsz = g(log_sz,    clone_mask)
        cop  = g(opacity,   clone_mask)
        noise_c = torch.randn_like(cxyz) * 0.003
        noise_c[:, 2] = 0.0

        kxyz = torch.cat([kxyz, cxyz + noise_c])
        kcr  = torch.cat([kcr,  ccr  ])
        klsx = torch.cat([klsx, clsx ])
        klsy = torch.cat([klsy, clsy ])
        klsz = torch.cat([klsz, clsz ])
        kop  = torch.cat([kop,  cop  ])

    # enforce hard cap
    if kxyz.shape[0] > args.max_splats:
        # keep highest-opacity ones when over cap
        keep_top = torch.argsort(torch.sigmoid(kop), descending=True)[:args.max_splats]
        kxyz = kxyz[keep_top]; kcr  = kcr[keep_top]
        klsx = klsx[keep_top]; klsy = klsy[keep_top]; klsz = klsz[keep_top]
        kop  = kop[keep_top]

    n_after = kxyz.shape[0]
    print(f"  Densify @ step {step}: {n_before} → {n_after} splats "
          f"(+{split_mask.sum().item()} split, +{clone_mask.sum().item()} cloned, "
          f"-{prune_mask.sum().item()} pruned)")

    # reassign globals as new leaf tensors 
    xyz       = kxyz.requires_grad_(True)
    color_raw = kcr.requires_grad_(True)
    log_sx    = klsx.requires_grad_(True)
    log_sy    = klsy.requires_grad_(True)
    log_sz    = klsz.requires_grad_(True)
    opacity   = kop.requires_grad_(True)

    xyz_grad_accum   = torch.zeros(xyz.shape[0], dtype=torch.float32, device=device)
    grad_accum_count = 0
    optimizer = make_optimizer()

# training loop
def train(num_steps: int):
    global xyz_grad_accum, grad_accum_count

    print(f"\n🚀 Training for {num_steps} steps | initial {xyz.shape[0]} splats | device={device}\n")

    for step in range(num_steps):
        optimizer.zero_grad()
        pred = render()

        # detail-weighted MSE: high-gradient regions count more
        sq_err = (pred - target) ** 2                    # (H, W, 3)
        loss   = (sq_err.mean(dim=2) * loss_weight).mean()
        loss.backward()

        # accumulate xyz gradient norms for densification decisions
        if xyz.grad is not None:
            xyz_grad_accum  += xyz.grad.detach().norm(dim=1)
            grad_accum_count += 1

        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:5d}/{num_steps} | Loss: {loss.item():.6f} | Splats: {xyz.shape[0]}")

        # densification
        if (args.densify_from <= step <= args.densify_until and
                step % args.densify_every == 0 and step > 0):
            densify_and_prune(step)

        # save preview
        if step % args.save_every == 0:
            out = (pred.detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(args.output_dir, f"step_{step:05d}.jpg"),
                cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
            )

    # final render & checkpoint
    with torch.no_grad():
        final = render()
    out = (final.cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, "final_broken.jpg"),
                cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    torch.save({
        "xyz":       xyz.detach(),
        "color_raw": color_raw.detach(),
        "log_sx":    log_sx.detach(),
        "log_sy":    log_sy.detach(),
        "log_sz":    log_sz.detach(),
        "opacity":   opacity.detach(),
    }, os.path.join(args.output_dir, "broken_checkpoint.pt"))

    print(f"\n Done. Final splat count: {xyz.shape[0]}")
    print(f"   Outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    train(args.iterations)