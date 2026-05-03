"""
reconstruct_splats.py  (FIXED)
──────────────────────────────
Key fixes over the original:
  1. The optimization no longer computes loss against the broken image inside
     the target mask (that was teaching the hand to disappear).
  2. Instead, we render the REFERENCE checkpoint and use that render as the
     supervision signal inside the target region — so we're pulling the
     transplanted splats toward what the intact hand should look like.
  3. A boundary-consistency loss is added at the EDGE of the target mask so
     the transplanted region blends smoothly with the surrounding broken image.
  4. Only xyz (position/alignment) is optimized; color/scale/opacity are frozen
     for the first `--freeze_appearance_steps` iterations so the hand shape
     doesn't get corrupted before it's properly aligned.
  5. Learning rates are much lower by default to avoid over-shooting.
  6. Step-0 render is saved before any gradient step so you always have the
     clean transplant as a baseline.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--ref",           type=str, required=True)
parser.add_argument("--broken",        type=str, required=True)
parser.add_argument("--ref_ckpt",      type=str, required=True)
parser.add_argument("--broken_ckpt",   type=str, default=None)
parser.add_argument("--src_mask",      type=str, default=None)
parser.add_argument("--tgt_mask",      type=str, default=None)
parser.add_argument("--iterations",    type=int,   default=3000)
# ── lower LRs than before; xyz only for first N steps ─────────────────────
parser.add_argument("--lr_xyz",        type=float, default=5e-5)   # was 3e-4
parser.add_argument("--lr_color",      type=float, default=1e-3)   # was 5e-3
parser.add_argument("--lr_scale",      type=float, default=5e-4)   # was 4e-3
parser.add_argument("--lr_opacity",    type=float, default=5e-4)   # was 4e-3
parser.add_argument("--freeze_appearance_steps", type=int, default=500,
                    help="Freeze color/scale/opacity for this many steps; "
                         "only align position first.")
parser.add_argument("--boundary_weight", type=float, default=0.3,
                    help="Weight of boundary-consistency loss term.")
parser.add_argument("--boundary_px",    type=int,   default=8,
                    help="Width (pixels) of the boundary ring around tgt_mask "
                         "used for blending loss.")
parser.add_argument("--max_dim",       type=int,   default=512)
parser.add_argument("--save_every",    type=int,   default=100)
parser.add_argument("--output_dir",    type=str,   default="./outputs")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
os.makedirs(args.output_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Image / mask I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_resize(path, max_dim):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w  = rgb.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return rgb


def load_mask(path, target_hw):
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    H, W = target_hw
    if raw.shape != (H, W):
        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)
    return raw > 10   # bool


def build_boundary_mask(mask_bool, px):
    """
    Returns a bool mask of the ring of `px` pixels just OUTSIDE the target
    mask. This ring lies on the broken image and has valid pixel colors —
    perfect for a blending/continuity loss.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0
    return dilated & ~mask_bool   # ring outside the mask


def build_camera(W, H):
    fx = fy = float(max(W, H) / (2.0 * np.tan(np.radians(30))))
    return fx, fy, W / 2.0, H / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian renderer
# ══════════════════════════════════════════════════════════════════════════════

def render_splats(xyz, color_raw, log_sx, log_sy, log_sz, opacity,
                  H, W, fx, fy, cx, cy, device):
    sx = torch.exp(log_sx); sy = torch.exp(log_sy); sz = torch.exp(log_sz)
    alpha = torch.sigmoid(opacity)
    c     = torch.sigmoid(color_raw)

    R_cam = torch.eye(3,   dtype=torch.float32, device=device)
    t_cam = torch.zeros(3, dtype=torch.float32, device=device)

    xyz_cam = xyz @ R_cam.T + t_cam
    xc, yc, zc = xyz_cam[:,0], xyz_cam[:,1], xyz_cam[:,2]
    valid = zc > 0.01
    if not valid.any():
        return torch.zeros((H, W, 3), dtype=torch.float32, device=device)

    xv, yv, zv    = xc[valid], yc[valid], zc[valid]
    sxv, syv, szv = sx[valid], sy[valid], sz[valid]
    av, cv_        = alpha[valid], c[valid]

    u  = xv * fx / zv + cx
    v  = yv * fy / zv + cy
    Nv = xv.shape[0]

    J = torch.zeros((Nv, 2, 3), dtype=torch.float32, device=device)
    J[:,0,0] =  fx / zv;  J[:,0,2] = -fx * xv / (zv * zv)
    J[:,1,1] =  fy / zv;  J[:,1,2] = -fy * yv / (zv * zv)

    cov3D = torch.zeros((Nv, 3, 3), dtype=torch.float32, device=device)
    cov3D[:,0,0] = sxv * sxv
    cov3D[:,1,1] = syv * syv
    cov3D[:,2,2] = szv * szv

    W_mat  = R_cam.unsqueeze(0).expand(Nv, 3, 3)
    Sigma2 = J @ (W_mat @ cov3D @ W_mat.transpose(1,2)) @ J.transpose(1,2)
    Sigma2 = Sigma2 + torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0) * 0.3

    det = (Sigma2[:,0,0]*Sigma2[:,1,1] - Sigma2[:,0,1]**2).clamp(min=1e-8)
    inv = torch.zeros_like(Sigma2)
    inv[:,0,0] =  Sigma2[:,1,1] / det
    inv[:,1,1] =  Sigma2[:,0,0] / det
    inv[:,0,1] = -Sigma2[:,0,1] / det
    inv[:,1,0] = -Sigma2[:,1,0] / det

    rad = 3.0 * torch.sqrt(torch.maximum(Sigma2[:,0,0], Sigma2[:,1,1]))

    ys_g, xs_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack([xs_g, ys_g], dim=-1)

    canvas = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    TILE = 64
    for y0 in range(0, H, TILE):
        y1 = min(H, y0+TILE)
        for x0 in range(0, W, TILE):
            x1 = min(W, x0+TILE)
            mt = (u+rad>x0)&(u-rad<x1)&(v+rad>y0)&(v-rad<y1)
            if not mt.any(): continue
            idx_  = torch.argsort(zv[mt])
            mu_t  = u[mt][idx_]; mv_t  = v[mt][idx_]
            inv_t = inv[mt][idx_]; a_t = av[mt][idx_]; c_t = cv_[mt][idx_]
            px    = pixels[y0:y1, x0:x1].reshape(-1, 2)
            dx    = px[:,0:1] - mu_t.unsqueeze(0)
            dy    = px[:,1:2] - mv_t.unsqueeze(0)
            maha  = (dx*dx*inv_t[:,0,0].unsqueeze(0) + dy*dy*inv_t[:,1,1].unsqueeze(0)
                     + 2.0*dx*dy*inv_t[:,0,1].unsqueeze(0))
            gauss    = torch.exp(-0.5*maha) * a_t.unsqueeze(0)
            T_after  = torch.cumprod(1.0-gauss, dim=1)
            T_before = torch.cat([torch.ones((px.shape[0],1), dtype=torch.float32, device=device),
                                  T_after[:,:-1]], dim=1)
            canvas[y0:y1, x0:x1] = (T_before*gauss @ c_t).reshape(y1-y0, x1-x0, 3)

    return torch.clamp(canvas, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Transplant helpers
# ══════════════════════════════════════════════════════════════════════════════

def project_uvs(xyz_t, fx, fy, cx, cy):
    x, y, z = xyz_t[:,0], xyz_t[:,1], xyz_t[:,2]
    z = z.clamp(min=0.01)
    return torch.stack([x*fx/z + cx, y*fy/z + cy], dim=1)


def mask_bbox(mask):
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def remap_xyz(xyz_t,
              fx_ref, fy_ref, cx_ref, cy_ref,
              fx_brk, fy_brk, cx_brk, cy_brk,
              src_mask, tgt_mask):
    uvs  = project_uvs(xyz_t, fx_ref, fy_ref, cx_ref, cy_ref)
    u_np = uvs[:,0].detach().cpu().numpy()
    v_np = uvs[:,1].detach().cpu().numpy()

    H_ref, W_ref = src_mask.shape
    in_src = src_mask[
        np.clip(v_np.astype(int), 0, H_ref-1),
        np.clip(u_np.astype(int), 0, W_ref-1),
    ]

    sx0,sy0,sx1,sy1 = mask_bbox(src_mask)
    tx0,ty0,tx1,ty1 = mask_bbox(tgt_mask)
    sw = max(sx1-sx0, 1); sh = max(sy1-sy0, 1)
    tw = max(tx1-tx0, 1); th = max(ty1-ty0, 1)

    u_norm = (u_np[in_src] - sx0) / sw
    v_norm = (v_np[in_src] - sy0) / sh

    xyz_sel = xyz_t[in_src].detach().clone()
    z_val   = xyz_sel[:,2]
    xyz_new = xyz_sel.clone()
    xyz_new[:,0] = (torch.tensor(u_norm*tw+tx0, dtype=torch.float32, device=xyz_t.device) - cx_brk) / fx_brk * z_val
    xyz_new[:,1] = (torch.tensor(v_norm*th+ty0, dtype=torch.float32, device=xyz_t.device) - cy_brk) / fy_brk * z_val

    return in_src, xyz_new


# ══════════════════════════════════════════════════════════════════════════════
# Reference render helper  (used to build supervision signal)
# ══════════════════════════════════════════════════════════════════════════════

def render_ref_into_target(ref_ckpt_data, src_mask, tgt_mask,
                            H_ref, W_ref, fx_ref, fy_ref, cx_ref, cy_ref,
                            H_brk, W_brk, fx_brk, fy_brk, cx_brk, cy_brk,
                            device):
    """
    Render the REFERENCE splats (transplanted into target space) to produce
    a clean supervision image for the target region.  This is what the
    optimization should be trying to match — NOT the broken image.
    """
    r_xyz = ref_ckpt_data["xyz"].to(device)
    r_cr  = ref_ckpt_data["color_raw"].to(device)
    r_lsx = ref_ckpt_data["log_sx"].to(device)
    r_lsy = ref_ckpt_data["log_sy"].to(device)
    r_lsz = ref_ckpt_data["log_sz"].to(device)
    r_op  = ref_ckpt_data["opacity"].to(device)

    in_src, xyz_t = remap_xyz(
        r_xyz, fx_ref, fy_ref, cx_ref, cy_ref,
        fx_brk, fy_brk, cx_brk, cy_brk,
        src_mask, tgt_mask)

    with torch.no_grad():
        ref_render = render_splats(
            xyz_t, r_cr[in_src], r_lsx[in_src], r_lsy[in_src],
            r_lsz[in_src], r_op[in_src],
            H_brk, W_brk, fx_brk, fy_brk, cx_brk, cy_brk, device)

    return ref_render   # (H_brk, W_brk, 3), no grad


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not args.src_mask or not args.tgt_mask:
        print("\nERROR: --src_mask and --tgt_mask are required.")
        sys.exit(1)

    # ── load images ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading images …")
    ref_rgb = load_resize(args.ref,    args.max_dim)
    brk_rgb = load_resize(args.broken, args.max_dim)
    H_ref, W_ref = ref_rgb.shape[:2]
    H_brk, W_brk = brk_rgb.shape[:2]
    fx_ref, fy_ref, cx_ref, cy_ref = build_camera(W_ref, H_ref)
    fx_brk, fy_brk, cx_brk, cy_brk = build_camera(W_brk, H_brk)

    # ── load masks ─────────────────────────────────────────────────────────────
    print("\n[2/5] Loading masks …")
    src_mask = load_mask(args.src_mask, (H_ref, W_ref))
    tgt_mask = load_mask(args.tgt_mask, (H_brk, W_brk))
    print(f"  Source mask : {src_mask.sum()} px  ({args.src_mask})")
    print(f"  Target mask : {tgt_mask.sum()} px  ({args.tgt_mask})")

    if not src_mask.any():
        sys.exit("ERROR: Source mask is empty.")
    if not tgt_mask.any():
        sys.exit("ERROR: Target mask is empty.")

    # boundary ring for blending loss
    boundary_mask = build_boundary_mask(tgt_mask, args.boundary_px)
    print(f"  Boundary ring: {boundary_mask.sum()} px")

    cv2.imwrite(os.path.join(args.output_dir, "src_mask_used.png"),  (src_mask*255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.output_dir, "tgt_mask_used.png"),  (tgt_mask*255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.output_dir, "boundary_mask.png"),  (boundary_mask*255).astype(np.uint8))

    # ── build supervision tensors ──────────────────────────────────────────────
    # FIX #1: supervision inside the target mask comes from the REFERENCE render,
    #         NOT from the broken image.  The broken image is only used for the
    #         boundary ring outside the mask.
    print("\n[3/5] Loading checkpoints …")
    ref_ckpt = torch.load(args.ref_ckpt, map_location=device, weights_only=True)

    print("  Rendering reference into target space for supervision …")
    ref_supervision = render_ref_into_target(
        ref_ckpt, src_mask, tgt_mask,
        H_ref, W_ref, fx_ref, fy_ref, cx_ref, cy_ref,
        H_brk, W_brk, fx_brk, fy_brk, cx_brk, cy_brk,
        device)   # (H_brk, W_brk, 3), no grad

    # save it so you can inspect what the optimizer is aiming for
    ref_sup_np = (ref_supervision.cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, "ref_supervision.jpg"),
                cv2.cvtColor(ref_sup_np, cv2.COLOR_RGB2BGR))
    print("  Saved ref_supervision.jpg — this is what the optimiser targets inside the mask")

    brk_tensor     = torch.tensor(brk_rgb,      dtype=torch.float32, device=device)
    tgt_weight     = torch.tensor(tgt_mask,      dtype=torch.float32, device=device)
    boundary_weight = torch.tensor(boundary_mask, dtype=torch.float32, device=device)

    # ── transplant ─────────────────────────────────────────────────────────────
    print("\n[4/5] Transplanting splats …")
    r_xyz = ref_ckpt["xyz"].to(device)
    r_cr  = ref_ckpt["color_raw"].to(device)
    r_lsx = ref_ckpt["log_sx"].to(device)
    r_lsy = ref_ckpt["log_sy"].to(device)
    r_lsz = ref_ckpt["log_sz"].to(device)
    r_op  = ref_ckpt["opacity"].to(device)
    print(f"  Ref: {r_xyz.shape[0]} splats")

    in_src, xyz_transplant = remap_xyz(
        r_xyz, fx_ref, fy_ref, cx_ref, cy_ref,
        fx_brk, fy_brk, cx_brk, cy_brk,
        src_mask, tgt_mask)
    n_transplant = int(in_src.sum())
    print(f"  Transplanting {n_transplant} splats")
    if n_transplant == 0:
        sys.exit("ERROR: No splats project into source mask.")

    cr_t  = r_cr[in_src].detach().clone()
    lsx_t = r_lsx[in_src].detach().clone()
    lsy_t = r_lsy[in_src].detach().clone()
    lsz_t = r_lsz[in_src].detach().clone()
    op_t  = r_op[in_src].detach().clone()

    # ── broken background splats ───────────────────────────────────────────────
    if args.broken_ckpt and os.path.exists(args.broken_ckpt):
        brk_ckpt = torch.load(args.broken_ckpt, map_location=device, weights_only=True)
        b_xyz = brk_ckpt["xyz"].to(device)
        b_cr  = brk_ckpt["color_raw"].to(device)
        b_lsx = brk_ckpt["log_sx"].to(device)
        b_lsy = brk_ckpt["log_sy"].to(device)
        b_lsz = brk_ckpt["log_sz"].to(device)
        b_op  = brk_ckpt["opacity"].to(device)
        print(f"  Broken: {b_xyz.shape[0]} splats")

        brk_uvs = project_uvs(b_xyz, fx_brk, fy_brk, cx_brk, cy_brk)
        bu_np   = brk_uvs[:,0].detach().cpu().numpy()
        bv_np   = brk_uvs[:,1].detach().cpu().numpy()
        in_tgt  = tgt_mask[
            np.clip(bv_np.astype(int), 0, H_brk-1),
            np.clip(bu_np.astype(int), 0, W_brk-1),
        ]
        keep = ~in_tgt
        print(f"  Removed {int(in_tgt.sum())} broken splats in target mask")

        base_xyz = b_xyz[keep].detach(); base_cr  = b_cr[keep].detach()
        base_lsx = b_lsx[keep].detach(); base_lsy = b_lsy[keep].detach()
        base_lsz = b_lsz[keep].detach(); base_op  = b_op[keep].detach()
    else:
        print("  No broken checkpoint — transplanted splats only")
        def _z(shape): return torch.zeros(shape, dtype=torch.float32, device=device)
        base_xyz=_z((0,3)); base_cr=_z((0,3))
        base_lsx=_z((0,)); base_lsy=_z((0,)); base_lsz=_z((0,)); base_op=_z((0,))

    n_base = base_xyz.shape[0]
    print(f"  Combined: {n_base} frozen + {n_transplant} trainable = {n_base+n_transplant} total")

    # ── save step-0 render BEFORE any gradients ────────────────────────────────
    # FIX #2: capture the clean transplant immediately; this is your best
    #         reference and should be the floor, not something to optimise away.
    t_xyz = xyz_transplant.clone()
    with torch.no_grad():
        step0 = render_splats(
            torch.cat([base_xyz, t_xyz]), torch.cat([base_cr, cr_t]),
            torch.cat([base_lsx, lsx_t]), torch.cat([base_lsy, lsy_t]),
            torch.cat([base_lsz, lsz_t]), torch.cat([base_op,  op_t]),
            H_brk, W_brk, fx_brk, fy_brk, cx_brk, cy_brk, device)
    s0 = (step0.cpu().numpy()*255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, "step_00000.jpg"),
                cv2.cvtColor(s0, cv2.COLOR_RGB2BGR))
    print(f"  Saved step_00000.jpg (clean transplant, no optimisation)")

    # ── make trainable tensors ─────────────────────────────────────────────────
    t_xyz = xyz_transplant.requires_grad_(True)
    # FIX #3: appearance params start frozen; unfrozen after freeze_appearance_steps
    t_cr  = cr_t.detach().clone()
    t_lsx = lsx_t.detach().clone()
    t_lsy = lsy_t.detach().clone()
    t_lsz = lsz_t.detach().clone()
    t_op  = op_t.detach().clone()

    def make_optimizer(include_appearance: bool):
        params = [{"params": t_xyz, "lr": args.lr_xyz}]
        if include_appearance:
            t_cr.requires_grad_(True)
            t_lsx.requires_grad_(True); t_lsy.requires_grad_(True); t_lsz.requires_grad_(True)
            t_op.requires_grad_(True)
            params += [
                {"params": t_cr,  "lr": args.lr_color},
                {"params": t_lsx, "lr": args.lr_scale},
                {"params": t_lsy, "lr": args.lr_scale},
                {"params": t_lsz, "lr": args.lr_scale},
                {"params": t_op,  "lr": args.lr_opacity},
            ]
        return optim.Adam(params)

    optimizer = make_optimizer(include_appearance=False)
    appearance_unlocked = False

    def render_combined():
        return render_splats(
            torch.cat([base_xyz, t_xyz]),
            torch.cat([base_cr,  t_cr]),
            torch.cat([base_lsx, t_lsx]),
            torch.cat([base_lsy, t_lsy]),
            torch.cat([base_lsz, t_lsz]),
            torch.cat([base_op,  t_op]),
            H_brk, W_brk, fx_brk, fy_brk, cx_brk, cy_brk, device)

    # ── training ───────────────────────────────────────────────────────────────
    print(f"\n[5/5] Fine-tuning {n_transplant} transplanted splats for {args.iterations} steps …")
    print(f"  Appearance (color/scale/opacity) frozen for first {args.freeze_appearance_steps} steps\n")

    for step in range(args.iterations):

        # FIX #4: unlock appearance after freeze period
        if step == args.freeze_appearance_steps and not appearance_unlocked:
            optimizer = make_optimizer(include_appearance=True)
            appearance_unlocked = True
            print(f"  Step {step}: appearance params unlocked")

        optimizer.zero_grad()
        pred = render_combined()

        # ── FIX #1 (core): loss inside mask targets the REFERENCE render,
        #                   NOT the broken image ─────────────────────────────
        sq_err_inner = ((pred - ref_supervision) ** 2).mean(dim=2)
        loss_inner   = (sq_err_inner * tgt_weight).sum() / tgt_weight.sum().clamp(min=1)

        # ── boundary loss: match the broken image in the ring just outside
        #    the mask so the transplant blends continuously ──────────────────
        sq_err_boundary = ((pred - brk_tensor) ** 2).mean(dim=2)
        loss_boundary   = (sq_err_boundary * boundary_weight).sum() / boundary_weight.sum().clamp(min=1)

        loss = loss_inner + args.boundary_weight * loss_boundary
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"  Step {step:5d}/{args.iterations}  |  "
                  f"Loss: {loss.item():.6f}  "
                  f"(inner={loss_inner.item():.6f}, "
                  f"boundary={loss_boundary.item():.6f})")

        if step % args.save_every == 0 and step > 0:
            out = (pred.detach().cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.output_dir, f"step_{step:05d}.jpg"),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    # ── final outputs ──────────────────────────────────────────────────────────
    with torch.no_grad():
        final = render_combined()
    out = (final.cpu().numpy()*255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, "reconstructed.jpg"),
                cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    vis = out.copy()
    vis[tgt_mask] = (vis[tgt_mask].astype(np.float32)*0.6
                     + np.array([0,200,200], dtype=np.float32)*0.4).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, "reconstructed_overlay.jpg"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    torch.save({
        "xyz":       torch.cat([base_xyz, t_xyz.detach()]),
        "color_raw": torch.cat([base_cr,  t_cr.detach()]),
        "log_sx":    torch.cat([base_lsx, t_lsx.detach()]),
        "log_sy":    torch.cat([base_lsy, t_lsy.detach()]),
        "log_sz":    torch.cat([base_lsz, t_lsz.detach()]),
        "opacity":   torch.cat([base_op,  t_op.detach()]),
        "meta": {"n_base": n_base, "n_transplant": n_transplant},
    }, os.path.join(args.output_dir, "reconstructed_checkpoint.pt"))

    print(f"\n✅  Done. Outputs → {args.output_dir}/")
    print(f"   step_00000.jpg              — clean transplant (baseline)")
    print(f"   ref_supervision.jpg         — what the optimiser targeted")
    print(f"   reconstructed.jpg           — final render")
    print(f"   reconstructed_overlay.jpg   — target region highlighted")
    print(f"   reconstructed_checkpoint.pt — full checkpoint")


if __name__ == "__main__":
    main()