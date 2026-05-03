import numpy as np


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,       2*x*z + 2*w*y],
        [2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z,   2*y*z - 2*w*x],
        [2*x*z - 2*w*y,      2*y*z + 2*w*x,       1 - 2*x*x - 2*y*y]
    ])


def read_cameras(path):
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = line.split()
            if len(d) < 5:
                continue
            cam_id  = int(d[0])
            model   = d[1]
            width   = int(d[2])
            height  = int(d[3])
            params  = list(map(float, d[4:]))
            cams[cam_id] = (model, width, height, params)
    return cams


def read_images(path):
    """
    Returns dict: image_id -> (qvec, tvec, cam_id, name)
    qvec: np.array [qw, qx, qy, qz]
    tvec: np.array [tx, ty, tz]
    """
    imgs = {}
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        d = line.split()
        if len(d) < 10:
            i += 1
            continue

        img_id = int(d[0])
        qvec   = np.array(list(map(float, d[1:5])))   # qw qx qy qz
        tvec   = np.array(list(map(float, d[5:8])))   # tx ty tz
        cam_id = int(d[8])
        name   = d[9]

        imgs[img_id] = (qvec, tvec, cam_id, name)
        i += 2  # skip the point2D line

    return imgs


def read_points3D(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = line.split()
            if len(d) < 7:
                continue
            xyz = np.array(list(map(float, d[1:4])), dtype=np.float32)
            rgb = np.array(list(map(int,   d[4:7])), dtype=np.float32) / 255.0
            pts.append((xyz, rgb))
    return pts


def get_intrinsics(cam_tuple):
    model, width, height, params = cam_tuple
    model = model.upper()

    if model == "SIMPLE_PINHOLE":
        # params: f, cx, cy
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy

    elif model == "PINHOLE":
        # params: fx, fy, cx, cy
        return params[0], params[1], params[2], params[3]

    elif model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        # params: f, cx, cy, k
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy

    elif model in ("RADIAL", "RADIAL_FISHEYE"):
        # params: f, cx, cy, k1, k2
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy

    elif model in ("OPENCV", "FULL_OPENCV"):
        # params: fx, fy, cx, cy, ...
        return params[0], params[1], params[2], params[3]

    else:
        # Fallback: assume (f, cx, cy, ...)
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy