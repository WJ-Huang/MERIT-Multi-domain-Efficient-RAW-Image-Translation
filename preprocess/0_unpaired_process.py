import os
import errno
from copy import deepcopy

import cv2
import numpy as np
import rawpy
from scipy.io import savemat

# ---------------------------
# 全局目标尺寸（强制偶数）/ Global target size (force even)
# ---------------------------
TARGET_W = 2000
TARGET_H = 1500
PAD_VALUE = 0.0

def _floor_even(x: int) -> int:
    x = int(x)
    return x - (x % 2)

TARGET_W = _floor_even(TARGET_W)
TARGET_H = _floor_even(TARGET_H)

# ---------------------------
# 基础工具 / Utilities
# ---------------------------
def apply_orient(arr, mode):
    """
    mode ∈ {'none','cw90','ccw90','rot180','hflip','vflip'}
    """
    if mode in (None, 'none'):
        return arr
    if mode == 'cw90':
        return np.rot90(arr, -1, axes=(0, 1))
    if mode == 'ccw90':
        return np.rot90(arr,  1, axes=(0, 1))
    if mode == 'rot180':
        return np.rot90(arr,  2, axes=(0, 1))
    if mode == 'hflip':
        return np.flip(arr, axis=1)
    if mode == 'vflip':
        return np.flip(arr, axis=0)
    raise ValueError(f"Unknown orientation mode: {mode}")

def check_dir(path_):
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def imwrite(filename, image):
    """保存 0~1 RGB 浮点图到 8-bit 文件 / Save RGB float [0,1] to 8-bit file."""
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0) * 255.0
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)

# ---------------------------
# RGGB 打包/解包 / RGGB pack helpers
# ---------------------------
def _cfa_to_pos_order(cfa):
    """
    将 2x2 CFA 编码转为位置索引 [R,G1,G2,B]（0..3）/ Map CFA to pos indices.
    支持两类输入:
      - 位置式 position-based: 包含 {0,1,2,3} 的排列
      - 颜色式 color-based: 0=R, 1/3=G, 2=B
    """
    cfa = np.asarray(cfa, dtype=int).ravel()
    if cfa.size != 4:
        raise ValueError(f"CFA must have 4 entries, got {cfa.size}: {cfa}")

    vals = np.unique(cfa)
    if set(vals.tolist()) == {0, 1, 2, 3} and np.all(np.sort(cfa) == np.array([0, 1, 2, 3])):
        return cfa

    if not set(vals.tolist()).issubset({0, 1, 2, 3}):
        raise ValueError(f"Unsupported CFA values: {vals}")

    r_pos = np.where(cfa == 0)[0]
    b_pos = np.where(cfa == 2)[0]
    g_pos = np.where((cfa == 1) | (cfa == 3))[0]

    if r_pos.size != 1 or b_pos.size != 1 or g_pos.size != 2:
        raise ValueError(f"Color-based CFA must contain 1×R, 2×G(1/3), 1×B. Got R:{r_pos}, G:{g_pos}, B:{b_pos}")

    g_pos = np.sort(g_pos)
    return np.array([int(r_pos[0]), int(g_pos[0]), int(g_pos[1]), int(b_pos[0])], dtype=int)

def pack_rggb(raw_bayer_2d, cfa):
    """
    Bayer 2D 马赛克 → 4 通道 RGGB 打包 / 2D Bayer mosaic → 4ch [R,G1,G2,B].
    输出形状 (H/2, W/2, 4)，通道顺序 [R, G1, G2, B]
    """
    if raw_bayer_2d.ndim != 2:
        raise ValueError(f"raw_image must be 2D, got shape {raw_bayer_2d.shape}")

    h, w = raw_bayer_2d.shape
    pos_order = _cfa_to_pos_order(cfa)
    idx = [(0, 0), (0, 1), (1, 0), (1, 1)]

    channels = []
    for pos in pos_order:
        rr, cc = idx[pos]
        ch = raw_bayer_2d[rr:h:2, cc:w:2].copy()
        channels.append(ch)

    base_shape = channels[0].shape
    if not all(ch.shape == base_shape for ch in channels):
        raise RuntimeError(f"Packed channel shapes mismatch: {[ch.shape for ch in channels]}")

    return np.stack(channels, axis=-1)

def from_rggb_to_rgb(rggb):
    """可视化：合并两路 G 并按 RGB 返回 / Merge G1,G2 → G and return RGB."""
    g = (rggb[..., 1] + rggb[..., 2]) / 2.0
    return np.stack([rggb[..., 0], g, rggb[..., 3]], axis=-1)

# ---------------------------
# Fine downsampling (downsampling only)
# ---------------------------
def _resize_rggb_downsample(raw_rggb: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    RGGB 四平面逐通道、抗混叠下采样 / Per-plane anti-alias downsampling for RGGB.
    """
    assert raw_rggb.ndim == 3 and raw_rggb.shape[2] == 4, f"Expect (H,W,4), got {raw_rggb.shape}"
    out = np.empty((new_h, new_w, 4), dtype=raw_rggb.dtype)
    for i in range(4):
        out[..., i] = cv2.resize(
            raw_rggb[..., i], (new_w, new_h), interpolation=cv2.INTER_AREA
        )
    return out

def fit_cover_center_crop_downsample_only(raw_rggb: np.ndarray, tw: int, th: int) -> np.ndarray:
    """
    覆盖缩放 + 居中裁剪（只下采样）/ Fit-to-cover + center-crop (downsample only).
    """
    assert raw_rggb.ndim == 3 and raw_rggb.shape[2] == 4, f"Expect (H,W,4), got {raw_rggb.shape}"
    h, w, _ = raw_rggb.shape

    if h < th or w < tw:
        raise RuntimeError(
            f"Input too small for downsample-only cover: got ({h},{w}), need >= ({th},{tw})."
        )

    scale = max(tw / w, th / h)
    new_w = _floor_even(int(round(w * scale)))
    new_h = _floor_even(int(round(h * scale)))

    if new_w < tw:
        new_w = tw
    if new_h < th:
        new_h = th

    if new_w < w or new_h < h:
        resized = _resize_rggb_downsample(raw_rggb, new_w, new_h)
    else:
        resized = raw_rggb

    y0 = (new_h - th) // 2
    x0 = (new_w - tw) // 2
    y1 = y0 + th
    x1 = x0 + tw
    cropped = resized[y0:y1, x0:x1, :]

    if cropped.shape[0] != th or cropped.shape[1] != tw:
        cropped_fixed = np.zeros((th, tw, 4), dtype=cropped.dtype)
        hh = min(th, cropped.shape[0])
        ww = min(tw, cropped.shape[1])
        cropped_fixed[:hh, :ww, :] = cropped[:hh, :ww, :]
        cropped = cropped_fixed

    return cropped.astype(np.float32)

# ---------------------------
# 数据目录 / Data directories
# ---------------------------
BASE_DIR   = './MDRAW'
RESULT_DIR = './results/MDRAW'

# 只处理 unpaired
cameras = ['huawei/', 'nikon/', 'canon/', 'iphone/', 'samsung/']

postfix_map = {
    'huawei/': '_A',
    'nikon/':  '_B',
    'iphone/': '_C',
    'samsung/':'_D',
    'canon/':  '_E'
}

# 相机元数据（白电平/黑电平/CFA）/ Camera metadata (WL/BL/CFA)
meta_data_huawei  = {'white_level': 4095,  'black_level': 256,  'cfa_pattern': np.array([3, 1, 2, 0])}
meta_data_nikon   = {'white_level': 16383, 'black_level': 1008, 'cfa_pattern': np.array([0, 1, 2, 3])}
meta_data_iphone  = {'white_level': 4095,  'black_level': 528,  'cfa_pattern': np.array([3, 1, 2, 0])}
meta_data_samsung = {'white_level': 1023,  'black_level': 64,   'cfa_pattern': np.array([2, 0, 3, 1])}
meta_data_canon   = {'white_level': 14274, 'black_level': 2046, 'cfa_pattern': np.array([2, 0, 3, 1])}

camera_meta = {
    'huawei/':  meta_data_huawei,
    'nikon/':   meta_data_nikon,
    'iphone/':  meta_data_iphone,
    'samsung/': meta_data_samsung,
    'canon/':   meta_data_canon
}

# 方向矫正 / Orientation per camera
ORIENT_PER_CAMERA = {
    'huawei/':  'none',
    'nikon/':   'rot180',
    'iphone/':  'none',
    'samsung/': 'none',
    'canon/':   'rot180'
}

def get_out_dirs(cam_tag):
    out_rggb_dir = os.path.join(RESULT_DIR, 'unpaired', cam_tag, 'raw-rggb')
    out_vis_dir  = os.path.join(RESULT_DIR, 'unpaired', cam_tag, 'vis')
    return out_rggb_dir, out_vis_dir

# ---------------------------
# 主流程 / Main pipeline
# ---------------------------
for _cam in cameras:
    postfix = postfix_map[_cam]

    path_to_raw_data = os.path.join(BASE_DIR, 'unpaired', _cam)
    in_dir = os.path.join(path_to_raw_data, 'raw')
    if not os.path.isdir(in_dir):
        print(f'[WARN] Skip: directory not found: {in_dir}')
        continue

    out_rggb_dir, out_vis_dir = get_out_dirs(_cam)
    check_dir(out_rggb_dir)
    check_dir(out_vis_dir)

    all_raw_img_paths = [
        os.path.join(in_dir, f) for f in os.listdir(in_dir)
        if os.path.splitext(f)[1].lower() in ('.dng', '.nef', '.cr2', '.cr3', '.arw', '.rw2')
    ]
    all_raw_img_paths.sort()

    meta_data = camera_meta[_cam]
    pos_order = deepcopy(meta_data['cfa_pattern'])

    for raw_img_path in all_raw_img_paths:
        try:
            with rawpy.imread(raw_img_path) as raw:
                data = raw.raw_image_visible.copy()
        except Exception as e:
            print(f'[ERROR] Failed to read: {raw_img_path} -> {e}')
            continue

        if data.ndim != 2:
            print(f'[WARN] Not a 2D Bayer mosaic, skipped: {raw_img_path} (shape={data.shape})')
            continue

        raw_bayer = data

        wl = float(meta_data['white_level'])
        bl = float(meta_data['black_level'])
        denom = max(wl - bl, 1.0)
        raw_bayer_norm = (raw_bayer.astype(np.float32) - bl) / denom
        raw_bayer_norm = np.clip(raw_bayer_norm, 0.0, 1.0)

        raw_rggb = pack_rggb(raw_bayer_norm, deepcopy(pos_order))
        raw_rggb = apply_orient(raw_rggb, ORIENT_PER_CAMERA.get(_cam, 'none'))

        try:
            raw_rggb_std = fit_cover_center_crop_downsample_only(
                raw_rggb, tw=TARGET_W, th=TARGET_H
            )
        except RuntimeError as e:
            print(f'[WARN] Skip (too small, no upsample): {raw_img_path} -> {e}')
            continue

        stem = os.path.splitext(os.path.basename(raw_img_path))[0]

        savemat(os.path.join(out_rggb_dir, stem + postfix + '.mat'),
                {"raw_rggb": raw_rggb_std.astype(np.float32)})

        vis_lin = np.clip(from_rggb_to_rgb(raw_rggb_std), 0.0, 1.0)
        vis = (vis_lin * 0.9) ** (1 / 1.6)
        vis = np.nan_to_num(vis, nan=0.0, posinf=1.0, neginf=0.0)
        imwrite(os.path.join(out_vis_dir, stem + postfix + '.jpg'), vis)

        print(stem)