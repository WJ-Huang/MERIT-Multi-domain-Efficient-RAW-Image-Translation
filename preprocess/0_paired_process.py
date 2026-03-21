#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import errno
import traceback
from copy import deepcopy
from itertools import combinations

import cv2
import numpy as np
import rawpy
from scipy.io import savemat

import torch
from kornia.feature import LoFTR

# ============ 全局参数 / Globals ============
TARGET_W = 2000
TARGET_H = 1500
PAD_VALUE = 0.0

RANSAC_REPROJ = 4.0          # RANSAC 重投影阈值(px) / reprojection threshold
PATCH_RGGB = 256             # RGGB patch 输出大小 / patch size
NMS_MIN_DIST = int(0.8 * PATCH_RGGB)  # 空间NMS最小间距 / spatial NMS min-dist
LOFTR_PRETRAINED = 'outdoor' # 'outdoor' or 'indoor'
LOFTR_CONF_THRESH = 0.2      # 置信度阈值（越低匹配越多）/ lower => more candidates

# 你可以改回 cuda；这里默认 CPU 方便无卡环境跑通
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _floor_even(x: int) -> int:
    x = int(x)
    return x - (x % 2)

TARGET_W = _floor_even(TARGET_W)
TARGET_H = _floor_even(TARGET_H)

# ============ 基础工具 / Utils ============
def apply_orient(arr, mode):
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

def imwrite(filename, image_rgb01):
    image = np.nan_to_num(image_rgb01, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0) * 255.0
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(filename, image)
    if not ok:
        print(f"[ERROR] cv2.imwrite 失败：{filename}")

def cam_tag_to_name(tag: str) -> str:
    """'huawei/' -> 'huawei' ；便于命名输出目录 / normalize tag for folder names"""
    return tag.strip('/')

# ============ RGGB 打包/解包 / Bayer pack/unpack ============
def _cfa_to_pos_order(cfa):
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
        raise ValueError("Color-based CFA must contain 1×R, 2×G(1/3), 1×B.")
    g_pos = np.sort(g_pos)
    return np.array([int(r_pos[0]), int(g_pos[0]), int(g_pos[1]), int(b_pos[0])], dtype=int)

def pack_rggb(raw_bayer_2d, cfa):
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
    g = (rggb[..., 1] + rggb[..., 2]) / 2.0
    return np.stack([rggb[..., 0], g, rggb[..., 3]], axis=-1)

# ============ 下采样统一化（逐通道） / Downsample-only normalize ============
def _resize_rggb_downsample(raw_rggb: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    assert raw_rggb.ndim == 3 and raw_rggb.shape[2] == 4
    out = np.empty((new_h, new_w, 4), dtype=raw_rggb.dtype)
    for i in range(4):
        out[..., i] = cv2.resize(raw_rggb[..., i], (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out

def fit_cover_center_crop_downsample_only(raw_rggb: np.ndarray, tw: int, th: int) -> np.ndarray:
    assert raw_rggb.ndim == 3 and raw_rggb.shape[2] == 4
    h, w, _ = raw_rggb.shape
    if h < th or w < tw:
        raise RuntimeError(f"Input too small for downsample-only cover: got ({h},{w}), need >= ({th},{tw}).")
    scale = max(tw / w, th / h)
    new_w = _floor_even(int(round(w * scale)))
    new_h = _floor_even(int(round(h * scale)))
    if new_w < tw: new_w = tw
    if new_h < th: new_h = th
    if new_w < w or new_h < h:
        resized = _resize_rggb_downsample(raw_rggb, new_w, new_h)
    else:
        resized = raw_rggb
    y0 = (new_h - th) // 2
    x0 = (new_w - tw) // 2
    cropped = resized[y0:y0+th, x0:x0+tw, :]
    if cropped.shape[0] != th or cropped.shape[1] != tw:
        cropped_fixed = np.zeros((th, tw, 4), dtype=cropped.dtype)
        hh = min(th, cropped.shape[0]); ww = min(tw, cropped.shape[1])
        cropped_fixed[:hh, :ww, :] = cropped[:hh, :ww, :]
        cropped = cropped_fixed
    return cropped.astype(np.float32)

# ============ LoFTR + RANSAC / Matching ============
def build_loftr():
    matcher = LoFTR(pretrained=LOFTR_PRETRAINED).to(DEVICE).eval()
    return matcher

def loftr_match(grayA_u8: np.ndarray, grayB_u8: np.ndarray, matcher: LoFTR,
                conf_thresh: float = LOFTR_CONF_THRESH, ransac_reproj: float = RANSAC_REPROJ):
    """输入 uint8 灰度，输出：KeyPoints、候选匹配、RANSAC 内点 / Input gray uint8; return KPs, matches, inliers"""
    img0 = torch.from_numpy(grayA_u8).float()[None, None, ...] / 255.0
    img1 = torch.from_numpy(grayB_u8).float()[None, None, ...] / 255.0
    img0 = img0.to(DEVICE)
    img1 = img1.to(DEVICE)
    with torch.no_grad():
        out = matcher({'image0': img0, 'image1': img1})
    if ('keypoints0' not in out) or (out['keypoints0'].numel() == 0):
        return [], [], [], []
    kpts0 = out['keypoints0'].detach().cpu().numpy()
    kpts1 = out['keypoints1'].detach().cpu().numpy()
    conf  = out['confidence'].detach().cpu().numpy()

    sel = conf >= conf_thresh
    kpts0 = kpts0[sel]; kpts1 = kpts1[sel]; conf = conf[sel]
    kpsA = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in kpts0]
    kpsB = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in kpts1]
    good = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=float(1.0 - conf[i])) for i in range(len(kpts0))]

    if kpts0.shape[0] < 4:
        return kpsA, kpsB, good, []

    H, mask = cv2.findHomography(kpts0.astype(np.float32), kpts1.astype(np.float32),
                                 cv2.RANSAC, ransacReprojThreshold=ransac_reproj,
                                 maxIters=3000, confidence=0.999)
    if mask is None:
        return kpsA, kpsB, good, []
    inliers = [good[i] for i, keep in enumerate(mask.ravel().tolist()) if keep]
    return kpsA, kpsB, good, inliers

# ============ 空间NMS / Spatial NMS ============
def nms_by_center_distance(kps1, kps2, matches, min_dist=NMS_MIN_DIST):
    kept, centers1, centers2 = [], [], []
    r2 = float(min_dist * min_dist)
    for m in sorted(matches, key=lambda mm: mm.distance):
        x1, y1 = kps1[m.queryIdx].pt
        x2, y2 = kps2[m.trainIdx].pt
        ok1 = all((x1 - cx1)**2 + (y1 - cy1)**2 >= r2 for (cx1, cy1) in centers1)
        if not ok1: continue
        ok2 = all((x2 - cx2)**2 + (y2 - cy2)**2 >= r2 for (cx2, cy2) in centers2)
        if not ok2: continue
        kept.append(m); centers1.append((x1, y1)); centers2.append((x2, y2))
    return kept

# ============ 成对同步裁剪 / Pairwise crop ============
def crop_pair_shift_sync(arr1, cx1, cy1, arr2, cx2, cy2, size=PATCH_RGGB):
    H1, W1 = arr1.shape[:2]; H2, W2 = arr2.shape[:2]; s = int(size)
    if W1 < s or H1 < s or W2 < s or H2 < s:
        raise RuntimeError("Image smaller than patch size; check TARGET size.")
    x01 = int(round(cx1)) - s // 2; y01 = int(round(cy1)) - s // 2
    x02 = int(round(cx2)) - s // 2; y02 = int(round(cy2)) - s // 2
    dx_lo = max(-x01, -x02); dx_hi = min(W1 - s - x01, W2 - s - x02)
    dx = 0 if dx_lo <= 0 <= dx_hi else (dx_lo if abs(dx_lo) < abs(dx_hi) else dx_hi)
    dy_lo = max(-y01, -y02); dy_hi = min(H1 - s - y01, H2 - s - y02)
    dy = 0 if dy_lo <= 0 <= dy_hi else (dy_lo if abs(dy_lo) < abs(dy_hi) else dy_hi)
    def clamp(v, lo, hi): return lo if v < lo else (hi if v > hi else v)
    x1 = clamp(x01 + int(dx), 0, W1 - s); y1 = clamp(y01 + int(dy), 0, H1 - s)
    x2 = clamp(x02 + int(dx), 0, W2 - s); y2 = clamp(y02 + int(dy), 0, H2 - s)
    return arr1[y1:y1+s, x1:x1+s, :].copy(), arr2[y2:y2+s, x2:x2+s, :].copy()

# ============ 连线可视化 / Visualization ============
def draw_matches(rgb1_01, rgb2_01, kps1, kps2, matches, inliers_only=True):
    h1, w1 = rgb1_01.shape[:2]; h2, w2 = rgb2_01.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = (np.clip(rgb1_01, 0, 1) * 255).astype(np.uint8)
    canvas[:h2, w1:w1+w2] = (np.clip(rgb2_01, 0, 1) * 255).astype(np.uint8)
    color = (0, 255, 0) if inliers_only else (255, 180, 0)
    color_bgr = (color[2], color[1], color[0])
    for m in matches:
        x1, y1 = kps1[m.queryIdx].pt
        x2, y2 = kps2[m.trainIdx].pt
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2 + w1)), int(round(y2)))
        cv2.circle(canvas, p1, 3, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 3, color_bgr, -1, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, color_bgr, 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

BASE_DIR   = './MDRAW'
RESULT_DIR = './results/MDRAW'

cameras   = ['huawei/', 'nikon/', 'iphone/', 'samsung/', 'canon/']

postfix_map = {
    'huawei/': '_A',
    'nikon/':  '_B',
    'iphone/': '_C',
    'samsung/':'_D',
    'canon/':  '_E'
}

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

# 与你之前设定一致 / keep consistent with your previous orientation settings
ORIENT_PER_CAMERA = {
    'huawei/':  'none',
    'nikon/':   'rot180',
    'iphone/':  'none',
    'samsung/': 'none',
    'canon/':   'rot180'
}

RAW_EXTS = ('.dng', '.nef', '.cr2', '.cr3', '.arw', '.rw2')

def list_raw_by_stem(folder):
    table = {}
    raw_dir = os.path.join(folder, 'raw')
    if not os.path.isdir(raw_dir):
        print(f"[ERROR] RAW 目录不存在：{raw_dir}")
        return table
    for f in os.listdir(raw_dir):
        p = os.path.join(raw_dir, f)
        if not os.path.isfile(p): continue
        if os.path.splitext(f)[1].lower() in RAW_EXTS:
            stem = os.path.splitext(f)[0]
            table[stem] = p
    return table

# ============ 读RAW→标准化RGGB / RAW->std RGGB ============
def load_std_rggb(path, meta, cfa, cam_tag):
    with rawpy.imread(path) as raw:
        data = raw.raw_image_visible.copy()
    wl = float(meta['white_level']); bl = float(meta['black_level'])
    raw_bayer_norm = np.clip((data.astype(np.float32) - bl) / max(wl - bl, 1.0), 0.0, 1.0)
    rggb = pack_rggb(raw_bayer_norm, deepcopy(cfa))
    rggb = apply_orient(rggb, ORIENT_PER_CAMERA.get(cam_tag, 'none'))
    rggb_std = fit_cover_center_crop_downsample_only(rggb, TARGET_W, TARGET_H)
    return rggb_std

# ============ 每一对相机的处理逻辑 / Per-pair processing ============
def process_one_pair(pair_name,
                     stem_list,
                     idx_by_cam,
                     camA, camB,
                     matcher,
                     out_root_std_raw,
                     out_root_std_jpg):
    """对相机对 (camA, camB) 的所有 common stem 做匹配与裁剪 / process one camera pair"""
    nameA, nameB = cam_tag_to_name(camA), cam_tag_to_name(camB)
    postfixA, postfixB = postfix_map[camA], postfix_map[camB]
    metaA, metaB = camera_meta[camA], camera_meta[camB]
    cfaA = deepcopy(metaA['cfa_pattern']); cfaB = deepcopy(metaB['cfa_pattern'])

    # pair 专属输出目录 / per-pair output dirs
    pair_root = os.path.join(RESULT_DIR, 'paired', f'{nameA}_{nameB}')

    # 真正给后续脚本读取的 paired patch 目录
    out_patch_A = os.path.join(pair_root, nameA)
    out_patch_B = os.path.join(pair_root, nameB)

    debug_pair_root = os.path.join(RESULT_DIR, 'paired', '_debug', f'{nameA}_{nameB}')
    out_match_dir   = os.path.join(debug_pair_root, 'matches')
    out_patch_vis_A = os.path.join(debug_pair_root, f'vis_{nameA}')
    out_patch_vis_B = os.path.join(debug_pair_root, f'vis_{nameB}')
    out_patch_vis_AB = os.path.join(debug_pair_root, f'vis_{nameA}_{nameB}')

    for d in [pair_root, out_patch_A, out_patch_B,
            debug_pair_root, out_match_dir, out_patch_vis_A, out_patch_vis_B, out_patch_vis_AB]:
        check_dir(d)

    total_pairs = 0
    total_patch_pairs = 0
    for stem in stem_list:
        total_pairs += 1
        pathA = idx_by_cam[camA][stem]
        pathB = idx_by_cam[camB][stem]
        print(f"\n[PAIR {pair_name}] {stem}")

        # 读 & 标准化 / load & normalize (并缓存 std 可选)
        try:
            rggbA = load_std_rggb(pathA, metaA, cfaA, camA)
            rggbB = load_std_rggb(pathB, metaB, cfaB, camB)
        except Exception as e:
            print(f"[ERROR] {pair_name}:{stem}: 标准化失败 -> {e}")
            print(traceback.format_exc()); continue

        # —— 保存标准化整图（只存一次每相机每 stem）/ save std once per camera per stem ——
        # 这里仍然在公共 std 目录下按 postfix 区分，避免不同 pair 重复保存
        try:
            # .mat
            savemat(os.path.join(out_root_std_raw, stem + postfixA + '.mat'), {"raw_rggb": rggbA.astype(np.float32)})
            savemat(os.path.join(out_root_std_raw, stem + postfixB + '.mat'), {"raw_rggb": rggbB.astype(np.float32)})
            # jpg
            visA = (np.clip(from_rggb_to_rgb(rggbA), 0.0, 1.0) * 0.9) ** (1/1.6)
            visB = (np.clip(from_rggb_to_rgb(rggbB), 0.0, 1.0) * 0.9) ** (1/1.6)
            imwrite(os.path.join(out_root_std_jpg, stem + postfixA + '.jpg'), np.nan_to_num(visA))
            imwrite(os.path.join(out_root_std_jpg, stem + postfixB + '.jpg'), np.nan_to_num(visB))
        except Exception as e:
            # 非致命，仅报告 / non-fatal
            print(f"[WARN] {pair_name}:{stem}: 保存整图失败 -> {e}")

        # —— LoFTR 匹配 / LoFTR matching ——
        try:
            rgbA01 = np.clip(from_rggb_to_rgb(rggbA), 0.0, 1.0)
            rgbB01 = np.clip(from_rggb_to_rgb(rggbB), 0.0, 1.0)
            grayA = cv2.cvtColor((rgbA01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            grayB = cv2.cvtColor((rgbB01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            kpsA, kpsB, good, inliers = loftr_match(grayA, grayB, matcher,
                                                     conf_thresh=LOFTR_CONF_THRESH,
                                                     ransac_reproj=RANSAC_REPROJ)
            print(f"[INFO] {pair_name}:{stem}: loftr good={len(good)}, inliers={len(inliers)}")
        except Exception as e:
            print(f"[ERROR] {pair_name}:{stem}: LoFTR 匹配失败 -> {e}")
            print(traceback.format_exc()); continue

        # —— 可视化 / Visualization ——
        try:
            if len(good) > 0:
                vis_ratio = draw_matches(rgbA01, rgbB01, kpsA, kpsB, good, inliers_only=False)
                imwrite(os.path.join(out_match_dir, f"{stem}_ratio.png"), vis_ratio)
            if len(inliers) > 0:
                vis_inl = draw_matches(rgbA01, rgbB01, kpsA, kpsB, inliers, inliers_only=True)
                imwrite(os.path.join(out_match_dir, f"{stem}_inliers.png"), vis_inl)
        except Exception as e:
            print(f"[WARN] {pair_name}:{stem}: 保存连线图失败 -> {e}")

        if len(inliers) == 0:
            print(f"[WARN] {pair_name}:{stem}: 无 RANSAC 内点，跳过裁剪。")
            continue

        # —— NMS 去重 / Spatial NMS ——
        try:
            inliers_nms = nms_by_center_distance(kpsA, kpsB, inliers, min_dist=NMS_MIN_DIST)
            print(f"[INFO] {pair_name}:{stem}: NMS 后剩余 {len(inliers_nms)}")
            if len(inliers_nms) == 0:
                print(f"[WARN] {pair_name}:{stem}: NMS 后无匹配点，跳过裁剪。")
                continue
        except Exception as e:
            print(f"[ERROR] {pair_name}:{stem}: NMS 失败 -> {e}")
            print(traceback.format_exc()); continue

        # —— 裁剪与保存 / Crop & save patches ——
        saved = 0
        for i, m in enumerate(inliers_nms):
            xA, yA = kpsA[m.queryIdx].pt
            xB, yB = kpsB[m.trainIdx].pt
            try:
                patchA, patchB = crop_pair_shift_sync(rggbA, xA, yA, rggbB, xB, yB, size=PATCH_RGGB)
            except Exception as e:
                print(f"[WARN] {pair_name}:{stem}: 第 {i} 个内点裁剪失败：{e}")
                continue

            # 1) RGGB .mat
            try:
                savemat(os.path.join(out_patch_A, f"{stem}_p{i:04d}{postfixA}.mat"),
                        {"raw_rggb": patchA.astype(np.float32)})
                savemat(os.path.join(out_patch_B, f"{stem}_p{i:04d}{postfixB}.mat"),
                        {"raw_rggb": patchB.astype(np.float32)})
            except Exception as e:
                print(f"[ERROR] {pair_name}:{stem}: 保存 patch MAT 失败（idx={i}）-> {e}")
                print(traceback.format_exc()); continue

            # 2) 可视化 / preview PNG
            try:
                visA_p = (np.clip(from_rggb_to_rgb(patchA), 0.0, 1.0) * 0.9) ** (1/1.6)
                visB_p = (np.clip(from_rggb_to_rgb(patchB), 0.0, 1.0) * 0.9) ** (1/1.6)
                imwrite(os.path.join(out_patch_vis_A,  f"{stem}_p{i:04d}{postfixA}.png"), np.nan_to_num(visA_p))
                imwrite(os.path.join(out_patch_vis_B,  f"{stem}_p{i:04d}{postfixB}.png"), np.nan_to_num(visB_p))
                imwrite(os.path.join(out_patch_vis_AB, f"{stem}_p{i:04d}{postfixA}{postfixB}.png"),
                        np.nan_to_num(np.concatenate([visA_p, visB_p], axis=1)))
            except Exception as e:
                print(f"[WARN] {pair_name}:{stem}: 保存 patch 可视化失败（idx={i}）-> {e}")

            saved += 1

        total_patch_pairs += saved
        if saved == 0:
            print(f"[WARN] {pair_name}:{stem}: 未成功保存任何 {PATCH_RGGB}×{PATCH_RGGB} patch。")
        else:
            print(f"[OK] {pair_name}:{stem}: 成功保存 patch 对数 = {saved}")

    return total_pairs, total_patch_pairs

# ============ 主流程 / Main ============
def main():
    matcher = build_lofTR_safe()

    # 公共输出：标准化整图（所有 pair 共享）
    out_rggb_std_dir = os.path.join(RESULT_DIR, 'paired', 'std_raw')
    out_vis_dir      = os.path.join(RESULT_DIR, 'paired', 'std_jpg')
    for d in [out_rggb_std_dir, out_vis_dir]:
        check_dir(d)

    # 为每个相机读取索引 / build index per camera
    idx_by_cam = {}
    for cam in cameras:
        base = os.path.join(BASE_DIR, 'paired', cam)
        idx_by_cam[cam] = list_raw_by_stem(base)
        print(f"[INFO] {cam}: found {len(idx_by_cam[cam])} raws")

    # 生成所有相机对（无序组合）/ all unordered camera pairs
    cam_pairs = list(combinations(cameras, 2))
    if not cam_pairs:
        print("[ERROR] cameras 里不到 2 个域，无法配对。")
        return

    grand_pairs = 0
    grand_patches = 0

    for camA, camB in cam_pairs:
        nameA, nameB = cam_tag_to_name(camA), cam_tag_to_name(camB)
        pair_name = f"{nameA}_{nameB}"
        # 只处理两个域之间**同名 stem** 的图像 / intersect stems
        common_stems = sorted(set(idx_by_cam[camA].keys()) & set(idx_by_cam[camB].keys()))
        print(f"\n========== [{pair_name}] common stems = {len(common_stems)} ==========")
        if len(common_stems) == 0:
            print(f"[WARN] {pair_name}: 无同名 stem，跳过。")
            continue

        total_pairs, total_patch_pairs = process_one_pair(
            pair_name=pair_name,
            stem_list=common_stems,
            idx_by_cam=idx_by_cam,
            camA=camA, camB=camB,
            matcher=matcher,
            out_root_std_raw=out_rggb_std_dir,
            out_root_std_jpg=out_vis_dir
        )
        print(f"[SUM] {pair_name}: 处理对数={total_pairs}, 保存 patch 对数={total_patch_pairs}")
        grand_pairs += total_pairs
        grand_patches += total_patch_pairs

    print("\n========== 全部相机对汇总 / Summary over all pairs ==========")
    print(f"[SUM] 总计处理的相机对内样本数（sum over pairs）：{grand_pairs}")
    print(f"[SUM] 总计保存的 {PATCH_RGGB}×{PATCH_RGGB} RGGB patch 对数：{grand_patches}")
    if grand_pairs == 0:
        print("[HINT] 没有任何 pair 被处理：检查 cameras 列表与各自 raw 目录的同名文件。")
    if grand_patches == 0:
        print("[HINT] 没有任何 patch：可能因无内点、NMS 后为空、或裁剪越界。可尝试降低 LOFTR_CONF_THRESH、增大 RANSAC_REPROJ、调小 NMS_MIN_DIST。")

def build_lofTR_safe():
    try:
        return build_loftr()
    except Exception as e:
        print(f"[ERROR] 初始化 LoFTR 失败：{e}")
        print("请确认已安装: pip install kornia ；并与当前 torch 版本兼容。")
        raise

if __name__ == "__main__":
    main()
