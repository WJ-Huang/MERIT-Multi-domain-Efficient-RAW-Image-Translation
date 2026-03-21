import os
import torch
import scipy.io
import argparse
from PIL import Image
import numpy as np
from torch import pixel_shuffle
import cv2


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


CAMERAS = ['iphone', 'huawei', 'nikon', 'samsung', 'canon']
RAW_VAR_CANDIDATES = ['raw_rggb']


def split_large_image(**kwargs):
    image, type = kwargs['image'], kwargs['type']
    if type == 'split':
        size = kwargs['patch_size']
        height, width = image.shape[:2]
        num_cols = width // size
        num_rows = height // size
        small_images = []
        for row in range(num_rows):
            for col in range(num_cols):
                left = col * size
                top = row * size
                right = left + size
                bottom = top + size
                small_image = image[top:bottom, left:right, ...]
                small_images.append(small_image)
        return small_images
    elif type == 'zoom':
        processed_image_size = kwargs['processed_image_size']
        return [cv2.resize(image, processed_image_size)]


def depth_to_space(feature_maps):
    if not torch.is_tensor(feature_maps):
        feature_maps = torch.tensor(feature_maps)
    feature_maps = pixel_shuffle(feature_maps, 2)
    feature_maps = np.array(feature_maps.to('cpu'))
    return feature_maps


def load_raw_rggb_from_mat(mat_path):
    """从 .mat 里取出 (H,W,4) 的 float32, [0,1] 的原始 RGGB"""
    data = scipy.io.loadmat(mat_path)
    for key in RAW_VAR_CANDIDATES:
        if key in data:
            arr = data[key]
            if hasattr(arr, 'dtype') and arr.dtype == object and arr.size == 1:
                arr = arr.item()
            arr = np.asarray(arr)
            if arr.ndim != 3 or arr.shape[2] != 4:
                raise ValueError(f"[ERROR] Variable '{key}' must be (H,W,4). Got {arr.shape} in {mat_path}")
            return arr.astype(np.float32)
    raise KeyError(f"[ERROR] None of {RAW_VAR_CANDIDATES} found in {mat_path}. Keys: {list(data.keys())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['zoom', 'split'], default='split')
    parser.add_argument('--processed_image_size', type=list, default=[512, 384])
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--parent_path', default='./results/MDRAW')
    parser.add_argument('--unpaired_data_path', default='unpaired')
    parser.add_argument('--processed_data_path', default='processed')
    parser.add_argument('--raw_path', default='raw-rggb')
    args = parser.parse_args()

    args.unpaired_data_path = os.path.join(args.parent_path, args.unpaired_data_path)
    args.processed_data_path = os.path.join(args.parent_path, args.processed_data_path)

    # 输出目录
    for cam in CAMERAS:
        ensure_dir(os.path.join(args.processed_data_path, 'unpaired', cam))
        ensure_dir(os.path.join(args.processed_data_path, 'unpaired', f'vis-{cam}'))

    # -----------------------------
    # 读取 {parent}/unpaired/{camera}/raw-rggb/*.mat 结构
    # -----------------------------
    for cam in CAMERAS:
        in_dir = os.path.join(args.unpaired_data_path, cam, args.raw_path)
        if not os.path.isdir(in_dir):
            print(f'[WARN] unpaired input dir not found: {in_dir}')
            continue

        raws = [f for f in os.listdir(in_dir) if f.lower().endswith('.mat')]
        raws.sort()
        print(f'[INFO] Unpaired {cam}: {len(raws)} files')

        index = 0
        for raw in raws:
            in_path = os.path.join(in_dir, raw)
            try:
                camera_raw = load_raw_rggb_from_mat(in_path)  # (H,W,4), [0,1]
            except Exception as e:
                print(f'[ERROR] Read MAT failed: {in_path} -> {e}')
                continue

            camera_raw_splits = split_large_image(
                image=camera_raw,
                patch_size=args.patch_size,
                type=args.type,
                processed_image_size=args.processed_image_size
            )

            for camera_raw_split in camera_raw_splits:
                camera_raw_split = camera_raw_split.transpose(2, 0, 1)
                np.save(
                    os.path.join(args.processed_data_path, 'unpaired', cam, f'{index}.npy'),
                    camera_raw_split
                )

                vis = depth_to_space(camera_raw_split).squeeze(0)
                image = Image.fromarray((np.clip(vis, 0.0, 1.0) * 255).astype(np.uint8))
                image.save(
                    os.path.join(args.processed_data_path, 'unpaired', f'vis-{cam}', f'{index}.png')
                )

                index += 1

            print(f'[INFO][unpaired] cam={cam} file={raw} patches={len(camera_raw_splits)}')