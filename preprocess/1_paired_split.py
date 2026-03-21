import os
import re
import argparse
import numpy as np
import scipy.io
from typing import List, Tuple, Dict
from PIL import Image
import cv2

# -------------------- Config & Patterns --------------------

RAW_VAR_CANDIDATES = ['raw_rggb']  # .mat variable candidates (e.g., RGGB 4-channels)
CAMERA_NAMES = ["huawei", "iphone", "nikon", "samsung", "canon"]

# Match ONE trailing suffix like: _A / -A / _iphone / -iphone (case-insensitive)
_SUFFIX_ONE_RE = re.compile(r"(?i)(?:[_-])(?:[A-E]|" + "|".join(CAMERA_NAMES) + r")$")


# -------------------- Utilities --------------------

def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def normalize_stem(filename: str) -> str:
    """
    Remove extension, then iteratively strip trailing camera/code suffixes
    like _A, -B, _iphone, -canon, etc., until none remain.
    Examples:
      '0001_A.mat'       -> '0001'
      'foo-iphone.MAT'   -> 'foo'
      'bar_A_iphone.mat' -> 'bar'
    """
    stem, _ = os.path.splitext(filename)
    while True:
        m = _SUFFIX_ONE_RE.search(stem)
        if not m:
            break
        stem = stem[:m.start()]  # strip the matched tail
    return stem

def load_raw_rggb_from_mat(mat_path: str) -> np.ndarray:
    """
    Load (H, W, 4) float32 RGGB array in [0,1] from a .mat file.
    """
    data = scipy.io.loadmat(mat_path)
    for key in RAW_VAR_CANDIDATES:
        if key in data:
            arr = data[key]
            if getattr(arr, 'dtype', None) == object and arr.size == 1:
                arr = arr.item()
            arr = np.asarray(arr)
            if arr.ndim != 3 or arr.shape[2] != 4:
                raise ValueError(
                    f"Variable '{key}' must be (H,W,4). Got {arr.shape} in {mat_path}"
                )
            return arr.astype(np.float32)
    raise KeyError(
        f"None of {RAW_VAR_CANDIDATES} found in {mat_path}. Keys: {list(data.keys())}"
    )

def split_or_resize(image: np.ndarray, mode: str, patch_size: int,
                    out_size_wh: Tuple[int, int]) -> List[np.ndarray]:
    """
    Either split into equal patches (mode='split') or resize to out_size_wh (mode='zoom').
    Returns a list of images.
    """
    if mode == 'split':
        H, W = image.shape[:2]
        ncols, nrows = W // patch_size, H // patch_size
        out = []
        for r in range(nrows):
            for c in range(ncols):
                y0, x0 = r * patch_size, c * patch_size
                out.append(image[y0:y0 + patch_size, x0:x0 + patch_size, :])
        return out
    elif mode == 'zoom':
        W, H = out_size_wh
        return [cv2.resize(image, (W, H))]
    else:
        return [image]

def rggb_depth_to_space_vis(x_4hw: np.ndarray) -> np.ndarray:
    """
    Quick visualization helper: (4,H,W) -> (2H,2W) grayscale mosaic.
    For sanity checks only.
    """
    R, G1, G2, B = x_4hw[0], x_4hw[1], x_4hw[2], x_4hw[3]
    H, W = R.shape
    out = np.zeros((H * 2, W * 2), dtype=np.float32)
    out[0::2, 0::2] = R
    out[0::2, 1::2] = G1
    out[1::2, 0::2] = G2
    out[1::2, 1::2] = B
    return out

def list_pair_folders(root_dir: str) -> List[Tuple[str, Tuple[str, str]]]:
    """
    Scan root_dir for subfolders that contain exactly two subdirectories.
    Treat each as a pair folder.
    Returns: [(pair_abs_path, (camA, camB)), ...]
    """
    out = []
    for name in sorted(os.listdir(root_dir)):
        pf = os.path.join(root_dir, name)
        if not os.path.isdir(pf):
            continue
        subdirs = [d for d in os.listdir(pf) if os.path.isdir(os.path.join(pf, d))]
        if len(subdirs) != 2:
            continue
        subdirs.sort()  # deterministic order
        out.append((pf, (subdirs[0], subdirs[1])))
    return out

def _stem_map(dir_path: str, debug: bool = False, label: str = "") -> Dict[str, str]:
    """
    Map *normalized* stem -> original filename for *.mat inside dir_path.
    Only compare by normalized stems.
    """
    out: Dict[str, str] = {}
    mats = [fn for fn in os.listdir(dir_path) if fn.lower().endswith(".mat")]
    for fn in mats:
        norm = normalize_stem(fn)
        # If multiple files collapse to the same normalized stem, keep the first.
        out.setdefault(norm, fn)
    if debug:
        sample = list(out.items())[:8]
        print(f"[DEBUG] {label} {dir_path} mats={len(mats)} normalized={len(out)} samples={sample}")
    return out


# -------------------- Core Conversion --------------------

def convert_pair_tree(paired_root: str, out_root: str, mode: str, patch_size: int,
                      out_size_wh: Tuple[int, int], save_vis: bool,
                      npy_subdir: str, vis_subdir: str, debug: bool = False):
    """
    Read pair-tree under `paired_root`, write outputs to mirrored tree under `out_root`.

    Inputs tree:
      paired_root/
        A_B/
          A/*.mat
          B/*.mat

    Outputs tree:
      out_root/
        A_B/
          A/{npy_subdir}/*.npy
          A/{vis_subdir}/*.png (if --save_vis)
          B/{npy_subdir}/*.npy
          B/{vis_subdir}/*.png (if --save_vis)
    """
    pairs = list_pair_folders(paired_root)
    if not pairs:
        print(f"[WARN] No valid pair folders under: {paired_root}")
        return

    print(f"[INFO] Found {len(pairs)} pair folders under {paired_root}")
    for pf_path, (camA, camB) in pairs:
        pair_name = os.path.basename(pf_path.rstrip(os.sep))

        # Input dirs
        A_dir_raw = os.path.join(pf_path, camA)
        B_dir_raw = os.path.join(pf_path, camB)

        # Output dirs (mirror under out_root)
        A_dir_npy = os.path.join(out_root, pair_name, camA, npy_subdir)
        B_dir_npy = os.path.join(out_root, pair_name, camB, npy_subdir)
        A_dir_vis = os.path.join(out_root, pair_name, camA, vis_subdir)
        B_dir_vis = os.path.join(out_root, pair_name, camB, vis_subdir)
        ensure_dir(A_dir_npy)
        ensure_dir(B_dir_npy)
        if save_vis:
            ensure_dir(A_dir_vis)
            ensure_dir(B_dir_vis)

        # Build normalized stem maps (compare only by normalized stems)
        A_map = _stem_map(A_dir_raw, debug=debug, label=f"{camA}")
        B_map = _stem_map(B_dir_raw, debug=debug, label=f"{camB}")
        common_stems = sorted(set(A_map.keys()) & set(B_map.keys()))

        print(f"[INFO][pair] {pair_name} | {camA} ↔ {camB} | common stems: {len(common_stems)}")

        for stem in common_stems:
            A_mat = os.path.join(A_dir_raw, A_map[stem])
            B_mat = os.path.join(B_dir_raw, B_map[stem])

            try:
                A_raw = load_raw_rggb_from_mat(A_mat)  # (H,W,4) in [0,1]
                B_raw = load_raw_rggb_from_mat(B_mat)
            except Exception as e:
                print(f"[ERROR] Read MAT failed: {stem} -> {e}")
                continue

            A_list = split_or_resize(A_raw, mode, patch_size, out_size_wh)
            B_list = split_or_resize(B_raw, mode, patch_size, out_size_wh)
            if len(A_list) != len(B_list):
                print(f"[ERROR] Mismatched counts for stem '{stem}': {len(A_list)} vs {len(B_list)}. Skipped.")
                continue

            if mode == 'split':
                for i, (Ai, Bi) in enumerate(zip(A_list, B_list)):
                    tag = f"{stem}_p{i:03d}"
                    a_np = Ai.transpose(2, 0, 1)  # (4,H,W)
                    b_np = Bi.transpose(2, 0, 1)
                    # Save NPY
                    np.save(os.path.join(A_dir_npy, f"{tag}.npy"), a_np)
                    np.save(os.path.join(B_dir_npy, f"{tag}.npy"), b_np)
                    # Save VIS (optional)
                    if save_vis:
                        va = rggb_depth_to_space_vis(a_np)
                        vb = rggb_depth_to_space_vis(b_np)
                        Image.fromarray((np.clip(va, 0, 1) * 255).astype(np.uint8)).save(os.path.join(A_dir_vis, f"{tag}.png"))
                        Image.fromarray((np.clip(vb, 0, 1) * 255).astype(np.uint8)).save(os.path.join(B_dir_vis, f"{tag}.png"))
                print(f"[INFO] Saved {len(A_list)} patches for stem '{stem}' "
                      f"to '{A_dir_npy}' and '{B_dir_npy}'.")
            else:
                a_np = A_list[0].transpose(2, 0, 1)
                b_np = B_list[0].transpose(2, 0, 1)
                # Save NPY
                np.save(os.path.join(A_dir_npy, f"{stem}.npy"), a_np)
                np.save(os.path.join(B_dir_npy, f"{stem}.npy"), b_np)
                # Save VIS (optional)
                if save_vis:
                    va = rggb_depth_to_space_vis(a_np)
                    vb = rggb_depth_to_space_vis(b_np)
                    Image.fromarray((np.clip(va, 0, 1) * 255).astype(np.uint8)).save(os.path.join(A_dir_vis, f"{stem}.png"))
                    Image.fromarray((np.clip(vb, 0, 1) * 255).astype(np.uint8)).save(os.path.join(B_dir_vis, f"{stem}.png"))
                print(f"[INFO] Saved 1 file for stem '{stem}' "
                      f"to '{A_dir_npy}' and '{B_dir_npy}'.")


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert paired .mat (RGGB) to .npy, writing to a mirrored tree under --out_root."
    )
    parser.add_argument(
        '--root',
        default='./results/MDRAW/paired',
        help='Input root of the pair-tree.'
    )
    parser.add_argument(
        '--out_root',
        default='./results/MDRAW/processed/test',
        help='Output root where the mirrored pair-tree will be written.'
    )
    parser.add_argument(
        '--type',
        choices=['split', 'zoom'],
        default='split',
        help="split: cut into equal patches; zoom: resize to --processed_image_size"
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=256,
        help='Patch size for split mode.'
    )
    parser.add_argument(
        '--processed_image_size',
        type=int,
        nargs=2,
        default=[512, 384],
        help='(W H) output size for zoom mode.'
    )
    parser.add_argument(
        '--save_vis',
        action='store_true',
        help='Also save quick PNG visualizations for sanity checks.'
    )
    parser.add_argument(
        '--npy_subdir',
        default='.',
        help='Subdirectory name (inside each camera folder) where .npy files are written.'
    )
    parser.add_argument(
        '--vis_subdir',
        default='vis',
        help='Subdirectory name (inside each camera folder) where visualization .png files are written.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print sample normalized stems and counts for each camera folder.'
    )
    args = parser.parse_args()

    in_root = os.path.abspath(args.root)
    out_root = os.path.abspath(args.out_root)

    if not os.path.isdir(in_root):
        raise FileNotFoundError(f"Input root not found: {in_root}")
    ensure_dir(out_root)

    convert_pair_tree(
        paired_root=in_root,
        out_root=out_root,
        mode=args.type,
        patch_size=args.patch_size,
        out_size_wh=tuple(args.processed_image_size),
        save_vis=args.save_vis,
        npy_subdir=args.npy_subdir,
        vis_subdir=args.vis_subdir,
        debug=args.debug
    )

if __name__ == '__main__':
    main()
