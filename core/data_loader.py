"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
import os
import random

from munch import Munch
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms


def listdir(dname):
    fnames = list(Path(dname).rglob('*.npy'))
    fnames.sort()
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        arr = np.load(str(fname))
        img = torch.from_numpy(arr).float()
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

class NpyFolder(data.Dataset):
    def __init__(self, root, transform=None, fixed_filenames=None):
        root = Path(root)
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        samples = []
        for cls in classes:
            cls_dir = root / cls
            for fn in cls_dir.rglob('*.npy'):
                name = fn.name
                if fixed_filenames is not None:
                    allow_list = fixed_filenames.get(cls, [])
                    if (name not in allow_list) and (Path(name).stem not in allow_list):
                        continue
                samples.append((fn, self.class_to_idx[cls]))
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        arr = np.load(str(path))               # shape (4, H, W) or similar
        img = torch.from_numpy(arr).float()    # convert to float tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples, self.targets = self._make_dataset(root)

    def _make_dataset(self, root):
        domains = sorted(os.listdir(root))
        fnames1, fnames2, labels = [], [], []
        for idx, domain in enumerate(domains):
            class_dir = os.path.join(root, domain)
            if not os.path.isdir(class_dir):
                continue
            files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            paths = [os.path.join(class_dir, f) for f in files]
            if len(paths) == 0:
                continue
            fnames1 += paths
            fnames2 += random.sample(paths, len(paths))
            labels += [idx] * len(paths)
        return list(zip(fnames1, fnames2)), labels

    def __getitem__(self, index):
        path1, path2 = self.samples[index]
        label = self.targets[index]
        arr1 = np.load(path1)
        arr2 = np.load(path2)
        img1 = torch.from_numpy(arr1).float()
        img2 = torch.from_numpy(arr2).float()
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label

    def __len__(self):
        return len(self.targets)

def _parse_pair_dir_name(dirname: str):
    parts = dirname.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid pair folder name: {dirname}. Expected 'src_tgt'.")
    return parts[0], parts[1]

class PairDirsPairedNpyDataset(data.Dataset):
    """
    目录结构（layout）:
      val_img_dir/
        src_tgt/
          src/*.npy
          tgt/*.npy
        ...

    返回（__getitem__ return）:
      imgs_dict: { src_domain: tensor(4,H,W), tgt_domain: tensor(4,H,W) }
      meta: { 'src': src_domain, 'tgt': tgt_domain, 'filename': fname }
    """
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Test root not found: {root}")

        samples = []
        for pair_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            try:
                src_domain, tgt_domain = _parse_pair_dir_name(pair_dir.name)
            except ValueError:
                continue
            src_dir = pair_dir / src_domain
            tgt_dir = pair_dir / tgt_domain
            if not (src_dir.is_dir() and tgt_dir.is_dir()):
                continue

            src_names = set(f.name for f in src_dir.glob('*.npy'))
            tgt_names = set(f.name for f in tgt_dir.glob('*.npy'))
            common = sorted(src_names & tgt_names)
            for fname in common:
                samples.append((src_domain, tgt_domain, src_dir/fname, tgt_dir/fname, fname))

        if not samples:
            raise RuntimeError(f"No paired .npy found under {root} with pairwise layout.")
        self.samples = samples

        self.pair_keys = [f"{s}->{t}" for (s, t, _, _, _) in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        src_domain, tgt_domain, p_src, p_tgt, fname = self.samples[idx]
        arr_src = np.load(str(p_src))  # (4,H,W)
        arr_tgt = np.load(str(p_tgt))
        img_src = torch.from_numpy(arr_src).float()
        img_tgt = torch.from_numpy(arr_tgt).float()
        if self.transform is not None:
            img_src = self.transform(img_src)
            img_tgt = self.transform(img_tgt)
        imgs = {src_domain: img_src, tgt_domain: img_tgt}
        meta = {'src': src_domain, 'tgt': tgt_domain, 'filename': fname}
        return imgs, meta

def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4, fixed_filenames=None):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    if (which == 'source') and (fixed_filenames is not None):
        transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],
                                 std =[0.5, 0.5, 0.5, 0.5]),
        ])
    else:
        crop = transforms.RandomResizedCrop(
            img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
        rand_crop = transforms.Lambda(
            lambda x: crop(x) if random.random() < prob else x)

        transform = transforms.Compose([
            rand_crop,
            transforms.Resize([img_size, img_size]),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5]),
        ])

    if which == 'source':
        dataset = NpyFolder(root, transform, fixed_filenames)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    if (which == 'source') and (fixed_filenames is not None):
        return data.DataLoader(dataset=dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers,
                               pin_memory=True,
                               drop_last=True)
    else:
        sampler = _make_balanced_sampler(dataset.targets)
        return data.DataLoader(dataset=dataset,
                               batch_size=batch_size,
                               sampler=sampler,
                               num_workers=num_workers,
                               pin_memory=True,
                               drop_last=True)

from collections import defaultdict

def get_test_loader_pairdirs(root, img_size=256, batch_size=32,
                             shuffle=False, num_workers=4):
    """
    新版测试 Loader（pairwise folders + grouped batch sampler）
    - 保证每个 batch 只包含同一对 (src, tgt)
    - 返回 (imgs_dict, meta)，其中 imgs_dict 仅含该对域名的两键
    """
    print('Preparing DataLoader for test (pairwise folders, grouped batches, 4-ch npy)...')

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],
                             std =[0.5, 0.5, 0.5, 0.5]),
    ])
    ds = PairDirsPairedNpyDataset(root=root, transform=transform)

    pair2idxs = defaultdict(list)
    for idx, key in enumerate(ds.pair_keys):
        pair2idxs[key].append(idx)

    batches = []
    for key, idxs in pair2idxs.items():
        for i in range(0, len(idxs), batch_size):
            batches.append(idxs[i:i+batch_size])

    class _PairBatchSampler(torch.utils.data.Sampler):
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            for b in self.batches:
                yield b

        def __len__(self):
            return len(self.batches)

    def _collate(batch):
        src = batch[0][1]['src']
        tgt = batch[0][1]['tgt']
        src_imgs = [item[0][src] for item in batch]
        tgt_imgs = [item[0][tgt] for item in batch]
        imgs_dict = {
            src: torch.stack(src_imgs, 0),
            tgt: torch.stack(tgt_imgs, 0),
        }
        meta = {
            'src': [src]*len(batch),
            'tgt': [tgt]*len(batch),
            'filename': [item[1]['filename'] for item in batch],
        }
        return imgs_dict, meta

    return data.DataLoader(dataset=ds,
                        batch_sampler=_PairBatchSampler(batches),
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=_collate)

class InputFetcher:
    def __init__(self, loader, loader_ref=None, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
