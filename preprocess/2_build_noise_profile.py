import os, argparse, numpy as np, torch
from tqdm import tqdm

def list_npy(dir_):
    return [os.path.join(dir_, f) for f in os.listdir(dir_) if f.endswith('.npy')]

def load_npy(path):
    return torch.from_numpy(np.load(path)).float()

def sobel_grad_mag(x):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    x4 = x.unsqueeze(0)
    gx = torch.nn.functional.conv2d(x4, kx.repeat(x4.shape[1],1,1,1), padding=1, groups=x4.shape[1])
    gy = torch.nn.functional.conv2d(x4, ky.repeat(x4.shape[1],1,1,1), padding=1, groups=x4.shape[1])
    g = torch.sqrt(gx*gx + gy*gy).squeeze(0)
    return g

def unfold_patches(x, patch, stride):
    p = x.unfold(1, patch, stride).unfold(2, patch, stride)
    c, nh, ph, nw, pw = p.shape
    p = p.contiguous().view(c, nh*nw, ph*pw)
    return p

def patch_mad_variance(patches):
    med = patches.median(dim=-1, keepdim=True).values
    mad = (patches - med).abs().median(dim=-1).values
    sigma = mad / 0.67448975
    var = sigma * sigma
    return var

def compute_grad_thresholds(files, patch, stride, keep_ratio, device, sample_ratio=1.0):
    per_ch = []
    ch_n = None
    with torch.no_grad():
        for p in tqdm(files, desc='scan_grad'):
            x = load_npy(p).to(device)
            if ch_n is None: ch_n = x.shape[0]
            g = sobel_grad_mag(x)
            gp = unfold_patches(g, patch, stride).mean(dim=-1).cpu()
            if sample_ratio < 1.0 and gp.shape[1] > 0:
                k = max(1, int(gp.shape[1] * sample_ratio))
                idx = torch.randperm(gp.shape[1])[:k]
                gp = gp[:, idx]
            per_ch.append(gp)
    if len(per_ch) == 0:
        return torch.zeros(ch_n)
    per_ch = torch.cat(per_ch, dim=1)
    q = torch.tensor([keep_ratio], dtype=torch.float32)
    th = torch.quantile(per_ch, q.to(per_ch.dtype), dim=1).squeeze(-1)
    return th

def fill_and_smooth(profile_1d, counts_1d, smooth_passes=1):
    y = profile_1d.clone().float()
    x = torch.arange(y.numel(), dtype=torch.float32)
    nz = (counts_1d > 0).nonzero(as_tuple=False).squeeze(-1)
    if nz.numel() == 0:
        return y
    if nz.numel() == 1:
        y[:] = y[nz[0]]
    else:
        xi = x
        xp = nz.float()
        fp = y[nz]
        y = torch.from_numpy(np.interp(xi.numpy(), xp.numpy(), fp.numpy())).to(y.dtype)
    if smooth_passes > 0:
        k = torch.tensor([1.,2.,1.]).view(1,1,-1) / 4.
        yv = y.view(1,1,-1)
        for _ in range(smooth_passes):
            yv = torch.nn.functional.pad(yv, (1,1), mode='replicate')
            yv = torch.nn.functional.conv1d(yv, k)
        y = yv.view(-1)
    return y

def build_profile(files, bins_n, patch, stride, keep_ratio, device, smooth_passes=1, sample_ratio=1.0):
    with torch.no_grad():
        x0 = load_npy(files[0]).to(device)
        C = x0.shape[0]
        edges = torch.linspace(0,1,bins_n+1, device=device)
        centers = (edges[:-1] + edges[1:]) / 2
        sum_vars = torch.zeros(C, bins_n, device=device)
        counts   = torch.zeros(C, bins_n, device=device)
        th = compute_grad_thresholds(files, patch, stride, keep_ratio, device, sample_ratio=sample_ratio).to(device)
        for p in tqdm(files, desc='accumulate'):
            x = load_npy(p).to(device)
            patches = unfold_patches(x, patch, stride)
            means = patches.mean(dim=-1)
            vars_ = patch_mad_variance(patches)
            g = sobel_grad_mag(x)
            gmean = unfold_patches(g, patch, stride).mean(dim=-1)
            mask = (gmean <= th.view(-1,1))
            means = torch.where(mask, means, torch.full_like(means, -1.0))
            vars_ = torch.where(mask, vars_, torch.zeros_like(vars_))
            idx = torch.bucketize(means, edges, right=True) - 1
            for c in range(C):
                valid = (idx[c] >= 0) & (idx[c] < bins_n) & mask[c]
                if valid.any():
                    ii = idx[c, valid].long()
                    sum_vars[c].index_add_(0, ii, vars_[c, valid])
                    counts[c].index_add_(0, ii, torch.ones(ii.numel(), dtype=counts.dtype, device=counts.device))
        prof = sum_vars / (counts + 1e-8)
        out = []
        for c in range(C):
            y = fill_and_smooth(prof[c].detach().cpu(), counts[c].detach().cpu(), smooth_passes=smooth_passes)
            out.append(y.unsqueeze(0))
        prof_filled = torch.cat(out, dim=0)
        return {
            'bins': centers.detach().cpu().float(),
            'profiles': prof_filled.float(),
            'counts': counts.cpu().float(),
            'meta': {
                'patch': patch, 'stride': stride, 'keep_ratio': keep_ratio,
                'smooth_passes': smooth_passes, 'sample_ratio': sample_ratio,
                'method': 'per-channel; MAD variance; flat-patch via Sobel percentile; interp+smooth'
            }
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--bins', type=int, default=100)
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--keep_ratio', type=float, default=0.3)
    ap.add_argument('--smooth_passes', type=int, default=1)
    ap.add_argument('--sample_ratio', type=float, default=1.0)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    files = list_npy(args.img_dir)
    if not files:
        raise RuntimeError('no npy files found')
    prof = build_profile(files, args.bins, args.patch, args.stride, args.keep_ratio, args.device,
                         smooth_passes=args.smooth_passes, sample_ratio=args.sample_ratio)
    torch.save(prof, args.out)
    print('saved to', args.out, '| shape:', tuple(prof['profiles'].shape))

if __name__ == '__main__':
    main()