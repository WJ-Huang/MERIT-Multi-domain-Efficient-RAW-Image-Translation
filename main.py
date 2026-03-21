import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader, get_test_loader_pairdirs
from core.solver import Solver
from pathlib import Path

import wandb

FIXINPUT={
    'iphone': ['74.npy', '151.npy', '268.npy','389.npy','541.npy','2565.npy','3634.npy',],
    'samsung': ['66.npy', '124.npy', '205.npy','408.npy','857.npy','1024.npy','2058.npy',],
    'huawei': ['139.npy', '661.npy', '2091.npy','1920.npy','3209.npy','3513.npy','3797.npy'],
    'nikon': ['139.npy', '661.npy', '2091.npy','1920.npy','3209.npy','3513.npy','3797.npy'],
    'canon': ['51.npy', '79.npy', '329.npy','757.npy','1205.npy','1455.npy','1541.npy',]
}

def str2bool(v):
    return v.lower() in ('true')

def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))
            and not d.startswith('vis-')]

def getDomains(root_dir: str):
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid directory: {root_dir}")
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if args.use_wandb:
        wandb.init(
            project='StarGAN-R2R',
            entity='bias-lab',
            config=vars(args),
            name="odb_final"
        )

    solver = Solver(args)

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.sample_dir)) == args.num_domains
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        val=get_train_loader(root=args.sample_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             fixed_filenames=FIXINPUT))

        solver.test_loader = get_test_loader_pairdirs(
            root=args.val_img_dir,
            img_size=args.img_size,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        solver.train(loaders)
        if args.use_wandb:
            wandb.finish()
        return
    elif args.mode == 'eval':
        solver.test_loader = get_test_loader_pairdirs(
            root=args.val_img_dir,
            img_size=args.img_size,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        solver.test()
        if args.use_wandb:
            wandb.finish()
        return
    else:
        raise ValueError(f"Unknown mode: {args.mode!r}. Expected 'train' or 'eval'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=5,
                        help='Number of domains')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=10,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=5,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--lambda_noise', type=float, default=1.0,
                        help='Weight for noise consistency loss')
    parser.add_argument('--lambda_id', type=float, default=1.0,
                        help='Weight for id identity loss')
    parser.add_argument('--lambda_cyc_ssim', type=float, default=0.1)
    parser.add_argument('--lambda_id_ssim', type=float, default=0.05)
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=200000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'eval'],
                        help='This argument is used in solver')
    parser.add_argument('--best_model', type=bool, default=True,
                        help='Whether to load best model')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='./results/MDRAW/processed/unpaired',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='./results/MDRAW/processed/test',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='./results/MDRAW/processed/unpaired',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/expr/checkpoints',
                        help='Directory for saving network checkpoints')
    parser.add_argument('--noise_profile_paths', type=str,
                            default='./preprocess/noise_profiles/iphone_profile.pt,./preprocess/noise_profiles/samsung_profile.pt,./preprocess/noise_profiles/huawei_profile.pt,./preprocess/noise_profiles/nikon_profile.pt,./preprocess/noise_profiles/canon_profile.pt')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='./results/expr/checkpoints',
                        help='Directory for saving generated images and videos')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=5000)

    parser.add_argument('--use_wandb', action='store_true')

    parser.add_argument('--noise_patch', type=int, default=16)
    parser.add_argument('--noise_stride', type=int, default=16)
    parser.add_argument('--noise_keep_ratio', type=float, default=0.3)
    parser.add_argument('--noise_use_mad', type=str2bool, default=True)

    args = parser.parse_args()
    main(args)
