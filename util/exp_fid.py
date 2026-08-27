"""FID of two image folders, same method as LGANs-TT realdata.py.

That script (run.py still launches `python realdata.py`) verifies with:

    from pytorch_fid import fid_score
    fid_score.calculate_fid_given_paths([real_dir, fake_dir], 50, device, 2048)

Both generated-vs-ref and VQGAN-recon-vs-ref go through fid_two_folders().
"""
import os

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from util.pytorch_fid_inception import InceptionV3

# pytorch-fid / realdata.py defaults
FID_DIMS = 2048
FID_BATCH_SIZE = 50
IMAGE_EXTENSIONS = ('bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp')
IMPLEMENTATION = (
    'pytorch-fid calculate_fid_given_paths (TF Inception 2015-12-05, dims=2048); '
    'same function as LGANs-TT realdata.py; used for both generated and VQGAN-recon FID'
)


class ImagePathDataset(Dataset):
    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB')
        return ToTensor()(img)


def list_image_files(folder):
    files = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower().lstrip('.')
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, name))
    if not files:
        raise FileNotFoundError('no images under {}'.format(folder))
    return files


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """pytorch-fid calculate_frechet_distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError('Imaginary component {}'.format(np.max(np.abs(covmean.imag))))
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def build_inception(device, dims=FID_DIMS):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    return model


def features_from_folder(folder, device, batch_size=FID_BATCH_SIZE, num_workers=1,
                         extractor=None, dims=FID_DIMS):
    files = list_image_files(folder)
    if extractor is None:
        extractor = build_inception(device, dims)
    if batch_size > len(files):
        batch_size = len(files)
    loader = DataLoader(
        ImagePathDataset(files),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    feats = np.empty((len(files), dims))
    start = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = extractor(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            feats[start:start + pred.shape[0]] = pred
            start += pred.shape[0]
    return feats


def stats_from_features(feats):
    feats = np.asarray(feats, dtype=np.float64)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def fid_two_folders(dir_a, dir_b, device, batch_size=FID_BATCH_SIZE, num_workers=1,
                    extractor=None, dims=FID_DIMS):
    """Same function for generated-vs-ref and VQGAN-recon-vs-ref."""
    if extractor is None:
        extractor = build_inception(device, dims)
    fa = features_from_folder(dir_a, device, batch_size, num_workers, extractor, dims)
    fb = features_from_folder(dir_b, device, batch_size, num_workers, extractor, dims)
    mu1, s1 = stats_from_features(fa)
    mu2, s2 = stats_from_features(fb)
    return frechet_distance(mu1, s1, mu2, s2)
