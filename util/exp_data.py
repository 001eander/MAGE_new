"""Frozen Tiny-ImageNet subsets and shared eval/train defaults.

Larger n is a prefix of smaller n (nested). Image A is n=1 and half-A-half-B A.
Image B is the second n=2 image and half-A-half-B B.
"""
import os

from PIL import Image
from torch.utils.data import Dataset

N_VALUES = (1, 2, 10, 100, 1000, 10000)
SPLIT_SEED = 0
IMAGE_A = 'train/n01443537/images/n01443537_0.JPEG'
IMAGE_B = 'train/n01629819/images/n01629819_0.JPEG'
SOURCE_NAME = 'tiny-imagenet-200'
SOURCE_NATIVE_SIZE = 64
TRAIN_EVAL_SIZE = 256
FID_COMPARABLE_TO_PAPER = False

# Official class-unconditional command in README.official.md.
DEFAULT_TEMPERATURE = 6.0
DEFAULT_NUM_ITER = 20
DEFAULT_ARGMAX = False
DEFAULT_GEN_SEED = 0
APPROX_COVER_MAX_ERRORS = 16
GRID_N = 8


def default_num_images(n):
    n = int(n)
    if n >= 10000:
        return 50000
    return max(1000, 50 * n)


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def default_splits_dir():
    return os.path.join(repo_root(), 'splits')


def default_source_root():
    return os.path.join(repo_root(), 'data', 'tiny-imagenet-200')


def default_ref_dir():
    return os.path.join(repo_root(), 'data', 'tiny-imagenet-val256')


def list_path(n, splits_dir=None):
    splits_dir = splits_dir or default_splits_dir()
    return os.path.join(splits_dir, 'n{}.txt'.format(int(n)))


def read_list(path):
    paths = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                paths.append(line.replace('\\', '/'))
    return paths


def load_n_list(n, splits_dir=None):
    return read_list(list_path(n, splits_dir))


def resolve_paths(rel_paths, source_root=None):
    source_root = source_root or default_source_root()
    return [os.path.join(source_root, p) for p in rel_paths]


def train_eval_transform(input_size=TRAIN_EVAL_SIZE):
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])


class ImageListDataset(Dataset):
    """RGB images from a frozen relative-path list. Label is unused (always 0)."""

    def __init__(self, rel_paths, source_root=None, transform=None):
        self.rel_paths = list(rel_paths)
        self.source_root = source_root or default_source_root()
        self.paths = resolve_paths(self.rel_paths, self.source_root)
        self.transform = transform
        missing = [p for p in self.paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                'missing {} image(s), first={}'.format(len(missing), missing[0]))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def generation_defaults(n):
    return {
        'temperature': DEFAULT_TEMPERATURE,
        'num_iter': DEFAULT_NUM_ITER,
        'argmax': DEFAULT_ARGMAX,
        'num_images': default_num_images(n),
        'seed': DEFAULT_GEN_SEED,
    }


def source_info():
    return {
        'dataset': SOURCE_NAME,
        'native_resolution': SOURCE_NATIVE_SIZE,
        'train_eval_resolution': TRAIN_EVAL_SIZE,
        'upsample': 'Resize({}, BICUBIC)+CenterCrop({})'.format(
            TRAIN_EVAL_SIZE, TRAIN_EVAL_SIZE),
        'fid_comparable_to_paper': FID_COMPARABLE_TO_PAPER,
        'note': (
            'Tiny-ImageNet originals are 64x64 and are upsampled to 256 before '
            'train/eval. FID on this set must not be compared to paper ImageNet-256.'
        ),
    }
