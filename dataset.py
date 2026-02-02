import os, random, math
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image

from torch import Tensor
# Used for conversion between PIL images and PyTorch tensors
PIL_to_Tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

Tensor_to_PIL = transforms.Compose([
    transforms.Lambda(lambda x: (x + 1) * 0.5),
    transforms.ToPILImage()
])

# Set random seed
random.seed('amane rinne')

class VIFDataset(Dataset):
    '''Infrared-Visible Fusion Dataset'''
    def __init__(self,
                labels: List[Dict[str, str]],
                *,
                vi_transform=None, ir_transform=None
    ) -> None:
        super().__init__()
        self.labels = labels
        self.image_size = (224, 224) # Unified size
        self.vi_transform = vi_transform or PIL_to_Tensor
        self.ir_transform = ir_transform or PIL_to_Tensor

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        item = self.labels[index]
        vi = Image.open(item['vi']).convert('L')
        ir = Image.open(item['ir']).convert('L')

        # Ensure image size is large enough
        if vi.size[0] < self.image_size[0] or vi.size[1] < self.image_size[1]:
            vi = vi.resize(self.image_size, Image.BICUBIC)
        if ir.size[0] < self.image_size[0] or ir.size[1] < self.image_size[1]:
            ir = ir.resize(self.image_size, Image.BICUBIC)

        box = self.random_area(vi.size) # Use random_area method to get a random crop area.
        with torch.no_grad():
            vi = self.vi_transform(vi)
            ir = self.ir_transform(ir)
            return F.crop(vi, *box), F.crop(ir, *box)
    
    def random_area(self, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        if image_size[0] < self.image_size[0] or image_size[1] < self.image_size[1]:
            return (0, 0, *self.image_size)
        i = random.randint(0, image_size[0] - self.image_size[0])
        j = random.randint(0, image_size[1] - self.image_size[1])
        return (i, j, *self.image_size)

    def __len__(self):
        return len(self.labels)
@dataclass
class ImageLabels:
    train: List[Dict[str, str]]
    valid: Optional[List[Dict[str, str]]] = None
    test: Optional[List[Dict[str, str]]] = None

def read_and_split(root: str, ratios: Tuple[float, float] | Tuple[float, float, float] | None):
   
    labels = parse_directory(root)
    if ratios is None: return ImageLabels(labels)
    assert sum(ratios) == 1, 'The sum of dataset split ratios must be 1'
    random.shuffle(labels)
    size = len(labels)
    train_offset = math.ceil(size * ratios[0])
    trainset = labels[:train_offset]
    if len(ratios) == 2:
        validset = labels[train_offset:]
        return ImageLabels(trainset, validset)
    else:
        valid_offset = math.floor(size * ratios[1]) + train_offset
        validset = labels[train_offset:valid_offset]
        testset = labels[valid_offset:]
        return ImageLabels(trainset, validset, testset)

def parse_directory(root: str) -> List[Dict[str, str]]:
    classes = os.listdir(root)
    print(f"Reading dataset from path: {root}")
    assert 'vi' in classes, 'Missing visible light dataset'
    assert 'ir' in classes, 'Missing infrared dataset'

    labels = []
    for filename in os.listdir(vipath := os.path.join(root, 'vi')):
        if os.path.isfile(irfile := os.path.join(root, 'ir', filename)):
            labels.append({
                'vi': os.path.join(vipath, filename),
                'ir': irfile
            })
    return labels

class Color:
    def __init__(self, kr = 0.299, kb = 0.114):
        self.kr = kr
        self.kb = kb

    def RGB_to_YCbCr(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r, g, b = image.chunk(3, dim=0)
        y = self.kr * r + (1 - self.kr - self.kb) * g + self.kb * b
        cb = 0.5 * (b - y)  / (1 - self.kb)
        cr = 0.5 * (r - y) / (1 - self.kr)
        return y, cb, cr

    def YCbCr_to_RGB(self, y: Tensor, cb: Tensor, cr: Tensor) -> Tensor:
        r = y + 2 * cr * (1 - self.kr)
        b = y + 2 * cb * (1 - self.kb)
        g = (y - self.kr * r - self.kb * b) / (1 - self.kb - self.kr)
        return torch.cat([r, g, b], dim=0)