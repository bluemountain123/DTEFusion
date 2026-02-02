import yaml
from os import path
from dataclasses import dataclass
from typing import Optional, Tuple, Iterator, Dict
from importlib import import_module

import torch
from torch.nn import Parameter, Module
from torch.utils.data import DataLoader, ConcatDataset

from dataset import VIFDataset, read_and_split, ImageLabels

class ModelConfig:
    def __init__(self, target: str, params: dict, pretrained: Optional[str]=None) -> None:
        self.target: str = target
        self.params = {}
        for k, v in params.items():
            if self.check(v):
                self.params[k] = ModelConfig(**v)
            else:
                self.params[k] = v
        self.pretrained: Optional[str] = pretrained

    @staticmethod
    def check(param: dict) -> bool:
        if type(param) is dict:
            if {'target', 'params'}.issubset(param.keys()):
                return True
        return False

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(target={self.target}, params={self.params}, pretrained={self.pretrained})'
    
class TrainArgs:
    def __init__(self,
        epochs: int,
        accumulation_steps: int,
        optimizer: dict,
    ) -> None:
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.optimizer = ModelConfig(**optimizer)
    def __repr__(self) -> str:
        return '\n'.join([
            'TrainArgs(',
            f'epochs={self.epochs}',
            f'accumulation_steps={self.accumulation_steps}',
            f'optimizer={self.optimizer}'
        ])
    

@dataclass
class DatasetArgs:
    url: Dict[str, str]
    batch_size: int
    shuffle: bool
    ratio: Optional[Tuple[float, float] | Tuple[float, float, float]] = None

@dataclass
class TrainConfig:
    model: ModelConfig
    train: TrainArgs
    dataset: DatasetArgs


def load_config(file: str='/home/raytrack/Fusion/CMfused11/config/model.yaml') -> TrainConfig:
    config_path = path.abspath(file)
    print(f"Config file path being used: {config_path}")

    assert path.isfile(file), FileNotFoundError
    with open(file) as f:
        config = yaml.full_load(f)
        model = ModelConfig(**config['model'])
        train = TrainArgs(**config['train'])
        dataset = DatasetArgs(**config['dataset'])
        return TrainConfig(model=model, train=train, dataset=dataset)

def load_model(config: ModelConfig, device: torch.device):
    kwargs = {}
    module_name, cls_name = config.target.rsplit('.', 1)
    print(module_name)
    print(cls_name)
    module = import_module(module_name)
    cls = getattr(module, cls_name)

    # Directly pass parameters to model constructor
    for k, v in config.params.items():
        kwargs[k] = load_model(v, device) if isinstance(v, ModelConfig) else v

    model: Module = cls(**kwargs)
    if config.pretrained is not None and path.isfile(config.pretrained) and hasattr(model, 'load_state_dict'):
        print(f'Loading pretrained weights from {config.pretrained}')
        state_dict = torch.load(config.pretrained)
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)

def load_dataset(config: DatasetArgs):
    # msrs_labels: ImageLabels = read_and_split(config.url['MSRS'], ratios=(0.8, 0.2))
    # Road_labels: ImageLabels = read_and_split(config.url['Road'], ratios=(0.8, 0.2))
    LLVIP_labels: ImageLabels = read_and_split(config.url['LLVIP'], ratios=(0.8, 0.2))

    # msrs_train = VIFDataset(msrs_labels.train)
    # Road_train = VIFDataset(Road_labels.train)
    LLVIP_train = VIFDataset(LLVIP_labels.train)

    # Create the trainset
    trainset = DataLoader(
        ConcatDataset([LLVIP_train]),
        batch_size=config.batch_size,
        shuffle=config.shuffle
    )

    # Handle validation data
    LLVIP_valid = None
    validset = None

    # if msrs_labels.valid is not None:
    #     msrs_valid = VIFDataset(msrs_labels.valid)
    # if Road_labels.valid is not None:
    #     Road_valid = VIFDataset(Road_labels.valid)
    if LLVIP_labels.valid is not None:
        LLVIP_valid = VIFDataset(LLVIP_labels.valid)

    # If all validation sets exist, create validset
    if LLVIP_valid:
        validset = DataLoader(
            ConcatDataset([LLVIP_valid]),
            batch_size=config.batch_size,
            shuffle=config.shuffle
        )

    return [trainset, validset] if validset else [trainset]


def load_optimizer(config: ModelConfig, params: Iterator[Parameter]):
    module_name, cls_name = config.target.rsplit('.', 1)
    module = import_module(module_name)
    optim = getattr(module, cls_name)
    return optim(params, **config.params)