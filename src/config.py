import argparse
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class DataConfig:
    path: str
    name: str
    batch_size: int


@dataclass
class NetConfig:
    name: str
    params: dict[str, Any]
    ckpt_path: str


@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    device: str
    init_method: str = "kmeans"


@dataclass
class ExperimentConfig:
    data: DataConfig
    net: NetConfig
    trainer: TrainerConfig
    wandb_mode: str
    tags: list[str]


def from_dict(cls, d: dict) -> ExperimentConfig:
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, val in d.items():
        if key in fieldtypes:
            ftype = fieldtypes[key]
            if hasattr(ftype, "__dataclass_fields__"):
                kwargs[key] = from_dict(ftype, val)
            else:
                kwargs[key] = val
    return cls(**kwargs)


@dataclass
class Args:
    cfg: str

    @classmethod
    def from_cli(cls) -> "Args":
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg", type=str, default="experiments/dummy.yaml")
        args = parser.parse_args()
        return Args(
            cfg=args.cfg,
        )


def parse_args() -> ExperimentConfig:
    args = Args.from_cli()
    with open(args.cfg) as file:
        cfg = from_dict(ExperimentConfig, yaml.safe_load(file))
    return cfg
