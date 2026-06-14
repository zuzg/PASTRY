from torch import nn

from src.architectures.superresolution import BicubicSR, ConvSR, NeuroSymbolicSR

PAVIA_CLASSES: list[str] = [
    "Undefined",
    "Water",
    "Trees",
    "Asphalt",
    "S-B Bricks",
    "Bitumen",
    "Tiles",
    "Shadows",
    "Meadows",
    "Bare Soil",
]

HOUSTON_CLASSES: list[str] = [
    "Unclassified",
    "Healthy grass",
    "Stressed grass",
    "Artificial turf",
    "Evergreen trees",
    "Deciduous trees",
    "Bare earth",
    "Water",
    "Residential buildings",
    "Non-residential buildings",
    "Roads",
    "Sidewalks",
    "Crosswalks",
    "Major thoroughfares",
    "Highways",
    "Railways",
    "Paved parking lots",
    "Unpaved parking lots",
    "Cars",
    "Trains",
    "Stadium seats",
]

HOUSTON_SMALL_CLASSES: list[str] = [
    "Undefined",
    "Grass healthy",
    "Grass unhealthy",
    "Trees",
    "Water",
    "Res. buildings",
    "Non-res. buildings",
    "Road",
]

MODELS_DICT: dict[str, nn.Module] = {
    "Bicubic": BicubicSR,
    "Conv": ConvSR,
    "Nesy": NeuroSymbolicSR,
}
