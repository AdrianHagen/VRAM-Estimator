from enum import Enum
import torch
from transformers import AdamW


class Optimizers(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


def get_optimizer(optimizer_enum, model_parameters, learning_rate=1e-3):
    """
    Maps an Optimizers enum value to a PyTorch optimizer.
    """
    if optimizer_enum == Optimizers.ADAM:
        return torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_enum == Optimizers.ADAMW:
        return AdamW(model_parameters, lr=learning_rate)
    elif optimizer_enum == Optimizers.SGD:
        return torch.optim.SGD(model_parameters, lr=learning_rate)
    elif optimizer_enum == Optimizers.ADAGRAD:
        return torch.optim.Adagrad(model_parameters, lr=learning_rate)
    elif optimizer_enum == Optimizers.ADAMAX:
        return torch.optim.Adamax(model_parameters, lr=learning_rate)
    elif optimizer_enum == Optimizers.RMSPROP:
        return torch.optim.RMSprop(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_enum}")


def get_memory_factor(optimizer_enum):
    if optimizer_enum == Optimizers.ADAM:
        return 2
    elif optimizer_enum == Optimizers.ADAMW:
        return 2
    elif optimizer_enum == Optimizers.SGD:
        return 1
    return 1
