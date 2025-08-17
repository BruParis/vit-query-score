# From mmaction repository
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor

from mmengine.config import ConfigDict

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]


def get_sinusoid_encoding(n_position: int, embed_dims: int) -> Tensor:
    """Generate sinusoid encoding table.

    Sinusoid encoding is a kind of relative position encoding method came from
    `Attention Is All You Need<https://arxiv.org/abs/1706.03762>`_.
    Args:
        n_position (int): The length of the input token.
        embed_dims (int): The position embedding dimension.
    Returns:
        :obj:`torch.FloatTensor`: The sinusoid encoding table of size
        (1, n_position, embed_dims)
    """

    vec = torch.arange(embed_dims, dtype=torch.float64)
    vec = (vec - vec % 2) / embed_dims
    vec = torch.pow(10000, -vec).view(1, -1)

    sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
    sinusoid_table[:, 0::2].sin_()  # dim 2i
    sinusoid_table[:, 1::2].cos_()  # dim 2i+1

    sinusoid_table = sinusoid_table.to(torch.float32)

    return sinusoid_table.unsqueeze(0)
