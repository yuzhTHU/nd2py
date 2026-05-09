# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from .ndformer_config import NDFormerModelConfig
from .ndformer_tokenizer import NDFormerTokenizer
from .ndformer_generator import NDFormerEqtreeGenerator, NDFormerGraphGenerator, NDFormerDataGenerator
from ...utils.lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ndformer_dataset import NDFormerDataset
    from .ndformer_model import NDFormerModel
    from .ndformer_mcts import NDFormerMCTS
    from .ndformer_model_flash_ansr import FlashANSRNDFormer

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "NDFormerDataset": (".ndformer_dataset", "nn"),
    "NDFormerModel": (".ndformer_model", "nn"),
    "NDFormerMCTS": (".ndformer_mcts", "nn"),
    "FlashANSRNDFormer": (".ndformer_model_flash_ansr", "nn"),
})
